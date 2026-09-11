#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS One-Command Multi-Speaker Fine-Tuning Script

End-to-end pipeline:
1. Discover speakers from ./audio_chunks/<speaker>/ (or a flat dir for one speaker)
2. Load pre-computed transcripts from chunks.jsonl (or transcribe with NeMo Parakeet TDT if missing)
3. Filter bad transcripts, write train_raw.jsonl
4. Extract audio_codes -> train_with_codes.jsonl
5. Fine-tune, with per-speaker centroid speaker embeddings baked into the checkpoint

Usage:
    python fine_tune_qwen.py --audio_dir ./audio_chunks --output_dir ./output_multi
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def configure_hf_cache():
    """Configure HuggingFace cache inside env to keep model weights local."""
    script_dir = Path(__file__).parent.absolute()
    hf_cache = script_dir / "env" / "hf_cache"
    if hf_cache.parent.exists():
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache))


def get_attention_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


configure_hf_cache()

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Signature version -- bump to force regeneration of cached jsonl files.
PIPELINE_VERSION = 3

ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,?!'\"-:;()")


def sub_talker_loss_finetune(talker, codec_ids, talker_hidden_states):
    """Correctly aligned replacement for talker.forward_sub_talker_finetune.

    The library version passes labels=codec_ids[:, 1:] into a HuggingFace
    ForCausalLMLoss, which shifts labels internally. But the logits it builds are
    already aligned -- logits[:, j] = lm_head[j](hidden[:, j+1]) predicts codebook
    j+1. The extra shift means every head is trained against the *next* codebook's
    target, codebook 1 is never supervised, and codebook 15 is scored against -100.

    Here we request logits only (labels=None) and compute the cross-entropy against
    the already-aligned targets ourselves.
    """
    num_groups = talker.config.num_code_groups

    embeds = [talker_hidden_states.unsqueeze(1)]
    for i in range(num_groups - 1):
        if i == 0:
            embeds.append(talker.get_input_embeddings()(codec_ids[:, :1]))
        else:
            embeds.append(talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, i : i + 1]))
    inputs_embeds = torch.cat(embeds, dim=1)

    outputs = talker.code_predictor.forward_finetune(inputs_embeds=inputs_embeds, labels=None)
    logits = outputs.logits                      # (N, num_groups-1, codebook_size)
    labels = codec_ids[:, 1:]                    # (N, num_groups-1), already aligned

    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )
    return logits, loss


def forward_batch(model, batch, sub_talker_weight: float):
    """Single forward pass. Returns (total_loss, codec0_loss, sub_talker_loss)."""
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding = model.speaker_encoder(
        ref_mels.to(model.device).to(model.dtype)
    ).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    # text_projection is applied on EVERY inference path in the library. Skipping it
    # here trains the talker against a different function than the one it generates
    # with (the dims happen to match on the 1.7B, so the mismatch is silent).
    # Project first, then mask: the MLP has a bias, so text_projection(0) != 0.
    input_text_embedding = (
        model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))
        * text_embedding_mask
    )
    input_codec_embedding = (
        model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    )
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
            codec_ids[:, :, i]
        )
        input_embeddings = input_embeddings + codec_i_embedding * codec_mask.unsqueeze(-1)

    # Pass full-length inputs and unshifted labels: ForCausalLMLoss does the single
    # shift internally. The previous code truncated inputs AND advanced labels, so
    # the model was trained to predict t+2 instead of t+1.
    outputs = model.talker(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=codec_0_labels,
        output_hidden_states=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states[0][-1]
    # Pair the hidden state that predicts frame p with frame p's codes.
    talker_hidden_states = hidden_states[:, :-1][codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    _, sub_talker_loss = sub_talker_loss_finetune(
        model.talker, talker_codec_ids, talker_hidden_states
    )

    total = outputs.loss + sub_talker_weight * sub_talker_loss
    return total, outputs.loss.detach(), sub_talker_loss.detach()


class Qwen3TTSPipeline:
    """End-to-end pipeline for Qwen3-TTS multi-speaker fine-tuning."""

    def __init__(
        self,
        audio_dir: str,
        ref_audio: str = None,
        ref_audio_dir: str = None,
        speaker_name: str = "my_speaker",
        output_dir: str = "./output",
        device: str = "cuda:0",
        tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        asr_model: str = "nvidia/parakeet-tdt-0.6b-v2",
        batch_size: int = 2,
        transcribe_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        lr: float = 2e-6,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        warmup_ratio: float = 0.05,
        save_every_n_epochs: int = 1,
        sub_talker_loss_weight: float = 0.3,
        val_frac: float = 0.06,
        num_workers: int = 4,
        centroid_clips: int = 64,
        max_drop_frac: float = 0.20,
        seed: int = 1234,
        speaker_prefix_template: str = "Speaker {speaker}: {text}",
        ignore_chunk_text: bool = False,
        force_retranscribe: bool = False,
        force_reencode: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.ref_audio_dir = Path(ref_audio_dir) if ref_audio_dir else None
        self.speaker_name = speaker_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.tokenizer_model_path = tokenizer_model_path
        self.init_model_path = init_model_path
        self.asr_model = asr_model
        self.batch_size = batch_size
        self.transcribe_batch_size = transcribe_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.save_every_n_epochs = save_every_n_epochs
        self.sub_talker_loss_weight = sub_talker_loss_weight
        self.val_frac = val_frac
        self.num_workers = num_workers
        self.centroid_clips = centroid_clips
        self.max_drop_frac = max_drop_frac
        self.seed = seed
        self.speaker_prefix_template = speaker_prefix_template
        self.ignore_chunk_text = ignore_chunk_text
        self.force_retranscribe = force_retranscribe
        self.force_reencode = force_reencode

        self.speakers: List[str] = []
        self.speaker_files: Dict[str, List[Path]] = {}
        self.speaker_ref_audios: Dict[str, Path] = {}
        self.manifest: Dict[str, dict] = {}
        self.chunk_params: Dict[str, Any] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_raw_jsonl = self.output_dir / "train_raw.jsonl"
        self.train_with_codes_jsonl = self.output_dir / "train_with_codes.jsonl"
        self.signature_file = self.output_dir / "data_signature.json"

        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        self.attn_implementation = get_attention_implementation()

    # ------------------------------------------------------------------ discovery

    def _speaker_of(self, path: Path) -> str:
        return path.parent.name if path.parent != self.audio_dir else self.speaker_name

    def _load_manifest(self) -> None:
        """Load per-chunk stats written by chunk_audio.py, if present."""
        self.manifest = {}
        for manifest_path in self.audio_dir.rglob("chunks.jsonl"):
            base = manifest_path.parent
            with open(manifest_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.manifest[str(base / row["file"])] = row

        for params_path in self.audio_dir.rglob("chunk_params.json"):
            try:
                self.chunk_params = json.loads(params_path.read_text())
                break
            except Exception:
                pass

        if self.manifest:
            n_text = sum(1 for r in self.manifest.values() if r.get("text"))
            print(f"Loaded chunk metadata for {len(self.manifest)} chunks ({n_text} with pre-computed text)")

    def validate_audio_files(self) -> List[Path]:
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")

        all_wavs = sorted(set(self.audio_dir.rglob("*.wav")) | set(self.audio_dir.rglob("*.WAV")))
        if not all_wavs:
            raise ValueError(f"No WAV files found in {self.audio_dir}")

        self._load_manifest()

        valid: List[Path] = []
        self.speaker_files = {}
        for wav in tqdm(all_wavs, desc="Validating audio"):
            try:
                torchaudio.load(str(wav))
            except Exception as exc:
                print(f"Warning: could not load {wav}: {exc}")
                continue
            self.speaker_files.setdefault(self._speaker_of(wav), []).append(wav)
            valid.append(wav)

        if not valid:
            raise ValueError(f"No valid audio files in {self.audio_dir}")

        self.speakers = sorted(self.speaker_files)
        print(f"\n{'='*60}")
        print(f"Discovered {len(self.speakers)} speaker(s), {len(valid)} chunks")
        for spk in self.speakers:
            print(f"  - {spk}: {len(self.speaker_files[spk])} chunks")
        print(f"{'='*60}\n")
        return valid

    def check_dependencies(self) -> None:
        """Report on the environment. Does NOT install -- this venv has pinned
        ARM64 wheels (flash-attn in particular) that a stray pip install would break."""
        print(f"\n{'='*60}\nEnvironment\n{'='*60}")
        required = ["torch", "torchaudio", "numpy", "librosa", "soundfile", "tqdm",
                    "transformers", "accelerate", "safetensors", "huggingface_hub", "qwen_tts"]
        missing = []
        for mod in required:
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if missing:
            raise SystemExit(
                f"Missing packages: {', '.join(missing)}\n"
                f"Install them yourself (pip install -r requirements.txt) -- this script "
                f"will not pip-install into your pinned ARM64 environment."
            )
        print(f"  all {len(required)} required packages present")
        print(f"  attention implementation: {self.attn_implementation}")
        print(f"  dtype: {self.torch_dtype}\n")

    # ------------------------------------------------------------------ ASR

    def transcribe_audio(self, audio_files: List[Path]) -> List[Dict[str, Any]]:
        results = []
        shown = Counter()
        todo = []

        chunk_asr = self.chunk_params.get("asr_model")
        if chunk_asr and chunk_asr != self.asr_model and not self.ignore_chunk_text:
            print(f"  NOTE: Chunks were generated with ASR model '{chunk_asr}', but pipeline "
                  f"is configured with '{self.asr_model}'. Using chunk transcripts. "
                  f"Pass --ignore_chunk_text to force re-transcribing with {self.asr_model}.")

        for path in audio_files:
            spk = self._speaker_of(path)
            row = self.manifest.get(str(path), {})
            chunk_text = row.get("text")
            if not self.ignore_chunk_text and chunk_text is not None and chunk_text.strip():
                results.append({
                    "audio": str(path),
                    "raw_text": chunk_text.strip(),
                    "speaker": spk,
                    "duration": self._duration_of(path),
                    "text_source": "chunk",
                })
                if shown[spk] < 5:
                    print(f"  [{spk} - from chunk] {path.name}: {chunk_text.strip()[:90]}")
                    shown[spk] += 1
            else:
                todo.append(path)

        if todo:
            import nemo.collections.asr as nemo_asr

            print(f"\n{'='*60}\nSTEP 1: Transcribing {len(todo)} file(s) with {self.asr_model}\n{'='*60}\n")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.asr_model)
            if self.device.startswith("cuda"):
                model = model.cuda()
            model.eval()

            paths = [str(p) for p in todo]
            with torch.inference_mode():
                transcripts = model.transcribe(paths, batch_size=self.transcribe_batch_size)
            if isinstance(transcripts, tuple):
                transcripts = transcripts[0]

            for path, text in zip(todo, transcripts):
                if hasattr(text, "text"):
                    text = text.text
                spk = self._speaker_of(path)
                results.append({
                    "audio": str(path),
                    "raw_text": text.strip(),
                    "speaker": spk,
                    "duration": self._duration_of(path),
                    "text_source": "asr",
                })
                if shown[spk] < 5:
                    print(f"  [{spk} - from ASR] {path.name}: {text.strip()[:90]}")
                    shown[spk] += 1

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\nReused pre-computed transcripts for all {len(results)} chunks (NeMo load skipped)")

        path_order = {str(p): i for i, p in enumerate(audio_files)}
        results.sort(key=lambda r: path_order.get(r["audio"], 0))

        print(f"\nProcessed {len(results)} files ({len(results) - len(todo)} reused, {len(todo)} transcribed)")
        return results

    def _duration_of(self, path: Path) -> float:
        row = self.manifest.get(str(path))
        if row:
            return float(row["duration_s"])
        info = torchaudio.info(str(path)) if hasattr(torchaudio, "info") else None
        if info is not None:
            return info.num_frames / info.sample_rate
        wav, sr = torchaudio.load(str(path))
        return wav.shape[-1] / sr

    @staticmethod
    def _drop_reason(text: str, duration: float) -> str:
        if not text:
            return "empty"
        words = text.split()
        if len(text) < 4 or len(words) < 2:
            return "too_short"
        if not re.search(r"[A-Za-z]", text):
            return "no_alpha"
        if sum(c not in ALLOWED_CHARS for c in text) / len(text) > 0.05:
            return "charset"

        stripped = text.strip()
        first_char = stripped[0]
        if not (first_char.isupper() or first_char.isdigit() or first_char in "\"'“‘("):
            return "not_capitalized"

        last_char = stripped[-1]
        if last_char in "\"'”’)":
            last_char = stripped[-2] if len(stripped) > 1 else ""
        if last_char not in ".?!":
            return "no_terminal_punct"

        last_word = re.sub(r"[^a-zA-Z]", "", words[-1]).lower()
        if last_word in ("and", "or", "but", "so", "because", "that", "with", "as", "if", "than"):
            return "hanging_conjunction"

        lowered = [w.lower().strip(".,?!\"'") for w in words]
        for n in (1, 2, 3):
            run = 1
            for i in range(n, len(lowered) - n + 1, n):
                if lowered[i : i + n] == lowered[i - n : i]:
                    run += 1
                    if run >= 3:
                        return "repetition"
                else:
                    run = 1
        if duration > 0:
            rate = len(text) / duration
            if rate < 2.0:
                return "rate_low"
            if rate > 25.0:
                return "rate_high"
        return ""

    def filter_transcripts(self, results: List[Dict[str, Any]]) -> Tuple[List[dict], List[dict]]:
        kept, dropped = [], []
        for row in results:
            reason = self._drop_reason(row["raw_text"], row["duration"])
            if reason:
                dropped.append({**row, "reason": reason})
            else:
                kept.append(row)

        by_spk = {}
        for row in results:
            by_spk.setdefault(row["speaker"], {"total": 0, "kept": 0, "secs": 0.0, "reasons": Counter()})
            by_spk[row["speaker"]]["total"] += 1
        for row in kept:
            by_spk[row["speaker"]]["kept"] += 1
            by_spk[row["speaker"]]["secs"] += row["duration"]
        for row in dropped:
            by_spk[row["speaker"]]["reasons"][row["reason"]] += 1

        print(f"\n{'='*60}\nTranscript filter\n{'='*60}")
        for spk, s in sorted(by_spk.items()):
            frac = 1 - s["kept"] / max(s["total"], 1)
            detail = ", ".join(f"{r}={c}" for r, c in sorted(s["reasons"].items())) or "none"
            print(f"  {spk}: kept {s['kept']}/{s['total']} ({s['secs']/60:.1f} min) "
                  f"| dropped {frac*100:.1f}% [{detail}]")

        valid_texts = [r["raw_text"] for r in kept if r.get("raw_text")]
        if valid_texts:
            term_frac = sum(bool(t.strip() and t.strip()[-1] in ".?!") for t in valid_texts) / len(valid_texts)
            cap_frac = sum(bool(t.strip() and t.strip()[0].isupper()) for t in valid_texts) / len(valid_texts)
            print(f"  Sentence quality: terminal {term_frac*100:.1f}% | capital {cap_frac*100:.1f}%")

        report = self.output_dir / "asr_report.json"
        with open(report, "w", encoding="utf-8") as f:
            json.dump({spk: {**s, "reasons": dict(s["reasons"])} for spk, s in by_spk.items()}, f, indent=2)
        if dropped:
            with open(self.output_dir / "dropped_chunks.jsonl", "w", encoding="utf-8") as f:
                for row in dropped:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        for spk, s in by_spk.items():
            frac = 1 - s["kept"] / max(s["total"], 1)
            if frac > self.max_drop_frac:
                raise ValueError(
                    f"Speaker '{spk}' lost {frac*100:.1f}% of chunks (limit "
                    f"{self.max_drop_frac*100:.0f}%). See {report}. This usually means a "
                    f"chunking parameter is wrong, not that the ASR is bad. "
                    f"Override with --max_drop_frac."
                )
        return kept, dropped

    # ------------------------------------------------------------------ references

    def _resolve_ref_override(self, speaker: str):
        """An explicit user-supplied reference for this speaker, or None."""
        if self.ref_audio_dir and self.ref_audio_dir.exists():
            for name in (f"{speaker}.wav", f"{speaker}.WAV", f"reference_{speaker}.wav"):
                cand = self.ref_audio_dir / name
                if cand.exists():
                    return cand
        if self.ref_audio and self.ref_audio.exists() and len(self.speakers) == 1:
            return self.ref_audio
        return None

    def _warn_if_band_limited(self, speaker: str, path: Path) -> None:
        """The speaker-encoder mel runs to fmax=12000, so a reference below 24 kHz
        leaves the top mel bins empty and produces a duller speaker embedding."""
        try:
            import soundfile as sf
            sr = sf.info(str(path)).samplerate
        except Exception:
            return
        if sr < 24000:
            print(f"    WARNING: reference for '{speaker}' is {sr} Hz ({path.name}). The "
                  f"speaker encoder models up to 12 kHz, so a {sr} Hz clip caps content at "
                  f"{sr // 2} Hz and yields a duller embedding. Prefer dropping "
                  f"--ref_audio_dir and letting the 24 kHz centroid pick the reference.")

    def _candidate_clips(self, speaker: str, kept: List[dict]) -> List[Path]:
        """Gate clips on chunk metadata + transcript, cheaply, with no audio I/O."""
        rows = [r for r in kept if r["speaker"] == speaker]
        if not rows:
            return []

        stats = [self.manifest.get(r["audio"]) for r in rows]
        if not any(stats):
            return [Path(r["audio"]) for r in rows if len(r["raw_text"].split()) >= 8]

        levels = [s["speech_dbfs"] for s in stats if s]
        snrs = sorted(s["snr_db"] for s in stats if s)
        median_level = float(np.median(levels))
        snr_q3 = snrs[int(len(snrs) * 0.75)] if snrs else -1e9

        cands = []
        for row, st in zip(rows, stats):
            if not st:
                continue
            if not (4.0 <= st["duration_s"] <= 8.0):
                continue
            if st["active_frac"] < 0.85:
                continue
            if st["snr_db"] < snr_q3:
                continue
            if abs(st["speech_dbfs"] - median_level) > 2.0:
                continue
            if len(row["raw_text"].split()) < 8:
                continue
            cands.append(Path(row["audio"]))

        if not cands:  # gate too strict -- relax to transcript length only
            cands = [Path(r["audio"]) for r in rows if len(r["raw_text"].split()) >= 8]
        if not cands:
            cands = [Path(r["audio"]) for r in rows]
        return sorted(cands)

    def _embed_clips(self, clips, model, device, dtype):
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        embs = []
        for path in clips:
            wav, sr = librosa.load(str(path), sr=None, mono=True)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=-1)
            if sr != 24000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
            mels = mel_spectrogram(
                torch.tensor(wav).unsqueeze(0).to(torch.float32),
                n_fft=1024, num_mels=128, sampling_rate=24000,
                hop_size=256, win_size=1024, fmin=0, fmax=12000,
            ).transpose(1, 2)
            with torch.no_grad():
                emb = model.speaker_encoder(mels.to(device).to(dtype))
            embs.append(emb[0].float().cpu())
        return torch.stack(embs)

    def compute_centroid_embedding(self, speaker, clips, model, device, dtype):
        """Mean speaker-encoder embedding over up to `centroid_clips` clips,
        rescaled to the median per-clip norm.
        """
        clips = sorted(clips)
        rng = random.Random(self.seed)
        sample = clips if len(clips) <= self.centroid_clips else rng.sample(clips, self.centroid_clips)
        sample = sorted(sample)

        emb = self._embed_clips(sample, model, device, dtype)
        norms = emb.norm(dim=1)
        centroid = emb.mean(dim=0)
        centroid = centroid / centroid.norm() * norms.median()

        unit = emb / norms.unsqueeze(1)
        cvec = centroid / centroid.norm()
        to_centroid = unit @ cvec
        k = len(emb)
        pair = unit @ unit.T
        mean_pair = float((pair.sum() - k) / (k * k - k)) if k > 1 else 1.0

        diag = {
            "n_clips": k,
            "mean_pairwise_cos": round(mean_pair, 4),
            "mean_cos_to_centroid": round(float(to_centroid.mean()), 4),
            "min_cos_to_centroid": round(float(to_centroid.min()), 4),
        }
        print(f"  - '{speaker}': {k} clips | pairwise cos {diag['mean_pairwise_cos']:.3f} "
              f"| to-centroid {diag['mean_cos_to_centroid']:.3f} (min {diag['min_cos_to_centroid']:.3f})")
        if diag["mean_cos_to_centroid"] < 0.80:
            print(f"    WARNING: mean cosine to centroid is {diag['mean_cos_to_centroid']:.3f} (<0.80). "
                  f"On a small speaker set this usually means chunks from another voice "
                  f"landed in '{speaker}'. Check the folder before trusting this checkpoint.")

        best = sample[int(to_centroid.argmax())]
        return centroid, diag, best

    # ------------------------------------------------------------------ data prep

    def create_train_jsonl(self, kept: List[dict]) -> None:
        print(f"\n{'='*60}\nSTEP 2: Writing train_raw.jsonl\n{'='*60}\n")

        for spk in self.speakers:
            override = self._resolve_ref_override(spk)
            if override is not None:
                self.speaker_ref_audios[spk] = override
            else:
                cands = self._candidate_clips(spk, kept)
                if cands:
                    self.speaker_ref_audios[spk] = cands[0]

        with open(self.train_raw_jsonl, "w", encoding="utf-8") as f:
            for row in kept:
                spk = row["speaker"]
                text = row["raw_text"]
                if self.speaker_prefix_template:
                    text = self.speaker_prefix_template.format(speaker=spk, text=text)
                f.write(json.dumps({
                    "audio": row["audio"],
                    "text": text,
                    "ref_audio": str(self.speaker_ref_audios.get(spk, row["audio"])),
                    "speaker": spk,
                    "duration": row["duration"],
                }, ensure_ascii=False) + "\n")
        print(f"Wrote {self.train_raw_jsonl} with {len(kept)} entries")

    def prepare_data(self) -> None:
        print(f"\n{'='*60}\nSTEP 3: Extracting audio codes\n{'='*60}\n")
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self.tokenizer_model_path, device_map=self.device
        )
        rows = [json.loads(line) for line in open(self.train_raw_jsonl, encoding="utf-8")]

        out, batch_rows, batch_audio = [], [], []
        BATCH = 32
        for row in tqdm(rows, desc="Encoding audio"):
            batch_rows.append(row)
            batch_audio.append(row["audio"])
            if len(batch_rows) >= BATCH:
                enc = tokenizer.encode(batch_audio)
                for code, r in zip(enc.audio_codes, batch_rows):
                    out.append({**r, "audio_codes": code.cpu().tolist()})
                batch_rows, batch_audio = [], []
        if batch_audio:
            enc = tokenizer.encode(batch_audio)
            for code, r in zip(enc.audio_codes, batch_rows):
                out.append({**r, "audio_codes": code.cpu().tolist()})

        with open(self.train_with_codes_jsonl, "w", encoding="utf-8") as f:
            for row in out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {self.train_with_codes_jsonl}")

        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ signatures

    def _audio_fingerprint(self) -> str:
        entries = []
        for p in sorted(self.audio_dir.rglob("*.wav")):
            st = p.stat()
            entries.append(f"{p.relative_to(self.audio_dir)}:{st.st_size}:{st.st_mtime_ns}")
        for chunks in sorted(self.audio_dir.rglob("chunks.jsonl")):
            entries.append(f"chunks:{chunks.read_text()}")
        for params in sorted(self.audio_dir.rglob("chunk_params.json")):
            entries.append(params.read_text())
        return hashlib.sha1("\n".join(entries).encode()).hexdigest()

    def _raw_signature(self) -> dict:
        n_with_text = sum(1 for r in self.manifest.values() if r.get("text"))
        if self.ignore_chunk_text or n_with_text == 0:
            chunk_text_mode = "asr"
        elif n_with_text == len(self.manifest):
            chunk_text_mode = "reuse"
        else:
            chunk_text_mode = "mixed"

        sig = {
            "version": PIPELINE_VERSION,
            "speakers": self.speakers,
            "prefix_template": self.speaker_prefix_template or "",
            "asr_model": self.asr_model,
            "max_drop_frac": self.max_drop_frac,
            "chunk_text_mode": chunk_text_mode,
            "ignore_chunk_text": self.ignore_chunk_text,
            "audio_fingerprint": self._audio_fingerprint(),
        }
        sig["hash"] = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()
        return sig

    def _codes_signature(self, raw_hash: str) -> dict:
        sig = {
            "version": PIPELINE_VERSION,
            "raw_hash": raw_hash,
            "tokenizer": self.tokenizer_model_path,
        }
        sig["hash"] = hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()
        return sig

    def _load_signatures(self) -> dict:
        if self.signature_file.exists():
            try:
                return json.loads(self.signature_file.read_text())
            except Exception:
                return {}
        return {}

    def _save_signature(self, key: str, sig: dict) -> None:
        data = self._load_signatures()
        data[key] = sig
        self.signature_file.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _explain_diff(stored: dict, current: dict) -> str:
        diffs = []
        for k in sorted(set(stored) | set(current)):
            if k == "hash":
                continue
            if stored.get(k) != current.get(k):
                a, b = stored.get(k), current.get(k)
                if k == "audio_fingerprint":
                    diffs.append("audio files or chunk manifests changed")
                else:
                    diffs.append(f"{k}: {a!r} -> {b!r}")
        return "; ".join(diffs) or "unknown change"

    # ------------------------------------------------------------------ training

    def train_model(self) -> None:
        print(f"\n{'='*60}\nSTEP 4: Fine-tuning\n{'='*60}\n")

        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from dataset import TTSDataset
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        from safetensors.torch import save_file
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from transformers import get_cosine_schedule_with_warmup

        set_seed(self.seed)

        logging_dir = self.output_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with="tensorboard",
            project_dir=str(logging_dir),
        )

        print(f"Loading model: {self.init_model_path} ({self.attn_implementation})")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            self.init_model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
        )
        config = Qwen3TTSConfig.from_pretrained(self.init_model_path)

        rows = [json.loads(line) for line in open(self.train_with_codes_jsonl, encoding="utf-8")]
        if not self.speakers:
            self.speakers = sorted({r.get("speaker", self.speaker_name) for r in rows})
        by_speaker: Dict[str, List[dict]] = {}
        for r in rows:
            by_speaker.setdefault(r.get("speaker", self.speaker_name), []).append(r)

        qwen3tts.model.speaker_encoder.requires_grad_(False)
        qwen3tts.model.speaker_encoder.to(accelerator.device)

        print(f"\nComputing centroid speaker embeddings ({self.centroid_clips} clips max)...")
        target_embeddings, speaker_meta = {}, {}
        ref_dir = self.output_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)

        for spk in self.speakers:
            clips = [Path(r["audio"]) for r in by_speaker.get(spk, [])]
            centroid, diag, best = self.compute_centroid_embedding(
                spk, clips, qwen3tts.model, accelerator.device, self.torch_dtype
            )
            target_embeddings[spk] = centroid

            override = self._resolve_ref_override(spk)
            if override is not None:
                self._warn_if_band_limited(spk, override)
            chosen = override if override is not None else best
            frozen = ref_dir / f"{spk}.wav"
            if Path(chosen).resolve() != frozen.resolve():
                shutil.copyfile(str(chosen), str(frozen))
            self.speaker_ref_audios[spk] = frozen
            speaker_meta[spk] = {**diag, "reference": str(frozen), "selected_from": str(chosen)}

        with open(self.output_dir / "speaker_meta.json", "w", encoding="utf-8") as f:
            json.dump(speaker_meta, f, indent=2)

        for r in rows:
            r["ref_audio"] = str(self.speaker_ref_audios[r.get("speaker", self.speaker_name)])

        train_rows, val_rows = self._split_train_val(by_speaker)
        print(f"\nTrain {len(train_rows)} samples | Val {len(val_rows)} samples")

        train_ds = TTSDataset(train_rows, qwen3tts.processor, config)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=train_ds.collate_fn, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        val_loader = None
        if val_rows:
            val_ds = TTSDataset(val_rows, qwen3tts.processor, config)
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False,
                collate_fn=val_ds.collate_fn, num_workers=min(2, self.num_workers),
                pin_memory=True,
            )

        params = [p for p in qwen3tts.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        steps_per_epoch = max(1, len(train_loader) // self.gradient_accumulation_steps)
        total_steps = max(1, steps_per_epoch * self.num_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * self.warmup_ratio)),
            num_training_steps=total_steps,
        )

        prepared = accelerator.prepare(qwen3tts.model, optimizer, train_loader, scheduler)
        model, optimizer, train_loader, scheduler = prepared
        if val_loader is not None:
            val_loader = accelerator.prepare(val_loader)

        accelerator.init_trackers("qwen3tts", config=self._hparams())

        best_eval, best_epoch, global_step = float("inf"), -1, 0
        model.train()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            for step, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    loss, ce, sub = forward_batch(model, batch, self.sub_talker_loss_weight)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    global_step += 1
                    accelerator.log({
                        "train/loss": loss.item(),
                        "train/codec0_loss": ce.item(),
                        "train/sub_talker_loss": sub.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + 1,
                    }, step=global_step)

                if step % 10 == 0:
                    accelerator.print(
                        f"  step {step:4d} | loss {loss.item():.4f} "
                        f"(codec0 {ce.item():.4f}, sub {sub.item():.4f}) "
                        f"| lr {scheduler.get_last_lr()[0]:.2e}"
                    )

            if val_loader is not None:
                metrics = self.evaluate(model, val_loader, accelerator)
                accelerator.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
                accelerator.print(
                    f"  EVAL loss {metrics['loss']:.4f} "
                    f"(codec0 {metrics['codec0_loss']:.4f}, sub {metrics['sub_talker_loss']:.4f})"
                )
                if metrics["loss"] < best_eval:
                    best_eval, best_epoch = metrics["loss"], epoch
                model.train()

            if (epoch + 1) % self.save_every_n_epochs == 0 and accelerator.is_main_process:
                self._save_checkpoint(accelerator, model, epoch, target_embeddings)

        accelerator.end_training()
        print("\nTraining complete!")
        if best_epoch >= 0:
            print(f"Best eval loss {best_eval:.4f} at epoch {best_epoch + 1} "
                  f"(checkpoint-epoch-{best_epoch})")

    def _hparams(self) -> dict:
        return {
            "lr": self.lr, "batch_size": self.batch_size, "num_epochs": self.num_epochs,
            "grad_accum": self.gradient_accumulation_steps, "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio, "sub_talker_loss_weight": self.sub_talker_loss_weight,
            "val_frac": self.val_frac, "seed": self.seed, "n_speakers": len(self.speakers),
            "speakers": ",".join(self.speakers), "asr_model": self.asr_model,
            "init_model": self.init_model_path, "attn": self.attn_implementation,
        }

    def _split_train_val(self, by_speaker: Dict[str, List[dict]]):
        """Per-speaker stratified split. Persisted so eval loss stays comparable
        across runs."""
        val_ids_file = self.output_dir / "val_ids.json"
        stored = None
        if val_ids_file.exists():
            try:
                stored = set(json.loads(val_ids_file.read_text()))
            except Exception:
                stored = None

        all_rows = [r for rows in by_speaker.values() for r in rows]
        if stored is not None and stored.issubset({r["audio"] for r in all_rows}):
            val = [r for r in all_rows if r["audio"] in stored]
            train = [r for r in all_rows if r["audio"] not in stored]
            if val and train:
                print(f"Reusing stored validation split ({len(val)} clips)")
                return train, val

        if self.val_frac <= 0:
            return all_rows, []

        rng = random.Random(self.seed)
        val_set = set()
        for spk, rows in sorted(by_speaker.items()):
            rows = sorted(rows, key=lambda r: r["audio"])
            n = int(round(len(rows) * self.val_frac))
            n = max(min(n, 24), min(8, len(rows) // 4))
            for r in rng.sample(rows, min(n, len(rows))):
                val_set.add(r["audio"])

        val = [r for r in all_rows if r["audio"] in val_set]
        train = [r for r in all_rows if r["audio"] not in val_set]
        val_ids_file.write_text(json.dumps(sorted(val_set), indent=2))
        return train, val

    @torch.no_grad()
    def evaluate(self, model, val_loader, accelerator) -> Dict[str, float]:
        model.eval()
        totals = {"loss": [], "codec0_loss": [], "sub_talker_loss": []}
        for batch in val_loader:
            loss, ce, sub = forward_batch(model, batch, self.sub_talker_loss_weight)
            for key, val in zip(totals, (loss.detach(), ce, sub)):
                totals[key].append(accelerator.gather_for_metrics(val.repeat(1)))
        return {k: torch.cat(v).float().mean().item() for k, v in totals.items()}

    def _save_checkpoint(self, accelerator, model, epoch, target_embeddings) -> None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import save_file

        out_dir = os.path.join(str(self.output_dir), f"checkpoint-epoch-{epoch}")
        base = (self.init_model_path if os.path.isdir(self.init_model_path)
                else snapshot_download(self.init_model_path))
        shutil.copytree(base, out_dir, dirs_exist_ok=True)

        with open(os.path.join(base, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["tts_model_type"] = "custom_voice"
        talker_cfg = cfg.get("talker_config", {})
        talker_cfg["spk_id"] = {spk.lower(): 3000 + i for i, spk in enumerate(self.speakers)}
        talker_cfg["spk_is_dialect"] = {spk.lower(): False for spk in self.speakers}
        cfg["talker_config"] = talker_cfg
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        unwrapped = accelerator.unwrap_model(model)
        state = {k: v.detach().to("cpu").to(torch.float32)
                 for k, v in unwrapped.state_dict().items()}
        for key in [k for k in state if k.startswith("speaker_encoder")]:
            del state[key]

        weight = state["talker.model.codec_embedding.weight"]
        for i, spk in enumerate(self.speakers):
            weight[3000 + i] = target_embeddings[spk].to(weight.device).to(weight.dtype)

        save_file(state, os.path.join(out_dir, "model.safetensors"))
        print(f"  saved checkpoint ({len(self.speakers)} speakers) -> {out_dir}")

    # ------------------------------------------------------------------ driver

    def run(self) -> None:
        print(f"\n{'='*60}\nQwen3-TTS Multi-Speaker Fine-Tuning\n{'='*60}")
        self.check_dependencies()
        audio_files = self.validate_audio_files()

        stored = self._load_signatures()
        raw_sig = self._raw_signature()

        need_transcribe = self.force_retranscribe or not self.train_raw_jsonl.exists()
        if not need_transcribe:
            prev = stored.get("raw", {})
            if prev.get("hash") != raw_sig["hash"]:
                print(f"\nCached train_raw.jsonl is stale ({self._explain_diff(prev, raw_sig)}); "
                      f"re-transcribing.")
                need_transcribe = True

        if need_transcribe:
            results = self.transcribe_audio(audio_files)
            kept, _ = self.filter_transcripts(results)
            if not kept:
                raise ValueError("Every transcript was filtered out. Check asr_report.json.")
            self.create_train_jsonl(kept)
            self._save_signature("raw", raw_sig)
        else:
            print(f"\nReusing {self.train_raw_jsonl.name}")

        codes_sig = self._codes_signature(raw_sig["hash"])
        need_encode = self.force_reencode or not self.train_with_codes_jsonl.exists()
        if not need_encode:
            prev = stored.get("codes", {})
            if prev.get("hash") != codes_sig["hash"]:
                print(f"Cached train_with_codes.jsonl is stale "
                      f"({self._explain_diff(prev, codes_sig)}); re-encoding.")
                need_encode = True
        if need_encode:
            self.prepare_data()
            self._save_signature("codes", codes_sig)
        else:
            print(f"Reusing {self.train_with_codes_jsonl.name}")

        self.train_model()
        print(f"\n{'='*60}\nPipeline complete. Checkpoints in {self.output_dir}\n{'='*60}\n")


def main():
    p = argparse.ArgumentParser(description="Qwen3-TTS Multi-Speaker Fine-Tuning Pipeline")

    p.add_argument("--audio_dir", type=str, required=True,
                   help="Chunk directory (with per-speaker subdirs, or flat for one speaker)")
    p.add_argument("--ref_audio", type=str, default=None,
                   help="Reference WAV override (single-speaker runs only)")
    p.add_argument("--ref_audio_dir", type=str, default=None,
                   help="Directory of per-speaker reference WAVs (<speaker>.wav)")
    p.add_argument("--speaker_name", type=str, default="my_speaker",
                   help="Speaker name used when audio_dir is flat")
    p.add_argument("--speaker_prefix_template", type=str, default="Speaker {speaker}: {text}",
                   help="Prompt conditioning template ('' to disable)")
    p.add_argument("--output_dir", type=str, default="./output")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    p.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2",
                   help="NeMo ASR model with punctuation and capitalization support.")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--transcribe_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-6,
                   help="Upstream default. Above ~1e-5 degrades speaker quality on small sets.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_every_n_epochs", type=int, default=2)
    p.add_argument("--sub_talker_loss_weight", type=float, default=0.3,
                   help="Upstream default. The library returns this loss unweighted.")
    p.add_argument("--val_frac", type=float, default=0.06, help="0 disables validation")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--centroid_clips", type=int, default=64,
                   help="Clips averaged into each speaker's baked-in embedding")
    p.add_argument("--max_drop_frac", type=float, default=0.20,
                   help="Fail if any speaker loses more than this fraction to the filter")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ignore_chunk_text", action="store_true",
                   help="Ignore pre-computed transcripts in chunks.jsonl and force ASR pass")
    p.add_argument("--force_retranscribe", action="store_true")
    p.add_argument("--force_reencode", action="store_true")

    args = p.parse_args()
    template = args.speaker_prefix_template if args.speaker_prefix_template.strip() else None

    Qwen3TTSPipeline(
        audio_dir=args.audio_dir, ref_audio=args.ref_audio, ref_audio_dir=args.ref_audio_dir,
        speaker_name=args.speaker_name, output_dir=args.output_dir, device=args.device,
        tokenizer_model_path=args.tokenizer_model_path, init_model_path=args.init_model_path,
        asr_model=args.asr_model, batch_size=args.batch_size,
        transcribe_batch_size=args.transcribe_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, lr=args.lr,
        weight_decay=args.weight_decay, num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio, save_every_n_epochs=args.save_every_n_epochs,
        sub_talker_loss_weight=args.sub_talker_loss_weight, val_frac=args.val_frac,
        num_workers=args.num_workers, centroid_clips=args.centroid_clips,
        max_drop_frac=args.max_drop_frac, seed=args.seed, speaker_prefix_template=template,
        ignore_chunk_text=args.ignore_chunk_text, force_retranscribe=args.force_retranscribe,
        force_reencode=args.force_reencode,
    ).run()


if __name__ == "__main__":
    main()
