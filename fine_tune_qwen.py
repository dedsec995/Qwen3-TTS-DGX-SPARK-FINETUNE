#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-TTS One-Command Fine-Tuning Script

This script provides a complete end-to-end fine-tuning pipeline:
1. Takes a directory of WAV files and a reference audio
2. Automatically transcribes using NeMo Parakeet
3. Creates the train_raw.jsonl
4. Prepares data (extracts audio_codes)
5. Trains the model

Usage:
    python train_parakeet.py \
        --audio_dir ./my_audio_files \
        --ref_audio ./reference.wav \
        --speaker_name my_voice \
        --output_dir ./output
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def configure_hf_cache():
    """Configure HuggingFace cache inside env to keep model weights local."""
    script_dir = Path(__file__).parent.absolute()
    hf_cache = script_dir / "env" / "hf_cache"

    # Only configure if env exists
    if hf_cache.parent.exists():
        hf_cache.mkdir(parents=True, exist_ok=True)
        # Only set HF_HOME - let HuggingFace manage subdirectories
        os.environ.setdefault("HF_HOME", str(hf_cache))


def get_attention_implementation():
    """Return best available attention implementation."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


# Configure HF cache before any HuggingFace imports
configure_hf_cache()

import torch
import torchaudio
from tqdm import tqdm


class Qwen3TTSPipeline:
    """End-to-end pipeline for Qwen3-TTS fine-tuning."""

    def __init__(
        self,
        audio_dir: str,
        ref_audio: str,
        speaker_name: str,
        output_dir: str = "./output",
        device: str = "cuda:0",
        tokenizer_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        batch_size: int = 2,
        transcribe_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        save_every_n_epochs: int = 1,
        language: str = "en",
    ):
        self.audio_dir = Path(audio_dir)
        self.ref_audio = Path(ref_audio)
        self.speaker_name = speaker_name
        self.output_dir = Path(output_dir)
        self.device = device
        self.tokenizer_model_path = tokenizer_model_path
        self.init_model_path = init_model_path
        self.batch_size = batch_size
        self.transcribe_batch_size = transcribe_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.save_every_n_epochs = save_every_n_epochs
        self.language = language

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Intermediate files
        self.train_raw_jsonl = self.output_dir / "train_raw.jsonl"
        self.train_with_codes_jsonl = self.output_dir / "train_with_codes.jsonl"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        # Detect attention implementation
        self.attn_implementation = get_attention_implementation()

    def validate_audio_files(self) -> List[Path]:
        """Find and validate all WAV files in the audio directory."""
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")

        wav_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.WAV"))

        if not wav_files:
            raise ValueError(f"No WAV files found in {self.audio_dir}")

        if not self.ref_audio.exists():
            raise ValueError(f"Reference audio not found: {self.ref_audio}")

        # Validate audio files can be loaded
        valid_files = []
        for wav_path in tqdm(wav_files, desc="Validating audio files"):
            try:
                torchaudio.load(str(wav_path))
                valid_files.append(wav_path)
            except Exception as e:
                print(f"Warning: Could not load {wav_path}: {e}")

        print(f"Found {len(valid_files)} valid audio files")
        return valid_files

    def check_dependencies(self) -> None:
        """Check and install all required dependencies."""
        print(f"\n{'='*60}")
        print("Checking and installing dependencies...")
        print(f"{'='*60}\n")

        # Core dependencies
        core_packages = [
            ("torch", "torch"),
            ("torchaudio", "torchaudio"),
            ("numpy", "numpy"),
            ("librosa", "librosa"),
            ("soundfile", "soundfile"),
            ("tqdm", "tqdm"),
        ]

        # ML/TTS dependencies
        ml_packages = [
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
            ("safetensors", "safetensors"),
            ("huggingface_hub", "huggingface-hub"),
            ("hf_transfer", "hf_transfer"),
        ]

        # Audio processing
        audio_packages = [
            ("qwen_tts", "qwen-tts"),
        ]

        all_packages = core_packages + ml_packages + audio_packages

        for module_name, package_name in all_packages:
            try:
                __import__(module_name)
                print(f"  {module_name} is installed")
            except ImportError:
                print(f"  Installing {package_name}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package_name, "-q"
                ])
                print(f"  {package_name} installed")

        # Check flash_attn (optional)
        try:
            import flash_attn  # noqa: F401
            print(f"  flash_attn is installed (using flash_attention_2)")
        except ImportError:
            print(f"  flash_attn not available (using eager attention - slower but compatible)")

        print("\nAll dependencies are ready!")
        print(f"Attention implementation: {self.attn_implementation}")
        print()

    def transcribe_with_parakeet(self, audio_files: List[Path]) -> List[Dict[str, Any]]:
        """
        Transcribe audio files using NeMo Parakeet.

        Returns a list of dictionaries with audio path and transcription.
        """
        import nemo.collections.asr as nemo_asr

        print(f"\n{'='*60}")
        print("STEP 1: Transcribing audio files with NeMo Parakeet")
        print(f"{'='*60}\n")

        # Load NeMo Parakeet model
        print("Loading NeMo Parakeet model (nvidia/parakeet-rnnt-1.1b)...")
        device = "cuda" if self.device.startswith("cuda") else "cpu"
        
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        if device == "cuda":
            asr_model = asr_model.cuda()

        results = []
        audio_paths = [str(p) for p in audio_files]
        
        print(f"Transcribing {len(audio_paths)} files...")
        transcripts = asr_model.transcribe(audio_paths, batch_size=self.transcribe_batch_size)
        
        if isinstance(transcripts, tuple):
            transcripts = transcripts[0]

        for audio_path, text in zip(audio_files, transcripts):
            if hasattr(text, 'text'):
                text = text.text
                
            results.append({
                "audio": str(audio_path),
                "text": text.strip(),
                "ref_audio": str(audio_path),
            })
            print(f"  {audio_path.name}: {text.strip()[:100]}...")

        # Cleanup model
        del asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nSuccessfully transcribed {len(results)} files")
        return results


    def create_train_jsonl(self, data: List[Dict[str, Any]]) -> None:
        """Create the train_raw.jsonl file."""
        print(f"\n{'='*60}")
        print("STEP 2: Creating train_raw.jsonl")
        print(f"{'='*60}\n")

        with open(self.train_raw_jsonl, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Created {self.train_raw_jsonl} with {len(data)} entries")

    def prepare_data(self) -> None:
        """Run the data preparation step to extract audio_codes."""
        print(f"\n{'='*60}")
        print("STEP 3: Preparing data (extracting audio_codes)")
        print(f"{'='*60}\n")

        # Import here to ensure qwen-tts is installed
        from qwen_tts import Qwen3TTSTokenizer

        # Load tokenizer
        print(f"Loading tokenizer: {self.tokenizer_model_path}")
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            self.tokenizer_model_path,
            device_map=self.device,
        )

        # Read and process JSONL
        total_lines = open(self.train_raw_jsonl).readlines()
        total_lines = [json.loads(line.strip()) for line in total_lines]

        final_lines = []
        batch_lines = []
        batch_audios = []
        BATCH_INFER_NUM = 32

        print(f"Processing {len(total_lines)} audio files...")

        for line in tqdm(total_lines, desc="Encoding audio"):
            batch_lines.append(line)
            batch_audios.append(line["audio"])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, line in zip(enc_res.audio_codes, batch_lines):
                    line["audio_codes"] = code.cpu().tolist()
                    final_lines.append(line)
                batch_lines.clear()
                batch_audios.clear()

        # Process remaining
        if len(batch_audios) > 0:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line["audio_codes"] = code.cpu().tolist()
                final_lines.append(line)

        # Write output
        final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]
        with open(self.train_with_codes_jsonl, "w", encoding="utf-8") as f:
            for line in final_lines:
                f.writelines(line + "\n")

        print(f"Created {self.train_with_codes_jsonl}")

        # Cleanup
        del tokenizer_12hz
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_model(self) -> None:
        """Run the fine-tuning step."""
        print(f"\n{'='*60}")
        print("STEP 4: Fine-tuning model")
        print(f"{'='*60}\n")

        # Import training modules
        from dataset import TTSDataset
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        from transformers import AutoConfig
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from accelerate import Accelerator
        from safetensors.torch import save_file

        # Setup accelerator
        logging_dir = self.output_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision="bf16",
            log_with="tensorboard",
            project_dir=str(logging_dir),
        )

        # Load model with detected attention implementation
        print(f"Loading model: {self.init_model_path}")
        print(f"Using attention implementation: {self.attn_implementation}")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            self.init_model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
        )

        config = AutoConfig.from_pretrained(self.init_model_path)

        # Load training data
        train_data = open(self.train_with_codes_jsonl).readlines()
        train_data = [json.loads(line) for line in train_data]

        print(f"Training on {len(train_data)} samples")

        # Create dataset and dataloader
        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )

        # Setup optimizer
        optimizer = AdamW(qwen3tts.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Prepare with accelerator
        model, optimizer, train_dataloader = accelerator.prepare(
            qwen3tts.model, optimizer, train_dataloader
        )

        # Extract target speaker embedding from the clean reference audio explicitly
        print("Extracting target speaker embedding from reference.wav...")
        import librosa
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        ref_audio_np, sr = librosa.load(str(self.ref_audio), sr=None, mono=True)
        if ref_audio_np.ndim > 1:
            ref_audio_np = np.mean(ref_audio_np, axis=-1)
        ref_audio_np = librosa.resample(ref_audio_np, orig_sr=sr, target_sr=24000)
        
        ref_mels = mel_spectrogram(
            torch.tensor(ref_audio_np).unsqueeze(0).to(torch.float32),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        
        # Now qwen3tts.model is on accelerator.device
        target_speaker_embedding = qwen3tts.model.speaker_encoder(
            ref_mels.to(accelerator.device).to(self.torch_dtype)
        ).detach()

        model.train()

        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    codec_ids = batch["codec_ids"]
                    ref_mels = batch["ref_mels"]
                    text_embedding_mask = batch["text_embedding_mask"]
                    codec_embedding_mask = batch["codec_embedding_mask"]
                    attention_mask = batch["attention_mask"]
                    codec_0_labels = batch["codec_0_labels"]
                    codec_mask = batch["codec_mask"]

                    # Get speaker embedding
                    speaker_embedding = model.speaker_encoder(
                        ref_mels.to(model.device).to(model.dtype)
                    ).detach()

                    input_text_ids = input_ids[:, :, 0]
                    input_codec_ids = input_ids[:, :, 1]

                    input_text_embedding = (
                        model.talker.model.text_embedding(input_text_ids)
                        * text_embedding_mask
                    )
                    input_codec_embedding = (
                        model.talker.model.codec_embedding(input_codec_ids)
                        * codec_embedding_mask
                    )
                    input_codec_embedding[:, 6, :] = speaker_embedding

                    input_embeddings = input_text_embedding + input_codec_embedding

                    # Add codec embeddings for each layer
                    for i in range(1, 16):
                        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[
                            i - 1
                        ](codec_ids[:, :, i])
                        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + codec_i_embedding

                    # Forward pass
                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True,
                    )

                    hidden_states = outputs.hidden_states[0][-1]
                    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                    talker_codec_ids = codec_ids[codec_mask]

                    sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )

                    loss = outputs.loss + sub_talker_loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                if step % 10 == 0:
                    accelerator.print(
                        f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                    )

            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0 and accelerator.is_main_process:
                output_dir = os.path.join(
                    str(self.output_dir), f"checkpoint-epoch-{epoch}"
                )

                # Get the actual model path (could be HF cache or local)
                from huggingface_hub import snapshot_download
                if os.path.isdir(self.init_model_path):
                    model_cache_path = self.init_model_path
                else:
                    # Download/get cached path for HuggingFace model
                    model_cache_path = snapshot_download(self.init_model_path)

                shutil.copytree(model_cache_path, output_dir, dirs_exist_ok=True)

                # Update config
                input_config_file = os.path.join(model_cache_path, "config.json")
                output_config_file = os.path.join(output_dir, "config.json")

                with open(input_config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)

                config_dict["tts_model_type"] = "custom_voice"

                talker_config = config_dict.get("talker_config", {})
                talker_config["spk_id"] = {self.speaker_name: 3000}
                talker_config["spk_is_dialect"] = {self.speaker_name: False}
                config_dict["talker_config"] = talker_config

                with open(output_config_file, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                # Save model weights
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {
                    k: v.detach().to("cpu").to(torch.float32)
                    for k, v in unwrapped_model.state_dict().items()
                }

                # Drop speaker encoder keys
                drop_prefix = "speaker_encoder"
                keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
                for k in keys_to_drop:
                    del state_dict[k]

                # Add speaker embedding
                weight = state_dict["talker.model.codec_embedding.weight"]
                state_dict["talker.model.codec_embedding.weight"][
                    3000
                ] = (
                    target_speaker_embedding[0]
                    .detach()
                    .to(weight.device)
                    .to(weight.dtype)
                )

                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)
                print(f"Saved checkpoint to {output_dir}")

        print("\nTraining complete!")

    def run(self) -> None:
        """Run the complete pipeline."""
        print(f"\n{'='*60}")
        print("Qwen3-TTS End-to-End Fine-Tuning Pipeline")
        print(f"{'='*60}\n")

        # Check dependencies
        self.check_dependencies()

        # Validate inputs
        audio_files = self.validate_audio_files()

        # Transcribe if needed
        if self.train_raw_jsonl.exists():
            print(f"\nSkipping transcription, found existing {self.train_raw_jsonl.name}")
        else:
            transcription_results = self.transcribe_with_parakeet(audio_files)

            if not transcription_results:
                raise ValueError("No transcriptions were generated. Please check your audio files.")

            # Create JSONL
            self.create_train_jsonl(transcription_results)

        # Prepare data if needed
        if self.train_with_codes_jsonl.exists():
            print(f"\nSkipping data prep, found existing {self.train_with_codes_jsonl.name}")
        else:
            self.prepare_data()

        # Train
        self.train_model()
        
        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"Checkpoints saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS End-to-End Fine-Tuning Pipeline"
    )

    # Input arguments
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing WAV files to use for training",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio file (WAV)",
    )
    parser.add_argument(
        "--speaker_name",
        type=str,
        default="my_speaker",
        help="Name for the speaker being cloned",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save checkpoints and intermediate files",
    )

    # Model arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training",
    )
    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path to initial model for fine-tuning",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--transcribe_batch_size",
        type=int,
        default=4,
        help="Batch size for NeMo Parakeet transcription",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )

    # Language arguments
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Language code (e.g., 'en', 'zh', 'es') or 'auto' for auto-detection",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = Qwen3TTSPipeline(
        audio_dir=args.audio_dir,
        ref_audio=args.ref_audio,
        speaker_name=args.speaker_name,
        output_dir=args.output_dir,
        device=args.device,
        tokenizer_model_path=args.tokenizer_model_path,
        init_model_path=args.init_model_path,
        batch_size=args.batch_size,
        transcribe_batch_size=args.transcribe_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        save_every_n_epochs=args.save_every_n_epochs,
        language=args.language,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
