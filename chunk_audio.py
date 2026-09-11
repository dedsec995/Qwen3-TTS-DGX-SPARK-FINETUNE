import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Everything downstream of this script is 24 kHz: the Qwen3-TTS audio tokenizer
# resamples to 24 kHz internally, extract_speaker_embedding hard-asserts sr == 24000,
# and the speaker-encoder mel is built with fmax=12000. Writing chunks at any lower
# rate silently discards the 8-12 kHz band that the model actually models.
TARGET_SR = 24000

AUDIO_GLOBS = ("*.wav", "*.WAV", "*.mp3", "*.flac", "*.m4a", "*.ogg")


def find_audio_files(directory):
    files = []
    for pattern in AUDIO_GLOBS:
        files.extend(directory.glob(pattern))
    return sorted(set(files))


def analyze_loudness(audio, frame_ms=50, gate_below_p95_db=25.0):
    """Frame-wise energy stats, in dBFS.

    Uses speech-active RMS rather than AudioSegment.dBFS, because the latter moves
    with however much silence a recording happens to contain -- which is exactly the
    thing that makes two speakers' levels look different when they are not.
    """
    samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)

    if samples.size == 0:
        return {"speech_dbfs": -120.0, "noise_dbfs": -120.0, "peak_dbfs": -120.0, "active_frac": 0.0}

    frame_len = max(1, int(audio.frame_rate * frame_ms / 1000))
    n_frames = len(samples) // frame_len
    if n_frames < 1:
        frames = samples[np.newaxis, :]
    else:
        frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)

    power = (frames ** 2).mean(axis=1)
    db = 10.0 * np.log10(np.maximum(power, 1e-12))

    gate = np.percentile(db, 95) - gate_below_p95_db
    active = db >= gate
    speech_power = power[active].mean() if active.any() else power.mean()

    return {
        "speech_dbfs": float(10.0 * np.log10(max(float(speech_power), 1e-12))),
        "noise_dbfs": float(np.percentile(db, 10)),
        "peak_dbfs": float(20.0 * np.log10(max(float(np.abs(samples).max()), 1e-12))),
        "active_frac": float(active.mean()),
    }


def normalize_loudness(audio, target_dbfs=-23.0, peak_ceiling_dbfs=-1.0):
    """Apply a single scalar gain to the whole file.

    Deliberately per-file, not per-chunk: per-chunk normalization flattens
    intra-utterance dynamics (the model learns every utterance has identical loudness)
    and boosts the noise floor of quiet takes by whatever gain they needed individually.
    """
    stats = analyze_loudness(audio)
    gain = target_dbfs - stats["speech_dbfs"]
    headroom = peak_ceiling_dbfs - stats["peak_dbfs"]
    clamped = gain > headroom
    if clamped:
        gain = headroom
    return audio.apply_gain(gain), gain, clamped, stats


def _capitalize_first_alpha(text: str) -> str:
    """Ensure the first alphabetical character in the text is uppercase."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            if ch.isupper():
                return text
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def sentence_stats(texts):
    """Calculate terminal punctuation, capitalization, and median word count."""
    valid = [t.strip() for t in texts if t and t.strip()]
    if not valid:
        return {"terminal_frac": 0.0, "capital_frac": 0.0, "median_words": 0.0}
    term = sum(bool(t[-1] in ".?!") for t in valid)
    cap = 0
    for t in valid:
        words = t.split()
        if not words:
            continue
        first_token = words[0].strip("\"'“‘([")
        if first_token and (first_token[0].isupper() or first_token[0].isdigit()):
            cap += 1
    word_counts = [len(t.split()) for t in valid]
    med_w = float(np.median(word_counts)) if word_counts else 0.0
    return {
        "terminal_frac": round(term / len(valid), 3),
        "capital_frac": round(cap / len(valid), 3),
        "median_words": round(med_w, 1),
    }


def _drop_and_pad(spans, total_ms, min_ms, pad_ms):
    """Drop spans below min_ms and apply clamped padding.

    Each item in spans must be a tuple/list whose first two elements are (start_ms, end_ms).
    Returns (padded_kept, dropped).
    """
    kept = [s for s in spans if (s[1] - s[0]) >= min_ms]
    dropped = [s for s in spans if (s[1] - s[0]) < min_ms]

    padded = []
    for i, item in enumerate(kept):
        start, end = item[0], item[1]
        pad_start = max(0, start - pad_ms)
        pad_end = min(total_ms, end + pad_ms)
        if i > 0:
            pad_start = max(pad_start, (kept[i - 1][1] + start) // 2)
        if i < len(kept) - 1:
            pad_end = min(pad_end, (end + kept[i + 1][0]) // 2)
        padded.append((pad_start, pad_end, item))

    return padded, dropped


def _split_group(group, max_ms):
    """Split one group of contiguous speech ranges so no span exceeds max_ms."""
    start, end = group[0][0], group[-1][1]
    if end - start <= max_ms:
        return [(start, end)]

    if len(group) == 1:
        spans = []
        pos = start
        while pos < end:
            spans.append((pos, min(pos + max_ms, end)))
            pos += max_ms
        return spans

    mid = (start + end) / 2.0
    best_i = min(
        range(len(group) - 1),
        key=lambda i: abs(((group[i][1] + group[i + 1][0]) / 2.0) - mid),
    )
    return _split_group(group[: best_i + 1], max_ms) + _split_group(group[best_i + 1 :], max_ms)


def pack_ranges(ranges, total_ms, min_ms, max_ms, pad_ms, merge_gap_ms):
    """Merge/split detected speech ranges into chunks with min_ms <= len <= max_ms.

    Returns (kept, dropped_short) as lists of (start_ms, end_ms).
    """
    if not ranges:
        return [], []

    groups = []
    current = [tuple(ranges[0])]
    for start, end in ranges[1:]:
        gap = start - current[-1][1]
        if gap <= merge_gap_ms and (end - current[0][0]) <= max_ms:
            current.append((start, end))
        else:
            groups.append(current)
            current = [(start, end)]
    groups.append(current)

    spans = []
    for group in groups:
        spans.extend(_split_group(group, max_ms))
    spans.sort()

    padded, dropped = _drop_and_pad(spans, total_ms, min_ms, pad_ms)
    kept = [(p[0], p[1]) for p in padded]
    return kept, dropped


# ------------------------------------------------------------------ ASR & Sentences


def load_asr_model(asr_model_name: str, device: str = "cuda:0"):
    """Load NeMo ASR model once."""
    import nemo.collections.asr as nemo_asr
    import torch

    print(f"Loading ASR model for sentence-aware chunking: {asr_model_name}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model_name)
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def _snap_to_zero_crossing(samples, target_idx, search_radius=240):
    """Snap target_idx to the nearest zero-crossing within search_radius samples."""
    lo = max(0, target_idx - search_radius)
    hi = min(len(samples) - 1, target_idx + search_radius)
    if hi <= lo:
        return target_idx
    window = samples[lo:hi]
    signs = np.signbit(window)
    crossings = np.where(signs[:-1] != signs[1:])[0]
    if len(crossings) == 0:
        return lo + int(np.argmin(np.abs(window)))
    best_c = min(crossings, key=lambda c: abs((lo + c) - target_idx))
    return lo + int(best_c)


def _find_acoustic_boundary_ms(
    samples,
    target_ms,
    search_left_ms,
    search_right_ms,
    is_end=True,
    frame_ms=10,
    sample_rate=TARGET_SR,
):
    """Find the quietest acoustic frame (energy valley) and snap to zero-crossing."""
    search_left_ms = max(0, search_left_ms)
    search_right_ms = max(search_left_ms, search_right_ms)
    if search_right_ms <= search_left_ms:
        return target_ms

    left_idx = int(search_left_ms * sample_rate / 1000)
    right_idx = int(search_right_ms * sample_rate / 1000)
    target_idx = int(target_ms * sample_rate / 1000)
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, frame_len // 2)

    window = samples[left_idx:right_idx]
    if len(window) < frame_len * 2:
        snapped_sample = _snap_to_zero_crossing(samples, target_idx)
        return int(round(snapped_sample * 1000.0 / sample_rate))

    n_frames = (len(window) - frame_len) // hop_len + 1
    frames = np.lib.stride_tricks.sliding_window_view(window, frame_len)[::hop_len]
    energies = np.mean(frames ** 2, axis=1)

    if is_end:
        # For end boundary, give preference to frames 40-120ms after target_ms to capture vocal decay
        target_frame_idx = max(0, int((target_idx - left_idx) / hop_len))
        decay_bonus = np.zeros_like(energies)
        min_decay_frames = int(40 * sample_rate / 1000 / hop_len)
        decay_bonus[: min(len(decay_bonus), target_frame_idx + min_decay_frames)] += 1e-4
        best_frame = int(np.argmin(energies + decay_bonus))
    else:
        best_frame = int(np.argmin(energies))

    best_sample = left_idx + best_frame * hop_len + frame_len // 2
    snapped_sample = _snap_to_zero_crossing(samples, best_sample)
    return int(round(snapped_sample * 1000.0 / sample_rate))


def apply_micro_fade(audio_chunk, fade_ms=10):
    """Apply smooth micro-fade to chunk edges to prevent boundary clicks."""
    if len(audio_chunk) < fade_ms * 2:
        return audio_chunk
    return audio_chunk.fade_in(fade_ms).fade_out(fade_ms)


def _split_sentence_at_clause_or_pause(
    seg,
    min_ms,
    max_ms,
    clause_min_pause_ms=120,
    silence_min_pause_ms=250,
    drop_unsplit_oversized=True,
):
    """Split an oversized sentence at natural clause boundaries or acoustic pauses.

    Hierarchy of candidate split points:
    1. Major clause punctuation (;, :, --, ...) or comma (,) with pause >= clause_min_pause_ms.
    2. Discourse conjunction (and, but, because, so, which, etc.) with pause >= clause_min_pause_ms.
    3. Comma with any pause >= 50ms.
    4. Unpunctuated acoustic silence valley >= silence_min_pause_ms.

    If no valid pause/clause exists:
    - If drop_unsplit_oversized: returns [] (drops run-on to prevent prosody corruption).
    - Else: falls back to midpoint cut.

    Splits normalize text: left fragment receives terminal '.', right fragment is capitalized.
    """
    start, end = seg["start_ms"], seg["end_ms"]
    dur = end - start
    if dur <= max_ms:
        return [seg]

    words = seg.get("words", [])
    if len(words) < 2:
        if drop_unsplit_oversized:
            return []
        spans = []
        pos = start
        while pos < end:
            p_end = min(pos + max_ms, end)
            spans.append({
                "start_ms": pos,
                "end_ms": p_end,
                "text": None,
                "words": [],
                "hard_cut": True,
            })
            pos += max_ms
        return spans

    CONJUNCTIONS = (
        "and", "but", "because", "so", "which", "although", "however",
        "whereas", "while", "then", "since", "yet", "or",
    )

    candidates = []
    mid = (start + end) / 2.0

    for i in range(len(words) - 1):
        w_left = words[i]
        w_right = words[i + 1]
        left_dur = w_left["end_ms"] - start
        right_dur = end - w_right["start_ms"]

        if left_dur < min_ms or left_dur > max_ms:
            continue
        if right_dur < min_ms and right_dur < max_ms:
            continue

        pause_ms = max(0, w_right["start_ms"] - w_left["end_ms"])
        token_left = w_left["word"].strip()
        token_right = w_right["word"].strip().lower().strip(".,?!\"'")

        score = 0
        split_type = "none"

        # Tier 1: Major clause punctuation or comma with pause
        if token_left and (token_left[-1] in ";:—–" or token_left.endswith("--") or token_left.endswith("...")):
            score = 1000 + min(pause_ms, 500)
            split_type = "major_punct"
        elif token_left and token_left[-1] == "," and pause_ms >= clause_min_pause_ms:
            score = 850 + min(pause_ms, 500)
            split_type = "comma_pause"
        # Tier 2: Discourse conjunction with acoustic pause
        elif token_right in CONJUNCTIONS and pause_ms >= clause_min_pause_ms:
            score = 700 + min(pause_ms, 500)
            split_type = "conjunction_pause"
        # Tier 3: Comma with minor pause
        elif token_left and token_left[-1] == "," and pause_ms >= 50:
            score = 550 + min(pause_ms, 500)
            split_type = "comma_minor"
        # Tier 4: Significant acoustic pause without punctuation
        elif pause_ms >= silence_min_pause_ms:
            score = 450 + min(pause_ms, 500)
            split_type = "acoustic_pause"

        if score > 0:
            dist_mid = abs(((w_left["end_ms"] + w_right["start_ms"]) / 2.0) - mid)
            adj_score = score - (dist_mid / 50.0)
            candidates.append((adj_score, i, split_type))

    if not candidates:
        if drop_unsplit_oversized:
            return []
        best_i = min(
            range(len(words) - 1),
            key=lambda i: abs(((words[i]["end_ms"] + words[i + 1]["start_ms"]) / 2.0) - mid),
        )
    else:
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_i = candidates[0][1]

    left_words = words[: best_i + 1]
    right_words = words[best_i + 1 :]

    # Normalize left text (must end in terminal punctuation)
    left_raw = " ".join(w["word"] for w in left_words).strip()
    if left_raw:
        if left_raw[-1] in ",;:":
            left_raw = left_raw[:-1] + "."
        elif left_raw[-1] not in ".?!":
            left_raw = left_raw + "."

    # Normalize right text (must start with capitalized letter)
    right_raw = " ".join(w["word"] for w in right_words).strip()
    if right_raw:
        right_raw = right_raw[0].upper() + right_raw[1:]

    left_seg = {
        "start_ms": left_words[0]["start_ms"],
        "end_ms": left_words[-1]["end_ms"],
        "text": left_raw,
        "words": left_words,
        "clause_split": True,
    }
    right_seg = {
        "start_ms": right_words[0]["start_ms"],
        "end_ms": right_words[-1]["end_ms"],
        "text": right_raw,
        "words": right_words,
        "clause_split": True,
    }

    right_results = _split_sentence_at_clause_or_pause(
        right_seg, min_ms, max_ms, clause_min_pause_ms, silence_min_pause_ms, drop_unsplit_oversized
    )
    if not right_results and drop_unsplit_oversized:
        return [left_seg]
    return [left_seg] + right_results


def pack_segments(
    segments,
    total_ms,
    min_ms,
    max_ms,
    pad_ms,
    segment_gap_ms,
    orphan_gap_ms,
    clause_min_pause_ms=120,
    silence_min_pause_ms=250,
    drop_unsplit_oversized=True,
):
    """Pack sentences into training chunks using clause-aware boundaries.

    Returns (kept_chunks, dropped_spans).
    Each item in kept_chunks is a dict with start_ms, end_ms, text, n_segments, clause_split, hard_cut.
    """
    if not segments:
        return [], []

    # 1. Forward merge sentences across gaps
    groups = []
    current = [segments[0]]
    for seg in segments[1:]:
        gap = seg["start_ms"] - current[-1]["end_ms"]
        curr_dur = current[-1]["end_ms"] - current[0]["start_ms"]
        allowed_gap = orphan_gap_ms if curr_dur < min_ms else segment_gap_ms

        if gap <= allowed_gap and (seg["end_ms"] - current[0]["start_ms"]) <= max_ms:
            current.append(seg)
        else:
            groups.append(current)
            current = [seg]
    groups.append(current)

    # 2. Split any group or single sentence that exceeds max_ms
    spans = []
    dropped_spans = []
    for group in groups:
        grp_start = group[0]["start_ms"]
        grp_end = group[-1]["end_ms"]
        valid_texts = [s["text"] for s in group if s.get("text")]
        grp_text = " ".join(valid_texts).strip() if valid_texts else None
        all_words = []
        for s in group:
            all_words.extend(s.get("words", []))

        if grp_end - grp_start <= max_ms:
            spans.append({
                "start_ms": grp_start,
                "end_ms": grp_end,
                "text": grp_text,
                "words": all_words,
                "n_segments": len(group),
                "clause_split": any(s.get("clause_split", False) for s in group),
                "hard_cut": False,
            })
        else:
            if len(group) > 1:
                mid = (grp_start + grp_end) / 2.0
                best_i = min(
                    range(len(group) - 1),
                    key=lambda i: abs(((group[i]["end_ms"] + group[i + 1]["start_ms"]) / 2.0) - mid),
                )
                left_k, left_d = pack_segments(
                    group[: best_i + 1], total_ms, min_ms, max_ms, pad_ms, segment_gap_ms, orphan_gap_ms,
                    clause_min_pause_ms, silence_min_pause_ms, drop_unsplit_oversized
                )
                right_k, right_d = pack_segments(
                    group[best_i + 1 :], total_ms, min_ms, max_ms, pad_ms, segment_gap_ms, orphan_gap_ms,
                    clause_min_pause_ms, silence_min_pause_ms, drop_unsplit_oversized
                )
                spans.extend(left_k)
                spans.extend(right_k)
                dropped_spans.extend(left_d)
                dropped_spans.extend(right_d)
            else:
                split_segs = _split_sentence_at_clause_or_pause(
                    group[0], min_ms, max_ms, clause_min_pause_ms, silence_min_pause_ms, drop_unsplit_oversized
                )
                if not split_segs:
                    dropped_spans.append((grp_start, grp_end))
                else:
                    for s in split_segs:
                        s["n_segments"] = 1
                        spans.append(s)

    # 3. Filter spans: must satisfy min_ms <= dur <= max_ms
    kept_chunks = []
    for s in spans:
        dur = s["end_ms"] - s["start_ms"]
        if dur >= min_ms:
            kept_chunks.append(s)
        else:
            dropped_spans.append((s["start_ms"], s["end_ms"]))

    kept_chunks.sort(key=lambda s: s["start_ms"])
    return kept_chunks, dropped_spans


def transcribe_windows_and_build_sentences(audio_file, audio_24k, ranges, asr_model, args):
    """Two-level sentence-aware transcription:
    1. Cut audio in silence into ~20s ASR windows.
    2. Transcribe windows with word timestamps.
    3. Reconstruct absolute timestamps and build sentences.
    """
    import torch

    win_min_ms = int(args.asr_window_sec * 500)
    win_max_ms = int(args.asr_window_sec * 1250)
    windows, _ = pack_ranges(ranges, len(audio_24k), win_min_ms, win_max_ms, pad_ms=0, merge_gap_ms=args.merge_gap_ms)
    if not windows:
        windows = [(0, len(audio_24k))]

    cache_dir = Path(args.asr_cache_dir) if args.asr_cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    st = audio_file.stat()
    file_tag = f"{audio_file.name}_{st.st_size}_{st.st_mtime_ns}"

    uncached_windows, uncached_audio_16k, win_results = [], [], {}
    for idx, (w_start, w_end) in enumerate(windows):
        key = hashlib.sha1(f"{file_tag}_{args.asr_model}_{w_start}_{w_end}".encode()).hexdigest()
        c_path = cache_dir / f"{key}.json" if cache_dir else None
        if c_path and c_path.exists():
            try:
                win_results[idx] = json.loads(c_path.read_text())
                continue
            except Exception:
                pass

        win_seg = audio_24k[w_start:w_end]
        samples_24k = np.frombuffer(win_seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        if win_seg.channels > 1:
            samples_24k = samples_24k.reshape(-1, win_seg.channels).mean(axis=1)

        samples_16k = librosa.resample(samples_24k, orig_sr=24000, target_sr=16000)
        uncached_windows.append((idx, w_start, w_end, c_path))
        uncached_audio_16k.append(samples_16k)

    if uncached_windows:
        batch_size = getattr(args, "asr_batch_size", 8)
        for i in range(0, len(uncached_audio_16k), batch_size):
            b_audio = uncached_audio_16k[i : i + batch_size]
            b_meta = uncached_windows[i : i + batch_size]
            with torch.inference_mode():
                hyps = asr_model.transcribe(b_audio, timestamps=True, batch_size=len(b_audio))

            for hyp, (idx, w_start, w_end, c_path) in zip(hyps, b_meta):
                text = hyp.text if hasattr(hyp, "text") else str(hyp)
                words = []
                if hasattr(hyp, "timestamp") and isinstance(hyp.timestamp, dict):
                    raw_words = hyp.timestamp.get("word", [])
                    for w in raw_words:
                        # Fallback for start/end if offset only
                        w_start_s = w.get("start", w.get("start_offset", 0) * 0.08)
                        w_end_s = w.get("end", w.get("end_offset", 0) * 0.08)
                        words.append({
                            "word": w.get("word", ""),
                            "start": float(w_start_s),
                            "end": float(w_end_s),
                        })

                res_data = {"text": text.strip(), "words": words}
                win_results[idx] = res_data
                if c_path:
                    c_path.write_text(json.dumps(res_data, ensure_ascii=False))

    # Map words to absolute file timestamps
    all_words = []
    for idx, (w_start, w_end) in enumerate(windows):
        w_res = win_results.get(idx, {})
        for w in w_res.get("words", []):
            abs_start = w_start + int(round(w["start"] * 1000))
            abs_end = w_start + int(round(w["end"] * 1000))
            all_words.append({
                "word": w["word"],
                "start_ms": abs_start,
                "end_ms": max(abs_start + 10, abs_end),
            })

    if not all_words:
        # Fallback: single segment over speech ranges
        return [{
            "start_ms": ranges[0][0],
            "end_ms": ranges[-1][1],
            "text": " ".join(win_results[i]["text"] for i in sorted(win_results) if win_results[i]["text"]).strip(),
            "words": [],
        }]

    # Construct sentences on terminal punctuation (. ? !)
    sentences = []
    curr_sentence = []
    for w in all_words:
        curr_sentence.append(w)
        word_token = w["word"].strip()
        if word_token and word_token[-1] in ".?!":
            sent_text = _capitalize_first_alpha(" ".join(cw["word"] for cw in curr_sentence).strip())
            sentences.append({
                "start_ms": curr_sentence[0]["start_ms"],
                "end_ms": curr_sentence[-1]["end_ms"],
                "text": sent_text,
                "words": list(curr_sentence),
            })
            curr_sentence = []

    if curr_sentence:
        raw_text = " ".join(cw["word"] for cw in curr_sentence).strip()
        if len(curr_sentence) >= 2 and raw_text:
            if raw_text[-1] in ",;:":
                raw_text = raw_text[:-1] + "."
            elif raw_text[-1] not in ".?!":
                raw_text = raw_text + "."
            sentences.append({
                "start_ms": curr_sentence[0]["start_ms"],
                "end_ms": curr_sentence[-1]["end_ms"],
                "text": _capitalize_first_alpha(raw_text),
                "words": list(curr_sentence),
            })

    return sentences


# ------------------------------------------------------------------ Main Chunking Logic


def chunk_audio_files(audio_files, output_path, args, asr_model=None):
    """Chunk one speaker's audio files into output_path. Returns list of manifest rows."""
    output_path.mkdir(parents=True, exist_ok=True)
    min_ms = int(args.min_length_sec * 1000)
    max_ms = int(args.max_length_sec * 1000)

    manifest = []
    chunk_counter = 0
    total_dropped_ms = 0
    n_clause_split = 0
    n_hard_cut = 0

    for audio_file in audio_files:
        print(f"  Processing {audio_file.name}...")
        try:
            audio = (
                AudioSegment.from_file(str(audio_file))
                .set_channels(1)
                .set_frame_rate(args.sample_rate)
                .set_sample_width(2)
                .remove_dc_offset()
            )
        except Exception as exc:
            print(f"    ERROR loading {audio_file}: {exc}")
            continue

        audio, gain_db, clamped, pre_stats = normalize_loudness(
            audio, args.target_dbfs, args.peak_ceiling_dbfs
        )
        stats = analyze_loudness(audio)

        print(
            f"    speech={pre_stats['speech_dbfs']:.1f} noise={pre_stats['noise_dbfs']:.1f} "
            f"peak={pre_stats['peak_dbfs']:.1f} dBFS -> gain {gain_db:+.1f} dB"
            + ("  [PEAK-CLAMPED: --target_dbfs is too hot]" if clamped else "")
        )

        silence_thresh = max(
            stats["noise_dbfs"] + 6.0, stats["speech_dbfs"] - args.silence_offset_db
        )

        ranges = detect_nonsilent(
            audio,
            min_silence_len=args.min_silence_ms,
            silence_thresh=silence_thresh,
            seek_step=args.seek_step_ms,
        )

        if not ranges:
            print(f"    WARNING: No speech detected in {audio_file.name}")
            continue

        samples_24k = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        if not args.no_sentence_aware and asr_model is not None:
            sentences = transcribe_windows_and_build_sentences(
                audio_file, audio, ranges, asr_model, args
            )
            kept, dropped = pack_segments(
                sentences,
                len(audio),
                min_ms,
                max_ms,
                args.pad_ms,
                args.segment_gap_ms,
                args.orphan_gap_ms,
                clause_min_pause_ms=args.clause_min_pause_ms,
                silence_min_pause_ms=args.silence_min_pause_ms,
                drop_unsplit_oversized=not args.no_drop_unsplit_oversized,
            )
            total_dropped_ms += sum(e - s for s, e in dropped)

            for k, chunk_meta in enumerate(kept):
                raw_start = chunk_meta["start_ms"]
                raw_end = chunk_meta["end_ms"]
                prev_end = kept[k - 1]["end_ms"] if k > 0 else 0
                next_start = kept[k + 1]["start_ms"] if k < len(kept) - 1 else len(audio)

                lead_min_ms = max(prev_end, raw_start - 250)
                snapped_start = _find_acoustic_boundary_ms(
                    samples_24k,
                    target_ms=raw_start,
                    search_left_ms=lead_min_ms,
                    search_right_ms=raw_start,
                    is_end=False,
                    sample_rate=args.sample_rate,
                )

                tail_max_ms = min(next_start, raw_end + 350)
                snapped_end = _find_acoustic_boundary_ms(
                    samples_24k,
                    target_ms=raw_end,
                    search_left_ms=raw_end,
                    search_right_ms=tail_max_ms,
                    is_end=True,
                    sample_rate=args.sample_rate,
                )

                if snapped_end - snapped_start < min_ms:
                    snapped_start = max(0, raw_start - args.pad_ms)
                    snapped_end = min(len(audio), raw_end + args.pad_ms)

                chunk = audio[snapped_start:snapped_end]
                chunk = apply_micro_fade(chunk, fade_ms=args.fade_ms)
                out_file = output_path / f"chunk_{chunk_counter:04d}.wav"
                chunk.export(str(out_file), format="wav")

                if chunk_meta.get("clause_split"):
                    n_clause_split += 1
                if chunk_meta.get("hard_cut"):
                    n_hard_cut += 1

                cstats = analyze_loudness(chunk)
                manifest.append({
                    "file": out_file.name,
                    "source": audio_file.name,
                    "start_ms": int(snapped_start),
                    "end_ms": int(snapped_end),
                    "duration_s": round((snapped_end - snapped_start) / 1000.0, 3),
                    "speech_dbfs": round(cstats["speech_dbfs"], 2),
                    "noise_dbfs": round(cstats["noise_dbfs"], 2),
                    "peak_dbfs": round(cstats["peak_dbfs"], 2),
                    "active_frac": round(cstats["active_frac"], 3),
                    "snr_db": round(cstats["speech_dbfs"] - cstats["noise_dbfs"], 2),
                    "text": chunk_meta.get("text"),
                    "n_segments": chunk_meta.get("n_segments", 1),
                })
                chunk_counter += 1

            print(
                f"    silence_thresh={silence_thresh:.1f} dBFS | {len(sentences)} sentence(s) "
                f"-> {len(kept)} chunks kept, {len(dropped)} below {args.min_length_sec}s dropped"
            )
        else:
            kept, dropped = pack_ranges(
                ranges, len(audio), min_ms, max_ms, args.pad_ms, args.merge_gap_ms
            )
            total_dropped_ms += sum(e - s for s, e in dropped)

            for k, (start_ms, end_ms) in enumerate(kept):
                prev_end = kept[k - 1][1] if k > 0 else 0
                next_start = kept[k + 1][0] if k < len(kept) - 1 else len(audio)

                lead_min_ms = max(prev_end, start_ms - 250)
                snapped_start = _find_acoustic_boundary_ms(
                    samples_24k,
                    target_ms=start_ms,
                    search_left_ms=lead_min_ms,
                    search_right_ms=start_ms,
                    is_end=False,
                    sample_rate=args.sample_rate,
                )

                tail_max_ms = min(next_start, end_ms + 350)
                snapped_end = _find_acoustic_boundary_ms(
                    samples_24k,
                    target_ms=end_ms,
                    search_left_ms=end_ms,
                    search_right_ms=tail_max_ms,
                    is_end=True,
                    sample_rate=args.sample_rate,
                )

                if snapped_end - snapped_start < min_ms:
                    snapped_start = max(0, start_ms - args.pad_ms)
                    snapped_end = min(len(audio), end_ms + args.pad_ms)

                chunk = audio[snapped_start:snapped_end]
                chunk = apply_micro_fade(chunk, fade_ms=args.fade_ms)
                out_file = output_path / f"chunk_{chunk_counter:04d}.wav"
                chunk.export(str(out_file), format="wav")

                cstats = analyze_loudness(chunk)
                manifest.append({
                    "file": out_file.name,
                    "source": audio_file.name,
                    "start_ms": int(snapped_start),
                    "end_ms": int(snapped_end),
                    "duration_s": round((snapped_end - snapped_start) / 1000.0, 3),
                    "speech_dbfs": round(cstats["speech_dbfs"], 2),
                    "noise_dbfs": round(cstats["noise_dbfs"], 2),
                    "peak_dbfs": round(cstats["peak_dbfs"], 2),
                    "active_frac": round(cstats["active_frac"], 3),
                    "snr_db": round(cstats["speech_dbfs"] - cstats["noise_dbfs"], 2),
                    "text": None,
                    "n_segments": 1,
                })
                chunk_counter += 1

            print(
                f"    silence_thresh={silence_thresh:.1f} dBFS | {len(ranges)} speech ranges "
                f"-> {len(kept)} chunks kept, {len(dropped)} below {args.min_length_sec}s dropped"
            )

    if manifest:
        with open(output_path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    durations = [row["duration_s"] for row in manifest]
    texts = [row["text"] for row in manifest if row.get("text")]
    s_stats = sentence_stats(texts)
    if durations:
        print(
            f"  -> {chunk_counter} chunks | duration min {min(durations):.1f}s "
            f"med {sorted(durations)[len(durations) // 2]:.1f}s max {max(durations):.1f}s "
            f"| total {sum(durations) / 60:.1f} min | dropped {total_dropped_ms / 1000:.0f}s"
        )
        if texts:
            print(
                f"  -> Sentence stats: terminal {s_stats['terminal_frac']*100:.1f}% "
                f"| capital {s_stats['capital_frac']*100:.1f}% | median words {s_stats['median_words']}"
            )

    report = {
        "n_chunks": len(manifest),
        "total_min": round(sum(durations) / 60.0, 2) if durations else 0.0,
        "dropped_sec": round(total_dropped_ms / 1000.0, 1),
        "n_clause_split": n_clause_split,
        "n_hard_cut": n_hard_cut,
        **s_stats,
    }
    with open(output_path / "chunk_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return manifest


def process_audio_files(args):
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        print(f"Input directory not found: {args.input_dir}")
        return

    speaker_subdirs = [
        d
        for d in sorted(input_path.iterdir())
        if d.is_dir() and not d.name.startswith(".") and find_audio_files(d)
    ]

    asr_model = None
    if not args.no_sentence_aware:
        try:
            asr_model = load_asr_model(args.asr_model, args.device)
        except Exception as exc:
            raise SystemExit(
                f"ERROR loading ASR model '{args.asr_model}': {exc}\n"
                f"To run without sentence-aware ASR, pass --no_sentence_aware."
            )

    total = 0
    if speaker_subdirs:
        print(
            f"Detected {len(speaker_subdirs)} speaker(s): {[d.name for d in speaker_subdirs]}"
        )
        for speaker_dir in speaker_subdirs:
            files = find_audio_files(speaker_dir)
            print(f"\n--- Speaker '{speaker_dir.name}' ({len(files)} file(s)) ---")
            rows = chunk_audio_files(files, output_path / speaker_dir.name, args, asr_model)
            total += len(rows)
    else:
        files = find_audio_files(input_path)
        if not files:
            print(f"No audio files found in {args.input_dir}")
            return
        print(f"Found {len(files)} file(s). Single-speaker (flat) mode.")
        rows = chunk_audio_files(files, output_path, args, asr_model)
        total += len(rows)

    output_path.mkdir(parents=True, exist_ok=True)
    params = {
        "version": 4,
        "sample_rate": args.sample_rate,
        "sentence_aware": not args.no_sentence_aware,
        "asr_model": args.asr_model if not args.no_sentence_aware else None,
        "asr_window_sec": args.asr_window_sec,
        "target_dbfs": args.target_dbfs,
        "peak_ceiling_dbfs": args.peak_ceiling_dbfs,
        "silence_offset_db": args.silence_offset_db,
        "min_silence_ms": args.min_silence_ms,
        "merge_gap_ms": args.merge_gap_ms,
        "segment_gap_ms": args.segment_gap_ms,
        "orphan_gap_ms": args.orphan_gap_ms,
        "clause_min_pause_ms": args.clause_min_pause_ms,
        "silence_min_pause_ms": args.silence_min_pause_ms,
        "fade_ms": args.fade_ms,
        "drop_unsplit_oversized": not args.no_drop_unsplit_oversized,
        "snap_ms": args.snap_ms,
        "pad_ms": args.pad_ms,
        "seek_step_ms": args.seek_step_ms,
        "min_length_sec": args.min_length_sec,
        "max_length_sec": args.max_length_sec,
    }
    with open(output_path / "chunk_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"\n=== Done. {total} chunks at {args.sample_rate} Hz in {args.output_dir} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split long audio into short 24 kHz TTS-ready WAV chunks with sentence-aware boundaries."
    )
    parser.add_argument("--input_dir", type=str, default="./raw_audio",
                        help="Directory with long audio files, or subdirectories per speaker")
    parser.add_argument("--output_dir", type=str, default="./audio_chunks",
                        help="Output directory for chunks")
    parser.add_argument("--min_length_sec", type=float, default=3.0,
                        help="Minimum chunk length; shorter spans are dropped, not kept")
    parser.add_argument("--max_length_sec", type=float, default=11.0,
                        help="Maximum chunk length (Qwen3-TTS handles up to 12s cleanly)")
    parser.add_argument("--sample_rate", type=int, default=TARGET_SR,
                        help="Output sample rate (Qwen3-TTS expects 24000)")
    parser.add_argument("--target_dbfs", type=float, default=-23.0,
                        help="Target speech-active RMS level per source file")
    parser.add_argument("--peak_ceiling_dbfs", type=float, default=-1.0,
                        help="Gain is clamped so true peak stays below this")
    parser.add_argument("--silence_offset_db", type=float, default=25.0,
                        help="Silence threshold = speech level minus this")
    parser.add_argument("--min_silence_ms", type=int, default=400,
                        help="Minimum silence length to split on")
    parser.add_argument("--merge_gap_ms", type=int, default=1400,
                        help="Merge adjacent silence speech ranges separated by at most this gap")
    parser.add_argument("--segment_gap_ms", type=int, default=1400,
                        help="Merge adjacent sentences separated by at most this gap")
    parser.add_argument("--orphan_gap_ms", type=int, default=3000,
                        help="Allow wider gap while group is below min_length_sec")
    parser.add_argument("--clause_min_pause_ms", type=int, default=120,
                        help="Minimum pause required to split an oversized sentence at a comma or clause mark")
    parser.add_argument("--silence_min_pause_ms", type=int, default=250,
                        help="Minimum acoustic pause required to split an oversized sentence without clause punctuation")
    parser.add_argument("--fade_ms", type=int, default=10,
                        help="Micro-fade in/out duration in milliseconds applied to chunk boundaries")
    parser.add_argument("--no_drop_unsplit_oversized", action="store_true",
                        help="Do not drop oversized run-on sentences without pauses; fall back to blind midpoint cut")
    parser.add_argument("--snap_ms", type=int, default=300,
                        help="Snap ASR cut points to nearest silence edge within this window")
    parser.add_argument("--pad_ms", type=int, default=200,
                        help="Silence padding kept on each side of a chunk")
    parser.add_argument("--seek_step_ms", type=int, default=10,
                        help="Silence-detection resolution (1 = pydub default, and very slow)")
    parser.add_argument("--asr_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2",
                        help="NeMo ASR model with punctuation and capitalization support")
    parser.add_argument("--asr_window_sec", type=float, default=20.0,
                        help="Length of ASR transcription windows in seconds")
    parser.add_argument("--asr_batch_size", type=int, default=8,
                        help="Batch size for window transcription")
    parser.add_argument("--asr_cache_dir", type=str, default=None,
                        help="Directory to cache window ASR transcripts (default: <output_dir>/.asr_cache)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_sentence_aware", action="store_true",
                        help="Disable sentence-aware ASR chunking and revert to silence-only chunking")
    args = parser.parse_args()

    if args.asr_cache_dir is None and not args.no_sentence_aware:
        args.asr_cache_dir = str(Path(args.output_dir) / ".asr_cache")

    process_audio_files(args)
