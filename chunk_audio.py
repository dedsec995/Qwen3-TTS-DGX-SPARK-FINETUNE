import argparse
import json
from pathlib import Path

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


def _split_group(group, max_ms):
    """Split one group of contiguous speech ranges so no span exceeds max_ms."""
    start, end = group[0][0], group[-1][1]
    if end - start <= max_ms:
        return [(start, end)]

    if len(group) == 1:
        # A genuinely continuous utterance longer than max_ms. Hard-cut it; a cut
        # mid-phoneme is acceptable here because this is rare in real speech.
        spans = []
        pos = start
        while pos < end:
            spans.append((pos, min(pos + max_ms, end)))
            pos += max_ms
        return spans

    # Split at the internal silence gap closest to the midpoint.
    mid = (start + end) / 2.0
    best_i = min(
        range(len(group) - 1),
        key=lambda i: abs(((group[i][1] + group[i + 1][0]) / 2.0) - mid),
    )
    return _split_group(group[: best_i + 1], max_ms) + _split_group(group[best_i + 1 :], max_ms)


def pack_ranges(ranges, total_ms, min_ms, max_ms, pad_ms, merge_gap_ms):
    """Merge/split detected speech ranges into chunks with min_ms <= len <= max_ms.

    Returns (kept, dropped_short) as lists of (start_ms, end_ms).
    Nothing shorter than min_ms is ever emitted -- there is no short-chunk escape hatch.
    """
    if not ranges:
        return [], []

    # 1. Merge forward across short gaps, never past max_ms.
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

    # 2. Split anything still oversize.
    spans = []
    for group in groups:
        spans.extend(_split_group(group, max_ms))
    spans.sort()

    # 3. Enforce the floor. Unconditionally.
    kept = [s for s in spans if (s[1] - s[0]) >= min_ms]
    dropped = [s for s in spans if (s[1] - s[0]) < min_ms]

    # 4. Pad last, clamped so padding never eats into a neighbouring kept span.
    padded = []
    for i, (start, end) in enumerate(kept):
        pad_start = max(0, start - pad_ms)
        pad_end = min(total_ms, end + pad_ms)
        if i > 0:
            pad_start = max(pad_start, (kept[i - 1][1] + start) // 2)
        if i < len(kept) - 1:
            pad_end = min(pad_end, (end + kept[i + 1][0]) // 2)
        padded.append((pad_start, pad_end))

    return padded, dropped


def chunk_audio_files(audio_files, output_path, args):
    """Chunk one speaker's audio files into output_path. Returns list of manifest rows."""
    output_path.mkdir(parents=True, exist_ok=True)
    min_ms = int(args.min_length_sec * 1000)
    max_ms = int(args.max_length_sec * 1000)

    manifest = []
    chunk_counter = 0
    total_dropped_ms = 0

    for audio_file in audio_files:
        print(f"  Processing {audio_file.name}...")
        try:
            # Convert at load, not at export: silence detection then runs on exactly
            # the signal that gets written, and detection cost drops ~4x.
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

        # Threshold relative to program loudness, with a floor just above the noise
        # floor so an unusually dynamic file cannot end up as one giant segment.
        silence_thresh = max(
            stats["noise_dbfs"] + 6.0, stats["speech_dbfs"] - args.silence_offset_db
        )

        # seek_step matters a lot: pydub's detect_silence slices at every 1 ms by
        # default, which is the real reason a full-length file takes ~35 s.
        ranges = detect_nonsilent(
            audio,
            min_silence_len=args.min_silence_ms,
            silence_thresh=silence_thresh,
            seek_step=args.seek_step_ms,
        )

        kept, dropped = pack_ranges(
            ranges, len(audio), min_ms, max_ms, args.pad_ms, args.merge_gap_ms
        )
        total_dropped_ms += sum(e - s for s, e in dropped)

        for start_ms, end_ms in kept:
            chunk = audio[start_ms:end_ms]
            out_file = output_path / f"chunk_{chunk_counter:04d}.wav"
            chunk.export(str(out_file), format="wav")

            cstats = analyze_loudness(chunk)
            manifest.append(
                {
                    "file": out_file.name,
                    "source": audio_file.name,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "duration_s": round((end_ms - start_ms) / 1000.0, 3),
                    "speech_dbfs": round(cstats["speech_dbfs"], 2),
                    "noise_dbfs": round(cstats["noise_dbfs"], 2),
                    "peak_dbfs": round(cstats["peak_dbfs"], 2),
                    "active_frac": round(cstats["active_frac"], 3),
                    "snr_db": round(cstats["speech_dbfs"] - cstats["noise_dbfs"], 2),
                }
            )
            chunk_counter += 1

        print(
            f"    silence_thresh={silence_thresh:.1f} dBFS | {len(ranges)} speech ranges "
            f"-> {len(kept)} chunks kept, {len(dropped)} below {args.min_length_sec}s dropped"
        )

    if manifest:
        with open(output_path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for row in manifest:
                f.write(json.dumps(row) + "\n")

    durations = [row["duration_s"] for row in manifest]
    if durations:
        print(
            f"  -> {chunk_counter} chunks | duration min {min(durations):.1f}s "
            f"med {sorted(durations)[len(durations) // 2]:.1f}s max {max(durations):.1f}s "
            f"| total {sum(durations) / 60:.1f} min | dropped {total_dropped_ms / 1000:.0f}s"
        )
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

    total = 0
    if speaker_subdirs:
        print(
            f"Detected {len(speaker_subdirs)} speaker(s): {[d.name for d in speaker_subdirs]}"
        )
        for speaker_dir in speaker_subdirs:
            files = find_audio_files(speaker_dir)
            print(f"\n--- Speaker '{speaker_dir.name}' ({len(files)} file(s)) ---")
            rows = chunk_audio_files(files, output_path / speaker_dir.name, args)
            total += len(rows)
    else:
        files = find_audio_files(input_path)
        if not files:
            print(f"No audio files found in {args.input_dir}")
            return
        print(f"Found {len(files)} file(s). Single-speaker (flat) mode.")
        rows = chunk_audio_files(files, output_path, args)
        total += len(rows)

    output_path.mkdir(parents=True, exist_ok=True)
    params = {
        "version": 2,
        "sample_rate": args.sample_rate,
        "target_dbfs": args.target_dbfs,
        "peak_ceiling_dbfs": args.peak_ceiling_dbfs,
        "silence_offset_db": args.silence_offset_db,
        "min_silence_ms": args.min_silence_ms,
        "merge_gap_ms": args.merge_gap_ms,
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
        description="Split long audio into short 24 kHz TTS-ready WAV chunks."
    )
    parser.add_argument("--input_dir", type=str, default="./raw_audio",
                        help="Directory with long audio files, or subdirectories per speaker")
    parser.add_argument("--output_dir", type=str, default="./audio_chunks",
                        help="Output directory for chunks")
    parser.add_argument("--min_length_sec", type=float, default=3.0,
                        help="Minimum chunk length; shorter spans are dropped, not kept")
    parser.add_argument("--max_length_sec", type=float, default=10.0,
                        help="Maximum chunk length")
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
                        help="Merge adjacent speech ranges separated by at most this gap. "
                             "Tuned on 48 kHz studio speech: 700 drops ~30%% of audio to "
                             "sub-min fragments, 1400 keeps ~85%% with natural pauses intact")
    parser.add_argument("--pad_ms", type=int, default=200,
                        help="Silence padding kept on each side of a chunk")
    parser.add_argument("--seek_step_ms", type=int, default=10,
                        help="Silence-detection resolution (1 = pydub default, and very slow)")
    args = parser.parse_args()

    process_audio_files(args)
