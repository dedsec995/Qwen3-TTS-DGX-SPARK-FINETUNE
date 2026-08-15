import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def chunk_wav_files(wav_files, output_path, min_length_sec=3.0, max_length_sec=10.0):
    """Split a list of WAV files into chunks of duration between min_length_sec and max_length_sec."""
    output_path.mkdir(parents=True, exist_ok=True)
    min_length_ms = int(min_length_sec * 1000)
    max_length_ms = int(max_length_sec * 1000)
    
    chunk_counter = 0
    for wav_file in wav_files:
        print(f"Processing {wav_file.name}...")
        try:
            audio = AudioSegment.from_file(str(wav_file))
        except Exception as e:
            print(f"Error loading {wav_file}: {e}")
            continue
        
        # Split on silence initially
        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=200 # keep some silence padding
        )
        
        # Combine chunks to get between min and max length
        combined_chunks = []
        current_chunk = AudioSegment.empty()
        
        for chunk in chunks:
            # If a single chunk is > max_length_ms, handle it carefully
            if len(chunk) > max_length_ms:
                # Try to re-split with lower silence threshold and duration
                sub_chunks = split_on_silence(
                    chunk,
                    min_silence_len=200,
                    silence_thresh=-35,
                    keep_silence=100
                )
                
                # If it still didn't split into smaller pieces, force split
                if len(sub_chunks) <= 1:
                    for i in range(0, len(chunk), max_length_ms):
                        sub_chunk = chunk[i:i+max_length_ms]
                        if len(sub_chunk) >= 1500: # only keep if it's decently long (>1.5s)
                            combined_chunks.append(sub_chunk)
                else:
                    # Successfully sub-split, aggregate these sub_chunks
                    sub_current = AudioSegment.empty()
                    for sub in sub_chunks:
                        if len(sub_current) + len(sub) <= max_length_ms:
                            sub_current += sub
                        else:
                            if len(sub_current) >= min_length_ms:
                                combined_chunks.append(sub_current)
                            elif len(sub_current) > 1500:
                                combined_chunks.append(sub_current)
                            sub_current = sub
                    if len(sub_current) > 1500:
                        combined_chunks.append(sub_current)
                continue

            # Normal aggregation logic for chunks <= max_length_ms
            if len(current_chunk) + len(chunk) <= max_length_ms:
                current_chunk += chunk
            else:
                # Exceeded max length, save current_chunk if long enough
                if len(current_chunk) >= min_length_ms:
                    combined_chunks.append(current_chunk)
                elif len(current_chunk) >= 1500:
                    # Keep slightly short chunks (>1.5s) to avoid data loss
                    combined_chunks.append(current_chunk)
                
                current_chunk = chunk
                
        # Handle the very last chunk
        if len(current_chunk) >= 1500:
            combined_chunks.append(current_chunk)
            
        for chunk in combined_chunks:
            # TTS models often want 16kHz mono or 24kHz
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            
            # Save
            out_file = output_path / f"chunk_{chunk_counter:04d}.wav"
            chunk.export(str(out_file), format="wav")
            chunk_counter += 1
            
    print(f"Created {chunk_counter} chunks in {output_path}")
    return chunk_counter

def process_audio_files(input_dir, output_dir, min_length_sec=3.0, max_length_sec=10.0):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory not found: {input_dir}")
        return

    # Check for speaker subdirectories (e.g. raw_audio/speaker1, raw_audio/speaker2)
    speaker_subdirs = [p for p in input_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    speaker_subdirs = [
        d for d in speaker_subdirs
        if list(d.glob("*.wav")) + list(d.glob("*.WAV")) + list(d.glob("*.mp3")) or list(d.glob("*.flac"))
    ]

    total_chunks = 0
    if speaker_subdirs:
        print(f"Detected {len(speaker_subdirs)} speaker subdirectories in {input_dir}: {[d.name for d in speaker_subdirs]}")
        for speaker_dir in speaker_subdirs:
            speaker_name = speaker_dir.name
            wav_files = (
                list(speaker_dir.glob("*.wav")) + list(speaker_dir.glob("*.WAV")) +
                list(speaker_dir.glob("*.mp3")) + list(speaker_dir.glob("*.flac"))
            )
            print(f"\n--- Chunking speaker '{speaker_name}' ({len(wav_files)} audio files) ---")
            speaker_output_dir = output_path / speaker_name
            total_chunks += chunk_wav_files(wav_files, speaker_output_dir, min_length_sec, max_length_sec)
    else:
        # Flat directory mode
        wav_files = (
            list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV")) +
            list(input_path.glob("*.mp3")) + list(input_path.glob("*.flac"))
        )
        if not wav_files:
            print(f"No audio files found in {input_dir}")
            return
            
        print(f"Found {len(wav_files)} files in {input_dir}. Chunking in single-speaker mode...")
        total_chunks = chunk_wav_files(wav_files, output_path, min_length_sec, max_length_sec)

    print(f"\n=== Done! Created {total_chunks} total chunks in {output_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split long audio files into short TTS-ready WAV chunks.")
    parser.add_argument("--input_dir", type=str, default="./raw_audio", help="Directory with long WAV/MP3/FLAC files (or subdirectories per speaker)")
    parser.add_argument("--output_dir", type=str, default="./audio_chunks", help="Output directory for short WAVs")
    parser.add_argument("--min_length_sec", type=float, default=3.0, help="Minimum chunk length in seconds")
    parser.add_argument("--max_length_sec", type=float, default=10.0, help="Maximum chunk length in seconds")
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir, args.min_length_sec, args.max_length_sec)
