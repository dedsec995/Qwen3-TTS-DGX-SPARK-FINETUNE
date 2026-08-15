import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def process_audio_files(input_dir, output_dir, min_length_sec=3.0, max_length_sec=10.0):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
        
    print(f"Found {len(wav_files)} files. Chunking...")
    
    min_length_ms = int(min_length_sec * 1000)
    max_length_ms = int(max_length_sec * 1000)
    
    chunk_counter = 0
    for wav_file in wav_files:
        print(f"Processing {wav_file.name}...")
        audio = AudioSegment.from_wav(str(wav_file))
        
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
            # TTS models often want 16kHz mono
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            
            # Save
            out_file = output_path / f"chunk_{chunk_counter:04d}.wav"
            chunk.export(str(out_file), format="wav")
            chunk_counter += 1
            
    print(f"Done! Created {chunk_counter} chunks in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw_audio", help="Directory with long WAV files")
    parser.add_argument("--output_dir", type=str, default="./audio_chunks", help="Output directory for short WAVs")
    parser.add_argument("--min_length_sec", type=float, default=3.0, help="Minimum chunk length in seconds")
    parser.add_argument("--max_length_sec", type=float, default=10.0, help="Maximum chunk length in seconds")
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir, args.min_length_sec, args.max_length_sec)
