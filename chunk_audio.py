import os
import argparse
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def process_audio_files(input_dir, output_dir, min_silence_len=500, silence_thresh=-40, target_length_ms=10000):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
        
    print(f"Found {len(wav_files)} files. Chunking...")
    
    chunk_counter = 0
    for wav_file in wav_files:
        print(f"Processing {wav_file.name}...")
        audio = AudioSegment.from_wav(str(wav_file))
        
        # Split on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=200 # keep some silence padding
        )
        
        # We might want to combine very short chunks to get closer to 5-15 seconds
        combined_chunks = []
        current_chunk = AudioSegment.empty()
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < target_length_ms:
                current_chunk += chunk
            else:
                if len(current_chunk) > 0:
                    combined_chunks.append(current_chunk)
                current_chunk = chunk
                
        if len(current_chunk) > 0:
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
    args = parser.parse_args()
    
    process_audio_files(args.input_dir, args.output_dir)
