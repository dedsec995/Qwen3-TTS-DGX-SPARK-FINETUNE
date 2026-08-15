import argparse
import os
from pathlib import Path
import torch
import soundfile as sf
import gc
from qwen_tts import Qwen3TTSModel


def test_base_model(text: str, ref_audio: str, ref_text: str, output_dir: Path):
    print("\n=== Testing Base Model (Zero Shot Voice Clone) ===")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    
    print("Supported speakers for base model:", model.get_supported_speakers())
    
    if ref_audio and os.path.exists(ref_audio):
        print(f"Generating voice clone using {ref_audio}...")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        out_file = output_dir / "output_base_model_voice_clone.wav"
        sf.write(str(out_file), wavs[0], sr)
        print(f"Saved {out_file}")
    else:
        print("Skipping base voice clone test: ref_audio not provided or not found.")
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Base model removed from memory.\n")


def test_finetuned_model(
    checkpoint_path: str,
    text: str,
    target_speakers: list = None,
    ref_audio: str = None,
    ref_text: str = None,
    output_dir: Path = Path("./")
):
    print(f"\n=== Testing Finetuned Multi-Speaker Model ===")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    model = Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    
    supported_speakers = model.get_supported_speakers()
    print(f"Supported speakers found in model config: {supported_speakers}")
    
    speakers_to_test = target_speakers if target_speakers else supported_speakers
    if not speakers_to_test:
        speakers_to_test = ["custom_speaker"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Test Custom Voice for each speaker
    print("\n--- Generating Custom Voice for each speaker ---")
    for spk in speakers_to_test:
        print(f"\nTesting speaker: '{spk}'")
        try:
            # Dynamic prompt conditioning prefix
            speaker_text = f"Speaker {spk}: {text}" if not text.startswith("Speaker ") else text
            
            wavs, sr = model.generate_custom_voice(
                text=speaker_text,
                language="English",
                speaker=spk,
            )
            out_file = output_dir / f"output_custom_voice_{spk}.wav"
            sf.write(str(out_file), wavs[0], sr)
            print(f"  [SUCCESS] Saved {out_file}")
        except Exception as e:
            print(f"  [ERROR] Generating custom voice for '{spk}': {e}")

    # 2. Test Zero-Shot Voice Clone if reference audio provided
    if ref_audio and os.path.exists(ref_audio):
        print("\n--- Testing Zero-Shot Voice Clone with Finetuned Model ---")
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            out_file = output_dir / "output_finetuned_voice_clone.wav"
            sf.write(str(out_file), wavs[0], sr)
            print(f"  [SUCCESS] Saved {out_file}")
        except Exception as e:
            print(f"  [ERROR] Generating voice clone: {e}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("\nFinetuned model removed from memory.\n")


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS fine-tuned multi-speaker checkpoints.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./output/checkpoint-epoch-2",
        help="Path to fine-tuned model checkpoint folder",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hi! I am super excited to demonstrate our new multi-speaker voice fine-tuning pipeline.",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        nargs="*",
        default=None,
        help="Specific speaker name(s) to test (defaults to all speakers in the checkpoint config)",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default="./reference.wav",
        help="Reference audio file for zero-shot test",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference audio transcript for zero-shot test",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_outputs",
        help="Directory to save generated audio samples",
    )
    parser.add_argument(
        "--test_base",
        action="store_true",
        help="Also test base model before testing fine-tuned model",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.test_base:
        test_base_model(args.text, args.ref_audio, args.ref_text, out_dir)

    test_finetuned_model(
        checkpoint_path=args.checkpoint_path,
        text=args.text,
        target_speakers=args.speaker,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
