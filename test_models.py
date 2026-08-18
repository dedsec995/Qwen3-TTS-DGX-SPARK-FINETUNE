import argparse
import gc
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def _load(checkpoint_path):
    return Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )


def _release(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def test_base_model(text, ref_audio, ref_text, output_dir, language):
    print("\n=== Base model (zero-shot voice clone) ===")
    model = _load("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print("Supported speakers:", model.get_supported_speakers())

    if ref_audio and os.path.exists(ref_audio):
        wavs, sr = model.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio, ref_text=ref_text
        )
        out = output_dir / "output_base_model_voice_clone.wav"
        sf.write(str(out), wavs[0], sr)
        print(f"Saved {out}")
    else:
        print("Skipped: --ref_audio not provided or missing.")
    _release(model)


def test_finetuned_model(checkpoint_path, text, target_speakers, ref_audio, ref_text,
                         output_dir, language, add_prefix):
    print(f"\n=== Fine-tuned model: {checkpoint_path} ===")

    cfg_path = Path(checkpoint_path) / "config.json"
    model_type = None
    if cfg_path.exists():
        model_type = json.loads(cfg_path.read_text()).get("tts_model_type")

    model = _load(checkpoint_path)
    supported = list(model.get_supported_speakers() or [])
    print(f"Speakers in checkpoint config: {supported}")

    speakers = target_speakers or supported
    if not speakers:
        print("No speakers registered in this checkpoint -- nothing to generate.")
        _release(model)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Custom voice (language={language!r}) ---")
    for spk in speakers:
        try:
            # The speaker embedding is injected at a sequence position that depends on
            # whether a language is set: position 6 for language="auto", position 7 once
            # an explicit language id is prepended. Training uses the "auto" layout, so
            # generating with an explicit language feeds the embedding into a slot the
            # model never saw.
            prompt = text
            if add_prefix and not text.startswith("Speaker "):
                prompt = f"Speaker {spk}: {text}"

            wavs, sr = model.generate_custom_voice(text=prompt, language=language, speaker=spk)
            out = output_dir / f"output_custom_voice_{spk}.wav"
            sf.write(str(out), wavs[0], sr)
            print(f"  [OK] {spk} -> {out}")
        except Exception as exc:
            print(f"  [ERROR] {spk}: {exc}")

    if ref_audio and os.path.exists(ref_audio):
        if model_type == "custom_voice":
            print("\n--- Zero-shot voice clone: skipped ---")
            print("  This checkpoint was saved with tts_model_type='custom_voice', which sets")
            print("  speaker_encoder=None on load, so generate_voice_clone cannot run. Use the")
            print("  base model for zero-shot cloning. This is expected, not a failure.")
        else:
            print("\n--- Zero-shot voice clone ---")
            try:
                wavs, sr = model.generate_voice_clone(
                    text=text, language=language, ref_audio=ref_audio, ref_text=ref_text
                )
                out = output_dir / "output_finetuned_voice_clone.wav"
                sf.write(str(out), wavs[0], sr)
                print(f"  [OK] -> {out}")
            except Exception as exc:
                print(f"  [ERROR] {exc}")

    _release(model)


def main():
    p = argparse.ArgumentParser(description="Test Qwen3-TTS fine-tuned multi-speaker checkpoints.")
    p.add_argument("--checkpoint_path", type=str, default="./output/checkpoint-epoch-9")
    p.add_argument("--text", type=str,
                   default="Hi! I am excited to demonstrate our multi-speaker voice fine-tuning pipeline.")
    p.add_argument("--speaker", type=str, nargs="*", default=None,
                   help="Speaker name(s) to test (default: all in the checkpoint config)")
    p.add_argument("--ref_audio", type=str, default="./reference.wav")
    p.add_argument("--ref_text", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./test_outputs")
    p.add_argument("--language", type=str, default="auto",
                   help="Keep 'auto' to match the training layout. An explicit language "
                        "shifts the speaker embedding slot from position 6 to 7.")
    p.add_argument("--no_prefix", action="store_true",
                   help="Do not prepend 'Speaker <name>: '. Use if you trained with "
                        "--speaker_prefix_template ''")
    p.add_argument("--test_base", action="store_true")
    args = p.parse_args()

    if args.language != "auto":
        print(f"WARNING: --language={args.language!r} moves the speaker embedding to sequence "
              f"position 7, but training placed it at position 6. Expect degraded output "
              f"unless you also changed the training layout.")

    out_dir = Path(args.output_dir)
    if args.test_base:
        test_base_model(args.text, args.ref_audio, args.ref_text, out_dir, args.language)

    test_finetuned_model(
        checkpoint_path=args.checkpoint_path, text=args.text, target_speakers=args.speaker,
        ref_audio=args.ref_audio, ref_text=args.ref_text, output_dir=out_dir,
        language=args.language, add_prefix=not args.no_prefix,
    )


if __name__ == "__main__":
    main()
