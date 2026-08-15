# Qwen3-TTS-DGX-SPARK-finetune

An end-to-end, ARM64-optimized fine-tuning pipeline for Qwen3-TTS (1.7B). This repository provides a streamlined way to fine-tune single or multiple speakers simultaneously into a unified checkpoint, specifically adapted for systems like the NVIDIA DGX Spark (Grace Blackwell / ARM64 architectures).

It uses flash attention 2 prebuilt wheels. If you want you can build them manually and install (Probably fry your CPU, Happy Cooking!!!)

## Features
- **Simultaneous Multi-Speaker Training**: Train multiple voices together to prevent catastrophic forgetting and create a single shared-weight checkpoint.
- **End-to-End Pipeline**: Automatically handles multi-speaker chunking, NeMo Parakeet transcription, speaker-conditioned prompt tagging, audio code extraction, and fine-tuning.
- **Dual-Inference Compatibility**: Supports both native `generate_custom_voice(speaker=...)` via registered `spk_id` slots and prompt-conditioned streaming inference (`"Speaker <name>: ..."`) for stacks like `faster-qwen3-tts`.
- **ARM64 Native**: Uses NVIDIA NeMo Parakeet to bypass x86 limitations.
- **Configurable Control**: Direct CLI controls for transcription batch size, training batch size, gradient accumulation, learning rate, weight decay, and checkpoint saving intervals.
- **Local Caching**: Stores model weights inside the local `./env/hf_cache` rather than bloating the host OS.

## Requirements
- Python 3.12
- PyTorch with CUDA 13 support
- `ffmpeg` (Required by pydub for audio chunking)

## Installation

A complete setup script is provided to automatically create a virtual environment and install all dependencies, including prebuilt ARM64 wheels for flash-attention.

1. Install system dependencies:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

2. Run the environment setup:
```bash
./setup_env.sh
```

3. Activate the environment:
```bash
source env/bin/activate
```

---

## Dataset Structure

### Multi-Speaker Setup (Recommended)
Place each speaker's long audio files in a dedicated subfolder within `./raw_audio`:

```text
raw_audio/
├── bob/
│   ├── interview_part1.wav
│   └── interview_part2.wav
└── alice/
    └── podcast_episode.wav
```

### Single-Speaker Setup
Place audio files directly in `./raw_audio`:
```text
raw_audio/
├── sample1.wav
└── sample2.wav
```

---

## Usage

### 1. Chunk Your Audio
TTS models require short audio segments (3–10 seconds). Run the chunking script; it automatically detects speaker subfolders and preserves the speaker hierarchy:

```bash
python chunk_audio.py --input_dir ./raw_audio --output_dir ./audio_chunks
```

The resulting `./audio_chunks` structure:
```text
audio_chunks/
├── bob/
│   ├── chunk_0000.wav
│   └── chunk_0001.wav
└── alice/
    ├── chunk_0000.wav
    └── chunk_0001.wav
```

### 2. Fine-Tune the Multi-Speaker Model
Run the end-to-end training pipeline. The script automatically discovers all speakers, conditions transcripts with `Speaker {name}: {text}`, extracts speaker embeddings, and trains the model:

```bash
python fine_tune_qwen.py \
    --audio_dir ./audio_chunks \
    --output_dir ./output_multi_speaker \
    --num_epochs 24 \
    --save_every_n_epochs 6 \
    --lr 3e-6 \
    --batch_size 12 \
    --transcribe_batch_size 8 \
    --gradient_accumulation_steps 4
```

> **Optional per-speaker reference audios**: If you have clean dedicated reference clips for each speaker (e.g. `references/bob.wav`, `references/alice.wav`), supply `--ref_audio_dir ./references`. Otherwise, the pipeline automatically extracts target embeddings from the speaker's audio chunks.

---

### 3. Test & Generate Audio

Use the multi-speaker testing script to synthesize speech for all trained voices:

```bash
python test_models.py \
    --checkpoint_path ./output_multi_speaker/checkpoint-epoch-23 \
    --text "Hello! This is a test of our unified multi-speaker fine-tuned model." \
    --output_dir ./test_outputs
```

#### Streaming Inference with `faster-qwen3-tts`
Because speakers are conditioned directly in the text representation, you can switch voices dynamically during streaming simply by prepending the speaker tag:

- **Bob**: `"Speaker bob: Welcome to the streaming audio pipeline."`
- **Alice**: `"Speaker alice: Welcome to the streaming audio pipeline."`

---

## Future
#### Do I intend to maintain this repository?
No. Unless it stops working for me.

#### Do I intend to include support for other platforms?
No. I have DGX Spark Grace Blackwell gb10 and I'll only include, it might work on other grace blackwell architectures.

## Credits & Acknowledgements
- **Alibaba Qwen Team**: For releasing the open-source [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models.
- **sruckh**: For the [Qwen3-TTS-finetune](https://github.com/sruckh/Qwen3-TTS-finetune) repository, whose training loop and `dataset.py` formed the baseline of this project.
- **NVIDIA NeMo Team**: For the Parakeet-TDT ASR models.

## License
This project is released under the MIT License. See the LICENSE file for details.

