# Qwen3-TTS-DGX-SPARK-finetune

An end-to-end, ARM64-optimized fine-tuning pipeline for Qwen3-TTS (1.7B). This repository provides a streamlined way to fine-tune single or multiple speakers simultaneously into a unified checkpoint, specifically adapted for systems like the NVIDIA DGX Spark (Grace Blackwell / ARM64 architectures).

It uses flash attention 2 prebuilt wheels. If you want you can build them manually and install (Probably fry your CPU, Happy Cooking!!!)

## Features
- **Simultaneous Multi-Speaker Training**: Train multiple voices together to prevent catastrophic forgetting and create a single shared-weight checkpoint.
- **Two-Level Sentence-Aware Audio Chunking**: Splits audio into ~20s transcription windows in silence, uses verbatim Parakeet-TDT ASR with word timestamps, and packs into 3–10s chunks cut cleanly on sentence boundaries (`.` `?` `!`).
- **Verbatim ASR with Punctuation & Disfluency Preservation**: Uses `nvidia/parakeet-tdt-0.6b-v2` to capture speech disfluencies (`um`, `uh`, stutters) and true punctuation/casing, eliminating breathless and robotic monotone TTS delivery.
- **Zero-Redundancy Pipeline**: `fine_tune_qwen.py` automatically reuses pre-computed transcripts from `chunks.jsonl`, skipping the 2.4 GB NeMo model loading entirely.
- **Centroid Speaker Embeddings**: Each speaker's baked-in embedding is averaged over 64 clips rather than taken from a single reference, and the reference clip itself is auto-selected as the one nearest that centroid.
- **Validation & Metrics**: Per-speaker stratified train/val split, eval loss every epoch, and TensorBoard logging of both losses and the learning rate.
- **Dual-Inference Compatibility**: Supports both native `generate_custom_voice(speaker=...)` via registered `spk_id` slots and prompt-conditioned streaming inference (`"Speaker <name>: ..."`) for stacks like `faster-qwen3-tts`.
- **ARM64 Native**: Optimized for Grace Blackwell architectures.
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

### 1. Chunk Your Audio (Sentence-Aware)
Run the chunking script; it detects speaker subfolders, transcribes ~20s windows in silence, and cuts chunks at natural sentence boundaries (`.` `?` `!`):

```bash
python chunk_audio.py --input_dir ./raw_audio --output_dir ./audio_chunks
```

The resulting `./audio_chunks` structure:
```text
audio_chunks/
├── bob/
│   ├── chunk_0000.wav
│   ├── chunk_0001.wav
│   └── chunks.jsonl      <-- contains audio stats and verbatim transcripts
└── alice/
    ├── chunk_0000.wav
    └── chunks.jsonl
```

### 2. Fine-Tune the Multi-Speaker Model
Run the end-to-end training pipeline. The script automatically discovers all speakers, loads pre-computed transcripts from `chunks.jsonl`, extracts speaker embeddings, and trains the model:

```bash
python fine_tune_qwen.py \
    --audio_dir ./audio_chunks \
    --output_dir ./output_multi \
    --num_epochs 10 \
    --save_every_n_epochs 2 \
    --lr 2e-6 \
    --batch_size 8 \
    --gradient_accumulation_steps 4
```

Defaults follow the upstream Qwen recipe. Learning rates above ~1e-5 degrade speaker
quality on small datasets, and over-training (24+ epochs on under an hour of audio) causes
catastrophic forgetting and robotic output — watch the eval loss instead:

```bash
tensorboard --logdir ./output_multi/logs
```

After the run, check `output_multi/asr_report.json` (what the transcript filter dropped and
why) and `output_multi/speaker_meta.json`. In the latter, `mean_cos_to_centroid` below 0.80
usually means chunks from another voice ended up in a speaker's folder; a clean
single-session recording scores around 0.99.

---

### 3. Test & Generate Audio

Use the multi-speaker testing script to synthesize speech for all trained voices:

```bash
python test_models.py \
    --checkpoint_path ./output_multi/checkpoint-epoch-9 \
    --text "Hello! This is a test of our unified multi-speaker fine-tuned model." \
    --output_dir ./test_outputs
```

Generation uses `language="auto"` by default, which is required: with an explicit language
the model prepends a language token and the speaker embedding moves from sequence position
6 to 7, which is not where training put it.

A fine-tuned checkpoint is saved as `tts_model_type="custom_voice"`, so the library sets
`speaker_encoder=None` on load and zero-shot `generate_voice_clone` is unavailable on it —
use the base model for zero-shot cloning. `test_models.py` detects this and skips that step.

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
- **Baseten**: For their Qwen3-TTS fine-tuning writeup, source of the centroid speaker-embedding approach.

## License
This project is released under the MIT License. See the LICENSE file for details.
