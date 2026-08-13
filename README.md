# Qwen3-TTS-DGX-SPARK-finetune

An end-to-end, ARM64-optimized fine-tuning pipeline for Qwen3-TTS (1.7B). This repository provides a streamlined way to clone voices using your own dataset, specifically adapted for systems like the NVIDIA DGX Spark (Grace Blackwell / ARM64 architectures).

It uses flash attention 2 prebuild wheels. If you want you can build them manually and install (Probabily fry your CPU, Happy Cooking!!!)

## Features
- **End-to-End Pipeline**: Handles transcription, tokenizer code extraction, and LoRA fine-tuning in a single execution.
- **ARM64 Native**: Uses NVIDIA NeMo Parakeet to bypass x86 limitations.
- **Configurable Control**: Direct CLI controls for transcription batch size, training batch size, gradient accumulation, learning rate, weight decay, and checkpoint saving intervals.
- **Local Caching**: Forces Hugging Face to store multi-gigabyte model weights inside the local `./env/hf_cache` rather than bloating the host OS.

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

## Usage

### 1. Prepare Your Data
TTS models require short audio segments. If you have long audio files (e.g., 30-minute podcasts), place them in a folder (e.g., `./raw_audio`) and use the included chunking script to slice them into 5-15 second WAV files.

```bash
python chunk_audio.py --input_dir ./raw_audio --output_dir ./audio_chunks
```

### 2. Prepare Reference Audio
Choose a single, clean 5-10 second clip of the target speaker and place it in the root directory as `reference.wav`. This acts as the anchor to generate speaker embeddings during the training loop.

### 3. Fine-Tune the Model
Run the main script to transcribe the chunks, extract audio tokens, and fine-tune the 1.7B model. 

```bash
python fine_tune_qwen.py \
    --audio_dir ./audio_chunks \
    --ref_audio ./reference.wav \
    --speaker_name my_voice \
    --output_dir ./output \
    --num_epochs 24 \
    --save_every_n_epochs 6 \
    --lr 3e-6 \
    --batch_size 12 \
    --transcribe_batch_size 8 \
    --gradient_accumulation_steps 4
```

## Future
#### Do I intent to maintain this repository?
No. Unless it stops working for me

#### Do I intent to include support for other platform?
No. I have DGX Spark Grace Blackwell gb10 and I'll only include, it might work on other grace blackwell architecture.


## Credits & Acknowledgements
- **Alibaba Qwen Team**: For releasing the incredible open-source [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models.
- **sruckh**: For the [Qwen3-TTS-finetune](https://github.com/sruckh/Qwen3-TTS-finetune) repository, whose unified Python training loop and `dataset.py` formed the baseline of this project.
- **NVIDIA NeMo Team**: For the Parakeet-TDT ASR models.

## License
This project is released under the MIT License. See the LICENSE file for details.
