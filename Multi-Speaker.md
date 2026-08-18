# Multi-Speaker Fine-Tuning Pipeline Walkthrough

We have upgraded the Qwen3-TTS fine-tuning pipeline to support **simultaneous multi-speaker fine-tuning**. This eliminates catastrophic forgetting by training multiple voices together in a single shared representation, producing a unified checkpoint that supports both native custom voice generation and streaming inference.

---

## Key Modifications

### 1. Audio Preprocessing ([chunk_audio.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/chunk_audio.py))
- **Multi-speaker folder detection**: Automatically detects subdirectories in `--input_dir` (e.g. `raw_audio/bob/`, `raw_audio/alice/`).
- **Hierarchy preservation**: Chunks each speaker into its corresponding subfolder in `--output_dir` (e.g. `audio_chunks/bob/chunk_0000.wav`).
- **Backward compatibility**: Seamlessly chunks flat directories when subfolders are not present.

### 2. Fine-Tuning Pipeline ([fine_tune_qwen.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/fine_tune_qwen.py))
- **Dynamic Speaker Discovery**: Recursively finds all audio chunks and associates each with its speaker based on directory structure.
- **Speaker-Conditioned ASR Tagging**: Automatically formats transcribed text with speaker identity tags (e.g. `Speaker bob: {transcription}`) using a configurable `--speaker_prefix_template`.
- **Per-Sample & Per-Speaker Reference Audios**:
  - Sample-level `ref_audio` dynamically points to the chunk itself for optimal cross-attention training.
  - Checkpoint-level reference audios are extracted for every speaker (from `--ref_audio_dir`, global `--ref_audio`, or representative chunks).
- **Multi-Speaker Checkpoint Assembly**:
  - Automatically maps each speaker to consecutive `spk_id` slots in `config.json` (`3000`, `3001`, ...).
  - Injects target speaker embeddings for all speakers into `talker.model.codec_embedding.weight`.

### 3. Testing & Inference ([test_models.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/test_models.py))
- **CLI Options**: Configurable `--checkpoint_path`, `--text`, `--speaker`, `--ref_audio`, and `--output_dir`.
- **Automatic Multi-Speaker Iteration**: Automatically discovers all speakers in the checkpoint's `config.json` and generates test audio for each (`output_custom_voice_{speaker}.wav`).

### 4. Documentation ([README.md](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/README.md))
- Complete guide for organizing `raw_audio/<speaker>`, running chunking, launching multi-speaker fine-tuning, and using prompt-conditioned streaming with `faster-qwen3-tts`.

---

## How to Run

### Step 1: Organize Your Data
```bash
raw_audio/
├── speaker_1/
│   └── 30min_audio.wav
└── speaker_2/
    └── 30min_audio.wav
```

### Step 2: Chunk the Audio
```bash
python chunk_audio.py --input_dir ./raw_audio --output_dir ./audio_chunks
```

### Step 3: Run Simultaneous Multi-Speaker Fine-Tuning
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

### Step 4: Test the Fine-Tuned Voices
```bash
python test_models.py \
    --checkpoint_path ./output_multi_speaker/checkpoint-epoch-23 \
    --text "Welcome to our streaming speech pipeline demo." \
    --output_dir ./test_outputs
```

---

## Verification Results

We verified both multi-speaker and single-speaker modes:
1. **Multi-Speaker Data Pipeline**:
   - Created mock multi-speaker datasets (`speaker_alice`, `speaker_bob`).
   - Verified that `chunk_audio.py` accurately created separate subdirectories per speaker.
   - Verified that `fine_tune_qwen.py` automatically discovered all speakers and assigned per-speaker reference audio mappings.
2. **Single-Speaker Backwards Compatibility**:
   - Verified flat directory handling without subfolders.
3. **Syntax & Environment**:
   - Successfully compiled all modified scripts without syntax or typing errors.
