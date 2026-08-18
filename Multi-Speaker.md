# Multi-Speaker Fine-Tuning Pipeline Walkthrough

We have upgraded the Qwen3-TTS fine-tuning pipeline to support **sentence-aware multi-speaker fine-tuning**. This eliminates catastrophic forgetting by training multiple voices together in a single shared representation, producing a unified checkpoint that supports both native custom voice generation and streaming inference.

---

## Key Modifications

### 1. Audio Preprocessing ([chunk_audio.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/chunk_audio.py))
- **Two-Level Sentence-Aware Chunking**:
  - Pre-segments audio in silence into ~20s windows and transcribes using `nvidia/parakeet-tdt-0.6b-v2` with word-level timestamps.
  - Reconstructs sentences on terminal punctuation (`.` `?` `!`) and packs into 3–10s chunks.
  - Snaps cut points to natural silence edges (`--snap_ms 300`) and splits oversized sentences at inter-word gaps (`_split_segment`).
- **Verbatim Transcripts in `chunks.jsonl`**: Directly writes audio statistics and verbatim transcripts (`text`, `n_segments`) into `chunks.jsonl`.
- **Multi-speaker folder detection**: Automatically detects subdirectories in `--input_dir` (e.g. `raw_audio/bob/`, `raw_audio/alice/`).
- **Backward compatibility**: Seamlessly chunks flat directories and supports `--no_sentence_aware` for silence-only mode.

### 2. Fine-Tuning Pipeline ([fine_tune_qwen.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/fine_tune_qwen.py))
- **Zero-Redundancy Transcript Passthrough**: Reuses pre-computed transcripts from `chunks.jsonl`, skipping NeMo loading (~2.4 GB VRAM saved).
- **Speaker-Conditioned ASR Tagging**: Formats transcribed text with speaker identity tags (e.g. `Speaker bob: {transcription}`) using configurable `--speaker_prefix_template`.
- **Centroid Speaker Embeddings**:
  - Automatically computes centroid speaker embeddings over 64 clips and rescales to median norm.
  - Selects the clip nearest to the centroid and freezes it under `output_multi/references/<speaker>.wav`.
- **Multi-Speaker Checkpoint Assembly**:
  - Automatically maps each speaker to consecutive `spk_id` slots in `config.json` (`3000`, `3001`, ...).
  - Injects target speaker embeddings for all speakers into `talker.model.codec_embedding.weight`.

### 3. Testing & Inference ([test_models.py](file:///home/dedsec995/voice_collection/Qwen3-TTS-finetune/test_models.py))
- **CLI Options**: Configurable `--checkpoint_path`, `--text`, `--speaker`, `--ref_audio`, and `--output_dir`.
- **Automatic Multi-Speaker Iteration**: Automatically discovers all speakers in the checkpoint's `config.json` and generates test audio for each (`output_custom_voice_{speaker}.wav`).

---

## How to Run

### Step 1: Organize Your Data
```bash
raw_audio/
├── bob/
│   └── 30min_audio.wav
└── alice/
    └── 30min_audio.wav
```

### Step 2: Chunk the Audio (Sentence-Aware)
```bash
python chunk_audio.py --input_dir ./raw_audio --output_dir ./audio_chunks
```

### Step 3: Run Simultaneous Multi-Speaker Fine-Tuning
```bash
python fine_tune_qwen.py \
    --audio_dir ./audio_chunks \
    --output_dir ./output_multi_speaker \
    --num_epochs 10 \
    --save_every_n_epochs 2 \
    --lr 2e-6 \
    --batch_size 8 \
    --gradient_accumulation_steps 4
```

### Step 4: Test the Fine-Tuned Voices
```bash
python test_models.py \
    --checkpoint_path ./output_multi_speaker/checkpoint-epoch-9 \
    --text "Welcome to our streaming speech pipeline demo." \
    --output_dir ./test_outputs
```
