# RAG-Lab Implementation Timeline

This timeline outlines milestones, estimates, and environment assumptions for implementing the RAG-Lab pipeline end-to-end.

## Assumptions
- Target OS: Linux
- GPU: CUDA available (full acceleration)
- Local-only models (no external APIs)
- Tools: `ffmpeg`, `pymediainfo`, `pysubs2/srt`, `sentence-transformers`, `torch/transformers`, `chromadb`, `opencv-python`, `ffmpeg-python`

## High-Level Estimate
- MVP: 4–5 days (35–45 hours)
- Production-ready: 1.5–2.5 weeks (60–90 hours) including tests, docs, and performance tuning

## Milestones & Estimates
1. **Setup & Tooling** (2–4h)
   - Python env, CUDA-enabled PyTorch, `ffmpeg`, basic project scaffolding
2. **Metadata Extraction** (2–4h)
   - Implement `ffprobe`/`pymediainfo` integration and normalized `Metadata` dataclass
3. **Subtitles + Text Embeddings** (3–5h)
   - Parse SRT/VTT via `pysubs2`/`srt`, embed with `sentence-transformers`
4. **Audio Segmentation + Embeddings** (6–10h)
   - Segment via fixed windows or silence detection, embed with CLAP/YAMNet
5. **Frame Extraction + Image Embeddings** (6–10h)
   - Extract frames at interval, embed via CLIP (ViT-B/32)
6. **ChromaDB Schema + Ingestion** (4–6h)
   - Define collections and metadata fields: `modality`, `start`, `end/timestamp`, `source`, `chunk_id`, `tags`
7. **Retrieval + Filtering** (4–6h)
   - Query by modality/time ranges; return nearest neighbors with linked metadata
8. **CLI + Config Integration** (3–4h)
   - Command-line runner for batch processing; leverage existing `config`
9. **Tests (Unit/Integration)** (6–8h)
   - Dataclass validation, embedding generation, ChromaDB ingestion/retrieval
10. **Performance Tuning** (6–10h)
   - GPU batching, multiprocessing for I/O-heavy steps, configurable sampling
11. **Docs & Examples** (3–4h)
   - Usage guide, sample commands, troubleshooting

## Proposed Timeline (Linux + GPU)
- **Day 1**: Setup, metadata extraction, subtitles parsing
- **Day 2**: Text embeddings + frame extraction baseline
- **Day 3**: CLIP image embeddings + audio segmentation
- **Day 4**: Audio embeddings + ChromaDB schema & ingestion
- **Day 5**: Retrieval APIs, CLI wiring, minimal docs
- **Days 6–10**: Tests, performance tuning, extended docs & examples

## Model Selections (Local)
- **Text**: `sentence-transformers/all-MiniLM-L6-v2` (fast, quality) or `multi-qa-MiniLM-L6-cos-v1`
- **Image**: CLIP `openai/clip-vit-base-patch32` (Hugging Face `transformers`)
- **Audio**: CLAP (e.g., LAION models) for audio embeddings; alternative YAMNet via TF if preferred

## ChromaDB Metadata Schema (per item)
- `id`: unique chunk/frame/segment id
- `modality`: `text|audio|video`
- `start`: float seconds
- `end` or `timestamp`: float seconds
- `source_path`: original file path
- `tags`: optional labels (speaker, scene, language)
- `extra`: any additional info (codec, resolution, etc.)

## Linux + GPU Setup (Poetry)
```bash
# System tools
sudo apt-get update && sudo apt-get install -y ffmpeg mediainfo

# Project deps
poetry install

# PyTorch with CUDA (adjust for your CUDA version)
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Add missing libs if not in pyproject
poetry add transformers pymediainfo pysubs2 opencv-python ffmpeg-python

# Verify GPU availability
poetry run python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY
```

## Next Steps
- Implement metadata and subtitles first (fast wins)
- Wire embeddings (text, image, audio) and ChromaDB ingestion
- Add CLI commands for batch processing and retrieval filters
