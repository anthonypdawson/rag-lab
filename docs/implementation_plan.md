# RAG-Lab Implementation Plan (Linux + GPU)

This document outlines scope, architecture, deliverables, timeline, and acceptance criteria for implementing RAG-Lab with local, GPU-accelerated models on Linux.

## Objectives
- Process video files into multimodal embeddings (text, audio, frames).
- Store embeddings in ChromaDB with rich metadata and timestamps.
- Provide retrieval with modality/time filters and a simple CLI.

## Scope
- Local-only models (no external APIs).
- Supported inputs: common video containers (MP4/MKV/AVI), subtitles (SRT/VTT).
- Embedding models: Sentence-Transformers (text), CLIP (images), CLAP/YAMNet (audio).

## Architecture
- **Extractor**: Metadata (ffprobe/MediaInfo), subtitles (pysubs2/srt), audio segments (ffmpeg), frames (ffmpeg/OpenCV).
- **Embedders**: 
  - Text: sentence-transformers `all-MiniLM-L6-v2` (default).
  - Image: CLIP ViT-B/32 via `transformers`.
  - Audio: CLAP (preferred) or YAMNet (alternative).
- **Storage**: ChromaDB collections with consistent metadata schema.
- **Orchestrator**: Pipeline in `rag_lab/core/pipeline.py` + CLI for batch runs.
- **Config/Logging**: Existing config and logger utilities.

## Deliverables
- Pipeline implementations for metadata, text, audio, video.
- ChromaDB ingestion/retrieval with filters and timestamps.
- CLI tool for processing and querying.
- Unit/integration tests and example scripts.
- Documentation: setup, usage, troubleshooting.

## Timeline (Linux + GPU)
- **Day 1**: Environment setup; metadata extraction; subtitle parsing.
- **Day 2**: Text embeddings; frame extraction baseline.
- **Day 3**: CLIP image embeddings; audio segmentation.
- **Day 4**: Audio embeddings; ChromaDB schema + ingestion.
- **Day 5**: Retrieval APIs; CLI wiring; minimal docs.
- **Days 6–10**: Tests, performance tuning (GPU batching, multiprocessing), expanded docs & examples.

## Milestones & Exit Criteria
- **M1: Extraction Ready**
  - Metadata, subtitles, frames, audio segments produced with timestamps.
- **M2: Embeddings Ready**
  - Text, image, audio embeddings generated locally; batchable on GPU.
- **M3: Storage Ready**
  - ChromaDB collections populated; metadata schema validated.
- **M4: Retrieval Ready**
  - Query by modality/time; returns items with linked metadata.
- **M5: Usability Ready**
  - CLI end-to-end run; docs and example commands.

## Metadata Schema (per item)
- `id`: unique id
- `modality`: `text|audio|video`
- `start`: float seconds
- `end` or `timestamp`: float seconds
- `source_path`: original file path
- `tags`: optional labels (speaker, scene, language)
- `extra`: optional technical details (codec, resolution, etc.)

## Dependencies
- `ffmpeg`, `pymediainfo`, `pysubs2`/`srt`, `sentence-transformers`, `transformers`, `torch`, `chromadb`, `opencv-python`, `ffmpeg-python`.

## Setup Checklist (Linux + CUDA, Poetry)
1. Install system tools: `ffmpeg` and `mediainfo`.
2. Install project dependencies: `poetry install`.
3. Install CUDA-enabled PyTorch in the Poetry environment:
  - `poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` (adjust CUDA version as needed)
4. Add missing libs (if not already in `pyproject.toml`):
  - `poetry add transformers pymediainfo pysubs2 opencv-python ffmpeg-python`
5. Verify GPU availability:
  - `poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"`

## Risks & Mitigations
- **Codec/subtitle variance**: Fallbacks and validation; log warnings.
- **Audio segmentation quality**: Configurable strategy (fixed window vs. silence detection).
- **Performance**: GPU batching, multiprocessing for I/O, adjustable sampling rate.

## Acceptance Criteria
- End-to-end CLI processes a sample video; stores and retrieves items from ChromaDB.
- Embeddings exist for all modalities; timestamps align across modalities.
- Docs enable a fresh Linux + GPU setup to run successfully.

## Communication & Iteration
- You run integration tests with your media set.
- Rapid fix cycles on reported issues; targeted patches and test updates.

## Start Signal
- Confirm Linux GPU environment and sample input files are ready.
- Proceed with M1 implementation (metadata + subtitles).
