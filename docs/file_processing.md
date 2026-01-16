# File Processing in RAG-Lab

This document describes how RAG-Lab processes video files, including extraction of multimodal data and metadata.

## Overview
RAG-Lab processes video files by extracting:
- Video frames (visual content)
- Audio segments (audio content)
- Subtitle text (textual content)
- File and media metadata (technical and descriptive)

All extracted data is linked by timestamps and stored in a vector database for efficient retrieval and cross-modal search.

## Steps in File Processing

1. **Input Video File**
   - Accepts various formats (e.g., MP4, MKV, AVI).

2. **Metadata Extraction**
   - Extracts technical metadata (duration, resolution, codecs, etc.) using tools like ffprobe or pymediainfo.
   - Optionally fetches descriptive metadata (title, year, cast) from external APIs for movies/TV shows.

3. **Frame Extraction**
   - Extracts video frames at configurable intervals.
   - Each frame is timestamped.

4. **Audio Segmentation**
   - Splits audio into segments (e.g., by time window or silence detection).
   - Each segment is timestamped.

5. **Subtitle/Text Extraction**
   - Extracts subtitle text and aligns it with timestamps.
   - Supports common subtitle formats (SRT, VTT, etc.).
   - If no subtitles exist, the system auto-generates an SRT via faster-whisper and persists it alongside the video using the suffix `._auto.srt`. Future runs will reuse this file rather than regenerating.
   - Each subtitle segment captures provenance metadata in the vector store: `origin` (external | embedded | auto) and `language` (ISO code when available, else `und`).

6. **Embedding Generation**
   - Generates vector embeddings for each modality using appropriate models.

7. **Storage in Vector Database**
   - Stores embeddings and metadata in ChromaDB, linking all data by time.

## Tools and Libraries
- **ffmpeg/ffprobe**: For frame, audio, and metadata extraction
- **pymediainfo**: For detailed media metadata
- **pysubs2, srt**: For subtitle parsing
- **ChromaDB**: For vector storage
- **Ollama**: For embedding generation

## Extensibility
- The pipeline can be extended to support new file types or metadata sources.
- Custom extraction or embedding steps can be added as needed.

---

For implementation details, see the code in the `rag_lab/core` and `rag_lab/utils` directories.