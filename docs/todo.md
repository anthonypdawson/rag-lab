# RAG-Lab TODOs

## Frame Processing
- Switch back to FFMPEG for frame extraction
- Save frames during embedding to data/video_id/frame_id
- Support frame recovery from original video using frame ID
- Include timestamp in vector metadata for reproducible frame lookup
- Enable frame reproduction from vector data
- Implement streaming frame extraction

## Audio
- Integrate CLAP for audio embedding and search
- Make audio processing optional in pipeline

## Performance & Optimization
- Optimize memory usage for long videos
- Add GPU-accelerated frame extraction option

## Error Handling
- Improve error handling for model loading

## LLM Integration
- Research and integrate multimodal LLM for text+image input

## Advanced Research & Temporal Semantics
- Implement temporal chunking using motion (optical flow), audio changes, or speech pauses
- Integrate motion-aware (spatiotemporal) embedding models (e.g., TimeSformer, VideoCLIP)
- Generate cross-modal text summaries/captions for video segments and store in vector DB
- Design prompt engineering strategies for temporal context (e.g., compare segments, describe changes)

---
Update this file as tasks are completed or new requirements are discovered.
