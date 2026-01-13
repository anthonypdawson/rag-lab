# Frame Processing Design

## Overview
Extract and embed video frames for visual content retrieval. Balance between temporal granularity, storage efficiency, and retrieval accuracy.

## Configuration Parameters

### Sampling Strategy
- **fps (frames per second)**: How many frames to extract per second of video
  - Options: 0.5, 1, 2, 5 fps
  - Default: 1 fps (reasonable balance)
  - Higher fps = more granular but more storage/compute

### Temporal Windowing
- **window_size (seconds)**: Timespan grouping for embeddings
  - Options: None (per-frame), 1s, 5s, 10s
  - Default: None (embed each frame individually)
  - Trade-off: Scene continuity vs. temporal precision

### Frame Selection Methods
1. **Uniform sampling**: Extract every N frames based on fps
2. **Keyframe detection**: Extract only significant frame changes (TODO: future enhancement)
3. **Scene boundary detection**: Extract frames at scene transitions (TODO: future enhancement)

## Processing Pipeline

### Phase 1: Frame Extraction
```
Input: video_path, fps, start_time, end_time
Output: List of frames with timestamps

1. Use ffmpeg to extract frames at specified fps
2. Save frames to temporary directory or keep in memory
3. Track timestamp for each frame
4. Return frame data + metadata
```

### Phase 2: Embedding Generation
```
Input: frames, window_size (optional)
Output: embeddings with metadata

Option A: Per-frame embedding (window_size=None)
- Embed each frame individually with CLIP
- Store with precise timestamp
- More granular search, more storage

Option B: Windowed embedding (window_size=5s)
- Group frames by time window
- Options for combining:
  a) Average frames → single embedding
  b) Embed each frame, average embeddings
  c) Select representative frame (middle of window)
- Less storage, captures scene context
```

### Phase 3: Storage in ChromaDB
```
Collection: "video"

Per-frame schema:
{
  id: "{video_hash}_frame_{frame_index}",
  embedding: [768-dim CLIP embedding],
  metadata: {
    video_id: hash,
    file_path: str,
    timestamp: float (seconds),
    frame_index: int,
    window_start: float (if windowed),
    window_end: float (if windowed),
    resolution: str (e.g., "1920x1080"),
    scene_id: int (optional, for future scene detection)
  }
}
```

## Implementation Strategy

### Step 1: Frame Extractor Class
```python
class FrameExtractor:
    def __init__(self, fps=1.0, output_format='numpy'):
        """
        fps: frames per second to extract
        output_format: 'numpy', 'pil', 'bytes'
        """
    
    def extract_frames(self, video_path, start=None, end=None):
        """
        Extract frames from video using ffmpeg
        Returns: List[Dict] with 'timestamp', 'frame_data', 'index'
        """
```

### Step 2: Video Embedder Class
```python
class VideoEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Load CLIP model"""
    
    def embed_frames(self, frames):
        """
        Embed list of frames
        Returns: embeddings (batch processing for efficiency)
        """
    
    def embed_and_store_frames(self, frames, collection, video_id, file_path):
        """
        Embed frames and store in ChromaDB
        Handles batching for GPU efficiency
        """
```

### Step 3: Pipeline Integration
```python
def process_video(self, file_path: str, metadata: Metadata) -> List[VideoFrame]:
    logger.info(f"Processing video frames for: {file_path}")
    
    # Get configuration
    fps = self.config.get('video_fps', 1.0)
    
    # Extract frames
    extractor = FrameExtractor(fps=fps)
    frames = extractor.extract_frames(file_path)
    
    if not frames:
        logger.warning(f"No frames extracted from: {file_path}")
        return []
    
    logger.info(f"Extracted {len(frames)} frames at {fps} fps")
    
    # Get collection
    collection = self.db_client.get_or_create_collection("video")
    
    # Embed and store
    embedder = VideoEmbedder()
    embedder.embed_and_store_frames(
        frames, collection, metadata.file_hash, file_path
    )
    
    # Return VideoFrame objects
    return [VideoFrame(timestamp=f['timestamp'], metadata=...) for f in frames]
```

## Performance Considerations

### GPU Batching
- Process frames in batches (e.g., 32 frames at a time)
- CLIP can efficiently batch encode multiple images
- Monitor GPU memory usage

### Memory Management
- For long videos: stream frames instead of loading all at once
- Use generators to yield frames incrementally
- Clean up temporary files after processing

### Storage Estimates
- 1 hour video @ 1 fps = 3,600 frames
- CLIP embedding = 512 dimensions × 4 bytes = 2KB per frame
- Total: ~7.2 MB for embeddings alone (plus metadata)
- @ 2 fps = ~14.4 MB per hour

## Configuration Example

```yaml
# config/config.yml
video:
  fps: 1.0  # frames per second
  window_size: null  # per-frame embedding (null) or window duration in seconds
  batch_size: 32  # frames per GPU batch
  max_dimension: 224  # resize frames to this max dimension for efficiency
  quality_threshold: null  # future: skip low-quality/blurry frames
```

## Testing Strategy
1. Test with short clip (10 seconds) - verify frame count and timestamps
2. Test with medium video (5 minutes) - check performance and GPU usage
3. Test with long video (1 hour) - verify memory handling
4. Verify embeddings are retrievable and timestamps align

## Future Enhancements
- Scene detection to extract only scene transitions
- Optical flow analysis to skip static scenes
- Object detection to add metadata tags
- Face detection for person-centric retrieval
- Quality filtering to skip blurry/corrupted frames

## Decision Points for User
1. **Default fps**: 0.5, 1, or 2 fps?
2. **Windowing**: Per-frame or windowed embeddings?
3. **Frame storage**: Keep extracted frames or discard after embedding?
4. **Resolution**: Resize frames before embedding for efficiency?

## Recommended Starting Configuration
```python
config = {
    'video_fps': 1.0,          # 1 frame per second
    'window_size': None,        # per-frame (more precise)
    'batch_size': 32,           # GPU batch size
    'max_dimension': 512,       # pre-resize for memory (CLIP resizes to 224 internally)
    'keep_frames': False        # discard after embedding
}
```
