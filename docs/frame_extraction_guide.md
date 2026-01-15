# Frame Extraction Guide

## Overview

The video pipeline now uses **ffmpeg** for reproducible frame extraction with precise timestamps. Frames are saved to disk with filenames that include both frame index and high-precision timestamps, enabling easy lookup and regeneration.

## Frame Naming Convention

Frames are saved with the following pattern:
```
data/{video_id}/frame_{index:06d}_{timestamp:.6f}.png
```

Example:
```
data/abc123def456/frame_000012_12.345678.png
```

Where:
- `video_id`: MD5 hash of the video file (first and last 100MB)
- `index`: Sequential frame number (0-indexed, zero-padded to 6 digits)
- `timestamp`: Precise timestamp in seconds (6 decimal places for microsecond precision)

## Metadata Stored

Each frame embedding includes the following metadata in ChromaDB:

- `video_id`: Unique identifier (hash) for the video
- `file_path`: Path to original video file
- `timestamp`: Precise timestamp in seconds
- `frame_index`: Sequential frame number
- `frame_filename`: Name of the saved frame file
- `frame_path`: Full path to saved frame file
- `sampling_rate`: FPS used for extraction (e.g., 1.0 = 1 frame/second)
- `max_dimension`: Max dimension frames were resized to (or None)

The pipeline also tracks extraction parameters:
- `extraction_fps`: FPS used for extraction
- `extraction_max_dimension`: Max dimension used
- `ffmpeg_version`: Version of ffmpeg used (for reproducibility)

## Usage

### Basic Frame Extraction

```python
from rag_lab.core.video_pipeline import FrameExtractor

extractor = FrameExtractor(
    fps=1.0,              # Extract 1 frame per second
    max_dimension=768,    # Resize frames to max 768 pixels (or None for auto-detect)
    output_dir="data"     # Base directory for saving frames
)

frames = extractor.extract_frames(
    video_path="/path/to/video.mp4",
    video_id="abc123def456"  # Video hash from metadata
)

# Each frame dict contains:
# {
#     "timestamp": 12.345678,
#     "filename": "frame_000012_12.345678.png",
#     "index": 12,
#     "frame_path": "data/abc123def456/frame_000012_12.345678.png"
# }
```

### Full Pipeline

```python
from rag_lab.core.pipeline import FileProcessingPipeline

pipeline = FileProcessingPipeline(config={
    'video_fps': 1.0,              # 1 frame per second
    'max_dimension': 768,          # Resize to 768px max
    'batch_size': 32,              # Process 32 frames at a time
    'frames_output_dir': 'data'    # Where to save frames
})

# Process entire video (extracts frames, embeds, stores in ChromaDB)
results = pipeline.process("/path/to/video.mp4")

# Access metadata
metadata = results['metadata']
print(f"Video hash: {metadata.file_hash}")
print(f"Extraction FPS: {metadata.extraction_fps}")
print(f"FFmpeg version: {metadata.ffmpeg_version}")

# Access frame results
video_frames = results['video']
for frame in video_frames[:5]:
    print(f"Frame {frame.timestamp}s: {frame.metadata['frame_path']}")
```

### Regenerating Missing Frames

If a frame file is missing during the query phase, you can regenerate it:

```python
from rag_lab.core.video_pipeline import regenerate_missing_frame

# Get metadata from ChromaDB query result
metadata = query_result['metadatas'][0]

# Regenerate the missing frame
frame_path = regenerate_missing_frame(
    video_path=metadata['file_path'],
    video_id=metadata['video_id'],
    timestamp=metadata['timestamp'],
    frame_index=metadata['frame_index'],
    fps=metadata['sampling_rate'],
    max_dimension=metadata.get('max_dimension'),
    output_dir='data'
)

if frame_path:
    print(f"Frame regenerated: {frame_path}")
    # Now you can load and use the frame
    from PIL import Image
    img = Image.open(frame_path)
else:
    print("Failed to regenerate frame")
```

### Using FrameExtractor Directly

```python
from rag_lab.core.video_pipeline import FrameExtractor

extractor = FrameExtractor(fps=1.0, max_dimension=768, output_dir="data")

# Regenerate a single frame
frame_path = extractor.regenerate_frame(
    video_path="/path/to/video.mp4",
    video_id="abc123def456",
    timestamp=12.345678,
    frame_index=12
)
```

## Reproducibility

To ensure reproducible frame extraction:

1. **Use the same ffmpeg version**: The pipeline stores the ffmpeg version in metadata
2. **Use the same extraction parameters**: FPS and max_dimension are stored in metadata
3. **Use the same video file**: Video hash is computed and stored

If you need to re-extract frames (e.g., after changing sampling rate):

1. Delete the old frame directory: `data/{video_id}/`
2. Delete old embeddings from ChromaDB
3. Re-run the pipeline with new parameters

## Querying and Using Frames

After extraction and embedding, you can query ChromaDB for similar frames:

```python
from rag_lab.core.chromadb_client import ChromaDBClient

db_client = ChromaDBClient()
collection = db_client.get_or_create_collection("video")

# Query with text
results = collection.query(
    query_texts=["a person walking"],
    n_results=5
)

# Check if frame files exist and regenerate if needed
for metadata in results['metadatas'][0]:
    frame_path = metadata['frame_path']
    
    if not os.path.exists(frame_path):
        print(f"Frame missing: {frame_path}")
        # Regenerate
        new_path = regenerate_missing_frame(
            video_path=metadata['file_path'],
            video_id=metadata['video_id'],
            timestamp=metadata['timestamp'],
            frame_index=metadata['frame_index'],
            fps=metadata['sampling_rate'],
            max_dimension=metadata.get('max_dimension')
        )
        frame_path = new_path
    
    # Use the frame
    from PIL import Image
    img = Image.open(frame_path)
    # ... do something with img
```

## FFmpeg Command Reference

The pipeline uses the following ffmpeg command structure:

```bash
ffmpeg -i input.mp4 \
  -vf "fps=1.0,scale='if(gt(iw,ih),768,-2)':'if(gt(ih,iw),768,-2)',showinfo" \
  -vsync 0 \
  -frame_pts 1 \
  data/video_id/frame_%d.png \
  -y
```

Key parameters:
- `-vf fps=X`: Extract X frames per second
- `-vf scale=...`: Resize frames maintaining aspect ratio
- `-vf showinfo`: Log precise timestamps to stderr
- `-vsync 0`: Passthrough timestamps without modification
- `-frame_pts 1`: Include PTS in output filename
- `-y`: Overwrite existing files

The `showinfo` filter outputs lines like:
```
n:12 pts:12345 pts_time:12.345678 ...
```

Which the pipeline parses to get precise timestamps for each frame.

## Best Practices

1. **Storage**: Frames are stored as PNG files. For large videos at high FPS, this can consume significant disk space. Monitor disk usage.

2. **Cleanup**: If re-processing a video with different parameters, clean up old frames first.

3. **Backup**: Consider backing up frame directories if regeneration from source video is expensive or if original video might be unavailable.

4. **Query Phase**: Always check if frame files exist before using them. Use `regenerate_missing_frame()` to restore missing files.

5. **Batch Processing**: When regenerating multiple frames, consider implementing a batch regeneration function to avoid repeated ffmpeg startup overhead.

6. **Version Tracking**: Store the ffmpeg version and pipeline version in your application logs for debugging reproducibility issues.
