# FFmpeg Frame Extraction Implementation - Summary

## Completed Changes

### 1. Video Pipeline Refactor (`rag_lab/core/video_pipeline.py`)

**Key Changes:**
- Replaced OpenCV-based frame extraction with ffmpeg-based extraction
- Frames are now saved to disk instead of kept in memory
- Added precise timestamp tracking using ffmpeg's showinfo filter
- Implemented `regenerate_frame()` method for on-demand frame regeneration

**New Frame Naming:**
- Pattern: `data/{video_id}/frame_{index:06d}_{timestamp:.6f}.png`
- Example: `data/abc123/frame_000012_12.345678.png`

**FrameExtractor Changes:**
- Added `output_dir` parameter to constructor
- `extract_frames()` now requires `video_id` parameter
- Returns frame dicts with: `timestamp`, `filename`, `index`, `frame_path`
- Added `_parse_showinfo()` to extract precise timestamps from ffmpeg output
- Added `regenerate_frame()` to recreate missing frames
- Updated `extract_frames_generator()` signature

**VideoEmbedder Changes:**
- `embed_and_store_frames()` now loads frames from disk instead of memory
- Added parameters: `sampling_rate`, `max_dimension`
- Handles missing frame files gracefully with warnings
- Stores comprehensive metadata including frame paths and extraction parameters

**New Module Features:**
- Added convenience function `regenerate_missing_frame()` at module level
- Updated module docstring with comprehensive documentation

### 2. Pipeline Core (`rag_lab/core/pipeline.py`)

**Metadata Dataclass:**
- Removed `year` field (not necessary)
- Added `extraction_fps` field
- Added `extraction_max_dimension` field
- Added `ffmpeg_version` field

**VideoFrame Metadata:**
- Now includes: `file_path`, `file_hash`, `frame_index`, `timestamp`, `sampling_rate`, `max_dimension`, `frame_filename`, `frame_path`

**FileProcessingPipeline:**
- Added `frames_output_dir` config parameter
- Updated `extract_metadata()` to capture extraction parameters and ffmpeg version
- Added `_get_ffmpeg_version()` helper method
- Updated `process_video()` to pass `video_id` to frame extraction
- Updated `process_video()` to pass all metadata parameters to embedding

### 3. Test Script (`scripts/test_frame_embedding.py`)

**Updates:**
- Added `frames_output_dir` to config
- Updated to use new `extract_frames()` API with `video_id`
- Updated to pass new parameters to `embed_and_store_frames()`
- Added output showing extraction parameters and frame file locations

### 4. Documentation

**New Files:**
- `docs/frame_extraction_guide.md`: Comprehensive guide for using the new frame extraction system

**Includes:**
- Frame naming conventions
- Metadata structure
- Usage examples for all components
- Regeneration workflow
- FFmpeg command reference
- Best practices

## Key Features Implemented

✅ **Reproducible Frame Extraction**
- Uses ffmpeg with consistent parameters
- Tracks ffmpeg version for reproducibility
- Stores extraction parameters in metadata

✅ **Precise Timestamps**
- Uses ffmpeg showinfo filter to capture exact pts_time
- Timestamps included in filenames (6 decimal places)
- Stored in both filename and metadata

✅ **Frame Storage**
- Organized in video-specific directories: `data/{video_id}/`
- Filenames include both index and timestamp
- Easy to locate frames from metadata

✅ **Frame Regeneration**
- `regenerate_frame()` method recreates individual frames
- Uses stored timestamp for exact reproduction
- Handles missing frames during query phase

✅ **Comprehensive Metadata**
- Each frame stores: timestamp, index, filename, path, sampling_rate, max_dimension
- Video-level metadata includes extraction parameters and ffmpeg version
- Enables validation and troubleshooting

✅ **Backward Compatibility**
- All changes are in the implementation
- External API remains similar (with added required parameters)
- Test scripts updated to demonstrate usage

## Usage Example

```python
from rag_lab.core.pipeline import FileProcessingPipeline

# Configure pipeline
pipeline = FileProcessingPipeline(config={
    'video_fps': 1.0,
    'max_dimension': 768,
    'batch_size': 32,
    'frames_output_dir': 'data'
})

# Process video
results = pipeline.process("/path/to/video.mp4")

# Frames are saved to: data/{video_hash}/frame_NNNNNN_T.TTTTTT.png
# Embeddings stored in ChromaDB with full metadata

# Later, if a frame is missing:
from rag_lab.core.video_pipeline import regenerate_missing_frame

frame_path = regenerate_missing_frame(
    video_path=original_video_path,
    video_id=video_hash,
    timestamp=12.345678,
    frame_index=12,
    fps=1.0,
    max_dimension=768
)
```

## Dependencies

**Required:**
- ffmpeg (must be installed and in PATH)
- PIL/Pillow (for image loading)
- transformers (for CLIP)
- torch (for embeddings)

**Optional:**
- tqdm (for progress bars)
- psutil (for auto-detecting memory-based max_dimension)

## Testing

Run the test script:
```bash
poetry run python scripts/test_frame_embedding.py /path/to/video.mp4
```

This will:
1. Extract metadata and compute video hash
2. Extract frames using ffmpeg
3. Save frames to `data/{video_hash}/`
4. Embed frames using CLIP
5. Store embeddings in ChromaDB
6. Verify storage by querying frame 0

## Migration Notes

If you have existing frame data:

1. **Old frames in memory**: Previous version didn't save frames to disk, so no migration needed.

2. **Old embeddings in ChromaDB**: These will have less metadata. Consider:
   - Leaving them as-is (will work but lack regeneration capability)
   - Re-processing videos to get full metadata
   - Adding a migration script to augment old metadata

3. **Config changes**: Add `frames_output_dir` to your config if not using default `'data'`.

## Future Enhancements

Possible improvements:
- Batch regeneration function for multiple missing frames
- Frame extraction change detection (detect when parameters change)
- Automatic cleanup of old frames when re-extracting
- Frame deduplication based on content hashing
- Support for other output formats (JPEG, WebP)
- Parallel frame extraction for long videos
- Progress tracking for frame regeneration
