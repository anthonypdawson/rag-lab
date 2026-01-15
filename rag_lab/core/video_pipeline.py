"""
video_pipeline.py

Frame extraction and video embedding utilities for RAG-Lab.
Extracts frames from video files using ffmpeg and generates embeddings using CLIP.

Frame Extraction:
- Uses ffmpeg for reproducible frame extraction with precise timestamps
- Saves frames to disk in pattern: data/{video_id}/frame_{index:06d}_{timestamp:.6f}.png
- Each frame filename includes both frame index and high-precision timestamp (pts_time)
- Supports regeneration of missing frames using stored metadata

Metadata Storage:
- Each frame embedding includes: video_id, file_path, timestamp, frame_index, 
  frame_filename, frame_path, sampling_rate, max_dimension
- Extraction parameters (fps, max_dimension, ffmpeg version) stored in pipeline Metadata

Reproducibility:
- Same ffmpeg version + same parameters = same frames
- Missing frames can be regenerated from original video using stored timestamp
- Use regenerate_frame() to restore missing frame files during query phase
"""

from typing import List, Dict, Any, Optional, Generator
import os
import logging
import math
import subprocess
import re
import json
from pathlib import Path

# Progress tracking
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Image processing
try:
    from PIL import Image
    import numpy as np
except ImportError:
    Image = None
    np = None

# System info
try:
    import psutil
except ImportError:
    psutil = None

# CLIP embedding
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    CLIPProcessor = None
    CLIPModel = None
    torch = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_lab.video_pipeline")


class FrameExtractor:
    """Extracts frames from video files using ffmpeg and saves them to disk."""
    
    def __init__(self, fps: float = 1.0, max_dimension: int = None, output_dir: str = "data"):
        """
        Args:
            fps: Frames per second to extract
            max_dimension: Max dimension to resize frames to (for memory efficiency).
                         If None, automatically determined based on available system memory.
                         CLIP will resize to 224x224 anyway, so this is just for memory optimization.
            output_dir: Base directory for saving extracted frames
        """
        self.fps = fps
        self.output_dir = output_dir
        
        # Auto-detect max_dimension based on system memory if not specified
        if max_dimension is None:
            max_dimension = self._auto_detect_max_dimension()
        
        self.max_dimension = max_dimension
        logger.info(f"FrameExtractor initialized with fps={self.fps}, max_dimension={self.max_dimension}")
    
    def _auto_detect_max_dimension(self) -> int:
        """
        Determine optimal max_dimension based on available system memory.
        More RAM = less aggressive resizing = better quality.
        """
        if psutil is None:
            logger.warning("psutil not available, using default max_dimension=512")
            return 512
        
        # Get total system memory in GB
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        if total_memory_gb >= 24:
            # High memory systems: minimal resizing
            return 1024
        elif total_memory_gb >= 12:
            # Medium memory systems: moderate resizing
            return 768
        else:
            # Lower memory systems: more aggressive resizing
            return 512
        
    def extract_frames(
        self, 
        video_path: str,
        video_id: str,
        start: Optional[float] = None, 
        end: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video using ffmpeg and save to disk.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video (typically hash)
            start: Start time in seconds (optional)
            end: End time in seconds (optional)
            
        Returns:
            List of dicts with 'timestamp', 'filename', 'index', 'frame_path'
        """
        logger.info(f"Extracting frames from {video_path} at {self.fps} fps using ffmpeg")
        
        # Create output directory for this video
        frames_dir = Path(self.output_dir) / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        # Use -vf fps=X to extract frames at specified rate
        # Use showinfo filter to capture precise timestamps
        # Use -frame_pts 1 to include pts in output filename pattern
        output_pattern = str(frames_dir / "frame_%d.png")
        
        # Build filter string
        vf_filters = [f"fps={self.fps}"]
        if self.max_dimension:
            vf_filters.append(f"scale='if(gt(iw,ih),{self.max_dimension},-2)':'if(gt(ih,iw),{self.max_dimension},-2)'")
        vf_filters.append("showinfo")
        vf_string = ",".join(vf_filters)
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", vf_string,
            "-vsync", "0",  # Passthrough timestamps
        ]
        
        if start is not None:
            cmd.extend(["-ss", str(start)])
        if end is not None:
            cmd.extend(["-to", str(end)])
        
        cmd.extend([
            "-frame_pts", "1",  # Include pts in filename
            output_pattern,
            "-y"  # Overwrite existing files
        ])
        
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Run ffmpeg and capture stderr for showinfo output
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            stderr_output = result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            return []
        
        # Parse showinfo output to get precise timestamps
        frame_info = self._parse_showinfo(stderr_output)
        
        # Match extracted frame files with their timestamps
        frames = []
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        
        for idx, frame_file in enumerate(frame_files):
            # Extract pts from filename (ffmpeg -frame_pts names files with pts value)
            pts_match = re.search(r'frame_(\d+)\.png', frame_file.name)
            if pts_match:
                pts = int(pts_match.group(1))
                # Find corresponding timestamp from showinfo
                timestamp = frame_info.get(idx, {}).get('pts_time', pts / 1000.0)
            else:
                timestamp = frame_info.get(idx, {}).get('pts_time', idx / self.fps)
            
            # Rename file to include timestamp for easy lookup
            new_filename = f"frame_{idx:06d}_{timestamp:.6f}.png"
            new_path = frames_dir / new_filename
            frame_file.rename(new_path)
            
            frames.append({
                "timestamp": timestamp,
                "filename": new_filename,
                "index": idx,
                "frame_path": str(new_path)
            })
        
        logger.info(f"Extracted {len(frames)} frames to {frames_dir}")
        return frames
    
    def _parse_showinfo(self, stderr_output: str) -> Dict[int, Dict[str, float]]:
        """
        Parse ffmpeg showinfo filter output to extract frame timestamps.
        
        Args:
            stderr_output: stderr from ffmpeg containing showinfo output
            
        Returns:
            Dict mapping frame index to timestamp info
        """
        frame_info = {}
        frame_idx = 0
        
        # Parse lines like: [Parsed_showinfo_2 @ 0x...] n:0 pts:12345 pts_time:12.345 ...
        for line in stderr_output.split('\n'):
            if 'Parsed_showinfo' in line and 'pts_time:' in line:
                # Extract pts_time
                pts_time_match = re.search(r'pts_time:([\d.]+)', line)
                n_match = re.search(r'n:(\d+)', line)
                
                if pts_time_match:
                    pts_time = float(pts_time_match.group(1))
                    n = int(n_match.group(1)) if n_match else frame_idx
                    frame_info[n] = {'pts_time': pts_time}
                    frame_idx += 1
        
        return frame_info
    
    def regenerate_frame(
        self,
        video_path: str,
        video_id: str,
        timestamp: float,
        frame_index: int,
        output_filename: str = None
    ) -> Optional[str]:
        """
        Regenerate a single frame at a specific timestamp from the original video.
        Used when a frame file is missing during query phase.
        
        Args:
            video_path: Path to original video file
            video_id: Unique identifier for the video
            timestamp: Exact timestamp in seconds
            frame_index: Frame index for consistent naming
            output_filename: Optional custom filename (otherwise uses standard pattern)
            
        Returns:
            Path to regenerated frame file, or None if failed
        """
        logger.info(f"Regenerating frame at timestamp {timestamp} from {video_path}")
        
        # Create output directory
        frames_dir = Path(self.output_dir) / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output filename
        if output_filename is None:
            output_filename = f"frame_{frame_index:06d}_{timestamp:.6f}.png"
        
        output_path = frames_dir / output_filename
        
        # Build ffmpeg command to extract single frame at specific timestamp
        vf_filters = []
        if self.max_dimension:
            vf_filters.append(f"scale='if(gt(iw,ih),{self.max_dimension},-2)':'if(gt(ih,iw),{self.max_dimension},-2)'")
        
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),  # Seek to timestamp
            "-i", video_path,
            "-frames:v", "1",  # Extract only 1 frame
        ]
        
        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
        
        cmd.extend([
            str(output_path),
            "-y"  # Overwrite if exists
        ])
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Regenerated frame saved to {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to regenerate frame: {e.stderr}")
            return None
    
    def extract_frames_generator(
        self, 
        video_path: str,
        video_id: str,
        start: Optional[float] = None, 
        end: Optional[float] = None,
        batch_size: int = 32
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Extract frames as a generator for memory efficiency on long videos.
        Yields batches of frames.
        """
        # For now, extract all and yield in batches
        # TODO: Implement true streaming extraction
        all_frames = self.extract_frames(video_path, video_id, start, end)
        
        for i in range(0, len(all_frames), batch_size):
            yield all_frames[i:i + batch_size]


class VideoEmbedder:
    """Handles video frame embedding using CLIP."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        """
        Args:
            model_name: HuggingFace CLIP model name
            device: 'cuda', 'cpu', or 'auto'
        """
        if CLIPProcessor is None or CLIPModel is None:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        logger.info(f"Loading CLIP model: {model_name}")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def embed(self, images: List) -> List[List[float]]:
        """
        Embed a list of PIL Images and return their embeddings.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of embeddings (as lists of floats)
        """
        logger.info(f"Embedding {len(images)} frames...")
        
        # Process images and move to GPU
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to GPU
        
        # Generate embeddings on GPU
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)  # Computed on GPU
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Move results back to CPU for storage (numpy/list conversion requires CPU)
        embeddings = image_features.cpu().numpy().tolist()
        logger.info("Frame embedding complete.")
        return embeddings
    
    def embed_and_store_frames(
        self, 
        frames: List[Dict[str, Any]], 
        chroma_collection, 
        video_id: str, 
        file_path: str,
        sampling_rate: float,
        max_dimension: Optional[int] = None,
        batch_size: int = 32
    ):
        """
        Embed frames from saved files and store them in ChromaDB with batching for efficiency.
        If a frame file is missing, it will be skipped (should be regenerated via regenerate_frame).
        
        Args:
            frames: List of dicts with 'timestamp', 'filename', 'index', 'frame_path'
            chroma_collection: ChromaDB collection object
            video_id: Unique identifier for the video (hash)
            file_path: Path to the video file
            sampling_rate: FPS used for extraction
            max_dimension: Max dimension frames were resized to (if any)
            batch_size: Number of frames to process at once
        """
        logger.info(f"Preparing to embed and store {len(frames)} frames for video: {file_path}")
        
        all_ids = []
        all_embeddings = []
        all_metadatas = []
        
        total_batches = (len(frames) + batch_size - 1) // batch_size
        
        # Process in batches with progress bar
        batch_iter = range(0, len(frames), batch_size)
        if tqdm:
            batch_iter = tqdm(batch_iter, total=total_batches, desc="Embedding frames", unit="batch")
        
        for i in batch_iter:
            batch = frames[i:i + batch_size]
            
            # Load images from disk
            images = []
            valid_frames = []
            for frame in batch:
                frame_path = frame.get("frame_path")
                if frame_path and os.path.exists(frame_path):
                    try:
                        img = Image.open(frame_path)
                        images.append(img)
                        valid_frames.append(frame)
                    except Exception as e:
                        logger.warning(f"Failed to load frame {frame_path}: {e}")
                else:
                    logger.warning(f"Frame file missing: {frame_path}. Skipping. Use regenerate_frame to restore.")
            
            if not images:
                continue
            
            # Generate embeddings
            embeddings = self.embed(images)
            
            # Prepare metadata and IDs
            for frame, embedding in zip(valid_frames, embeddings):
                frame_id = f"{video_id}_frame_{frame['index']}"
                metadata = {
                    "video_id": video_id,
                    "file_path": file_path,
                    "timestamp": frame["timestamp"],
                    "frame_index": frame["index"],
                    "frame_filename": frame["filename"],
                    "frame_path": frame["frame_path"],
                    "sampling_rate": sampling_rate,
                    "max_dimension": max_dimension,
                }
                
                all_ids.append(frame_id)
                all_embeddings.append(embedding)
                all_metadatas.append(metadata)
        
        # Store all embeddings in ChromaDB
        logger.info("Storing frame embeddings in ChromaDB...")
        chroma_collection.add(
            ids=all_ids,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        logger.info(f"Stored {len(all_embeddings)} frame embeddings in ChromaDB.")


# Module-level convenience function for frame regeneration
def regenerate_missing_frame(
    video_path: str,
    video_id: str,
    timestamp: float,
    frame_index: int,
    fps: float = 1.0,
    max_dimension: int = None,
    output_dir: str = "data"
) -> Optional[str]:
    """
    Convenience function to regenerate a missing frame file.
    Use this during the query phase when a frame file is missing.
    
    Args:
        video_path: Path to original video file
        video_id: Unique identifier for the video (hash)
        timestamp: Exact timestamp in seconds (from metadata)
        frame_index: Frame index (from metadata)
        fps: Sampling rate used for original extraction
        max_dimension: Max dimension used for original extraction
        output_dir: Base directory for frames
        
    Returns:
        Path to regenerated frame file, or None if failed
        
    Example:
        >>> frame_path = regenerate_missing_frame(
        ...     video_path="/path/to/video.mp4",
        ...     video_id="abc123",
        ...     timestamp=12.345678,
        ...     frame_index=12,
        ...     fps=1.0,
        ...     max_dimension=768
        ... )
    """
    extractor = FrameExtractor(fps=fps, max_dimension=max_dimension, output_dir=output_dir)
    return extractor.regenerate_frame(video_path, video_id, timestamp, frame_index)
