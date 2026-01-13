"""
video_pipeline.py

Frame extraction and video embedding utilities for RAG-Lab.
Extracts frames from video files and generates embeddings using CLIP.
"""

from typing import List, Dict, Any, Optional, Generator
import os
import logging
import math

# Progress tracking
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Image processing
try:
    from PIL import Image
    import numpy as np
    import cv2
except ImportError:
    Image = None
    np = None
    cv2 = None

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
    """Extracts frames from video files using OpenCV."""
    
    def __init__(self, fps: float = 1.0, max_dimension: int = None):
        """
        Args:
            fps: Frames per second to extract
            max_dimension: Max dimension to resize frames to (for memory efficiency).
                         If None, automatically determined based on available system memory.
                         CLIP will resize to 224x224 anyway, so this is just for memory optimization.
        """
        self.fps = fps
        
        # Auto-detect max_dimension based on system memory if not specified
        if max_dimension is None:
            max_dimension = self._auto_detect_max_dimension()
        
        self.max_dimension = max_dimension
        logger.info(f"FrameExtractor initialized with max_dimension={self.max_dimension}")
    
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
        start: Optional[float] = None, 
        end: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video using OpenCV.
        
        Args:
            video_path: Path to video file
            start: Start time in seconds (optional)
            end: End time in seconds (optional)
            
        Returns:
            List of dicts with 'timestamp', 'frame_data' (PIL Image), 'index'
        """
        if Image is None or cv2 is None:
            raise ImportError("PIL and opencv-python are required. Install with: pip install Pillow opencv-python")
        
        logger.info(f"Extracting frames from {video_path} at {self.fps} fps")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Calculate frame interval
        frame_interval = int(video_fps / self.fps) if self.fps > 0 else 1
        
        # Set start position
        start_frame = int((start or 0.0) * video_fps)
        end_frame = int((end or duration) * video_fps) if end else total_frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_idx = 0
        current_frame_num = start_frame
        
        # Progress bar for frame extraction
        total_expected = (end_frame - start_frame) // frame_interval
        pbar = tqdm(total=total_expected, desc="Extracting frames", unit="frame") if tqdm else None
        
        while current_frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame at interval
            if (current_frame_num - start_frame) % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                h, w = frame_rgb.shape[:2]
                if max(h, w) > self.max_dimension:
                    scale = self.max_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Convert to PIL Image
                img = Image.fromarray(frame_rgb)
                
                # Calculate timestamp
                timestamp = current_frame_num / video_fps
                
                frames.append({
                    "timestamp": timestamp,
                    "frame_data": img,
                    "index": frame_idx
                })
                frame_idx += 1
                
                if pbar:
                    pbar.update(1)
            
            current_frame_num += 1
        
        if pbar:
            pbar.close()
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def extract_frames_generator(
        self, 
        video_path: str, 
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
        all_frames = self.extract_frames(video_path, start, end)
        
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
        batch_size: int = 32
    ):
        """
        Embed frames and store them in ChromaDB with batching for efficiency.
        
        Args:
            frames: List of dicts with 'timestamp', 'frame_data' (PIL Image), 'index'
            chroma_collection: ChromaDB collection object
            video_id: Unique identifier for the video (hash)
            file_path: Path to the video file
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
            
            # Extract images from batch
            images = [frame["frame_data"] for frame in batch]
            
            # Generate embeddings
            embeddings = self.embed(images)
            
            # Prepare metadata and IDs
            for frame, embedding in zip(batch, embeddings):
                frame_id = f"{video_id}_frame_{frame['index']}"
                metadata = {
                    "video_id": video_id,
                    "file_path": file_path,
                    "timestamp": frame["timestamp"],
                    "frame_index": frame["index"],
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
        logger.info(f"Stored {len(frames)} frame embeddings in ChromaDB.")
