"""
pipeline.py: Multimodal file processing pipeline for RAG-Lab

This module defines a pipeline to process video files, extract multimodal data (text, audio, video), generate embeddings for each modality, and prepare data for storage in a vector database.
"""


from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import os
import logging

from rag_lab.core.text_pipeline import SubtitleParser, TextEmbedder
from rag_lab.core.video_pipeline import FrameExtractor, VideoEmbedder
from rag_lab.core.chromadb_client import ChromaDBClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_lab.pipeline")

@dataclass
class Metadata:
    file_path: str
    file_hash: Optional[str] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None
    codec: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TextSegment:
    text: str
    start: float
    end: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AudioSegment:
    start: float
    end: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VideoFrame:
    timestamp: float
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class FileProcessingPipeline:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.subtitle_parser = SubtitleParser()
        self.text_embedder = TextEmbedder()
        self.frame_extractor = None  # Lazy init to avoid loading CLIP if not needed
        self.video_embedder = None
        self.db_client = ChromaDBClient()

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point: processes a video file and returns embeddings and metadata for each modality.
        """
        logger.info(f"Starting processing for file: {file_path}")
        # 1. Extract metadata once
        metadata = self.extract_metadata(file_path)
        logger.info("Metadata extraction complete.")
        # 2. Extract and embed subtitles
        text_results = self.process_text(file_path, metadata)
        logger.info("Text processing complete.")
        # 3. Extract and embed audio
        audio_results = self.process_audio(file_path, metadata)
        logger.info("Audio processing complete.")
        # 4. Extract and embed video frames
        video_results = self.process_video(file_path, metadata)
        logger.info("Video processing complete.")
        # 5. Combine results
        logger.info("All processing steps complete.")
        return {
            "metadata": metadata,
            "text": text_results,
            "audio": audio_results,
            "video": video_results,
        }


    def compute_md5(self, file_path: str, chunk_size: int = 8192, region_size: int = 100 * 1024 * 1024) -> str:
        """
        Compute a combined md5 hash of the first and last region_size bytes of a file.
        If the file is smaller than 2 * region_size, hash the whole file.
        """
        logger.info(f"Computing file hash for: {file_path}")
        file_size = None
        try:
            file_size = os.path.getsize(file_path)
        except Exception:
            logger.warning(f"Could not get file size for: {file_path}")
        if file_size is not None and file_size > 2 * region_size:
            md5_first = hashlib.md5()
            md5_last = hashlib.md5()
            with open(file_path, "rb") as f:
                # First region
                first_bytes = f.read(region_size)
                md5_first.update(first_bytes)
                # Last region
                f.seek(-region_size, os.SEEK_END)
                last_bytes = f.read(region_size)
                md5_last.update(last_bytes)
            combined = md5_first.hexdigest() + md5_last.hexdigest()
            logger.info(f"Computed partial file hash for large file: {file_path}")
            return combined
        else:
            md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    md5.update(data)
            logger.info(f"Computed full file hash for: {file_path}")
            return md5.hexdigest()

    def extract_metadata(self, file_path: str) -> Metadata:
        logger.info(f"Extracting metadata for: {file_path}")
        # TODO: Implement metadata extraction (ffprobe, pymediainfo, etc.)
        file_hash = self.compute_md5(file_path)
        return Metadata(file_path=file_path, file_hash=file_hash)

    def process_text(self, file_path: str, metadata: Metadata) -> List[TextSegment]:
        logger.info(f"Processing text for: {file_path}")
        
        video_hash = metadata.file_hash
        
        # Parse subtitles using SubtitleParser
        segments = self.subtitle_parser.parse(file_path)
        if not segments:
            logger.warning(f"No subtitles found for: {file_path}")
            return []
        
        logger.info(f"Parsed {len(segments)} subtitle segments.")
        
        # Get or create subtitles collection
        collection = self.db_client.get_or_create_collection("subtitles")
        
        # Embed and store in ChromaDB
        self.text_embedder.embed_and_store_subtitles(segments, collection, video_hash, file_path)
        logger.info(f"Embedded and stored {len(segments)} subtitle segments.")
        
        # Convert to TextSegment format for return
        text_segments = []
        for seg in segments:
            text_segments.append(TextSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                metadata={
                    "file_path": file_path,
                    "file_hash": video_hash
                }
            ))
        
        return text_segments

    def process_audio(self, file_path: str, metadata: Metadata) -> List[AudioSegment]:
        logger.info(f"Processing audio for: {file_path}")
        # TODO: Segment audio, generate embeddings
        # video_hash = metadata.file_hash
        return []

    def process_video(self, file_path: str, metadata: Metadata) -> List[VideoFrame]:
        logger.info(f"Processing video frames for: {file_path}")
        
        video_hash = metadata.file_hash
        
        # Get configuration
        fps = self.config.get('video_fps', 1.0)
        max_dimension = self.config.get('max_dimension', None)  # None = auto-detect based on RAM
        batch_size = self.config.get('batch_size', 32)
        
        # Lazy initialization of video components
        if self.frame_extractor is None:
            self.frame_extractor = FrameExtractor(fps=fps, max_dimension=max_dimension)
        if self.video_embedder is None:
            self.video_embedder = VideoEmbedder()
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(file_path)
        
        if not frames:
            logger.warning(f"No frames extracted from: {file_path}")
            return []
        
        logger.info(f"Extracted {len(frames)} frames at {fps} fps")
        
        # Get or create video collection
        collection = self.db_client.get_or_create_collection("video")
        
        # Embed and store
        self.video_embedder.embed_and_store_frames(
            frames, collection, video_hash, file_path, batch_size=batch_size
        )
        logger.info(f"Embedded and stored {len(frames)} frame embeddings.")
        
        # Return VideoFrame objects
        video_frames = []
        for frame in frames:
            video_frames.append(VideoFrame(
                timestamp=frame["timestamp"],
                metadata={
                    "file_path": file_path,
                    "file_hash": video_hash,
                    "frame_index": frame["index"]
                }
            ))
        
        return video_frames
