"""
pipeline.py: Multimodal file processing pipeline for RAG-Lab

This module defines a pipeline to process video files, extract multimodal data (text, audio, video), generate embeddings for each modality, and prepare data for storage in a vector database.
"""


from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import os
import logging
import json
from pathlib import Path

from rag_lab.core.text_pipeline import SubtitleParser, TextEmbedder
from rag_lab.core.video_pipeline import FrameExtractor, VideoEmbedder
from rag_lab.core.audio_pipeline import AudioExtractor, AudioDescriber, AudioEmbedder
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
    # Frame extraction parameters for reproducibility
    extraction_fps: Optional[float] = None
    extraction_max_dimension: Optional[int] = None
    ffmpeg_version: Optional[str] = None
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
        # Lazy init: load text embedder only when needed in process_text
        self.text_embedder = None
        self.frame_extractor = None  # Lazy init to avoid loading CLIP if not needed
        self.video_embedder = None
        # Audio processing components (lazy init)
        self.audio_extractor = None
        self.audio_describer = None
        self.audio_embedder = None
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
        
        # Get ffmpeg version for reproducibility tracking
        ffmpeg_version = self._get_ffmpeg_version()
        
        # Get extraction parameters from config
        extraction_fps = self.config.get('video_fps', 1.0)
        extraction_max_dimension = self.config.get('max_dimension', None)
        
        return Metadata(
            file_path=file_path, 
            file_hash=file_hash,
            extraction_fps=extraction_fps,
            extraction_max_dimension=extraction_max_dimension,
            ffmpeg_version=ffmpeg_version
        )
    
    def _get_ffmpeg_version(self) -> Optional[str]:
        """Get ffmpeg version for reproducibility tracking."""
        try:
            import subprocess
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Extract version from first line (e.g., "ffmpeg version 4.4.2")
            first_line = result.stdout.split('\\n')[0]
            return first_line.strip()
        except Exception as e:
            logger.warning(f"Could not determine ffmpeg version: {e}")
            return None

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

        # Lazy-load text embedder, preferring a locally downloaded model
        if self.text_embedder is None:
            model_name = self._resolve_text_model()
            self.text_embedder = TextEmbedder(model_name=model_name)

        # Embed and store in ChromaDB
        self.text_embedder.embed_and_store_subtitles(segments, collection, video_hash, file_path)
        logger.info(f"Embedded and stored {len(segments)} subtitle segments.")
        
        # Convert to TextSegment format for return, carrying provenance
        text_segments: List[TextSegment] = []
        for seg in segments:
            text_segments.append(TextSegment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                metadata={
                    "file_path": file_path,
                    "file_hash": video_hash,
                    "origin": seg.get("origin", "unknown"),
                    "language": seg.get("language", "und"),
                    "subtitle_source_path": seg.get("source_path"),
                }
            ))

        return text_segments

    def _resolve_text_model(self) -> str:
        """Resolve a local model path under './models' before falling back to HF id.

        Preference order:
        1) If 'text_model' is a valid directory path, use it directly.
        2) If './models/<basename>' exists, use that path.
        3) If './models/<org>/<basename>' exists, use that path.
        4) Otherwise, return the original spec (HF id or short name).
        """
        spec = self.config.get("text_model", "all-MiniLM-L6-v2")
        try:
            # Direct path provided
            if isinstance(spec, str) and os.path.isdir(spec):
                return spec

            models_dir = self.config.get("models_dir", "models")

            # Derive basename from HF id like 'sentence-transformers/all-MiniLM-L6-v2'
            parts = spec.replace("\\", "/").split("/")
            base_name = parts[-1].replace(":", "_")
            org_name = parts[-2] if len(parts) > 1 else None

            # Check ./models/<basename>
            candidate = os.path.join(models_dir, base_name)
            if os.path.isdir(candidate):
                return candidate

            # Check ./models/<org>/<basename>
            if org_name:
                nested = os.path.join(models_dir, org_name, base_name)
                if os.path.isdir(nested):
                    return nested
        except Exception:
            pass
        return spec

    def process_audio(self, file_path: str, metadata: Metadata) -> List[AudioSegment]:
        logger.info(f"Processing audio for: {file_path}")

        video_hash = metadata.file_hash

        # Get configuration
        chunk_duration = self.config.get('audio_chunk_duration', 10.0)
        overlap = self.config.get('audio_overlap', 2.0)
        temp_dir = self.config.get('temp_dir', 'data/tmp')
        audio_batch_size = self.config.get('audio_batch_size', 8)
        # Note: Default to 'audio-caption' since speech is handled by subtitle pipeline
        audio_model_provider = self.config.get('audio_model_provider', 'audio-caption')  # 'audio-caption', 'whisper', or 'combined'
        whisper_model = self.config.get('whisper_model', 'base')

        # Lazy initialization of audio components
        if self.audio_extractor is None:
            self.audio_extractor = AudioExtractor(chunk_duration=chunk_duration, overlap=overlap)
        if self.audio_describer is None:
            self.audio_describer = AudioDescriber(
                model_provider=audio_model_provider,
                model_name=whisper_model
            )
        if self.audio_embedder is None:
            audio_model = self._resolve_text_model()  # Reuse text model resolver
            self.audio_embedder = AudioEmbedder(model_name=audio_model)

        audio_path = None
        chunks = []
        chunk_dir = None
        try:
            # Extract audio from video (with video_hash for reusability)
            audio_path = self.audio_extractor.extract_audio(
                file_path, 
                temp_dir=temp_dir, 
                video_hash=video_hash
            )
            # Chunk audio into segments (with video_hash for reusability)
            chunks = self.audio_extractor.chunk_audio(
                audio_path, 
                temp_base=temp_dir, 
                video_hash=video_hash
            )
            if not chunks:
                logger.warning(f"No audio chunks created for: {file_path}")
                return []

            logger.info(f"Created {len(chunks)} audio chunks")

            # Generate descriptions for each chunk (batched for efficiency)
            descriptions = self.audio_describer.describe_batch(chunks, batch_size=audio_batch_size)

            # Get or create audio collection
            collection = self.db_client.get_or_create_collection("audio")

            # Embed and store
            self.audio_embedder.embed_and_store_audio(
                chunks, descriptions, collection, video_hash, file_path
            )
            logger.info(f"Embedded and stored {len(chunks)} audio descriptions.")

            # Optionally save segments to JSON for review
            if self.config.get('save_audio_segments', False):
                self._save_audio_segments_json(file_path, video_hash, chunks, descriptions)

            # Return AudioSegment objects
            audio_segments = []
            for chunk, description in zip(chunks, descriptions):
                audio_segments.append(AudioSegment(
                    start=chunk["start"],
                    end=chunk["end"],
                    metadata={
                        "file_path": file_path,
                        "file_hash": video_hash,
                        "duration": chunk["duration"],
                        "chunk_index": chunk["index"],
                        "description": description,
                    }
                ))
            return audio_segments
        finally:
            # Cleanup temp audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned up temp audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp audio file: {e}")
            # Cleanup temp audio chunk files and directory
            if chunks:
                chunk_dir = os.path.dirname(chunks[0]['chunk_path'])
                try:
                    import shutil
                    if os.path.exists(chunk_dir) and 'audio_chunks' in chunk_dir:
                        shutil.rmtree(chunk_dir)
                        logger.info(f"Cleaned up audio chunks directory: {chunk_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove audio chunks directory: {e}")

    def process_video(self, file_path: str, metadata: Metadata) -> List[VideoFrame]:
        logger.info(f"Processing video frames for: {file_path}")
        
        video_hash = metadata.file_hash
        
        # Get configuration
        fps = self.config.get('video_fps', 1.0)
        max_dimension = self.config.get('max_dimension', None)  # None = auto-detect based on RAM
        batch_size = self.config.get('batch_size', 32)
        
        # Lazy initialization of video components
        if self.frame_extractor is None:
            output_dir = self.config.get('frames_output_dir', 'data')
            self.frame_extractor = FrameExtractor(fps=fps, max_dimension=max_dimension, output_dir=output_dir)
        if self.video_embedder is None:
            self.video_embedder = VideoEmbedder()
        
        # Extract frames
        frames = self.frame_extractor.extract_frames(file_path, video_hash)
        
        if not frames:
            logger.warning(f"No frames extracted from: {file_path}")
            return []
        
        logger.info(f"Extracted {len(frames)} frames at {fps} fps")
        
        # Get or create video collection
        collection = self.db_client.get_or_create_collection("video")
        
        # Embed and store
        self.video_embedder.embed_and_store_frames(
            frames, collection, video_hash, file_path, 
            sampling_rate=fps, max_dimension=max_dimension, batch_size=batch_size
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
                    "frame_index": frame["index"],
                    "timestamp": frame["timestamp"],
                    "sampling_rate": fps,
                    "max_dimension": max_dimension,
                    "frame_filename": frame["filename"],
                    "frame_path": frame["frame_path"]
                }
            ))
        
        return video_frames

    def _save_audio_segments_json(
        self,
        file_path: str,
        video_hash: str,
        chunks: List[Dict[str, Any]],
        descriptions: List[str]
    ):
        """Save audio segments to JSON file for review."""
        output_data = {
            "video_path": file_path,
            "video_hash": video_hash,
            "total_chunks": len(chunks),
            "segments": [
                {
                    "index": chunk["index"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "duration": chunk["duration"],
                    "description": desc
                }
                for chunk, desc in zip(chunks, descriptions)
            ]
        }
        
        output_dir = self.config.get('audio_segments_output_dir', 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_dir,
            f"{video_hash}_audio_segments.json"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved audio segments to: {output_file}")
