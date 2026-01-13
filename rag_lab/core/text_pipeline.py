"""
text_pipeline.py

Subtitle parsing and text embedding utilities for RAG-Lab.
Automatically generates subtitles if missing using faster-whisper.
"""

from typing import List, Dict, Any, Optional
import os
import logging

# Subtitle parsing
try:
    import pysubs2
except ImportError:
    pysubs2 = None

try:
    import srt
except ImportError:
    srt = None

# Text embedding
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Auto subtitle generation
try:
    from .auto_subtitle import AutoSubtitleGenerator
except ImportError:
    AutoSubtitleGenerator = None

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Optional: suppress tokenizers warning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_lab.text_pipeline")

class SubtitleParser:
    """Handles parsing of SRT/VTT subtitle files. Generates subtitles if missing."""
    def __init__(self, backend: str = "pysubs2", auto_generate: bool = True, model_size: str = "base"):
        self.backend = backend
        self.auto_generate = auto_generate
        self.model_size = model_size
        self._auto_sub = None
        if auto_generate and AutoSubtitleGenerator is not None:
            self._auto_sub = AutoSubtitleGenerator(model_size=model_size)

    def parse(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Parse subtitles for a video file, prioritizing:
        1. External subtitle file (SRT/VTT/ASS/etc.)
        2. Embedded subtitles (prefer English, else first)
        3. Auto-generate with faster-whisper if none found
        Returns list of segments with text and timestamps.
        Automatically cleans up any temp files generated during extraction.
        """
        temp_files = []
        # 1. Search for external subtitle file
        base, _ = os.path.splitext(video_path)
        for ext in [".srt", ".vtt", ".ass", ".sub", ".ssa"]:
            candidate = base + ext
            if os.path.exists(candidate):
                logger.info(f"Found external subtitle file: {candidate}")
                return self._parse_file(candidate)

        # 2. Check for embedded subtitles
        try:
            from rag_lab.core.video_decoder import VideoDecoder
        except ImportError:
            VideoDecoder = None
        if VideoDecoder is not None:
            decoder = VideoDecoder(video_path)
            tracks = decoder.metadata.subtitle_tracks if decoder.metadata else []
            if tracks:
                logger.info(f"Found embedded subtitle tracks: {tracks}")
                # Prefer English
                selected = None
                for lang in tracks:
                    if lang.lower() == "en":
                        selected = lang
                        break
                if not selected:
                    selected = tracks[0]
                # Extract embedded subtitles to temp file
                temp_srt = os.path.splitext(video_path)[0] + "._embedded.srt"
                out_path = decoder.extract_subtitles(output_srt=temp_srt)
                if out_path and os.path.exists(out_path):
                    logger.info(f"Extracted embedded subtitles to: {out_path}")
                    temp_files.append(out_path)
                    segments = self._parse_file(out_path)
                    self._cleanup_temp_files(temp_files)
                    return segments

        # 3. Auto-generate with faster-whisper
        if self.auto_generate and self._auto_sub is not None:
            logger.info(f"No subtitles found. Generating with faster-whisper for: {video_path}")
            gen_path = os.path.splitext(video_path)[0] + "._auto.srt"
            out_path = self._auto_sub.generate(video_path, output_path=gen_path)
            if out_path and os.path.exists(out_path):
                temp_files.append(out_path)
                segments = self._parse_file(out_path)
                self._cleanup_temp_files(temp_files)
                return segments

        logger.warning(f"No subtitles found for video: {video_path}")
        return []

    def _cleanup_temp_files(self, temp_files: list):
        for f in temp_files:
            try:
                os.remove(f)
                logger.info(f"Cleaned up temp file: {f}")
            except Exception as e:
                logger.warning(f"Failed to remove temp file {f}: {e}")

    def _parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a subtitle file into segments."""
        if self.backend == "pysubs2" and pysubs2:
            subs = pysubs2.load(file_path)
            return [
                {
                    "start": event.start / 1000.0,
                    "end": event.end / 1000.0,
                    "text": event.text
                }
                for event in subs
            ]
        elif self.backend == "srt" and srt:
            with open(file_path, "r", encoding="utf-8") as f:
                subs = list(srt.parse(f.read()))
            return [
                {
                    "start": sub.start.total_seconds(),
                    "end": sub.end.total_seconds(),
                    "text": sub.content
                }
                for sub in subs
            ]
        else:
            raise ImportError("No suitable subtitle parser backend available.")

class TextEmbedder:
    """Handles text embedding using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        logger.info(f"Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return their embeddings."""
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        logger.info("Text embedding complete.")
        return embeddings

    def embed_and_store_subtitles(self, segments: List[Dict[str, Any]], chroma_collection, video_id: str, file_path: str):
        """
        Embed subtitle segments and store them in ChromaDB.
        segments: list of dicts with 'text', 'start', 'end', etc.
        chroma_collection: ChromaDB collection object
        video_id: unique identifier for the video (hash)
        file_path: path to the video file
        """
        logger.info(f"Preparing to embed and store {len(segments)} subtitle segments for video: {file_path}")
        texts = [seg["text"] for seg in segments]
        embeddings = self.embed(texts)
        metadatas = [
            {
                "video_id": video_id,
                "file_path": file_path,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "language": seg.get("language", "und"),
            }
            for seg in segments
        ]
        ids = [f"{video_id}_sub_{i}" for i in range(len(segments))]
        logger.info("Storing subtitle embeddings in ChromaDB...")
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(segments)} subtitle embeddings in ChromaDB.")
