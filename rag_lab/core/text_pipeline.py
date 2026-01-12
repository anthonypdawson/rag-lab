"""
text_pipeline.py

Subtitle parsing and text embedding utilities for RAG-Lab.
Automatically generates subtitles if missing using faster-whisper.
"""

from typing import List, Dict, Any, Optional
import os

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

class SubtitleParser:
    """Handles parsing of SRT/VTT subtitle files. Generates subtitles if missing."""
    def __init__(self, backend: str = "pysubs2", auto_generate: bool = True, model_size: str = "base"):
        self.backend = backend
        self.auto_generate = auto_generate
        self.model_size = model_size
        self._auto_sub = None
        if auto_generate and AutoSubtitleGenerator is not None:
            self._auto_sub = AutoSubtitleGenerator(model_size=model_size)

    def parse(self, file_path: str, video_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parse subtitle file and return list of segments with text and timestamps.
        If file_path does not exist and auto_generate is enabled, generate subtitles from video_path.
        """
        if not os.path.exists(file_path):
            if self.auto_generate and self._auto_sub and video_path:
                print(f"Subtitle file {file_path} not found. Generating with faster-whisper...")
                file_path = self._auto_sub.generate(video_path, output_path=file_path)
            else:
                raise FileNotFoundError(f"Subtitle file {file_path} not found and auto-generation is disabled or unavailable.")
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
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return their embeddings."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

# Example usage (to be removed or moved to tests):
# parser = SubtitleParser()
# segments = parser.parse("example.srt", video_path="example.mp4")
# embedder = TextEmbedder()
# embeddings = embedder.embed([seg["text"] for seg in segments])
