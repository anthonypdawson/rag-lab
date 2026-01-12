"""
auto_subtitle.py

Automatic subtitle generation using faster-whisper (preferred for speed/efficiency).
"""

from typing import Optional
import os

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

import datetime

def _format_timestamp(seconds: float) -> str:
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    ms = int((td.total_seconds() - total_seconds) * 1000)
    return f"{str(td)[:-3].rjust(8,'0')},{ms:03d}"

class AutoSubtitleGenerator:
    """Generates subtitles (SRT) from audio/video files using faster-whisper."""
    def __init__(self, model_size: str = "base", device: str = "auto"):
        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed. Please install with 'pip install faster-whisper'.")
        self.model = WhisperModel(model_size, device=device)

    def generate(self, input_path: str, output_path: Optional[str] = None, language: Optional[str] = None) -> str:
        """
        Generate subtitles for the given file. Returns the path to the SRT file.
        If output_path is not provided, creates one next to input_path.
        """
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".srt"
        segments, _ = self.model.transcribe(input_path, language=language, beam_size=5)
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, segment in enumerate(segments, 1):
                start = _format_timestamp(segment.start)
                end = _format_timestamp(segment.end)
                text = segment.text.strip().replace('\n', ' ')
                f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
        return output_path
