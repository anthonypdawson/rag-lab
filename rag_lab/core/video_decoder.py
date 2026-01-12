"""
video_decoder.py: Video decoding and metadata extraction utilities for RAG-Lab

This module provides functions to extract metadata, decode video frames, extract audio, and handle subtitles from video files.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
@dataclass
class VideoMetadata:
    file_path: str
    duration: Optional[float] = None
    framerate: Optional[float] = None
    resolution: Optional[str] = None
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    has_subtitles: bool = False
    subtitle_tracks: List[str] = field(default_factory=list)
    external_subtitle: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
import cv2
import numpy as np
from pymediainfo import MediaInfo
import ffmpeg
import os

class VideoDecoder:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.metadata = self.extract_metadata()

    def extract_metadata(self) -> VideoMetadata:
        """
        Extracts metadata such as duration, framerate, resolution, codecs, and stream info.
        Also checks for external subtitle files.
        """
        media_info = MediaInfo.parse(self.file_path)
        video_track = next((t for t in media_info.tracks if t.track_type == 'Video'), None)
        audio_track = next((t for t in media_info.tracks if t.track_type == 'Audio'), None)
        subtitle_tracks = [t for t in media_info.tracks if t.track_type == 'Text']
        external_sub = self.find_external_subtitles()
        has_subtitles = len(subtitle_tracks) > 0 or external_sub is not None
        return VideoMetadata(
            file_path=self.file_path,
            duration=float(video_track.duration) / 1000 if video_track and video_track.duration else None,
            framerate=float(video_track.frame_rate) if video_track and video_track.frame_rate else None,
            resolution=f"{video_track.width}x{video_track.height}" if video_track and video_track.width and video_track.height else None,
            video_codec=video_track.codec if video_track and hasattr(video_track, 'codec') else None,
            audio_codec=audio_track.codec if audio_track and hasattr(audio_track, 'codec') else None,
            has_subtitles=has_subtitles,
            subtitle_tracks=[t.language or 'und' for t in subtitle_tracks],
            external_subtitle=external_sub,
            extra={},
        )

    def decode_frames(self, every_n_frames: int = 1, max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Decodes video frames at a given interval.
        Returns a list of dicts: { 'frame': np.ndarray, 'timestamp': float }
        """
        cap = cv2.VideoCapture(self.file_path)
        frames = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % every_n_frames == 0:
                timestamp = frame_count / fps if fps else None
                frames.append({'frame': frame, 'timestamp': timestamp})
                if max_frames and len(frames) >= max_frames:
                    break
            frame_count += 1
        cap.release()
        return frames

    def extract_audio(self, output_wav: Optional[str] = None) -> str:
        """
        Extracts audio to a .wav file. Returns the path to the audio file.
        """
        if output_wav is None:
            output_wav = os.path.splitext(self.file_path)[0] + '_audio.wav'
        (
            ffmpeg
            .input(self.file_path)
            .output(output_wav, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_wav

    def find_external_subtitles(self) -> Optional[str]:
        """
        Searches for external subtitle files in the same directory as the video file.
        Returns the path to the first found subtitle file, or None if not found.
        """
        base, _ = os.path.splitext(self.file_path)
        exts = [".srt", ".ass", ".vtt", ".sub", ".ssa"]
        for ext in exts:
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        # Also check for files with the same base name but different case or language tags
        directory = os.path.dirname(self.file_path)
        basename = os.path.basename(base)
        for fname in os.listdir(directory):
            lower = fname.lower()
            if lower.startswith(basename.lower()) and any(lower.endswith(e) for e in exts):
                return os.path.join(directory, fname)
        return None
    
    def extract_subtitles(self, output_srt: Optional[str] = None) -> Optional[str]:
        """
        Extracts subtitles to an .srt file if present. Returns the path or None.
        Checks for embedded and external subtitles.
        """
        # Prefer embedded subtitles if present
        if self.metadata.has_subtitles and self.metadata.subtitle_tracks:
            if output_srt is None:
                output_srt = os.path.splitext(self.file_path)[0] + '.srt'
            (
                ffmpeg
                .input(self.file_path)
                .output(output_srt, map='0:s:0')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_srt if os.path.exists(output_srt) else None
        # Otherwise, return external subtitle file if found
        if self.metadata.external_subtitle:
            return self.metadata.external_subtitle
        return None
