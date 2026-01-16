"""
audio_pipeline.py

Audio extraction, chunking, and description generation for RAG-Lab.
Extracts audio from video, chunks it into segments, generates AI descriptions,
and embeds those descriptions for storage in ChromaDB.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import tempfile
import subprocess

# Audio processing
try:
    import ffmpeg
except ImportError:
    ffmpeg = None

# Text embedding (for audio descriptions)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Audio captioning/tagging
try:
    from transformers import pipeline as hf_pipeline
    from transformers import AutoProcessor, AutoModel
except ImportError:
    hf_pipeline = None
    AutoProcessor = None
    AutoModel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_lab.audio_pipeline")


class AudioExtractor:
    """Extracts audio from video files and chunks it into segments."""
    
    def __init__(self, chunk_duration: float = 10.0, overlap: float = 2.0):
        """
        Initialize audio extractor.
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between consecutive chunks in seconds
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
    def extract_audio(self, video_path: str, output_path: Optional[str] = None, temp_dir: str = "data/tmp", video_hash: Optional[str] = None) -> str:
        """
        Extract audio from video file to WAV format.
        
        Args:
            video_path: Path to the video file
            output_path: Optional path for output audio file. If None, creates temp file.
            temp_dir: Directory for temporary files (default: data/tmp)
            video_hash: Optional video hash to use in filename for reusability
            
        Returns:
            Path to the extracted audio file
        """
        if ffmpeg is None:
            raise ImportError("ffmpeg-python is not installed")
            
        if output_path is None:
            # Create temp file in project directory with video hash if provided
            os.makedirs(temp_dir, exist_ok=True)
            if video_hash:
                output_path = os.path.join(temp_dir, f"{video_hash}_audio.wav")
                # If audio already extracted, reuse it
                if os.path.exists(output_path):
                    logger.info(f"Reusing existing extracted audio: {output_path}")
                    return output_path
            else:
                fd, output_path = tempfile.mkstemp(suffix='.wav', dir=temp_dir)
                os.close(fd)
            
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        
        try:
            # Extract audio as mono 16kHz WAV (standard for speech models)
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Audio extracted successfully to {output_path}")
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error: {e.stderr.decode()}")
            raise
            
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            return 0.0
            
    def chunk_audio(self, audio_path: str, output_dir: Optional[str] = None, temp_base: str = "data/tmp", video_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save audio chunks. If None, creates in temp_base.
            temp_base: Base directory for temp files (default: data/tmp)
            video_hash: Optional video hash to use in directory name for reusability
            
        Returns:
            List of dicts with chunk metadata: {
                'chunk_path': str,
                'start': float,
                'end': float,
                'duration': float,
                'index': int
            }
        """
        if output_dir is None:
            os.makedirs(temp_base, exist_ok=True)
            if video_hash:
                output_dir = os.path.join(temp_base, f"{video_hash}_audio_chunks")
                os.makedirs(output_dir, exist_ok=True)
                # Check if chunks already exist
                if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
                    logger.info(f"Reusing existing audio chunks from: {output_dir}")
                    # Load existing chunks
                    chunks = []
                    for filename in sorted(os.listdir(output_dir)):
                        if filename.endswith('.wav'):
                            match = os.path.splitext(filename)[0].split('_')
                            if len(match) >= 2 and match[0] == 'chunk':
                                chunk_path = os.path.join(output_dir, filename)
                                # We'll need to recompute timestamps, but for now just return paths
                                pass
                    # If we can't reliably reconstruct, just re-chunk
            else:
                output_dir = tempfile.mkdtemp(prefix='audio_chunks_', dir=temp_base)
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        duration = self.get_audio_duration(audio_path)
        if duration == 0:
            logger.warning("Could not determine audio duration")
            return []
            
        chunks = []
        chunk_index = 0
        step = self.chunk_duration - self.overlap
        start_time = 0.0
        
        logger.info(f"Chunking audio: duration={duration}s, chunk_duration={self.chunk_duration}s, overlap={self.overlap}s")
        
        while start_time < duration:
            end_time = min(start_time + self.chunk_duration, duration)
            chunk_duration = end_time - start_time
            
            # Skip very short final chunks
            if chunk_duration < 1.0:
                break
                
            chunk_filename = f"chunk_{chunk_index:04d}.wav"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            # Extract chunk using ffmpeg
            try:
                (
                    ffmpeg
                    .input(audio_path, ss=start_time, t=chunk_duration)
                    .output(chunk_path, acodec='pcm_s16le', ac=1, ar='16000')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
                
                chunks.append({
                    'chunk_path': chunk_path,
                    'start': start_time,
                    'end': end_time,
                    'duration': chunk_duration,
                    'index': chunk_index
                })
                
                chunk_index += 1
            except ffmpeg.Error as e:
                logger.error(f"Failed to extract chunk at {start_time}s: {e.stderr.decode()}")
                
            start_time += step
            
        logger.info(f"Created {len(chunks)} audio chunks")
        return chunks


class AudioDescriber:
    """Generates descriptions of audio content using AI models.
    
    Note: Speech/dialogue is handled by the subtitle pipeline.
    This focuses on non-speech audio: music, sound effects, ambient sounds.
    """
    
    def __init__(self, model_provider: str = "audio-caption", model_name: Optional[str] = None):
        """
        Initialize audio describer.
        
        Args:
            model_provider: Which model to use:
                - "audio-caption": Non-speech audio only (music, sounds) - RECOMMENDED
                - "whisper": Speech transcription (redundant with subtitle pipeline)
                - "combined": Both speech and audio (use if subtitle pipeline disabled)
            model_name: Specific model name/size (provider-dependent)
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self._model = None
        self._whisper_model = None
        
    def _load_audio_caption_model(self):
        """Lazy load audio captioning model."""
        if self._model is None:
            if hf_pipeline is None:
                raise ImportError("transformers is not installed")
            
            logger.info("Loading audio captioning model (this may take a moment)...")
            # Use audio-to-text model for captioning
            # Options: "facebook/wav2vec2-base-960h" for simple, or more advanced models
            # For now, we'll use a simple audio classification approach with descriptions
            try:
                # Try to use an audio captioning model if available
                self._model = hf_pipeline(
                    "audio-classification",
                    model="MIT/ast-finetuned-audioset-10-10-0.4593"
                )
                logger.info("Audio classification model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load advanced audio model: {e}")
                logger.info("Falling back to basic audio processing")
                self._model = None
        return self._model
        
    def _load_whisper(self):
        """Lazy load Whisper model for speech transcription."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                model_size = self.model_name or "base"
                logger.info(f"Loading Whisper model: {model_size}")
                try:
                    self._whisper_model = WhisperModel(model_size, device="cuda", compute_type="float16")
                except Exception:
                    logger.warning("CUDA not available for Whisper, using CPU")
                    self._whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            except ImportError:
                raise ImportError("faster-whisper is not installed")
        return self._whisper_model
        
    def describe_audio(self, audio_path: str) -> str:
        """
        Generate a description of the audio content.
        
        Describes both speech and non-speech audio (music, sounds, ambient noise).
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Text description of the audio content
        """
        if self.model_provider == "whisper":
            return self._describe_with_whisper(audio_path)
        elif self.model_provider == "audio-caption":
            return self._describe_with_audio_caption(audio_path)
        elif self.model_provider == "combined":
            return self._describe_combined(audio_path)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _describe_with_audio_caption(self, audio_path: str) -> str:
        """Generate description using audio captioning/classification."""
        model = self._load_audio_caption_model()
        
        if model is None:
            # Fallback: use a simple heuristic approach
            return self._describe_simple(audio_path)
        
        try:
            # Run audio classification
            results = model(audio_path, top_k=3)
            
            # Format results as description
            labels = [r['label'] for r in results]
            scores = [r['score'] for r in results]
            
            # Create natural language description
            if scores[0] > 0.5:
                description = f"Audio contains: {labels[0]}"
            else:
                description = f"Audio contains: {', '.join(labels[:2])}"
            
            return description
            
        except Exception as e:
            logger.warning(f"Audio captioning failed: {e}")
            return self._describe_simple(audio_path)
    
    def _describe_simple(self, audio_path: str) -> str:
        """Simple fallback description based on audio analysis."""
        # This is a placeholder - in practice, you'd want to:
        # 1. Load audio and compute features (RMS, spectral centroid, etc.)
        # 2. Classify as speech/music/noise
        # 3. Return appropriate description
        return "[Audio content detected]"
    
    def _describe_combined(self, audio_path: str) -> str:
        """Generate description combining speech transcription and audio events."""
        parts = []
        
        # Get speech transcription
        speech_text = self._describe_with_whisper(audio_path)
        if speech_text and speech_text != "[No speech detected]":
            parts.append(f"Speech: {speech_text}")
        
        # Get audio events
        audio_desc = self._describe_with_audio_caption(audio_path)
        if audio_desc and audio_desc != "[Audio content detected]":
            parts.append(f"Audio: {audio_desc}")
        
        if not parts:
            return "[No significant audio detected]"
        
        return " | ".join(parts)
            
    def _describe_with_whisper(self, audio_path: str) -> str:
        """Generate speech transcription using Whisper."""
        model = self._load_whisper()
        
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        # Combine all segments into one transcription
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            
        transcription = " ".join(text_parts).strip()
        
        # If no speech detected, return a note
        if not transcription:
            transcription = "[No speech detected]"
            
        logger.debug(f"Transcribed audio: {transcription[:100]}...")
        return transcription
        
    def describe_batch(self, audio_chunks: List[Dict[str, Any]], batch_size: int = 8) -> List[str]:
        """
        Generate descriptions for a batch of audio chunks with GPU-efficient batching.
        
        Args:
            audio_chunks: List of chunk metadata dicts with 'chunk_path' key
            batch_size: Number of chunks to process in parallel (default: 8)
            
        Returns:
            List of descriptions, one per chunk
        """
        if self.model_provider == "audio-caption":
            return self._describe_batch_parallel(audio_chunks, batch_size)
        else:
            # Sequential for other providers
            descriptions = []
            for i, chunk in enumerate(audio_chunks):
                logger.info(f"Describing audio chunk {i+1}/{len(audio_chunks)}")
                description = self.describe_audio(chunk['chunk_path'])
                descriptions.append(description)
            return descriptions
    
    def _describe_batch_parallel(self, audio_chunks: List[Dict[str, Any]], batch_size: int) -> List[str]:
        """Process audio chunks in parallel batches for better GPU efficiency."""
        model = self._load_audio_caption_model()
        
        if model is None:
            # Fallback to sequential
            return [self._describe_simple(chunk['chunk_path']) for chunk in audio_chunks]
        
        descriptions = []
        total = len(audio_chunks)
        
        for i in range(0, total, batch_size):
            batch = audio_chunks[i:i+batch_size]
            batch_end = min(i+batch_size, total)
            logger.info(f"Describing audio chunks {i+1}-{batch_end}/{total}")
            
            try:
                # Process batch in parallel
                audio_paths = [chunk['chunk_path'] for chunk in batch]
                batch_results = model(audio_paths, top_k=3, batch_size=len(audio_paths))
                
                # Format results
                for results in batch_results:
                    if isinstance(results, list) and len(results) > 0:
                        labels = [r['label'] for r in results[:2]]
                        description = f"Audio contains: {', '.join(labels)}"
                    else:
                        description = "[Audio content detected]"
                    descriptions.append(description)
                    
            except Exception as e:
                logger.warning(f"Batch processing failed: {e}, falling back to sequential")
                for chunk in batch:
                    descriptions.append(self._describe_with_audio_caption(chunk['chunk_path']))
        
        return descriptions


class AudioEmbedder:
    """Embeds audio descriptions and stores them in ChromaDB."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize audio embedder.
        
        Args:
            model_name: Sentence transformer model for embedding descriptions
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed")
        logger.info(f"Loading audio description embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of audio descriptions."""
        logger.info(f"Embedding {len(texts)} audio descriptions...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        logger.info("Audio description embedding complete.")
        return embeddings
        
    def embed_and_store_audio(
        self,
        chunks: List[Dict[str, Any]],
        descriptions: List[str],
        chroma_collection,
        video_id: str,
        file_path: str
    ):
        """
        Embed audio descriptions and store them in ChromaDB.
        
        Args:
            chunks: List of audio chunk metadata dicts
            descriptions: List of audio descriptions (one per chunk)
            chroma_collection: ChromaDB collection object
            video_id: Unique identifier for the video (hash)
            file_path: Path to the video file
        """
        logger.info(f"Preparing to embed and store {len(descriptions)} audio descriptions for: {file_path}")
        
        embeddings = self.embed(descriptions)
        
        metadatas = [
            {
                "video_id": video_id,
                "file_path": file_path,
                "start": chunk["start"],
                "end": chunk["end"],
                "duration": chunk["duration"],
                "chunk_index": chunk["index"],
                "type": "audio_description",
            }
            for chunk in chunks
        ]
        
        ids = [f"{video_id}_audio_{i}" for i in range(len(chunks))]
        
        logger.info("Storing audio embeddings in ChromaDB...")
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=descriptions,
            metadatas=metadatas
        )
        logger.info(f"Stored {len(descriptions)} audio embeddings in ChromaDB.")
