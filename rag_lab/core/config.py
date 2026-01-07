from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration for multimodal processing"""
    chunk_size: int = 4000  # Maximum characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum characters for a valid chunk
    processing_mode: str = 'summary'  # default processing mode
    output_format: str = 'json'  # json, txt, markdown
    save_chunks: bool = False  # Whether to save individual chunks
    parallel_processing: bool = False  # Future feature
