"""
test_frame_embedding.py: Test frame extraction and embedding for RAG-Lab
"""

import sys
import os
from rag_lab.core.video_pipeline import FrameExtractor, VideoEmbedder
from rag_lab.core.chromadb_client import ChromaDBClient
from rag_lab.core.pipeline import FileProcessingPipeline

def main(video_path):
    # Extract metadata (including md5 hash)
    pipeline = FileProcessingPipeline(config={
        'video_fps': 1.0,  # 1 frame per second
        'max_dimension': None,  # Auto-detect based on system RAM (or set explicitly like 512, 768, 1024)
        'batch_size': 32,  # Process 32 frames at a time
        'frames_output_dir': 'data'  # Directory where frames will be saved
    })
    
    print("Extracting metadata...")
    metadata = pipeline.extract_metadata(video_path)
    video_hash = metadata.file_hash
    file_path = metadata.file_path
    fps = metadata.extraction_fps
    max_dimension = metadata.extraction_max_dimension
    
    print(f"Video hash: {video_hash}")
    print(f"Extraction FPS: {fps}")
    print(f"Max dimension: {max_dimension}")
    
    # Extract frames
    print("Extracting frames using ffmpeg...")
    extractor = FrameExtractor(fps=fps, max_dimension=max_dimension, output_dir='data')
    frames = extractor.extract_frames(video_path, video_hash)
    
    if not frames:
        print("No frames extracted from video.")
        return
    
    print(f"Extracted {len(frames)} frames.")
    print(f"Frame timestamps: {[f['timestamp'] for f in frames[:5]]}..." if len(frames) > 5 else f"Frame timestamps: {[f['timestamp'] for f in frames]}")
    print(f"Frame files saved to: data/{video_hash}/")
    
    # Initialize embedder and ChromaDB client
    print("Loading CLIP model...")
    embedder = VideoEmbedder()
    db_client = ChromaDBClient()
    collection = db_client.get_or_create_collection("video")
    
    # Embed and store
    print("Embedding and storing frames...")
    embedder.embed_and_store_frames(
        frames, collection, video_hash, file_path,
        sampling_rate=fps, max_dimension=max_dimension, batch_size=32
    )
    print(f"Embedded and stored {len(frames)} frames for video: {video_path}")
    
    # Query to verify
    print("\nVerifying storage...")
    results = collection.get(ids=[f"{video_hash}_frame_0"])
    if results['ids']:
        print(f"Successfully verified frame 0 in ChromaDB")
        print(f"Metadata: {results['metadatas'][0]}")
    else:
        print("Warning: Could not find frame 0 in ChromaDB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: poetry run python scripts/test_frame_embedding.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
