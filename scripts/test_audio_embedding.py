"""
test_audio_embedding.py: Test audio processing and embedding for RAG-Lab
"""

import sys
import os
import json
from pathlib import Path
from rag_lab.core.audio_pipeline import AudioExtractor, AudioDescriber, AudioEmbedder
from rag_lab.core.chromadb_client import ChromaDBClient
from rag_lab.core.pipeline import FileProcessingPipeline

def main(video_path):
    # Extract metadata (including md5 hash)
    pipeline = FileProcessingPipeline(config={
        'audio_chunk_duration': 10.0,        # 10 second chunks
        'audio_overlap': 2.0,                 # 2 second overlap
        'audio_model_provider': 'audio-caption', # Focus on non-speech audio (speech handled by subtitles)
        'whisper_model': 'base'               # Only used if provider is 'whisper' or 'combined'
    })
    
    metadata = pipeline.extract_metadata(video_path)
    video_hash = metadata.file_hash
    file_path = metadata.file_path

    print(f"Processing audio for video: {video_path}")
    print(f"Video hash: {video_hash}")
    
    # Extract audio
    extractor = AudioExtractor(chunk_duration=10.0, overlap=2.0)
    audio_path = extractor.extract_audio(video_path)
    print(f"Extracted audio to: {audio_path}")
    
    # Chunk audio
    chunks = extractor.chunk_audio(audio_path)
    print(f"Created {len(chunks)} audio chunks")
    
    if not chunks:
        print("No audio chunks created. Exiting.")
        return
    
    # Generate descriptions (non-speech audio only, speech is in subtitles)
    describer = AudioDescriber(model_provider="audio-caption", model_name="base")
    descriptions = describer.describe_batch(chunks)
    
    print("\nAudio descriptions:")
    for i, (chunk, desc) in enumerate(zip(chunks, descriptions)):
        print(f"  Chunk {i} ({chunk['start']:.1f}s - {chunk['end']:.1f}s): {desc[:80]}...")
    
    # Save results to JSON for review
    output_data = {
        "video_path": video_path,
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
    
    output_file = Path(video_path).stem + "_audio_segments.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved audio segments to: {output_file}")
    
    # Initialize embedder and ChromaDB client
    embedder = AudioEmbedder()
    db_client = ChromaDBClient()
    collection = db_client.get_or_create_collection("audio")
    
    # Embed and store
    embedder.embed_and_store_audio(chunks, descriptions, collection, video_hash, file_path)
    print(f"\nEmbedded and stored {len(chunks)} audio descriptions for video: {video_path}")
    
    # Cleanup
    try:
        os.remove(audio_path)
        print(f"Cleaned up temp audio file: {audio_path}")
    except Exception as e:
        print(f"Warning: Could not remove temp audio file: {e}")
    
    # Cleanup audio chunk files
    if chunks:
        chunk_dir = os.path.dirname(chunks[0]['chunk_path'])
        try:
            import shutil
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
                print(f"Cleaned up audio chunks directory: {chunk_dir}")
        except Exception as e:
            print(f"Warning: Could not remove audio chunks directory: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_audio_embedding.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
