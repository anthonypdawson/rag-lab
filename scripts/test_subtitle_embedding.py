"""
test_subtitle_embedding.py: Test embedding subtitles and storing in ChromaDB for RAG-Lab
"""

import sys
import os
from rag_lab.core.text_pipeline import SubtitleParser, TextEmbedder
from rag_lab.core.chromadb_client import ChromaDBClient
from rag_lab.core.pipeline import FileProcessingPipeline

def main(video_path):
    # Extract metadata (including md5 hash)
    pipeline = FileProcessingPipeline()
    metadata = pipeline.extract_metadata(video_path)
    video_hash = metadata.file_hash
    file_path = metadata.file_path

    # Parse subtitles using new logic
    parser = SubtitleParser()
    segments = parser.parse(video_path)
    if not segments:
        print("No subtitles found in video or external file.")
        return
    print(f"Parsed {len(segments)} subtitle segments.")

    # Initialize embedder and ChromaDB client
    embedder = TextEmbedder()
    db_client = ChromaDBClient()
    collection = db_client.get_or_create_collection("subtitles")

    # Embed and store
    embedder.embed_and_store_subtitles(segments, collection, video_hash, file_path)
    print(f"Embedded and stored {len(segments)} subtitles for video: {video_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_subtitle_embedding.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
