"""
chromadb_client.py: ChromaDB client and collection management for RAG-Lab
"""

import chromadb
import os


class ChromaDBClient:
    ACCEPTED_COLLECTIONS = {"subtitles", "audio", "video"}

    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "chroma_db")
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create_collection(self, name: str, metadata: dict = None):
        if name not in self.ACCEPTED_COLLECTIONS:
            raise ValueError(f"Collection '{name}' is not accepted. Allowed: {self.ACCEPTED_COLLECTIONS}")
        if not metadata:
            metadata = {"type": name}
        return self.client.get_or_create_collection(name=name, metadata=metadata)

