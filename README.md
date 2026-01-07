# RAG-Lab

## Project Overview

RAG-Lab is a Retrieval-Augmented Generation (RAG) system designed to process video files by extracting and embedding multimodal data—video frames, audio segments, and subtitle text. The system generates vector embeddings for each modality and stores them in a vector database (ChromaDB), linking all data by their corresponding timestamps.

### Key Features
- **Video Processing:** Extracts frames from video files and generates embeddings for visual content.
- **Audio Processing:** Segments audio tracks and generates embeddings for audio content.
- **Subtitle/Text Processing:** Extracts subtitle text and generates embeddings for textual content.
- **Vector Database Integration:** Stores all embeddings in ChromaDB, with metadata linking video, audio, and text by time.
- **Ollama Model Provider:** Uses Ollama for generating embeddings and other model-based tasks.

### Typical Workflow
1. Input a video file.
2. Extract video frames, audio segments, and subtitle text, each with timestamps.
3. Generate embeddings for each modality using appropriate models.
4. Store embeddings in ChromaDB, linking them by time for efficient retrieval and cross-modal search.

### Use Cases
- Semantic search across video, audio, and text.
- Context-aware retrieval for generative AI applications.
- Multimodal data analysis and indexing.

---

For setup instructions and usage examples, see the documentation or ask for code samples.