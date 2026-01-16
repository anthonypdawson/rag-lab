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

## Dependencies

Core dependencies (see pyproject.toml):

- faiss-cpu (>=1.13.2,<2.0.0)
- numpy (>=2.4.0,<3.0.0)
- pandas (>=2.3.3,<3.0.0)
- scikit-learn (>=1.8.0,<2.0.0)
- sentence-transformers (>=5.2.0,<6.0.0)
- tqdm (>=4.67.1,<5.0.0)
- requests (>=2.32.5,<3.0.0)
- chromadb (>=1.4.0,<2.0.0)
- transformers (>=4.40.0,<5.0.0)
- pymediainfo (>=6.0.0,<7.0.0)
- pysubs2 (>=1.6.0,<2.0.0)
- opencv-python (>=4.8.0,<5.0.0)
- ffmpeg-python (>=0.2.0,<0.3.0)
- faster-whisper

**Note:** PyTorch is required but not listed in pyproject.toml to allow installation of the CUDA-enabled version.

## Setup Checklist


1. Install system tools: `ffmpeg` and `mediainfo`.

2. Install project dependencies:
	```bash
	poetry install
	```

3. Install CUDA-enabled PyTorch in the Poetry environment (example for CUDA 13.0):
	```bash
	poetry run pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	```
	Adjust the CUDA version as needed for your system. See https://pytorch.org/get-started/locally/ for details.

4. Add missing libraries (if not already in pyproject.toml):
	```bash
	poetry add transformers pymediainfo pysubs2 opencv-python ffmpeg-python
	```

5. Pre-download the sentence-transformers embedding model (optional but recommended to avoid download timeouts):
	```bash
	poetry run huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/all-MiniLM-L6-v2
	```
	The pipeline will automatically detect and use the local model from `./models/all-MiniLM-L6-v2` if present, otherwise it will download from Hugging Face.

6. Install the correct NVIDIA CUDA Toolkit for GPU support with faster-whisper:
	- See [CUDA setup guidance](https://github.com/SYSTRAN/faster-whisper/issues/1276) for details on required CUDA versions and installation steps for Windows and Linux.
	- Make sure your CUDA version matches your PyTorch and faster-whisper requirements.
	- Add the CUDA bin directory to your system PATH (or install to the correct paths according to the issue link).

7. Verify GPU availability:
	```bash
	poetry run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	```

---

For setup instructions and usage examples, see the documentation or ask for code samples.