"""
Script to test video metadata extraction using VideoDecoder.

Usage:
    python test_video_metadata.py <video_file>
"""

import sys
from rag_lab.core.video_decoder import VideoDecoder

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_video_metadata.py <video_file>")
        sys.exit(1)
    video_path = sys.argv[1]
    decoder = VideoDecoder(video_path)
    metadata = decoder.metadata
    print("Extracted Metadata:")
    print(metadata)

if __name__ == "__main__":
    main()
