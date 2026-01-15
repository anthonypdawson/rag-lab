"""
regenerate_frames.py: Utility to regenerate missing frames from ChromaDB metadata

This script checks the ChromaDB video collection for frames with missing files
and regenerates them from the original video using stored metadata.

Usage:
    poetry run python scripts/regenerate_frames.py [--collection video] [--video-id VIDEO_ID] [--dry-run]
"""

import sys
import os
import argparse
from pathlib import Path

from rag_lab.core.chromadb_client import ChromaDBClient
from rag_lab.core.video_pipeline import regenerate_missing_frame


def check_and_regenerate_frames(
    collection_name: str = "video",
    video_id: str = None,
    dry_run: bool = False
):
    """
    Check for missing frame files and regenerate them.
    
    Args:
        collection_name: ChromaDB collection to check
        video_id: Optional video ID to check (if None, checks all)
        dry_run: If True, only report missing frames without regenerating
    """
    db_client = ChromaDBClient()
    collection = db_client.get_or_create_collection(collection_name)
    
    # Get all frame entries
    print(f"Fetching frames from collection '{collection_name}'...")
    
    if video_id:
        # Get frames for specific video
        results = collection.get(
            where={"video_id": video_id},
            include=["metadatas"]
        )
    else:
        # Get all frames
        results = collection.get(include=["metadatas"])
    
    if not results['ids']:
        print("No frames found in collection.")
        return
    
    print(f"Found {len(results['ids'])} frames in collection.")
    
    # Check for missing files
    missing_frames = []
    for frame_id, metadata in zip(results['ids'], results['metadatas']):
        frame_path = metadata.get('frame_path')
        if frame_path and not os.path.exists(frame_path):
            missing_frames.append({
                'id': frame_id,
                'metadata': metadata
            })
    
    if not missing_frames:
        print("✓ All frame files exist!")
        return
    
    print(f"\n⚠ Found {len(missing_frames)} missing frame files.")
    
    if dry_run:
        print("\nDry run - would regenerate the following frames:")
        for frame in missing_frames[:10]:  # Show first 10
            metadata = frame['metadata']
            print(f"  - Frame {metadata.get('frame_index')}: {metadata.get('frame_path')}")
        if len(missing_frames) > 10:
            print(f"  ... and {len(missing_frames) - 10} more")
        return
    
    # Regenerate missing frames
    print("\nRegenerating missing frames...")
    success_count = 0
    fail_count = 0
    
    for i, frame_info in enumerate(missing_frames):
        metadata = frame_info['metadata']
        
        print(f"\n[{i+1}/{len(missing_frames)}] Regenerating frame {metadata.get('frame_index')}...")
        print(f"  Video: {metadata.get('video_id')}")
        print(f"  Timestamp: {metadata.get('timestamp')}")
        
        try:
            new_path = regenerate_missing_frame(
                video_path=metadata['file_path'],
                video_id=metadata['video_id'],
                timestamp=metadata['timestamp'],
                frame_index=metadata['frame_index'],
                fps=metadata.get('sampling_rate', 1.0),
                max_dimension=metadata.get('max_dimension'),
                output_dir='data'  # Assuming default
            )
            
            if new_path:
                print(f"  ✓ Regenerated: {new_path}")
                success_count += 1
            else:
                print(f"  ✗ Failed to regenerate")
                fail_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Regeneration complete:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Total: {len(missing_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description="Check and regenerate missing frame files from ChromaDB metadata"
    )
    parser.add_argument(
        '--collection',
        default='video',
        help='ChromaDB collection name (default: video)'
    )
    parser.add_argument(
        '--video-id',
        help='Regenerate frames only for specific video ID (hash)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only report missing frames without regenerating'
    )
    
    args = parser.parse_args()
    
    check_and_regenerate_frames(
        collection_name=args.collection,
        video_id=args.video_id,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
