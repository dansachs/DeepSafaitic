#!/usr/bin/env python3
"""
List all available model checkpoints with timestamps.
"""

import sys
from pathlib import Path
from model_utils import list_models, find_latest_model

def main():
    """List models in checkpoint directory."""
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    else:
        # Default locations
        checkpoint_dir = Path("models")
        if not checkpoint_dir.exists():
            checkpoint_dir = Path("/content/drive/MyDrive/safaitic_project")
    
    if not checkpoint_dir.exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        print("\nUsage: python list_models.py [checkpoint_directory]")
        return 1
    
    print("=" * 70)
    print("Available Model Checkpoints")
    print("=" * 70)
    print(f"Directory: {checkpoint_dir}\n")
    
    models = list_models(checkpoint_dir)
    
    if not models:
        print("No models found.")
        print("\nExpected format: safaitic_matcher_YYYY-MM-DD_HH-MM-SS.pth")
        return 1
    
    print(f"Found {len(models)} model(s):\n")
    
    for i, (timestamp, model_path) in enumerate(models, 1):
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"{i}. {model_path.name}")
        print(f"   Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Path: {model_path}")
        print()
    
    # Show latest
    latest = find_latest_model(checkpoint_dir)
    if latest:
        print("=" * 70)
        print(f"Latest model: {latest.name}")
        print(f"Use this path: {latest}")
        print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

