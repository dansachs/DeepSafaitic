"""
Utility functions for model checkpoint management with timestamps.
"""

from pathlib import Path
from datetime import datetime
import re


def get_timestamped_model_name(base_name="safaitic_matcher", extension=".pth"):
    """
    Generate a timestamped model filename.
    
    Args:
        base_name: Base name for the model (default: "safaitic_matcher")
        extension: File extension (default: ".pth")
    
    Returns:
        Timestamped filename, e.g., "safaitic_matcher_2024-12-30_14-30-45.pth"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}{extension}"


def find_latest_model(checkpoint_dir, base_name="safaitic_matcher", extension=".pth"):
    """
    Find the most recent model checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for models
        base_name: Base name pattern to match (default: "safaitic_matcher")
        extension: File extension (default: ".pth")
    
    Returns:
        Path to latest model, or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Pattern to match: safaitic_matcher_YYYY-MM-DD_HH-MM-SS.pth
    pattern = re.compile(rf"{re.escape(base_name)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}){re.escape(extension)}")
    
    models = []
    for file_path in checkpoint_dir.glob(f"{base_name}_*{extension}"):
        match = pattern.match(file_path.name)
        if match:
            timestamp_str = match.group(1)
            try:
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                models.append((timestamp, file_path))
            except ValueError:
                # Skip if timestamp can't be parsed
                continue
    
    if not models:
        # Fallback: check for non-timestamped version
        fallback = checkpoint_dir / f"{base_name}{extension}"
        if fallback.exists():
            return fallback
        return None
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x[0], reverse=True)
    return models[0][1]


def list_models(checkpoint_dir, base_name="safaitic_matcher", extension=".pth"):
    """
    List all model checkpoints in a directory, sorted by date (newest first).
    
    Args:
        checkpoint_dir: Directory to search
        base_name: Base name pattern
        extension: File extension
    
    Returns:
        List of (timestamp, path) tuples, sorted newest first
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return []
    
    pattern = re.compile(rf"{re.escape(base_name)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}){re.escape(extension)}")
    
    models = []
    for file_path in checkpoint_dir.glob(f"{base_name}_*{extension}"):
        match = pattern.match(file_path.name)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                models.append((timestamp, file_path))
            except ValueError:
                continue
    
    # Also check for non-timestamped version
    fallback = checkpoint_dir / f"{base_name}{extension}"
    if fallback.exists():
        # Use file modification time as timestamp
        mtime = datetime.fromtimestamp(fallback.stat().st_mtime)
        models.append((mtime, fallback))
    
    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x[0], reverse=True)
    return models

