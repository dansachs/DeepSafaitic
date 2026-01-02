#!/usr/bin/env python3
"""
Setup script for local Safaitic scanner environment.

Creates necessary directories and helps organize files.
"""

import os
from pathlib import Path
import sys


def create_directory_structure():
    """Create necessary directories for local setup."""
    directories = [
        "models",
        "scan_results",
        "stone_images"
    ]
    
    created = []
    existing = []
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            existing.append(dir_name)
        else:
            dir_path.mkdir(exist_ok=True)
            created.append(dir_name)
    
    print("Directory Structure:")
    print("=" * 50)
    if created:
        print(f"✓ Created: {', '.join(created)}")
    if existing:
        print(f"  Already exists: {', '.join(existing)}")
    print()


def check_model_checkpoint():
    """Check if model checkpoint exists."""
    model_path = Path("models/safaitic_matcher.pth")
    
    print("Model Checkpoint:")
    print("=" * 50)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Found: {model_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ Missing: {model_path}")
        print()
        print("  To download from Google Drive:")
        print("  1. Go to: https://drive.google.com")
        print("  2. Navigate to: My Drive/safaitic_project/")
        print("  3. Download: safaitic_matcher.pth")
        print("  4. Save to: models/safaitic_matcher.pth")
        print()
        print("  Or use gdown:")
        print("    pip install gdown")
        print("    gdown FILE_ID -O models/safaitic_matcher.pth")
    print()


def check_glyphs_directory():
    """Check if cleaned_glyphs directory exists and has correct structure."""
    glyphs_dir = Path("cleaned_glyphs")
    
    print("Reference Glyphs Directory:")
    print("=" * 50)
    
    if not glyphs_dir.exists():
        print(f"✗ Missing: {glyphs_dir}")
        print()
        print("  Options:")
        print("  1. Copy from SQL dataset:")
        print("     cp -r /path/to/sql/dataset/cleaned_glyphs ./cleaned_glyphs")
        print()
        print("  2. Create symlink (recommended):")
        print("     ln -s /path/to/sql/dataset/cleaned_glyphs ./cleaned_glyphs")
        print()
        print("  3. Use absolute path in scan_and_visualize.py")
        return
    
    # Check structure
    glyph_folders = [d for d in glyphs_dir.iterdir() if d.is_dir()]
    
    if len(glyph_folders) == 0:
        print(f"✗ Empty: {glyphs_dir}")
        print("  No glyph folders found")
        return
    
    print(f"✓ Found: {glyphs_dir}")
    print(f"  Glyph folders: {len(glyph_folders)}")
    
    # Check for ideal.png files
    ideal_count = 0
    square_count = 0
    
    for folder in glyph_folders:
        if (folder / "ideal.png").exists():
            ideal_count += 1
        if (folder / "square.png").exists():
            square_count += 1
    
    print(f"  ideal.png files: {ideal_count}/{len(glyph_folders)}")
    print(f"  square.png files: {square_count}/{len(glyph_folders)}")
    
    if ideal_count < len(glyph_folders):
        print()
        print("  ⚠️  Warning: Some glyphs missing ideal.png")
        print("     ideal.png is required for classification")
    print()


def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'cv2': 'OpenCV (opencv-python)',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'albumentations': 'Albumentations'
    }
    
    print("Python Dependencies:")
    print("=" * 50)
    
    missing = []
    installed = []
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            installed.append(package_name)
        except ImportError:
            missing.append(package_name)
    
    if installed:
        print(f"✓ Installed: {', '.join(installed)}")
    
    if missing:
        print(f"✗ Missing: {', '.join(missing)}")
        print()
        print("  Install with:")
        print(f"    pip install {' '.join(missing)}")
    else:
        print("  All dependencies installed!")
    print()


def create_config_template():
    """Create a config.py template if it doesn't exist."""
    config_path = Path("config.py")
    
    if config_path.exists():
        print("Config file already exists: config.py")
        return
    
    template = '''"""
Configuration file for local Safaitic scanner.

Update paths to match your setup.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Model checkpoint (download from Google Drive)
MODEL_PATH = PROJECT_ROOT / "models" / "safaitic_matcher.pth"

# Reference glyphs directory
# Option 1: Use local cleaned_glyphs folder
GLYPHS_DIR = PROJECT_ROOT / "cleaned_glyphs"

# Option 2: Use absolute path to SQL dataset
# GLYPHS_DIR = Path("/path/to/sql/dataset/cleaned_glyphs")

# Output directory for scan results
OUTPUT_DIR = PROJECT_ROOT / "scan_results"

# Default scanning parameters
DEFAULT_CONFIDENCE_THRESHOLD = 1.0
DEFAULT_MIN_CONTOUR_AREA = 100
'''
    
    config_path.write_text(template)
    print("✓ Created config.py template")
    print("  Update paths in config.py to match your setup")
    print()


def main():
    """Run all setup checks."""
    print("=" * 50)
    print("Safaitic Scanner - Local Setup Check")
    print("=" * 50)
    print()
    
    create_directory_structure()
    check_model_checkpoint()
    check_glyphs_directory()
    check_dependencies()
    
    # Ask if user wants config template
    if not Path("config.py").exists():
        response = input("Create config.py template? (y/n): ").strip().lower()
        if response == 'y':
            create_config_template()
    
    print("=" * 50)
    print("Setup check complete!")
    print()
    print("Next steps:")
    print("1. Download model checkpoint to models/safaitic_matcher.pth")
    print("2. Set up cleaned_glyphs directory (copy or symlink from SQL dataset)")
    print("3. Run: python scan_and_visualize.py stone.jpg --checkpoint models/safaitic_matcher.pth --glyphs cleaned_glyphs")
    print()


if __name__ == "__main__":
    main()

