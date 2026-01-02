# Local Setup Guide for Safaitic Scanner

## Directory Structure

For local use, organize your files like this:

```
glyph-training/                    # Your project root (current directory)
├── model.py                       # Model architecture (already here)
├── dataset.py                     # Dataset class (already here)
├── scan_and_visualize.py          # Scanner script (already here)
│
├── cleaned_glyphs/                 # Reference glyph images (for classification)
│   ├── alif/
│   │   ├── ideal.png
│   │   └── square.png
│   ├── b/
│   │   ├── ideal.png
│   │   └── square.png
│   └── ... (all 28 glyph folders)
│
├── models/                        # Create this folder for model checkpoints
│   └── safaitic_matcher.pth       # Download from Google Drive
│
├── scan_results/                   # Output directory (created automatically)
│   ├── stone_scan_visualization.png
│   └── transcription.txt
│
└── stone_images/                   # Optional: organize your stone images here
    └── stone_001.jpg
```

## Step-by-Step Setup

### 1. Download Model Checkpoint from Google Drive

**Option A: Using Google Drive Web Interface**
1. Go to: https://drive.google.com
2. Navigate to: `My Drive/safaitic_project/`
3. Download `safaitic_matcher.pth`
4. Save it to: `glyph-training/models/safaitic_matcher.pth`

**Option B: Using gdown (Command Line)**
```bash
# Install gdown if needed
pip install gdown

# Get the file ID from Google Drive (right-click file > Get link)
# Replace FILE_ID with your actual file ID
gdown FILE_ID -O models/safaitic_matcher.pth
```

**Option C: Using Python Script**
```python
from google_drive_downloader import GoogleDriveDownloader as gdd

# Get file ID from Google Drive share link
file_id = 'YOUR_FILE_ID_HERE'
gdd.download_file_from_google_drive(
    file_id=file_id,
    dest_path='./models/safaitic_matcher.pth'
)
```

### 2. Set Up Your Dataset

You mentioned getting the dataset from SQL in a different directory. Here are your options:

#### Option A: Copy/Symlink to Current Directory (Recommended)

If your SQL dataset is at `/path/to/sql/dataset/cleaned_glyphs`:

```bash
# Copy (if you want a separate copy)
cp -r /path/to/sql/dataset/cleaned_glyphs ./cleaned_glyphs

# OR create a symlink (saves space, always uses latest from SQL)
ln -s /path/to/sql/dataset/cleaned_glyphs ./cleaned_glyphs
```

#### Option B: Use Absolute Path in Script

You can keep the dataset where it is and use absolute paths:

```python
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="stone.jpg",
    model_checkpoint="./models/safaitic_matcher.pth",
    glyphs_dir="/path/to/sql/dataset/cleaned_glyphs",  # Absolute path
    output_dir="./scan_results"
)
```

#### Option C: Create a Config File

Create `config.py`:
```python
# config.py
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "safaitic_matcher.pth"
GLYPHS_DIR = Path("/path/to/sql/dataset/cleaned_glyphs")  # Your SQL dataset path
OUTPUT_DIR = PROJECT_ROOT / "scan_results"
```

Then use it:
```python
from config import MODEL_PATH, GLYPHS_DIR, OUTPUT_DIR
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="stone.jpg",
    model_checkpoint=str(MODEL_PATH),
    glyphs_dir=str(GLYPHS_DIR),
    output_dir=str(OUTPUT_DIR)
)
```

### 3. Verify Dataset Structure

Make sure your `cleaned_glyphs` directory has this structure:
```
cleaned_glyphs/
├── alif/
│   ├── ideal.png
│   └── square.png
├── b/
│   ├── ideal.png
│   └── square.png
├── ... (all 28 glyphs)
```

**Required files:**
- Each glyph folder must have at least `ideal.png` (used as reference)
- `square.png` is optional but recommended

### 4. Test the Setup

Run a quick test:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Test import
python -c "from scan_and_visualize import load_model_from_drive; print('✓ Import successful')"

# Test with a sample image (if you have one)
python scan_and_visualize.py stone_image.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output scan_results
```

## Quick Reference: File Locations

| File/Folder | Location | Notes |
|------------|----------|-------|
| **Model checkpoint** | `models/safaitic_matcher.pth` | Download from Google Drive |
| **Reference glyphs** | `cleaned_glyphs/` | Can be symlinked from SQL dataset |
| **Stone images** | `stone_images/` or anywhere | Any path works |
| **Output** | `scan_results/` | Created automatically |

## Command Line Examples

### Basic Usage
```bash
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

### With Custom Paths
```bash
python scan_and_visualize.py /path/to/stone.jpg \
    --checkpoint /absolute/path/to/models/safaitic_matcher.pth \
    --glyphs /path/to/sql/dataset/cleaned_glyphs \
    --output /path/to/results
```

### Python Script Usage
```python
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="stone.jpg",
    model_checkpoint="models/safaitic_matcher.pth",
    glyphs_dir="cleaned_glyphs",  # or absolute path to SQL dataset
    output_dir="scan_results",
    confidence_threshold=1.0,
    min_contour_area=100
)

print(f"Detected {results['num_detected']} glyphs")
print(f"Transcription: {results['transcription']}")
```

## Troubleshooting

### "cleaned_glyphs not found"
- Check the path: `ls cleaned_glyphs/` should show glyph folders
- Use absolute path: `--glyphs /full/path/to/cleaned_glyphs`
- Verify structure: each folder should have `ideal.png`

### "Model checkpoint not found"
- Download from Google Drive first
- Check path: `ls models/safaitic_matcher.pth`
- Use absolute path if needed

### "No module named 'model'"
- Make sure you're in the `glyph-training` directory
- Or add it to Python path: `export PYTHONPATH=/path/to/glyph-training:$PYTHONPATH`

## Recommended Directory Structure Summary

```
glyph-training/
├── models/
│   └── safaitic_matcher.pth          # ← Download from Drive
├── cleaned_glyphs/                    # ← Symlink or copy from SQL dataset
│   ├── alif/
│   ├── b/
│   └── ...
├── scan_and_visualize.py
├── model.py
├── dataset.py
└── scan_results/                      # ← Created automatically
```

