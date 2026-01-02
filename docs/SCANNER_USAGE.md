# Safaitic Stone Scanner - Usage Guide

## Overview

The `scan_and_visualize.py` script processes full stone images to detect and classify Safaitic glyphs using your trained Siamese model. It provides step-by-step visualization of the detection process.

## Features

1. **Step A (Binary Mask)**: Converts stone image to high-contrast black-and-white using adaptive thresholding
2. **Step B (Contour Detection)**: Finds all potential glyph regions (RED boxes)
3. **Step C (Classification)**: Uses Siamese model to classify each region (GREEN boxes for high-confidence matches)
4. **Visualization**: Creates a 4-panel visualization showing the entire process
5. **Transcription Export**: Saves the detected glyph sequence to a text file
6. **Ground Truth Comparison**: Highlights disagreements in YELLOW if ground truth CSV is provided

## Installing Dependencies

```bash
pip install torch torchvision opencv-python pillow matplotlib numpy pandas albumentations
```

## Importing Model from Google Drive

The script automatically handles both **Google Colab** and **local** environments:

### Option 1: Google Colab (Recommended)

The script automatically:
- Detects if you're running in Colab
- Mounts Google Drive (if not already mounted)
- Uses default Drive paths:
  - Model: `/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth`
  - Glyphs: `/content/drive/MyDrive/safaitic_project/cleaned_glyphs`

**Usage in Colab:**
```python
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="/content/drive/MyDrive/safaitic_project/stone_image.jpg",
    model_checkpoint="/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth",
    glyphs_dir="/content/drive/MyDrive/safaitic_project/cleaned_glyphs",
    output_dir="/content/drive/MyDrive/safaitic_project/scan_results"
)
```

### Option 2: Local Usage

**Step 1: Download the model checkpoint from Google Drive**
- Go to your Google Drive: `MyDrive/safaitic_project/`
- Download `safaitic_matcher.pth` to your local machine

**Step 2: Run the script**
```bash
python scan_and_visualize.py stone_image.jpg \
    --checkpoint ./safaitic_matcher.pth \
    --glyphs ./cleaned_glyphs \
    --output ./scan_results
```

Or in Python:
```python
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="stone_image.jpg",
    model_checkpoint="./safaitic_matcher.pth",  # Local path
    glyphs_dir="./cleaned_glyphs",  # Local path
    output_dir="./scan_results"
)
```

## Command Line Usage

```bash
python scan_and_visualize.py <image_path> [options]
```

### Arguments

- `image`: Path to stone image (required)

### Options

- `--checkpoint PATH`: Path to model checkpoint (default: `safaitic_matcher.pth`)
- `--glyphs PATH`: Path to cleaned_glyphs directory (default: `cleaned_glyphs`)
- `--output PATH`: Output directory for results (default: `scan_results`)
- `--confidence FLOAT`: Confidence threshold - max distance for a match (default: 1.0)
  - Lower values = stricter matching (fewer false positives)
  - Higher values = more lenient matching (more detections, but may include false positives)
- `--min-area INT`: Minimum contour area in pixels (default: 100)
  - Increase to filter out small noise
  - Decrease to detect smaller glyphs
- `--ground-truth PATH`: Path to CSV file with ground truth transcription

## Ground Truth CSV Format

If you have manual transcriptions, create a CSV file with the following format:

```csv
box_index,glyph_name
0,alif
1,b
2,t
3,th
```

Where:
- `box_index`: Index of the box in the detected sequence (0-based)
- `glyph_name`: Expected glyph name (must match folder names in cleaned_glyphs)

Boxes where the AI's prediction disagrees with ground truth will be highlighted in **YELLOW** in the visualization.

## Output Files

The script generates the following files in the output directory:

1. **`stone_scan_visualization.png`**: 4-panel visualization showing:
   - Panel 1: Original image
   - Panel 2: Binary mask (digital rubbing)
   - Panel 3: All candidate boxes (RED)
   - Panel 4: Final transcription with classified glyphs (GREEN, or YELLOW if disagreement)

2. **`transcription.txt`**: Text file containing:
   - Number of glyphs detected
   - Glyph sequence (space-separated)
   - Detailed breakdown with positions and confidence scores

## Example: Full Workflow

### In Google Colab

```python
# Cell 1: Install and mount
!pip install opencv-python -q
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Run scanner
from scan_and_visualize import scan_stone

results = scan_stone(
    image_path="/content/drive/MyDrive/safaitic_project/stone_001.jpg",
    model_checkpoint="/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth",
    glyphs_dir="/content/drive/MyDrive/safaitic_project/cleaned_glyphs",
    output_dir="/content/drive/MyDrive/safaitic_project/scan_results",
    confidence_threshold=1.0,
    min_contour_area=100,
    ground_truth_csv="/content/drive/MyDrive/safaitic_project/ground_truth_001.csv"
)

print(f"Detected {results['num_detected']} glyphs")
print(f"Transcription: {results['transcription']}")
```

### Locally

```bash
# Download model from Drive first, then:
python scan_and_visualize.py stone_001.jpg \
    --checkpoint safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output scan_results \
    --confidence 1.0 \
    --min-area 100 \
    --ground-truth ground_truth_001.csv
```

## Tuning Parameters

### Confidence Threshold

- **Too low (< 0.8)**: May miss valid glyphs (false negatives)
- **Too high (> 1.5)**: May include noise as glyphs (false positives)
- **Recommended**: Start with 1.0, adjust based on results

### Minimum Contour Area

- **Too high**: May miss small glyphs
- **Too low**: May detect noise as glyphs
- **Recommended**: Start with 100, adjust based on image resolution

## Troubleshooting

### "Checkpoint not found"

**In Colab:**
- Make sure Drive is mounted: `drive.mount('/content/drive')`
- Verify the path: `ls /content/drive/MyDrive/safaitic_project/`
- Check that `safaitic_matcher.pth` exists

**Locally:**
- Download the checkpoint from Google Drive first
- Use absolute path: `--checkpoint /full/path/to/safaitic_matcher.pth`

### "No glyphs detected"

- Try lowering `--confidence` threshold
- Try lowering `--min-area` threshold
- Check that the binary mask (Panel 2) shows clear glyph shapes
- The image may need preprocessing (contrast adjustment, etc.)

### "CUDA out of memory"

- Use CPU instead: The script will auto-detect, but you can force CPU by setting `CUDA_VISIBLE_DEVICES=""`
- Process smaller image regions
- Reduce batch processing (if implemented in future versions)

## Advanced: Custom Model Loading

If you need more control over model loading:

```python
from scan_and_visualize import load_model_from_drive, load_reference_embeddings

# Load model manually
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model_from_drive(
    checkpoint_path="/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth",
    device=device,
    embedding_dim=512
)

# Load reference embeddings
embeddings = load_reference_embeddings(
    model=model,
    glyphs_dir="/content/drive/MyDrive/safaitic_project/cleaned_glyphs",
    device=device
)
```

## Next Steps

After scanning, you can:
1. Review the visualization to see detection quality
2. Compare with ground truth to identify systematic errors
3. Adjust confidence threshold and retry
4. Use the transcription for further analysis
5. Fine-tune the model if certain glyphs are consistently misclassified

