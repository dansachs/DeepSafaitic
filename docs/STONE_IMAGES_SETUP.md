# Setting Up Full Stone Images from SQL

## Current Status ✅

- ✅ **Model downloaded:** `models/safaitic_matcher.pth` (140MB)
- ✅ **Reference glyphs:** `cleaned_glyphs/` (already exists)
- ⚠️ **Stone images:** Need to set up from SQL

## Directory Structure

```
glyph-training/
├── models/
│   └── safaitic_matcher.pth          # ✅ Already downloaded
│
├── cleaned_glyphs/                    # ✅ Reference dataset (already exists)
│   ├── alif/
│   │   └── ideal.png
│   └── ... (28 glyphs)
│
├── stone_images/                       # ← CREATE THIS for your SQL stone images
│   ├── stone_001.jpg
│   ├── stone_002.jpg
│   └── ...
│
└── scan_results/                      # ← Created automatically when scanning
    ├── stone_scan_visualization.png
    └── transcription.txt
```

## Setting Up Stone Images from SQL

### Option 1: Create stone_images Folder (Recommended)

```bash
# Create directory
mkdir -p stone_images

# Copy or symlink from SQL directory
# Replace /path/to/sql/stone_images with your actual SQL path
ln -s /path/to/sql/stone_images/* stone_images/
# OR
cp /path/to/sql/stone_images/* stone_images/
```

### Option 2: Use SQL Directory Directly

You can keep the stone images in your SQL directory and use absolute paths:

```bash
python scan_and_visualize.py /path/to/sql/stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output scan_results
```

### Option 3: Batch Process from SQL Directory

Create a script to process all stones:

```python
# batch_scan_stones.py
from pathlib import Path
from scan_and_visualize import scan_stone

# SQL directory with stone images
sql_stone_dir = Path("/path/to/sql/stone_images")
output_base = Path("scan_results")

# Process all images
for stone_image in sql_stone_dir.glob("*.jpg"):
    print(f"\nProcessing: {stone_image.name}")
    
    # Create output subdirectory for each stone
    output_dir = output_base / stone_image.stem
    
    results = scan_stone(
        image_path=str(stone_image),
        model_checkpoint="models/safaitic_matcher.pth",
        glyphs_dir="cleaned_glyphs",
        output_dir=str(output_dir)
    )
    
    print(f"✓ Completed: {stone_image.name}")
    print(f"  Detected {results['num_detected']} glyphs")
```

## Quick Test

Once you have stone images set up:

```bash
# Single image
python scan_and_visualize.py stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs

# Or with absolute path to SQL directory
python scan_and_visualize.py /path/to/sql/stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

## What You Need

| Item | Status | Location |
|------|--------|----------|
| **Model** | ✅ Ready | `models/safaitic_matcher.pth` |
| **Reference Glyphs** | ✅ Ready | `cleaned_glyphs/` |
| **Stone Images** | ⚠️ Need to set up | `stone_images/` or SQL directory |

## Next Steps

1. **Identify your SQL stone images directory:**
   ```bash
   # Find where your SQL stores stone images
   # Example: /Users/dansachs/Documents/sql_stones/
   ```

2. **Choose your approach:**
   - **Symlink** (saves space, always current): `ln -s /sql/path/* stone_images/`
   - **Copy** (independent, uses space): `cp /sql/path/* stone_images/`
   - **Use directly** (no setup needed): Use full paths in commands

3. **Test with one image:**
   ```bash
   python scan_and_visualize.py <path_to_stone_image> \
       --checkpoint models/safaitic_matcher.pth \
       --glyphs cleaned_glyphs
   ```

## Example: If SQL Directory is `/Users/dansachs/Documents/sql_stones/`

```bash
# Create symlink
mkdir -p stone_images
ln -s /Users/dansachs/Documents/sql_stones/*.jpg stone_images/

# Or use directly
python scan_and_visualize.py /Users/dansachs/Documents/sql_stones/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

That's it! The scanner will process your stone images and create visualizations in `scan_results/`.

