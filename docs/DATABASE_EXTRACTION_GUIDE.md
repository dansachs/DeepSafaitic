# Extracting Stone Images from SQLite Database

## Database Structure

Your database (`safaitic.db`) contains:
- **Table:** `inscriptions`
- **Rows:** 22,742 inscriptions
- **Image column:** `image_url` (contains URLs to ociana.osu.edu)

## Options for Getting Images

### Option 1: Download from URLs (Recommended for small batches)

Download images on-demand as you need them:

```bash
# Download first 10 images as a test
python3 extract_stone_images_from_db.py \
    --download-urls \
    --limit 10
```

**Note:** Downloading all 22,742 images will take significant time and bandwidth.

### Option 2: Check for Local Copies

If you have local copies of the images somewhere, point the script to that directory:

```bash
# If images are in a local directory
python3 extract_stone_images_from_db.py \
    --local-base "/path/to/local/images"
```

### Option 3: Extract Specific Inscriptions

Query the database for specific inscriptions and download only those:

```python
import sqlite3
from extract_stone_images_from_db import extract_images_from_db

conn = sqlite3.connect('/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db')
cursor = conn.cursor()

# Get IDs of inscriptions you want
cursor.execute("SELECT id FROM inscriptions WHERE status = 'active' LIMIT 100")
ids = [row[0] for row in cursor.fetchall()]

# Download only those
# (You'd need to modify the script to accept a list of IDs)
```

## Quick Start

### 1. Inspect Database (No downloads)

```bash
python3 extract_stone_images_from_db.py --inspect-only
```

### 2. Test Download (First 5 images)

```bash
python3 extract_stone_images_from_db.py \
    --download-urls \
    --limit 5
```

### 3. Full Extraction (All 22,742 images)

**⚠️ Warning:** This will download ~22,742 images. This may take hours and use significant bandwidth.

```bash
python3 extract_stone_images_from_db.py --download-urls
```

## Script Options

```bash
python3 extract_stone_images_from_db.py [OPTIONS]

Options:
  --table TEXT              Table name (default: inscriptions)
  --image-column TEXT       Image column (default: image_url)
  --id-column TEXT          ID column (default: id)
  --output TEXT             Output directory (default: stone_images)
  --local-base TEXT         Base directory for relative paths
  --download-urls           Download from URLs (default: False)
  --inspect-only            Only inspect, don't extract
  --limit INTEGER           Limit number of images to extract
```

## Recommended Workflow

1. **Test with a few images first:**
   ```bash
   python3 extract_stone_images_from_db.py --download-urls --limit 10
   ```

2. **Verify the images look correct:**
   ```bash
   ls -lh stone_images/
   ```

3. **Test scanning:**
   ```bash
   python scan_and_visualize.py stone_images/stone_*.jpg \
       --checkpoint models/safaitic_matcher.pth \
       --glyphs cleaned_glyphs
   ```

4. **If everything works, download more:**
   ```bash
   # Download in batches to avoid overwhelming the server
   python3 extract_stone_images_from_db.py --download-urls --limit 100
   ```

## Alternative: Use URLs Directly

If you don't want to download all images, you could modify the scanner to work with URLs directly, but this would require internet access and be slower.

## Current Status

- ✅ Database found: `/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db`
- ✅ 22,742 inscriptions with image URLs
- ⚠️ Images are URLs (not local files)
- ⚠️ Need to download or find local copies

