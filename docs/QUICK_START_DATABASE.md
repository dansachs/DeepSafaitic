# Quick Start: Extract Images from Database

## Your Setup

- ✅ **Database:** `/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db`
- ✅ **Model:** `models/safaitic_matcher.pth` (ready)
- ✅ **Reference glyphs:** `cleaned_glyphs/` (ready)
- ⚠️ **Stone images:** Need to extract from database (22,742 URLs)

## Step 1: Test with 5 Images

```bash
python3 extract_stone_images_from_db.py \
    --download-urls \
    --limit 5
```

This will:
- Download 5 images from ociana.osu.edu
- Save them to `stone_images/` directory
- Name them as `stone_<id>.jpg`

## Step 2: Test Scanning

```bash
# Test with one of the downloaded images
python scan_and_visualize.py stone_images/stone_*.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

## Step 3: Download More (If Test Works)

```bash
# Download 100 images
python3 extract_stone_images_from_db.py --download-urls --limit 100

# Or download all (22,742 images - will take time!)
python3 extract_stone_images_from_db.py --download-urls
```

## What the Script Does

1. Connects to your SQLite database
2. Queries the `inscriptions` table for `image_url` values
3. Downloads images from URLs (if `--download-urls` is set)
4. Saves them to `stone_images/` directory
5. Names files as `stone_<id>.jpg`

## Troubleshooting

**"No images found"**
- Check database path is correct
- Verify table name is `inscriptions`

**"Download errors"**
- Check internet connection
- Some URLs might be broken (script will skip them)

**"Images not working"**
- Verify images downloaded correctly: `ls -lh stone_images/`
- Check file sizes (should be > 0 bytes)

## Next Steps

Once you have images in `stone_images/`, you can:
1. Scan individual stones
2. Batch process multiple stones
3. Compare with ground truth transcriptions

See `SCANNER_USAGE.md` for detailed scanning instructions.

