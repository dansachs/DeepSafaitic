# Quick Setup - Where to Put Files

## TL;DR - File Locations

```
glyph-training/                    # Your current directory
│
├── models/                        # ✅ Model already downloaded
│   └── safaitic_matcher.pth      # ✅ 140MB - ready to use
│
├── cleaned_glyphs/                # ✅ Reference glyphs (already exists)
│   ├── alif/
│   │   └── ideal.png
│   └── ... (all 28 glyphs)
│
├── stone_images/                  # ← SET UP: Full stone images from SQL
│   ├── stone_001.jpg
│   └── ...
│
└── scan_results/                  # ← CREATED AUTOMATICALLY
```

## Step 1: Model ✅ Already Done!

Your model is already downloaded at `models/safaitic_matcher.pth` (140MB)

## Step 2: Set Up Full Stone Images from SQL

**If your SQL stone images are at:** `/path/to/sql/stone_images/`

**Option A: Create stone_images folder and symlink (Recommended)**
```bash
mkdir -p stone_images
ln -s /path/to/sql/stone_images/* stone_images/
```

**Option B: Use Absolute Path (No setup needed)**
Just use the full path when running the script:
```bash
python scan_and_visualize.py /path/to/sql/stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

**Note:** `cleaned_glyphs/` already exists in this directory - no setup needed for reference glyphs!

## Step 3: Verify Setup

Run the setup checker:
```bash
python setup_local.py
```

## Step 4: Test

```bash
# If you created stone_images folder:
python scan_and_visualize.py stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs

# Or with absolute path to SQL directory:
python scan_and_visualize.py /path/to/sql/stone_images/stone_001.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

## Summary

| What | Status | Location |
|------|--------|----------|
| **Model** | ✅ Ready | `models/safaitic_matcher.pth` (already downloaded) |
| **Reference Glyphs** | ✅ Ready | `cleaned_glyphs/` (already exists) |
| **Stone Images** | ⚠️ Set up needed | `stone_images/` or use SQL path directly |

**See `STONE_IMAGES_SETUP.md` for detailed instructions on organizing stone images from SQL.**

That's it! 🎉

