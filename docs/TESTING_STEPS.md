# Step-by-Step Testing Guide for scan_and_visualize.py

## Quick Start Checklist

- [ ] Model checkpoint exists
- [ ] Reference glyphs directory exists
- [ ] At least one test stone image available
- [ ] Dependencies installed

## Step 1: Verify Prerequisites (2 minutes)

### Check Model
```bash
# Check if model exists
ls -lh models/safaitic_matcher.pth

# Or list all models
python list_models.py models/
```

**Expected:** Should see your model file (140MB+)

### Check Reference Glyphs
```bash
# Check cleaned_glyphs directory
ls cleaned_glyphs/

# Should see 28 folders (alif, b, t, etc.)
```

**Expected:** 28 glyph folders, each with `ideal.png`

### Check Dependencies
```bash
python -c "import torch, cv2, albumentations, matplotlib; print('✓ All dependencies installed')"
```

**Expected:** No errors

## Step 2: Get Test Images (5 minutes)

### Option A: Extract from Database (Recommended)
```bash
# Extract 5 test images from your database
python3 extract_stone_images_from_db.py \
    --download-urls \
    --limit 5
```

**Expected:** Images saved to `stone_images/` directory

### Option B: Use Existing Images
If you already have stone images somewhere:
```bash
# Create directory
mkdir -p stone_images

# Copy or symlink your images
cp /path/to/your/images/*.jpg stone_images/
```

## Step 3: Run First Test (2 minutes)

### Quick Test Script
```bash
# Use the quick test script
bash quick_test.sh stone_images/stone_1.jpg
```

**Or manually:**
```bash
python scan_and_visualize.py stone_images/stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output test_results
```

**Expected Output:**
```
============================================================
SAFAITIC STONE SCANNER
============================================================
Image: stone_images/stone_1.jpg
Model: models/safaitic_matcher.pth
Glyphs: cleaned_glyphs

Loading stone image...
✓ Image loaded: (height, width, 3) (mode: RGB)

Using device: cuda  (or cpu)
Loading model from: models/safaitic_matcher.pth
✓ Model loaded successfully on cuda
  Validation loss: X.XXXX
  Trained for X epochs

Loading reference embeddings...
  ✓ alif
  ✓ b
  ... (28 total)

============================================================
STEP A: Creating Binary Mask
============================================================
✓ Binary mask created

============================================================
STEP B: Finding All Candidates
============================================================
✓ Found X candidate boxes

============================================================
STEP C: Classifying with Siamese Model
============================================================
Classifying X candidate boxes...
  Box 0: alif (distance: 0.XXX)
  Box 1: b (distance: 0.XXX)
  ...
✓ Classified X high-confidence glyphs

============================================================
GENERATING VISUALIZATION
============================================================
✓ Visualization saved to: test_results/stone_scan_visualization.png

============================================================
EXPORTING TRANSCRIPTION
============================================================
✓ Transcription saved to: test_results/transcription.txt

============================================================
SCAN COMPLETE
============================================================
Detected X glyphs
Results saved to: test_results
```

## Step 4: Review Results (5 minutes)

### Check Visualization
```bash
# Open the visualization
open test_results/stone_scan_visualization.png

# Or on Linux:
xdg-open test_results/stone_scan_visualization.png
```

**What to Check:**

1. **Panel 1 (Original Image):**
   - ✓ Image loads correctly
   - ✓ No distortion or errors

2. **Panel 2 (Binary Mask):**
   - ✓ Glyphs visible as white shapes on black
   - ✓ Not too noisy (should see clear glyph outlines)
   - ⚠️ If all black/white: thresholding may need adjustment

3. **Panel 3 (All Candidates - RED boxes):**
   - ✓ RED boxes cover actual glyphs
   - ⚠️ Too many boxes on noise? → Increase `--min-area`
   - ⚠️ Missing glyphs? → Decrease `--min-area`

4. **Panel 4 (Final - GREEN boxes):**
   - ✓ GREEN boxes on real glyphs
   - ✓ Labels look reasonable
   - ⚠️ Too many false positives? → Lower `--confidence`
   - ⚠️ Missing glyphs? → Raise `--confidence`

### Check Transcription
```bash
cat test_results/transcription.txt
```

**Expected:**
```
Safaitic Stone Transcription
==================================================

Number of glyphs detected: X

Glyph sequence:
alif b t th ...

Detailed breakdown:
  1. alif (distance: 0.XXX, pos: (X, Y))
  2. b (distance: 0.XXX, pos: (X, Y))
  ...
```

## Step 5: Tune Parameters (10-15 minutes)

### If Too Many False Positives

**Symptoms:**
- Many RED boxes on noise/background
- GREEN boxes on non-glyph regions
- Low precision

**Fix:**
```bash
# Increase min-area, lower confidence
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --confidence 0.8 \
    --min-area 200 \
    --output test_results_strict
```

### If Missing Glyphs

**Symptoms:**
- Known glyphs not detected
- Few or no GREEN boxes
- Low recall

**Fix:**
```bash
# Decrease min-area, raise confidence
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --confidence 1.2 \
    --min-area 50 \
    --output test_results_lenient
```

### Test Different Thresholds

```bash
# Test multiple confidence thresholds
for conf in 0.8 1.0 1.2 1.5; do
    echo "Testing confidence: $conf"
    python scan_and_visualize.py stone_1.jpg \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --confidence $conf \
        --output "test_results_conf_$conf"
done

# Compare results
ls -lh test_results_conf_*/
```

## Step 6: Test Edge Cases (Optional, 10 minutes)

### Test Different Image Formats
```bash
# If you have RGBA images
python scan_and_visualize.py rgba_image.png ...

# If you have grayscale images
python scan_and_visualize.py grayscale.jpg ...
```

### Test Large Images
```bash
# Test with high-resolution image
python scan_and_visualize.py large_stone.jpg ...
```

**Monitor:** Watch for memory issues (should be handled automatically now)

### Test with Ground Truth (If Available)

Create `ground_truth.csv`:
```csv
box_index,glyph_name
0,alif
1,b
2,t
```

Run:
```bash
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --ground-truth ground_truth.csv \
    --output test_results_with_gt
```

**Check:** YELLOW boxes indicate disagreements

## Step 7: Batch Test (If Everything Works)

Once you're happy with parameters:

```bash
# Process multiple images
for img in stone_images/*.jpg; do
    echo "Processing: $img"
    python scan_and_visualize.py "$img" \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --confidence 1.0 \
        --min-area 100 \
        --output "scan_results/$(basename $img .jpg)"
done
```

## Troubleshooting

### "Model not found"
```bash
# Check model location
ls models/

# List available models
python list_models.py models/

# Use latest automatically
python scan_and_visualize.py stone.jpg \
    --checkpoint models/ \
    --glyphs cleaned_glyphs
```

### "No glyphs detected"
1. Check binary mask (Panel 2) - are glyphs visible?
2. Lower `--min-area` (try 50)
3. Check image quality/contrast
4. May need to adjust adaptive thresholding in code

### "Too many false positives"
1. Increase `--min-area` (try 200)
2. Lower `--confidence` (try 0.8)
3. Check binary mask quality

### "CUDA out of memory"
```bash
# Use CPU instead
CUDA_VISIBLE_DEVICES="" python scan_and_visualize.py ...
```

### "Image format error"
- Should be fixed now, but if issues persist:
  - Convert image to RGB: `convert image.png -background white -alpha remove image_rgb.jpg`
  - Or use PIL to convert before scanning

## Success Criteria

You're ready to use the scanner when:
- ✅ No Python errors
- ✅ Binary mask shows glyphs clearly
- ✅ RED boxes cover actual glyphs (not just noise)
- ✅ GREEN boxes are on real glyphs with reasonable labels
- ✅ Processing time is acceptable (< 1 minute per image)
- ✅ Results match expectations

## Next Steps After Testing

1. **Document optimal parameters** for your image set
2. **Create batch processing script** for production
3. **Set up monitoring** for large-scale processing
4. **Create validation pipeline** to catch errors

## Quick Reference

**Basic test:**
```bash
bash quick_test.sh stone_images/stone_1.jpg
```

**With custom parameters:**
```bash
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --confidence 1.0 \
    --min-area 100 \
    --output results
```

**List available models:**
```bash
python list_models.py models/
```

**Check results:**
```bash
open test_results/stone_scan_visualization.png
cat test_results/transcription.txt
```

Good luck! 🎉

