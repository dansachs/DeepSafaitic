# Testing Guide for scan_and_visualize.py

## Overview

This guide covers testing strategies, expected bugs, and best practices for the Safaitic stone scanner.

## Prerequisites Checklist

Before testing, verify:
- [ ] Model checkpoint exists: `models/safaitic_matcher.pth`
- [ ] Reference glyphs exist: `cleaned_glyphs/` with all 28 glyphs
- [ ] At least one test stone image available
- [ ] Required packages installed: `torch`, `torchvision`, `opencv-python`, `albumentations`, `matplotlib`, `pillow`, `numpy`

## Testing Workflow

### Phase 1: Basic Functionality Test

**Goal:** Verify the script runs without errors on a single image.

```bash
# Test with one image
python scan_and_visualize.py stone_images/stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output test_results
```

**Expected Output:**
- Model loads successfully
- Binary mask is created
- Contours are detected
- At least some boxes are classified
- Visualization is generated
- Transcription file is created

**Success Criteria:**
- No Python errors
- `test_results/stone_scan_visualization.png` exists
- `test_results/transcription.txt` exists
- Visualization shows 4 panels

### Phase 2: Parameter Tuning Test

**Goal:** Find optimal parameters for your images.

```bash
# Test different confidence thresholds
for threshold in 0.8 1.0 1.2 1.5; do
    python scan_and_visualize.py stone_images/stone_1.jpg \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --confidence $threshold \
        --output test_results_threshold_$threshold
done

# Test different min-area values
for area in 50 100 200 500; do
    python scan_and_visualize.py stone_images/stone_1.jpg \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --min-area $area \
        --output test_results_area_$area
done
```

**What to Look For:**
- **Confidence threshold too low (< 0.8):** Few/no detections, high precision but low recall
- **Confidence threshold too high (> 1.5):** Many false positives, noise detected as glyphs
- **Min-area too low (< 50):** Too many small noise contours detected
- **Min-area too high (> 500):** Large glyphs detected, but small ones missed

### Phase 3: Visual Inspection Test

**Goal:** Manually verify detection quality.

1. Open `stone_scan_visualization.png`
2. Check each panel:
   - **Panel 1 (Original):** Image loads correctly
   - **Panel 2 (Binary Mask):** Glyphs are clearly visible as white on black
   - **Panel 3 (All Candidates):** RED boxes cover actual glyphs (not just noise)
   - **Panel 4 (Final):** GREEN boxes are on real glyphs with correct labels

**Common Issues:**
- Binary mask too noisy → Adjust adaptive thresholding parameters
- Too many false positives → Lower confidence threshold or increase min-area
- Missing glyphs → Raise confidence threshold or decrease min-area
- Wrong classifications → Model may need retraining or confidence threshold adjustment

### Phase 4: Batch Processing Test

**Goal:** Process multiple images without errors.

```bash
# Process 10 images
for img in stone_images/stone_{1..10}.jpg; do
    python scan_and_visualize.py "$img" \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --output "scan_results/$(basename $img .jpg)"
done
```

**What to Monitor:**
- Memory usage (watch for OOM errors)
- Processing time per image
- Consistent results across images

## Expected Bugs and Issues

### 1. Image Loading Errors

**Symptom:**
```
PIL.UnidentifiedImageError: cannot identify image file
```

**Causes:**
- Corrupted image file
- Unsupported image format
- Image is actually a URL (not downloaded)

**Solutions:**
```python
# Check image before processing
from PIL import Image
img = Image.open("stone_1.jpg")
img.verify()  # Will raise exception if corrupted
```

### 2. Empty Contours / No Detections

**Symptom:**
- Binary mask is all black or all white
- No RED boxes in Panel 3
- "Found 0 candidate boxes"

**Causes:**
- Image is too dark/bright
- Adaptive thresholding parameters too strict
- Image resolution too low
- Glyphs blend into background

**Solutions:**
```python
# Adjust adaptive thresholding in step_a_binary_mask():
# Increase blockSize for larger images
# Adjust C value (higher = more sensitive)
binary = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=15,  # Try 11, 15, 21
    C=3  # Try 2, 3, 5
)
```

### 3. Too Many False Positives

**Symptom:**
- Many RED boxes on noise/background
- GREEN boxes on non-glyph regions
- Low precision

**Solutions:**
- Increase `min_contour_area` (e.g., 200 instead of 100)
- Lower `confidence_threshold` (e.g., 0.8 instead of 1.0)
- Improve binary mask quality (preprocessing)

### 4. Missing Glyphs (False Negatives)

**Symptom:**
- Known glyphs not detected
- Low recall

**Solutions:**
- Decrease `min_contour_area` (e.g., 50 instead of 100)
- Increase `confidence_threshold` (e.g., 1.2 instead of 1.0)
- Check binary mask - glyphs may not be visible

### 5. Model Loading Errors

**Symptom:**
```
RuntimeError: Error(s) in loading state_dict
KeyError: 'model_state_dict'
```

**Causes:**
- Checkpoint format mismatch
- Model architecture changed
- Corrupted checkpoint file

**Solutions:**
```python
# Verify checkpoint structure
import torch
checkpoint = torch.load("models/safaitic_matcher.pth", map_location='cpu')
print(checkpoint.keys())  # Should include 'model_state_dict'
```

### 6. Memory Errors (OOM)

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Causes:**
- Image too large
- Batch processing too many images
- GPU memory leak

**Solutions:**
- Process images one at a time
- Resize large images before processing
- Use CPU instead: `CUDA_VISIBLE_DEVICES="" python scan_and_visualize.py ...`
- Clear GPU cache between images

### 7. Classification Errors

**Symptom:**
- Glyphs classified as wrong letters
- High distance scores even for correct matches

**Causes:**
- Model not trained well enough
- Image preprocessing mismatch (different from training)
- Confidence threshold too high

**Solutions:**
- Check model validation loss (should be < 2.0)
- Verify image preprocessing matches training (128x128, ImageNet normalization)
- Lower confidence threshold
- Retrain model with more data

### 8. Path/Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'model'
ImportError: cannot import name 'SafaiticSiameseNet'
```

**Solutions:**
- Run from project root directory
- Add to PYTHONPATH: `export PYTHONPATH=/path/to/glyph-training:$PYTHONPATH`
- Verify `model.py` exists in current directory

## Best Practices

### 1. Start Small

**Always test with 1-2 images first:**
```bash
# Test single image
python scan_and_visualize.py stone_images/stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

### 2. Use Test Images with Known Ground Truth

**Create a small test set:**
- 5-10 images you've manually transcribed
- Mix of easy and difficult images
- Various image qualities

**Compare results:**
```bash
# Create ground truth CSV
echo "box_index,glyph_name" > ground_truth.csv
echo "0,alif" >> ground_truth.csv
echo "1,b" >> ground_truth.csv
# ... etc

# Run with ground truth
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --ground-truth ground_truth.csv
```

### 3. Monitor Processing Time

**Track performance:**
```bash
time python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

**Expected times:**
- Model loading: 2-5 seconds
- Image preprocessing: < 1 second
- Contour detection: < 1 second
- Classification (per box): 0.1-0.5 seconds
- Total (small image, 10 boxes): 5-10 seconds

### 4. Save Intermediate Results

**Modify script to save binary mask:**
```python
# In step_a_binary_mask(), add:
cv2.imwrite("debug_binary_mask.png", binary_mask)
```

**This helps debug:**
- Is thresholding working?
- Are glyphs visible in binary mask?
- Too much noise?

### 5. Validate Input Images

**Check image properties:**
```python
from PIL import Image
img = Image.open("stone_1.jpg")
print(f"Size: {img.size}")
print(f"Mode: {img.mode}")
print(f"Format: {img.format}")

# Verify it's reasonable
assert img.size[0] > 100 and img.size[1] > 100, "Image too small"
assert img.mode in ['RGB', 'RGBA', 'L'], f"Unexpected mode: {img.mode}"
```

### 6. Test Edge Cases

**Test with:**
- Very large images (> 5000px)
- Very small images (< 500px)
- Low contrast images
- High noise images
- Images with no glyphs
- Images with many overlapping glyphs

### 7. Compare Multiple Runs

**Test consistency:**
```bash
# Run same image twice
python scan_and_visualize.py stone_1.jpg ... --output run1
python scan_and_visualize.py stone_1.jpg ... --output run2

# Compare results (should be identical for deterministic steps)
diff run1/transcription.txt run2/transcription.txt
```

**Note:** Contour detection may vary slightly due to OpenCV's implementation, but classifications should be consistent.

### 8. Log Everything

**Add logging to track issues:**
```python
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('scan.log')])
```

## Debugging Checklist

When something goes wrong:

1. **Check image loads:**
   ```python
   from PIL import Image
   img = Image.open("stone_1.jpg")
   img.show()  # Visual check
   ```

2. **Check binary mask:**
   - Save binary mask to file
   - Verify glyphs are visible
   - Adjust thresholding if needed

3. **Check contours:**
   - Print number of contours found
   - Visualize contours on original image
   - Verify boxes make sense

4. **Check model:**
   - Verify model loads without errors
   - Test with a known glyph image
   - Check embedding dimensions match

5. **Check classifications:**
   - Print distance scores for each box
   - Verify reference embeddings loaded correctly
   - Check confidence threshold value

## Performance Benchmarks

**Expected performance (CPU, single image):**
- Small image (1000x1000px, 5 glyphs): 5-10 seconds
- Medium image (2000x2000px, 15 glyphs): 15-30 seconds
- Large image (4000x4000px, 30 glyphs): 30-60 seconds

**Expected performance (GPU, single image):**
- Small image: 2-5 seconds
- Medium image: 5-15 seconds
- Large image: 15-30 seconds

**Bottlenecks:**
- Model inference (largest bottleneck)
- Contour detection (scales with image size)
- Image preprocessing (usually fast)

## Next Steps After Testing

Once testing is complete:

1. **Document optimal parameters** for your image set
2. **Create a batch processing script** for production use
3. **Set up monitoring** for production runs
4. **Create validation pipeline** to catch errors early
5. **Optimize** for your specific use case

## Quick Test Script

Save as `quick_test.sh`:

```bash
#!/bin/bash
# Quick test script

IMAGE=$1
if [ -z "$IMAGE" ]; then
    IMAGE="stone_images/stone_1.jpg"
fi

echo "Testing with: $IMAGE"
echo "===================="

python scan_and_visualize.py "$IMAGE" \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output test_results \
    --confidence 1.0 \
    --min-area 100

echo ""
echo "Results saved to: test_results/"
echo "Check: test_results/stone_scan_visualization.png"
```

Usage: `bash quick_test.sh stone_images/stone_1.jpg`

