# Next Steps for scan_and_visualize.py

## Quick Start Checklist

- [ ] **Extract test images** from database (5-10 images)
- [ ] **Run quick test:** `bash quick_test.sh stone_images/stone_1.jpg`
- [ ] **Review visualization:** Check all 4 panels look correct
- [ ] **Tune parameters:** Adjust confidence threshold and min-area
- [ ] **Test with ground truth:** Compare results with known transcriptions
- [ ] **Scale up:** Process more images once parameters are tuned

## Immediate Next Steps

### 1. Extract Test Images (5 minutes)

```bash
# Download 5 test images from database
python3 extract_stone_images_from_db.py \
    --download-urls \
    --limit 5
```

### 2. Run First Test (2 minutes)

```bash
# Quick test with first image
bash quick_test.sh stone_images/stone_1.jpg

# Or manually:
python scan_and_visualize.py stone_images/stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output test_results
```

### 3. Review Results (5 minutes)

1. Open `test_results/stone_scan_visualization.png`
2. Check each panel:
   - **Panel 1:** Original image loads correctly
   - **Panel 2:** Binary mask shows glyphs clearly
   - **Panel 3:** RED boxes cover actual glyphs
   - **Panel 4:** GREEN boxes are correct with labels

3. Read `test_results/transcription.txt`
4. Compare with what you see in the image

### 4. Tune Parameters (10-15 minutes)

**If too many false positives:**
```bash
# Lower confidence threshold
python scan_and_visualize.py stone_1.jpg \
    --confidence 0.8 \
    --min-area 200
```

**If missing glyphs:**
```bash
# Raise confidence threshold, lower min-area
python scan_and_visualize.py stone_1.jpg \
    --confidence 1.2 \
    --min-area 50
```

**If binary mask is poor:**
- May need to adjust adaptive thresholding in code
- See `EXPECTED_BUGS.md` for details

### 5. Test with Ground Truth (Optional, 10 minutes)

Create a CSV with known transcriptions:
```csv
box_index,glyph_name
0,alif
1,b
2,t
```

Run with ground truth:
```bash
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --ground-truth ground_truth.csv
```

YELLOW boxes indicate disagreements.

## Expected Issues & Solutions

### Issue: No glyphs detected

**Symptoms:**
- Binary mask is all black/white
- No RED boxes in Panel 3
- "Found 0 candidate boxes"

**Solutions:**
1. Check binary mask quality (save it to file)
2. Adjust adaptive thresholding parameters
3. Lower `--min-area` (try 50 instead of 100)
4. Check image quality/contrast

### Issue: Too many false positives

**Symptoms:**
- Many RED boxes on noise
- GREEN boxes on non-glyph regions

**Solutions:**
1. Increase `--min-area` (try 200)
2. Lower `--confidence` (try 0.8)
3. Improve image preprocessing

### Issue: Wrong classifications

**Symptoms:**
- Glyphs classified as wrong letters
- High distance scores

**Solutions:**
1. Check model validation loss (should be < 2.0)
2. Lower `--confidence` threshold
3. Verify image preprocessing matches training
4. May need model retraining

### Issue: Memory errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Use CPU: `CUDA_VISIBLE_DEVICES="" python scan_and_visualize.py ...`
2. Process smaller images
3. Resize large images before processing

See `EXPECTED_BUGS.md` for complete list.

## Best Practices

### 1. Start Small
- Test with 1-2 images first
- Verify everything works before scaling up

### 2. Use Known Test Images
- Create a small test set (5-10 images)
- Manually transcribe them for comparison

### 3. Document Parameters
- Keep notes on what works for your images
- Different image types may need different settings

### 4. Monitor Performance
- Track processing time per image
- Watch for memory issues
- Log errors for debugging

### 5. Validate Results
- Compare with ground truth when available
- Review visualizations carefully
- Check transcription files

## Testing Workflow

```
1. Extract 5 test images
   ↓
2. Run quick_test.sh on first image
   ↓
3. Review visualization
   ↓
4. Tune parameters if needed
   ↓
5. Test with ground truth (if available)
   ↓
6. Process remaining test images
   ↓
7. Document optimal parameters
   ↓
8. Scale up to full dataset
```

## Performance Expectations

**Single Image (CPU):**
- Small (1000x1000px, 5 glyphs): 5-10 seconds
- Medium (2000x2000px, 15 glyphs): 15-30 seconds
- Large (4000x4000px, 30 glyphs): 30-60 seconds

**Single Image (GPU):**
- Small: 2-5 seconds
- Medium: 5-15 seconds
- Large: 15-30 seconds

**Bottlenecks:**
- Model inference (largest)
- Contour detection (scales with image size)
- Image preprocessing (usually fast)

## Files Created

- ✅ `TESTING_GUIDE.md` - Comprehensive testing guide
- ✅ `EXPECTED_BUGS.md` - Known issues and fixes
- ✅ `quick_test.sh` - Quick test script
- ✅ `NEXT_STEPS_SCANNER.md` - This file

## Quick Reference

**Test single image:**
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

**With ground truth:**
```bash
python scan_and_visualize.py stone_1.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --ground-truth ground_truth.csv
```

## Success Criteria

You're ready to scale up when:
- ✅ No Python errors
- ✅ Binary mask shows glyphs clearly
- ✅ RED boxes cover actual glyphs (not just noise)
- ✅ GREEN boxes are on real glyphs with reasonable labels
- ✅ Processing time is acceptable
- ✅ Results match expectations (or you understand why they don't)

## Getting Help

1. **Check `TESTING_GUIDE.md`** for detailed testing procedures
2. **Check `EXPECTED_BUGS.md`** for known issues and fixes
3. **Review visualization** - the 4 panels show exactly what's happening
4. **Check logs** - error messages usually point to the issue
5. **Start with simple images** - test with high-quality, clear images first

## Next Phase: Production Use

Once testing is complete:
1. Create batch processing script
2. Set up monitoring/logging
3. Optimize for your specific use case
4. Create validation pipeline
5. Document your workflow

Good luck! 🎉

