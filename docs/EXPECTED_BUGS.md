# Expected Bugs and Known Issues

## Critical Bugs (Must Fix)

### 1. Empty ROI Handling
**Location:** `step_c_classify_boxes()` line 284

**Issue:** If ROI extraction fails (e.g., box coordinates out of bounds), the script continues but may crash later.

**Fix:**
```python
# Add bounds checking
roi = image[y:y+h, x:x+w]
if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
    print(f"  Warning: Box {i} out of bounds, skipping")
    continue
```

### 2. Model Device Mismatch
**Location:** `step_c_classify_boxes()` line 290

**Issue:** If model is on GPU but reference embeddings are on CPU (or vice versa), distance calculation will fail.

**Fix:**
```python
# Ensure embeddings are on same device
roi_embedding = model(roi_tensor).squeeze(0)
for glyph_name, ref_embedding in reference_embeddings.items():
    ref_embedding = ref_embedding.to(device)  # Move to same device
    distance = euclidean_distance(...)
```

### 3. Image Format Assumptions
**Location:** `step_a_binary_mask()` line ~150

**Issue:** Assumes RGB format. May fail with grayscale or RGBA images.

**Fix:**
```python
# Handle different image formats
if len(image.shape) == 2:
    gray = image  # Already grayscale
elif len(image.shape) == 3:
    if image.shape[2] == 4:
        # RGBA - convert to RGB first
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
```

## High Priority Bugs (Should Fix)

### 4. Memory Leak in Batch Processing
**Location:** `step_c_classify_boxes()` line 297

**Issue:** Embeddings accumulate in GPU memory if processing many boxes.

**Fix:**
```python
# Clear cache periodically
if i % 50 == 0 and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 5. Aspect Ratio Filter Too Strict
**Location:** `step_b_find_contours()` line 231

**Issue:** Some glyphs may be wider than 5:1 ratio (e.g., long horizontal lines).

**Current:** `aspect_ratio < 0.2 or aspect_ratio > 5.0`

**Potential Fix:**
```python
# More lenient for horizontal glyphs
if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # More lenient
    continue
```

### 6. Reading Order Sort May Be Wrong
**Location:** `step_b_find_contours()` line 237

**Issue:** Simple row grouping may not match actual reading order (right-to-left, top-to-bottom for Safaitic).

**Current:** `boxes.sort(key=lambda b: (b[1] // 50, b[0]))`

**Potential Fix:**
```python
# More sophisticated reading order
# Group by approximate row (with tolerance)
# Then sort within row (right-to-left for Safaitic)
def reading_order_key(box):
    x, y, w, h = box
    row = y // (h * 2)  # More adaptive row grouping
    return (row, -x)  # Negative x for right-to-left
boxes.sort(key=reading_order_key)
```

## Medium Priority Issues

### 7. No Progress Bar for Classification
**Location:** `step_c_classify_boxes()` line 279

**Issue:** No feedback when processing many boxes (can take minutes).

**Fix:**
```python
from tqdm import tqdm
for i, (x, y, w, h) in enumerate(tqdm(boxes, desc="Classifying")):
    ...
```

### 8. Confidence Threshold Not Validated
**Location:** `scan_stone()` function

**Issue:** No validation that confidence_threshold is reasonable (should be 0-2 for normalized embeddings).

**Fix:**
```python
if not 0 <= confidence_threshold <= 2.0:
    raise ValueError(f"confidence_threshold must be between 0 and 2.0, got {confidence_threshold}")
```

### 9. No Error Recovery for Individual Boxes
**Location:** `step_c_classify_boxes()` line 288

**Issue:** If one box fails, entire process stops (should continue with other boxes).

**Current:** Try/except continues, but could be more robust.

**Fix:** Already handled, but could add retry logic.

### 10. Binary Mask Parameters Hardcoded
**Location:** `step_a_binary_mask()` line ~150

**Issue:** Adaptive threshold parameters (blockSize=11, C=2) may not work for all images.

**Fix:** Make parameters configurable:
```python
def step_a_binary_mask(image: np.ndarray, block_size=11, c=2) -> np.ndarray:
    ...
```

## Low Priority / Edge Cases

### 11. Very Large Images May Cause Memory Issues
**Location:** Throughout

**Issue:** Images > 5000px may cause OOM errors.

**Fix:** Add image resizing option:
```python
MAX_IMAGE_SIZE = 4000
if max(image.shape[:2]) > MAX_IMAGE_SIZE:
    scale = MAX_IMAGE_SIZE / max(image.shape[:2])
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    image = cv2.resize(image, new_size)
```

### 12. No Validation of Reference Embeddings
**Location:** `load_reference_embeddings()` line ~120

**Issue:** If reference embeddings fail to load, error may not be clear.

**Fix:** Add validation:
```python
if len(embeddings) == 0:
    raise ValueError("No reference embeddings loaded! Check cleaned_glyphs directory.")
```

### 13. Visualization May Fail with Many Boxes
**Location:** `create_visualization()` line 341

**Issue:** Too many boxes (> 100) may make visualization unreadable.

**Fix:** Limit number of boxes shown or make visualization scrollable.

### 14. No Check for Empty Classified Boxes
**Location:** `create_visualization()` line 370

**Issue:** If no boxes classified, visualization still created but may be confusing.

**Fix:** Add warning message:
```python
if len(classified_boxes) == 0:
    print("⚠️  Warning: No glyphs detected. Check:")
    print("   - Binary mask quality")
    print("   - Confidence threshold (try lowering)")
    print("   - Min contour area (try lowering)")
```

## Testing-Specific Issues

### 15. Non-Deterministic Contour Detection
**Issue:** OpenCV contour detection may vary slightly between runs.

**Impact:** Low - classifications should be consistent.

**Workaround:** Accept minor variations in box coordinates.

### 16. Model Inference Non-Deterministic (if dropout enabled)
**Issue:** If model has dropout in eval mode, results may vary.

**Fix:** Ensure `model.eval()` is called (already done).

## Recommended Fixes Priority

1. **Fix #2 (Device Mismatch)** - Will cause crashes
2. **Fix #3 (Image Format)** - Common issue with different image types
3. **Fix #1 (Empty ROI)** - Prevents crashes
4. **Fix #4 (Memory Leak)** - Important for batch processing
5. **Fix #7 (Progress Bar)** - Better UX
6. **Fix #8 (Threshold Validation)** - Prevents user errors
7. **Fix #10 (Configurable Parameters)** - Better flexibility

## Quick Fixes Script

Save as `apply_quick_fixes.py`:

```python
#!/usr/bin/env python3
"""Apply quick fixes to scan_and_visualize.py"""

# This would be a script to automatically apply the fixes above
# For now, apply manually or use search_replace in your editor
```

## Reporting Bugs

When reporting bugs, include:
1. Full error traceback
2. Image properties (size, format)
3. Command used
4. Expected vs actual behavior
5. System info (OS, Python version, PyTorch version)

