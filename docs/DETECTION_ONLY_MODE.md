# Detection Only Mode - Focus on Perfecting Glyph Detection

## New Approach

Instead of trying to classify glyphs immediately, we're breaking the problem into stages:

1. **Stage 1: Detection** (Current Focus) ✅
   - Just detect WHERE glyphs are
   - Perfect the detection first
   - No classification yet

2. **Stage 2: Text Direction** (Next)
   - Determine reading order
   - Know where to start and where to go
   - Handle right-to-left, top-to-bottom, etc.

3. **Stage 3: Character Recognition** (Later)
   - Once detection is perfect, add classification
   - Use the trained model or train a new one

## New Script: `scan_detection_only.py`

This script focuses ONLY on detection - no model loading, no classification.

### Usage

```bash
python scan_detection_only.py stone_images/stone_16820.jpg
```

### Options

```bash
python scan_detection_only.py stone.jpg \
    --output detection_results \
    --min-area 30 \
    --dilate 0 \
    --no-filter-rulers  # if you want to see rulers
```

### What It Does

1. **Step A:** Creates binary mask (same as before)
2. **Step B:** Detects glyph locations (contours)
3. **Visualization:** Shows 4 panels:
   - Original image
   - Binary mask
   - Detected glyphs (GREEN boxes with numbers)
   - Detection statistics

### Output

- `detection_visualization.png` - 4-panel visualization
- `detection_results.txt` - List of all detected boxes with coordinates

## Advantages

1. **Faster** - No model loading, no classification
2. **Simpler** - Focus on one problem at a time
3. **Easier to debug** - Can see exactly what's being detected
4. **Iterative improvement** - Perfect detection before moving on

## Next Steps

Once detection is perfect:

1. **Add reading order detection**
   - Detect text direction (right-to-left for Safaitic)
   - Order boxes correctly
   - Handle multi-line inscriptions

2. **Then add classification**
   - Use your trained model
   - Or train a new one on real stone glyphs
   - Classify each detected box

## Comparison

**Old approach (scan_and_visualize.py):**
- Detection + Classification in one step
- Model must be loaded
- Slower, more complex

**New approach (scan_detection_only.py):**
- Detection only
- No model needed
- Faster, simpler, easier to perfect

## Quick Test

```bash
# Test detection only
python scan_detection_only.py stone_images/stone_16820.jpg

# View results
open detection_results/detection_visualization.png
cat detection_results/detection_results.txt
```

Focus on getting the GREEN boxes to perfectly match all glyph locations before worrying about what they are!

