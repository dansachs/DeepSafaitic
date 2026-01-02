# Ruler Filtering Feature

## Problem

Stone images often contain rulers or measuring devices for scale reference. These long, thin objects are being detected as glyphs by the contour detection, causing false positives.

## Solution

Added automatic ruler detection and filtering to `step_b_find_contours()`.

## How It Works

The `is_ruler_like()` function detects rulers based on multiple criteria:

### 1. Extreme Aspect Ratio
- **Rulers:** Typically 20:1 or more (very long and thin)
- **Glyphs:** Usually < 10:1
- **Filter:** Objects with aspect ratio > 15:1 are filtered

### 2. Length
- **Rulers:** Often 300px+ in length
- **Glyphs:** Typically < 200px
- **Filter:** Very long objects (> 300px) with high aspect ratio (> 8:1) are filtered

### 3. Position
- **Rulers:** Often placed at image edges
- **Glyphs:** Usually in center of image
- **Filter:** Long, thin objects near edges (within 5% of border) are filtered

### 4. Straightness
- **Rulers:** Straight lines
- **Glyphs:** Curved shapes
- **Filter:** Very straight objects (> 250px, aspect ratio > 8:1) are filtered

## Usage

### Default (Ruler Filtering Enabled)

```bash
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs
```

Rulers are automatically filtered out.

### Disable Ruler Filtering

If you want to see all detections (including rulers):

```bash
python scan_and_visualize.py stone.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --no-filter-rulers
```

### In Python

```python
from scan_and_visualize import scan_stone

# With ruler filtering (default)
results = scan_stone(
    image_path="stone.jpg",
    model_checkpoint="models/safaitic_matcher.pth",
    glyphs_dir="cleaned_glyphs",
    filter_rulers=True  # Default
)

# Without ruler filtering
results = scan_stone(
    image_path="stone.jpg",
    model_checkpoint="models/safaitic_matcher.pth",
    glyphs_dir="cleaned_glyphs",
    filter_rulers=False
)
```

## Output

When rulers are filtered, you'll see:

```
STEP B: Finding All Candidates
============================================================
  Filtered out 2 ruler-like objects
✓ Found 8 candidate boxes
```

## Tuning

If the filter is too aggressive (filtering real glyphs) or too lenient (missing rulers), you can adjust thresholds in `is_ruler_like()`:

- **Aspect ratio threshold:** Change `15.0` to higher/lower
- **Length threshold:** Change `300` to higher/lower  
- **Edge threshold:** Change `0.05` (5%) to higher/lower
- **Straightness threshold:** Change `0.1` (10% deviation) to higher/lower

## Testing

Test with images that have rulers:

```bash
# Test with ruler filtering
python scan_and_visualize.py stone_with_ruler.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs

# Compare without filtering
python scan_and_visualize.py stone_with_ruler.jpg \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --no-filter-rulers \
    --output test_results_no_filter
```

Check the visualizations to see if rulers are properly filtered.

