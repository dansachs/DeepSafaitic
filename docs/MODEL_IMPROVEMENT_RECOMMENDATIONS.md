# Model Improvement Recommendations

## Current Situation

**Problem:** Model detects boxes correctly but classifies glyphs completely incorrectly.

**Evidence:**
- Ground truth: "l mnʿt bn ʿtq" 
- Model output: "alif r r n l d_dot" (completely wrong)
- Model is finding boxes but not recognizing what's in them

## Root Cause Analysis

### Why Classification is Failing

1. **Domain Gap:** Model was trained on clean `ideal.png` images (128x128, white background, perfect glyphs)
   - Real stone images are: degraded, varying lighting, complex backgrounds, different scales
   - The model never saw real stone images during training

2. **Preprocessing Mismatch:** 
   - Training: Clean ideal.png → resize to 128x128 → normalize
   - Inference: Degraded stone ROI → resize to 128x128 → normalize
   - The degraded ROIs look nothing like the training data

3. **Scale Issues:**
   - Training glyphs are normalized and centered
   - Stone glyphs vary in size, rotation, and position within the box

## Recommended Solutions

### Option 1: Two-Stage Approach (RECOMMENDED)

**Stage 1: Detection (Current - Working)**
- Use contour detection to find glyph locations ✓ (This is working!)

**Stage 2: Classification (Needs Improvement)**
- Instead of using the Siamese model directly, use a different approach:

#### 2A: Fine-tune on Real Stone Data
- Extract glyph ROIs from stone images
- Manually label them (or use your database transliterations)
- Fine-tune the model on real stone glyphs
- This bridges the domain gap

#### 2B: Use a Classification Model Instead
- Train a standard classifier (not Siamese) on:
  - Input: Glyph ROI from stone image
  - Output: Glyph class (alif, b, t, etc.)
- This is simpler and more direct than similarity matching

#### 2C: Hybrid Approach
- Keep Siamese model for similarity
- But train it on pairs of: (real stone glyph, ideal reference)
- This teaches it to match degraded glyphs to clean references

### Option 2: Improve Current Siamese Model

**A. Data Augmentation During Training**
- The current augmentation is good, but may not be aggressive enough
- Add more stone-like augmentations:
  - Background textures
  - Lighting variations
  - More aggressive noise

**B. Train on Real Stone Glyphs**
- Extract glyphs from stone images
- Create pairs: (stone glyph, ideal reference)
- Retrain the model on these pairs

**C. Adjust Confidence Threshold**
- Current threshold (1.0) may be too strict
- Try lowering to 0.8 or even 0.6
- But this might increase false positives

### Option 3: Start from Scratch (If Needed)

If the current approach fundamentally doesn't work:

**New Architecture:**
1. **Object Detection Model** (YOLO, Faster R-CNN)
   - Detects and classifies glyphs in one step
   - Trained end-to-end on stone images

2. **Segmentation Model** (U-Net, Mask R-CNN)
   - Segments glyph regions
   - Then classifies each region

3. **Sequence Model** (CRNN, Transformer)
   - Treats inscription as a sequence
   - Uses context to improve recognition

## Immediate Next Steps

### Step 1: Analyze Current Failures

Run the comparison script:
```bash
python3 compare_results.py
```

This will show:
- What the model predicted vs. ground truth
- Where it's failing
- Patterns in errors

### Step 2: Extract Training Data from Stones

Create a script to:
1. Extract glyph ROIs from stone images (using current detection)
2. Match them to ground truth transliterations
3. Create a labeled dataset of real stone glyphs

### Step 3: Fine-tune Model

Options:
- **Quick fix:** Fine-tune on real stone glyphs (few hundred examples)
- **Better fix:** Retrain with mixed data (ideal + stone glyphs)
- **Best fix:** Train a new classifier specifically for stone glyphs

## Recommended Path Forward

**Short-term (This Week):**
1. ✅ Extract ground truth from database (DONE)
2. Analyze failure patterns
3. Extract labeled glyph ROIs from stone images
4. Fine-tune model on 50-100 real stone glyph examples

**Medium-term (Next Week):**
1. Collect more labeled stone glyph data
2. Retrain model with mixed dataset (ideal + stone)
3. Test on validation set

**Long-term (If Needed):**
1. Consider switching to object detection architecture
2. Or use sequence-to-sequence model for full inscriptions

## Code to Get Started

I'll create scripts for:
1. Extracting labeled glyph ROIs from stone images
2. Fine-tuning the model on real data
3. Testing improved model

The key insight: **The detection is working, but the classification needs real stone data to learn from.**

