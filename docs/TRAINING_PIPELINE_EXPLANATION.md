# Safaitic Glyph Siamese Network Training Pipeline - Technical Explanation

## Overview

This notebook trains a **Siamese Neural Network** to learn embeddings for Safaitic alphabet glyphs. The goal is to create a model that can determine whether two glyph images represent the same letter or different letters, even when one image is heavily degraded (simulating real-world stone carving conditions).

**Core Concept:** Siamese networks use a shared backbone to generate embeddings for pairs of images. If two images are of the same glyph, their embeddings should be close together in the embedding space. If they're different glyphs, their embeddings should be far apart.

---

## 1. Data Structure and Organization

### Input Data Format
- **Location:** `/content/drive/MyDrive/safaitic_project/cleaned_glyphs/`
- **Structure:** Each glyph has its own subfolder (e.g., `alif/`, `h/`, `s1/`, etc.)
- **Files per glyph:**
  - `ideal.png` - Clean, normalized 128x128 PNG (used as anchor/reference)
  - `square.png` - Alternative variant of the same glyph (used for positive pairs)

### Data Collection
The `SafaiticSiameseDataset` class:
1. Scans the root directory for all subfolders
2. Collects paths to all `ideal.png` files (these become anchors)
3. Collects paths to all `square.png` files (these become pair candidates)
4. Creates mappings:
   - `ideal_paths`: List of all ideal.png paths (one per glyph)
   - `all_paths`: List of all images (ideal + square) for negative pair sampling
   - `label_to_ideal_indices`: Maps glyph name to indices in ideal_paths
   - `label_to_all_indices`: Maps glyph name to indices in all_paths

**Key Point:** The dataset contains ~28 unique glyphs, but generates 10,000 pairs per epoch through random sampling and augmentation.

---

## 2. Image Preprocessing Pipeline

### Base Transform (Applied to All Images)
```python
A.Compose([
    A.Resize(128, 128),  # Ensures all images are exactly 128x128
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ToTensorV2()  # Converts numpy array to PyTorch tensor (C, H, W format)
])
```

**Why ImageNet normalization?** The model uses a pretrained ResNet18 backbone, which was trained on ImageNet. Using the same normalization ensures the pretrained weights work correctly.

### Anchor Images (Always Clean)
- **Source:** Always `ideal.png` files
- **Processing:** Only base transform (resize + normalize + tensor conversion)
- **No augmentation applied** - anchors remain clean and consistent

### Pair Images (Augmented)
- **Source:** Can be either `ideal.png` or `square.png` from the same or different glyph
- **Processing:** Base transform + aggressive augmentation pipeline ("The Ager")

---

## 3. Aggressive Augmentation Pipeline ("The Ager")

The augmentation pipeline simulates real-world degradation of stone carvings. It's applied **only to pair images, never to anchors**.

### Augmentation Steps (in order):

1. **SafeRotate (360° rotation)**
   - Rotates image up to 360 degrees
   - `border_mode=1` (cv2.BORDER_CONSTANT) with `value=255` (white)
   - **Purpose:** Simulates glyphs carved at different angles
   - **Critical:** White border prevents black wedges during rotation

2. **ElasticTransform (Elastic Deformation)**
   - `alpha=120, sigma=6` (120 * 0.05)
   - `alpha_affine=3.6` (120 * 0.03)
   - `border_mode=1, value=255` (white fill)
   - **Purpose:** Simulates warping, cracking, and distortion of stone over time
   - **Effect:** Creates curved distortions in the glyph lines

3. **RandomScale (10% scaling)**
   - `scale_limit=0.1` (92-108% of original size)
   - **Purpose:** Simulates perspective differences and carving depth variations

4. **GaussianBlur (Weathering)**
   - `blur_limit=(1, 3)` pixels
   - **Purpose:** Simulates erosion and weathering of stone surfaces

5. **RandomBrightnessContrast**
   - `brightness_limit=0.3, contrast_limit=0.3`
   - **Purpose:** Simulates lighting variations and shadow effects

6. **GaussNoise (Stone Texture)**
   - `var_limit=(10.0, 50.0)`
   - **Purpose:** Adds random noise to simulate stone grain and texture

7. **CoarseDropout (Damage/Patches)**
   - `max_holes=4, max_height=15, max_width=15`
   - `fill_value=255` (white patches)
   - **Purpose:** Simulates chips, cracks, and missing stone patches

8. **ISONoise (Camera/Scanning Artifacts)**
   - `color_shift=(0.01, 0.05), intensity=(0.1, 0.5)`
   - **Purpose:** Simulates scanning artifacts and camera noise

9. **Resize (128, 128) - CRITICAL**
   - Resizes back to 128x128 after all augmentations
   - **Why critical:** Some augmentations (especially RandomScale) can change image dimensions
   - **Without this:** PyTorch DataLoader would fail when trying to batch different-sized images

10. **Normalize + ToTensor**
    - Final normalization and tensor conversion

**Key Design Decision:** The anchor is always clean (`ideal.png` with no augmentation), while the pair is always augmented. This simulates the real-world scenario: you have a clean reference glyph and need to match it against a degraded/weathered carving.

---

## 4. Dataset Class: Pair Generation Logic

### Dataset Size
- **`__len__()` returns:** `10000` (not the number of glyphs)
- **Why:** Generates 10,000 random pairs per epoch for robust training
- **With batch_size=32:** ~312 batches per epoch

### `__getitem__(idx)` - Pair Generation

**Step 1: Random Anchor Selection**
```python
anchor_idx = random.randint(0, len(self.ideal_paths) - 1)
anchor_path = self.ideal_paths[anchor_idx]
anchor_label = self.ideal_labels[anchor_idx]
```
- Randomly selects one of the 28 glyphs as the anchor
- Always uses `ideal.png` (clean version)

**Step 2: Positive/Negative Decision (50/50 Balance)**
```python
is_positive = random.random() < 0.5
```

**Step 3A: Positive Pair (Same Glyph)**
- `label = 1`
- Selects another image from the **same glyph** (prefers `square.png` if available, otherwise `ideal.png`)
- Both anchor and pair are the same glyph, but pair is augmented

**Step 3B: Negative Pair (Different Glyph)**
- `label = 0`
- Randomly selects a **different glyph** from the dataset
- Uses any available image from that different glyph (ideal or square)
- Pair is augmented

**Step 4: Apply Transforms**
- Anchor: Base transform only (clean)
- Pair: Base transform + aggressive augmentation (degraded)

**Step 5: Return**
```python
return anchor_tensor, pair_tensor, torch.tensor(label, dtype=torch.float32)
```

**Key Points:**
- Each call to `__getitem__` generates a **completely random pair**
- The same `idx` can return different pairs on different epochs
- 50/50 balance is maintained through random sampling, not deterministic indexing
- Augmentation is applied **on-the-fly** during data loading (different each time)

---

## 5. Model Architecture

### SafaiticSiameseNet

**Backbone: ResNet18 (Pretrained)**
- Uses `torchvision.models.resnet18(pretrained=True)`
- Removes the final classification layer (fc)
- Extracts features from the avgpool layer (512-dimensional feature vector)

**Embedding Head:**
```python
nn.Sequential(
    nn.Flatten(),                    # Flatten 512-dim feature vector
    nn.Linear(512, 512),             # First linear layer
    nn.ReLU(inplace=True),           # Activation
    nn.Dropout(0.3),                 # Regularization
    nn.Linear(512, 512),             # Second linear layer
    L2Norm(dim=1)                    # L2 normalization (unit sphere)
)
```

**Output:** 512-dimensional embedding vector, normalized to unit length

**Why L2 Normalization?**
- Projects embeddings onto a unit sphere
- Makes distance calculations more stable
- Euclidean distance between normalized vectors ranges from 0 to 2

### Forward Pass
1. Input: `(batch_size, 3, 128, 128)` tensor
2. ResNet18 backbone: `(batch_size, 512, 1, 1)`
3. Embedding head: `(batch_size, 512)` normalized vector

---

## 6. Loss Function: Contrastive Loss

### Formula
```
For positive pairs (label = 1):
  loss = distance²

For negative pairs (label = 0):
  loss = max(0, margin - distance)²
  where margin = 2.0
```

### Behavior
- **Positive pairs:** Minimizes distance (pulls embeddings together)
- **Negative pairs:** Maximizes distance up to the margin (pushes embeddings apart)
- **Margin (2.0):** Once negative pairs are 2.0 units apart, no further penalty

### Distance Metric
- **Euclidean distance** between normalized embeddings
- Range: 0 to 2 (since embeddings are L2-normalized)

---

## 7. Training Loop Details

### Epoch Structure

**For each epoch (1 to num_epochs):**

1. **Training Phase**
   - Model in `train()` mode
   - Iterates through all batches in `train_loader`
   - For each batch:
     - Forward pass: Get embeddings for anchor and pair
     - Compute contrastive loss
     - Backward pass: Compute gradients
     - **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)` prevents exploding gradients
     - Update weights via Adam optimizer
   - Returns average training loss

2. **Validation Phase** (every 2 epochs, or first 5 epochs)
   - Model in `eval()` mode
   - No gradient computation (`torch.no_grad()`)
   - Computes loss and accuracy
   - **Accuracy calculation:**
     - Distance < 1.0 → Predicted "Same" (positive)
     - Distance ≥ 1.0 → Predicted "Different" (negative)
     - Compares predictions to true labels

3. **Learning Rate Scheduling**
   - `ReduceLROnPlateau` scheduler
   - Monitors validation loss
   - Reduces LR by 50% if no improvement for 5 epochs
   - **Manual logging:** Prints when LR changes

4. **Early Stopping Check**
   - If validation loss improves: Reset patience counter, save best checkpoint
   - If validation loss doesn't improve: Increment patience counter
   - If patience ≥ 15: Stop training early

5. **Checkpointing**
   - **Best checkpoint:** Saved whenever validation loss improves
   - **Periodic checkpoints:** Saved every 10 epochs
   - **Checkpoint contents:**
     - Epoch number
     - Model state dict
     - Optimizer state dict
     - Validation loss and accuracy

6. **GPU Memory Monitoring** (every 5 epochs)
   - Prints allocated and max allocated memory
   - Clears GPU cache to prevent memory leaks

7. **Visualization** (every 10 epochs)
   - Generates sample pairs with embedding distances
   - Saves to Drive: `visualization_epoch_N.png`

### Training Statistics
- **Train Loss:** Average contrastive loss over all training batches
- **Val Loss:** Average contrastive loss over validation batches
- **Val Accuracy:** Percentage of pairs correctly classified (distance < 1.0 for positive, ≥ 1.0 for negative)

---

## 8. Data Loading and Batching

### DataLoader Configuration
```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,                    # Shuffles pairs each epoch
    num_workers=4,                  # Parallel data loading
    pin_memory=True,                # Faster GPU transfer
    persistent_workers=True         # Keeps workers alive between epochs
)
```

### Batch Structure
- **Batch size:** 32 pairs
- **Batch format:** Tuple of 3 tensors
  - `anchor`: `(32, 3, 128, 128)` - Clean anchor images
  - `pair`: `(32, 3, 128, 128)` - Augmented pair images
  - `label`: `(32,)` - Binary labels (1=positive, 0=negative)

### Train/Val Split
- **80/20 split:** 8,000 training pairs, 2,000 validation pairs per epoch
- **Validation dataset:** Uses `augment=False` for proper evaluation
- **Why:** Validation should test on clean images to measure true model performance

---

## 9. Checkpoint Resuming

### How It Works
```python
resume_from_checkpoint = False  # Set to True to resume

if resume_from_checkpoint and checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['val_loss']
```

**Resuming:**
- Loads model weights
- Loads optimizer state (including learning rate, momentum)
- Continues from `start_epoch` to `num_epochs`
- Preserves `best_val_loss` for early stopping

---

## 10. Confusion Matrix Generation

### Process

1. **Load Trained Model**
   - Loads best checkpoint
   - Sets model to `eval()` mode

2. **Generate Embeddings for All Glyphs**
   - For each glyph folder:
     - Load `ideal.png` (clean version)
     - Apply base transform (no augmentation)
     - Get embedding from model
   - Result: Dictionary mapping glyph name → 512-dim embedding

3. **Compute Distance Matrix**
   - For each pair of glyphs (i, j):
     - Compute Euclidean distance between their embeddings
   - Result: N×N matrix where N = number of glyphs (~28)

4. **Generate Confusion Matrix**
   - **Threshold:** 1.0 (same as training)
   - If distance < 1.0: Mark as "confused" (model thinks they're the same)
   - If distance ≥ 1.0: Not confused (model correctly distinguishes them)

5. **Analysis**
   - Finds top 10 most confused pairs (lowest distances)
   - Prints summary statistics
   - Visualizes distance matrix and confusion matrix

**Interpretation:**
- **Distance < 1.0:** Model thinks these glyphs are the same (confusion)
- **Distance ≥ 1.0:** Model correctly distinguishes them
- **Lower distance = More similar** in the embedding space

---

## 11. Key Design Decisions and Rationale

### Why Siamese Network?
- **Problem:** Need to match degraded glyphs to clean references
- **Solution:** Learn embeddings where same glyphs are close, different glyphs are far
- **Advantage:** Can compare any two glyphs without retraining

### Why Always Clean Anchors?
- **Real-world scenario:** You have a clean reference (ideal.png)
- **Task:** Match degraded carvings to clean references
- **Training:** Anchor = clean reference, Pair = degraded version

### Why Aggressive Augmentation?
- **Goal:** Model must work on heavily degraded real-world carvings
- **Augmentation simulates:**
  - Weathering and erosion
  - Cracking and warping
  - Lighting variations
  - Scanning artifacts
- **Result:** Model learns robust features that work despite degradation

### Why 10,000 Pairs Per Epoch?
- **Small dataset:** Only 28 unique glyphs
- **Solution:** Generate many random pairs through augmentation
- **Benefit:** Each epoch sees different combinations, preventing overfitting
- **Result:** ~312 batches per epoch (vs. 1 batch with 28 samples)

### Why Contrastive Loss?
- **Explicitly optimizes distance:** Minimizes for positive, maximizes for negative
- **Margin-based:** Prevents negative pairs from being pushed infinitely far
- **Works well with normalized embeddings:** Distance range is bounded (0-2)

### Why L2 Normalization?
- **Stability:** Normalized embeddings are more stable during training
- **Distance interpretation:** Euclidean distance has clear meaning (0-2 range)
- **Prevents embedding collapse:** Forces model to use full embedding space

---

## 12. Training Hyperparameters

- **Batch size:** 32
- **Learning rate:** 0.001 (Adam optimizer)
- **Learning rate scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Embedding dimension:** 512
- **Contrastive loss margin:** 2.0
- **Distance threshold:** 1.0 (for accuracy calculation)
- **Early stopping patience:** 15 epochs
- **Gradient clipping:** max_norm=1.0

---

## 13. Output Files

All saved to `/content/drive/MyDrive/safaitic_project/`:

1. **`safaitic_matcher.pth`** - Best model checkpoint (updated when val loss improves)
2. **`safaitic_matcher_epoch_N.pth`** - Periodic checkpoints (every 10 epochs)
3. **`visualization_epoch_N.png`** - Sample pair visualizations (every 10 epochs)
4. **`training_curves.png`** - Loss and accuracy plots (end of training)
5. **`confusion_matrix.png`** - Confusion matrix visualization (generated separately)

---

## 14. Current Limitations and Potential Issues

### Known Limitations:
1. **Small dataset:** Only 28 unique glyphs (though 10,000 pairs per epoch helps)
2. **Fixed threshold:** Distance threshold of 1.0 is hardcoded (could be learned)
3. **Binary classification:** Only distinguishes same/different (could extend to multi-class)
4. **Augmentation intensity:** Very aggressive (may be too harsh for some use cases)

### Potential Issues:
1. **Overfitting:** Model might memorize specific glyphs rather than learning general features
2. **Embedding collapse:** All embeddings might cluster together (mitigated by L2 normalization)
3. **Validation accuracy:** Low accuracy (16-33%) suggests model may not be learning effectively
4. **Distance threshold:** 1.0 might not be optimal (could tune based on validation)

---

## 15. Data Flow Summary

```
Raw Images (ideal.png, square.png)
    ↓
Dataset.__getitem__()
    ↓
Random Pair Selection (positive or negative)
    ↓
Image Loading (PIL)
    ↓
Anchor: Base Transform Only
Pair: Base Transform + Aggressive Augmentation
    ↓
PyTorch Tensors (3, 128, 128)
    ↓
DataLoader Batching (32 pairs)
    ↓
Model Forward Pass (ResNet18 + Embedding Head)
    ↓
Embeddings (512-dim, normalized)
    ↓
Euclidean Distance Calculation
    ↓
Contrastive Loss
    ↓
Backward Pass + Gradient Clipping
    ↓
Optimizer Step (Adam)
    ↓
Repeat for all batches
    ↓
Validation (every 2 epochs)
    ↓
Checkpointing (best + periodic)
    ↓
Early Stopping Check
```

---

## 16. Next Steps for Improvement

### Potential Enhancements:
1. **Triplet Loss:** Use triplets (anchor, positive, negative) instead of pairs
2. **Hard Negative Mining:** Focus training on difficult negative pairs
3. **Multi-Scale Features:** Combine features from multiple ResNet layers
4. **Attention Mechanisms:** Add attention to focus on important glyph regions
5. **Data Augmentation Tuning:** Adjust augmentation intensity based on validation performance
6. **Distance Metric Learning:** Learn the distance function instead of using Euclidean
7. **Ensemble Methods:** Combine multiple models for better accuracy
8. **Transfer Learning:** Fine-tune ResNet18 specifically for glyph recognition
9. **Active Learning:** Identify which glyph pairs need more training data
10. **Threshold Tuning:** Learn optimal distance threshold instead of fixed 1.0

---

This pipeline is designed to be robust, with extensive checkpointing, monitoring, and early stopping to handle long training runs. The aggressive augmentation ensures the model can handle real-world degradation, while the Siamese architecture allows flexible comparison of any glyph pairs.

