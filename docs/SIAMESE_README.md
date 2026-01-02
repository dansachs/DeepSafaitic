# Safaitic Siamese Network - Training Pipeline

## Overview
This pipeline trains a Siamese network to learn embeddings for Safaitic glyphs by comparing "Ideal/Square" versions with aggressively augmented versions.

## Files Created

### 1. `model.py`
- **SafaiticSiameseNet**: Siamese network using ResNet18 backbone
  - Pretrained ResNet18 (ImageNet weights)
  - Removes final classification layer
  - Adds custom embedding layer (512 dimensions)
  - L2 normalization for unit sphere embeddings
  
- **ContrastiveLoss**: Loss function for Siamese training
  - Margin: 2.0 (configurable)
  - Minimizes distance for positive pairs
  - Maximizes distance for negative pairs (with margin)

- **euclidean_distance**: Function to compute distance between embeddings

### 2. `dataset.py`
- **SafaiticSiameseDataset**: Dataset class for pair generation
  - **Positive Pairs (Label=1)**: Same glyph (ideal + augmented)
  - **Negative Pairs (Label=0)**: Different glyphs (ideal + augmented from different letter)
  
- **Aggressive Augmentation ("The Ager")** using Albumentations:
  - 360° rotation (SafeRotate)
  - Elastic distortion (warping)
  - Random scaling
  - Gaussian blur (weathering)
  - Brightness/contrast variations
  - GaussNoise (stone texture)
  - CoarseDropout (occlusions/damage)
  - ISONoise (additional texture)

### 3. `colab_train.py`
- Google Colab optimized training script
- Features:
  - Automatic Google Drive mounting
  - GPU support (CUDA)
  - Training/validation loop
  - Checkpoint saving (best model to Google Drive)
  - Visualization after each epoch (5 sample pairs)
  - Training curves plotting

## Usage

### In Google Colab:

1. **Upload files to Colab:**
   ```python
   # Upload model.py, dataset.py, colab_train.py
   # Upload cleaned_glyphs/ directory
   ```

2. **Install dependencies:**
   ```python
   !pip install torch torchvision albumentations tqdm matplotlib pillow
   ```

3. **Run training:**
   ```python
   !python colab_train.py
   ```

### Locally:

1. **Install dependencies:**
   ```bash
   pip install torch torchvision albumentations tqdm matplotlib pillow
   ```

2. **Run training:**
   ```bash
   python colab_train.py
   ```
   (Will skip Google Drive mounting automatically)

## Hyperparameters

- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 0.001 (with ReduceLROnPlateau scheduler)
- **Embedding Dimension**: 512
- **Contrastive Loss Margin**: 2.0
- **Train/Val Split**: 80/20

## Outputs

- **Checkpoint**: `safaitic_matcher.pth` (saved to Google Drive in Colab)
- **Visualizations**: `visualization_epoch_X.png` (every 5 epochs)
- **Training Curves**: `training_curves.png`

## Model Architecture

```
Input (3, 128, 128)
    ↓
ResNet18 Backbone (pretrained)
    ↓
Global Average Pooling
    ↓
Linear(512 → 512)
    ↓
ReLU + Dropout(0.3)
    ↓
Linear(512 → 512)
    ↓
L2 Normalization
    ↓
Embedding (512-dim)
```

## Training Strategy

1. **Positive pairs**: Learn that same glyph (even with heavy augmentation) should have similar embeddings
2. **Negative pairs**: Learn that different glyphs should have distant embeddings
3. **Augmentation**: "The Ager" ensures model is robust to real-world variations (rotation, distortion, noise, wear)

## Expected Results

- Model learns to recognize glyphs despite heavy augmentation
- Embedding distances: 
  - Positive pairs: < 1.0 (same glyph)
  - Negative pairs: > 1.0 (different glyphs)
- Validation accuracy should improve over epochs

## Next Steps

After training:
1. Use model for glyph matching/retrieval
2. Fine-tune on real Safaitic inscription images
3. Build search/retrieval system using embeddings
4. Evaluate on test set of real-world images

