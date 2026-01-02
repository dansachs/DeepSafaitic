# Google Colab Execution Roadmap

## Step-by-Step Instructions

Copy and paste these code blocks into your Google Colab notebook in order.

---

### **Step 1: Install Dependencies**

```python
# Install required packages
!pip install torch torchvision albumentations tqdm matplotlib pillow numpy -q

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### **Step 2: Mount Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
import os
if os.path.exists('/content/drive/MyDrive'):
    print("✓ Google Drive mounted successfully")
else:
    print("✗ Google Drive mount failed")
```

---

### **Step 3: Upload Dataset**

**Option A: Upload via Colab UI**
1. Click the folder icon in the left sidebar
2. Click "Upload to session storage"
3. Upload your `cleaned_glyphs` folder (or zip file)

**Option B: Upload from Google Drive**
```python
# If you've already uploaded to Drive
import shutil
drive_path = '/content/drive/MyDrive/safaitic_project/cleaned_glyphs'
local_path = '/content/cleaned_glyphs'

if os.path.exists(drive_path):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    shutil.copytree(drive_path, local_path)
    print(f"✓ Copied dataset from Drive to {local_path}")
else:
    print("Dataset not found in Drive. Please upload manually.")
```

**Option C: Unzip if uploaded as zip**
```python
# If you uploaded cleaned_glyphs.zip
import zipfile
zip_path = '/content/cleaned_glyphs.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('/content/')
    print("✓ Extracted dataset")
```

---

### **Step 4: Upload Code Files**

Upload these files to Colab (via UI or code):
- `model.py`
- `dataset.py`
- `colab_train.py`

**Or create them directly in Colab:**
```python
# You can copy-paste the contents of each file into separate cells
# Or use the file upload feature
```

---

### **Step 5: Verify Dataset Structure**

```python
from pathlib import Path

dataset_path = Path('/content/cleaned_glyphs')
if not dataset_path.exists():
    dataset_path = Path('./cleaned_glyphs')  # Try current directory

if dataset_path.exists():
    # Count files
    png_files = list(dataset_path.rglob('*.png'))
    folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"✓ Dataset found: {len(folders)} folders, {len(png_files)} PNG files")
    
    # Show sample structure
    print("\nSample folders:")
    for folder in sorted(folders)[:5]:
        files = list(folder.glob('*.png'))
        print(f"  {folder.name}: {len(files)} files")
else:
    print("✗ Dataset not found! Please upload cleaned_glyphs folder")
```

---

### **Step 6: Run Training**

```python
# Import and run training
from colab_train import main

# This will:
# - Mount Drive (if not already mounted)
# - Load dataset
# - Train model
# - Save checkpoints to Drive
# - Generate visualizations

main()
```

---

### **Step 7: Validation Cell (Run After 2 Minutes of Training)**

**Create this as a separate cell that you can run during training:**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataset import SafaiticSiameseDataset
from model import SafaiticSiameseNet, euclidean_distance

# Load the latest checkpoint
checkpoint_path = Path('/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth')

if checkpoint_path.exists():
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SafaiticSiameseNet(embedding_dim=512).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Validation accuracy: {checkpoint['val_accuracy']:.4f}")
    
    # Load dataset and visualize sample pairs
    dataset = SafaiticSiameseDataset(root_dir='cleaned_glyphs', augment=True)
    samples = dataset.get_sample_pairs(num_pairs=5)
    
    # Create visualization
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    
    with torch.no_grad():
        for i, sample_data in enumerate(samples):
            if len(sample_data) == 4:
                anchor_np, pair_np, label, hard_negative = sample_data
            else:
                anchor_np, pair_np, label = sample_data
                hard_negative = False
            
            # Convert to tensors and get embeddings
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            anchor_tensor = torch.from_numpy(anchor_np).permute(2, 0, 1).unsqueeze(0).float()
            pair_tensor = torch.from_numpy(pair_np).permute(2, 0, 1).unsqueeze(0).float()
            
            anchor_tensor = (anchor_tensor - mean) / std
            pair_tensor = (pair_tensor - mean) / std
            
            anchor_tensor = anchor_tensor.to(device)
            pair_tensor = pair_tensor.to(device)
            
            emb1 = model(anchor_tensor)
            emb2 = model(pair_tensor)
            distance = euclidean_distance(emb1, emb2).item()
            
            # Display
            axes[i, 0].imshow(anchor_np)
            axes[i, 0].set_title("Anchor (Clean)", fontweight='bold')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(pair_np)
            pair_type = "Positive" if label == 1 else "Negative"
            if hard_negative:
                pair_type += " (Hard)"
            axes[i, 1].set_title(f"Augmented ({pair_type})", fontweight='bold')
            axes[i, 1].axis('off')
            
            axes[i, 2].text(0.5, 0.5,
                           f"Distance: {distance:.3f}\n"
                           f"Label: {int(label)}\n"
                           f"Predicted: {'Same' if distance < 1.0 else 'Different'}",
                           ha='center', va='center', fontsize=11,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[i, 2].axis('off')
    
    plt.suptitle('Model Validation: Sample Pairs with Embedding Distances\n'
                 'Check that model is learning correctly',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Plot training progress if available
    print("\n" + "="*70)
    print("Model is seeing correctly if:")
    print("  - Positive pairs have distance < 1.0")
    print("  - Negative pairs have distance > 1.0")
    print("  - Hard negatives are challenging (distance closer to threshold)")
    print("="*70)
    
else:
    print("✗ Checkpoint not found. Training may not have started yet.")
    print("  Run the training cell first and wait a few minutes.")
```

---

### **Step 8: Visualize Training Samples (Optional - During Training)**

You can visualize raw training samples (before model processing) to see the augmentation effects. This is useful for monitoring what the model is seeing.

```python
# Import the visualization function
from colab_train import visualize_training_samples
from dataset import SafaiticSiameseDataset

# Create a dataset instance (same as training)
dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)

# Visualize 10 sample pairs
visualize_training_samples(dataset, num_samples=10)
```

**Note:** This will display images **inline in the Colab notebook** so you can see them immediately. The images show:
- **Anchor (Clean)**: The original, clean glyph
- **Augmented**: The same glyph after "The Ager" augmentations (for positive pairs) or a different glyph (for negative pairs)

You can run this cell at any time during or before training to verify the augmentations are working correctly.

---

### **Step 9: Generate Confusion Matrix (After Training)**

After training is complete, generate a confusion matrix to see which glyphs are being confused with each other, especially the problematic pairs like (s1, s2) and (h, h_dot).

```python
# Install seaborn if not already installed (optional, for better visualization)
!pip install seaborn -q

# Run the confusion matrix generator
from generate_confusion_matrix import main
main()
```

**Or run directly:**
```python
!python generate_confusion_matrix.py
```

**What it does:**
- Loads your trained model
- Generates embeddings for all glyphs (using ideal.png as reference)
- Computes pairwise distances between all glyphs
- Creates a confusion matrix showing which glyphs are confused (distance < threshold)
- Highlights problematic pairs: (s1, s2), (h, h_dot), etc.
- Saves visualization to Drive: `confusion_matrix.png`

**Output includes:**
- **Distance Matrix**: Heatmap showing all pairwise distances (lower = more similar)
- **Confusion Matrix**: Binary matrix showing which pairs are confused (red = confused)
- **Summary**: Text output showing most confused pairs and status of problematic pairs
- **Highlights**: Dashed red borders around problematic pairs that are still confused

**Interpretation:**
- ✅ **NOT CONFUSED**: The model correctly distinguishes between these similar glyphs
- ❌ **CONFUSED**: The model still struggles with these pairs (may need more training or different approach)

---

### **Step 10: Monitor Training Progress**

```python
# Check training logs and visualizations
import os
from pathlib import Path

checkpoint_dir = Path('/content/drive/MyDrive/safaitic_project')

# List all checkpoints
checkpoints = list(checkpoint_dir.glob('*.pth'))
visualizations = list(checkpoint_dir.glob('visualization_*.png'))

print(f"Checkpoints: {len(checkpoints)}")
print(f"Visualizations: {len(visualizations)}")

if visualizations:
    print("\nLatest visualization:")
    latest_vis = max(visualizations, key=os.path.getctime)
    print(f"  {latest_vis.name}")
    
    # Display it
    from IPython.display import Image, display
    display(Image(str(latest_vis)))
```

---

## Quick Start (All-in-One)

If you want to run everything at once:

```python
# 1. Install
!pip install torch torchvision albumentations tqdm matplotlib pillow numpy -q

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Upload your files (model.py, dataset.py, colab_train.py, cleaned_glyphs/)
#    Use the file upload UI or copy-paste code

# 4. Run training
from colab_train import main
main()
```

---

## Troubleshooting

**Issue: "Dataset not found"**
- Make sure `cleaned_glyphs` folder is in `/content/` or current directory
- Check folder structure: `cleaned_glyphs/alif/ideal.png`, etc.

**Issue: "CUDA out of memory"**
- Reduce `batch_size` in `colab_train.py` (try 16 or 8)
- Reduce `num_workers` to 2

**Issue: "Checkpoint not saving"**
- Verify Drive is mounted: `os.path.exists('/content/drive/MyDrive')`
- Check write permissions

**Issue: "Slow training"**
- Ensure GPU is enabled: Runtime → Change runtime type → GPU
- Check `pin_memory=True` is set in DataLoader

---

## Expected Output

After running training, you should see:
- Training progress with loss values
- Validation accuracy improving
- Checkpoints saved to Drive
- **Visualizations displayed inline every 5 epochs** (images appear directly in the notebook)
- Training curves plot at the end (also displayed inline)

**Visualization Features:**
- Images are automatically displayed inline in Colab using `plt.show()`
- Visualizations are saved to Drive AND shown in the notebook
- You can see sample pairs with embedding distances during training
- Use `visualize_training_samples()` to preview raw augmentations anytime

