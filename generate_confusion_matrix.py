"""
Generate Confusion Matrix for Safaitic Glyph Recognition.
Tests the trained model to identify which glyphs are being confused with each other,
with special focus on hard negative pairs like (s1, s2) and (h, h_dot).
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed. Using matplotlib for heatmap.")
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Google Colab imports
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

from model import SafaiticSiameseNet, euclidean_distance


def load_model(checkpoint_path, device, embedding_dim=512):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device (cuda or cpu)
        embedding_dim: Dimension of embeddings (default: 512)
    
    Returns:
        Loaded model
    """
    model = SafaiticSiameseNet(embedding_dim=embedding_dim).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Validation Loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    print(f"  Validation Accuracy: {checkpoint.get('val_accuracy', 'unknown'):.4f}")
    
    return model


def load_glyph_image(image_path, transform):
    """
    Load and transform a single glyph image.
    
    Args:
        image_path: Path to image file
        transform: Albumentations transform
    
    Returns:
        Transformed image tensor
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    transformed = transform(image=img_array)
    return transformed['image']


def get_glyph_embeddings(model, root_dir, device, use_ideal=True):
    """
    Get embeddings for all glyphs.
    
    Args:
        model: Trained model
        root_dir: Root directory containing glyph subfolders
        use_ideal: If True, use ideal.png; if False, use square.png
    
    Returns:
        Dictionary mapping glyph name to embedding tensor
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise ValueError(f"Root directory not found: {root_dir}")
    
    # Transform for images (no augmentation, just normalization)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    embeddings = {}
    glyph_names = []
    
    # Get all subdirectories (one per letter)
    for letter_dir in sorted(root_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        
        letter_name = letter_dir.name
        
        # Use ideal.png as reference (clean version)
        if use_ideal:
            image_path = letter_dir / "ideal.png"
        else:
            image_path = letter_dir / "square.png"
        
        if not image_path.exists():
            print(f"  Warning: {image_path} not found, skipping {letter_name}")
            continue
        
        # Load and transform image
        img_tensor = load_glyph_image(image_path, transform)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get embedding
        with torch.no_grad():
            embedding = model(img_tensor)
        
        embeddings[letter_name] = embedding.cpu()
        glyph_names.append(letter_name)
    
    print(f"✓ Generated embeddings for {len(embeddings)} glyphs")
    return embeddings, sorted(glyph_names)


def compute_distance_matrix(embeddings, glyph_names):
    """
    Compute pairwise distance matrix between all glyphs.
    
    Args:
        embeddings: Dictionary mapping glyph name to embedding tensor
        glyph_names: Sorted list of glyph names
    
    Returns:
        Distance matrix (n_glyphs x n_glyphs)
    """
    n = len(glyph_names)
    distance_matrix = np.zeros((n, n))
    
    for i, name1 in enumerate(glyph_names):
        for j, name2 in enumerate(glyph_names):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                emb1 = embeddings[name1]
                emb2 = embeddings[name2]
                # Compute Euclidean distance
                distance = euclidean_distance(emb1, emb2).item()
                distance_matrix[i, j] = distance
    
    return distance_matrix


def generate_confusion_matrix(distance_matrix, glyph_names, threshold=1.0):
    """
    Generate confusion matrix from distance matrix.
    Two glyphs are considered "confused" if their distance is below threshold.
    
    Args:
        distance_matrix: Pairwise distance matrix
        glyph_names: List of glyph names
        threshold: Distance threshold for "same" classification
    
    Returns:
        Confusion matrix (binary: 1 = confused, 0 = not confused)
    """
    n = len(glyph_names)
    confusion_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:  # Don't mark diagonal (same glyph)
                # If distance is below threshold, they're confused
                if distance_matrix[i, j] < threshold:
                    confusion_matrix[i, j] = 1
    
    return confusion_matrix


def highlight_problematic_pairs(confusion_matrix, glyph_names, problematic_pairs):
    """
    Create a mask to highlight problematic pairs in the confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        glyph_names: List of glyph names
        problematic_pairs: List of tuples of problematic pairs, e.g., [('s1', 's2'), ('h', 'h_dot')]
    
    Returns:
        Highlight mask (1 where problematic pairs are confused, 0 otherwise)
    """
    n = len(glyph_names)
    highlight_mask = np.zeros((n, n))
    
    name_to_idx = {name: i for i, name in enumerate(glyph_names)}
    
    for pair in problematic_pairs:
        name1, name2 = pair
        if name1 in name_to_idx and name2 in name_to_idx:
            idx1 = name_to_idx[name1]
            idx2 = name_to_idx[name2]
            # Mark both directions
            if confusion_matrix[idx1, idx2] > 0:
                highlight_mask[idx1, idx2] = 1
            if confusion_matrix[idx2, idx1] > 0:
                highlight_mask[idx2, idx1] = 1
    
    return highlight_mask


def plot_confusion_matrix(distance_matrix, confusion_matrix, glyph_names, 
                         highlight_mask=None, threshold=1.0, save_path=None):
    """
    Plot confusion matrix with distance values and highlights.
    
    Args:
        distance_matrix: Pairwise distance matrix
        confusion_matrix: Binary confusion matrix
        glyph_names: List of glyph names
        highlight_mask: Mask for highlighting problematic pairs
        threshold: Distance threshold used
        save_path: Path to save the plot
    """
    n = len(glyph_names)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Distance Matrix (heatmap)
    ax1 = axes[0]
    if HAS_SEABORN:
        sns.heatmap(distance_matrix, 
                    xticklabels=glyph_names, 
                    yticklabels=glyph_names,
                    annot=True, 
                    fmt='.2f',
                    cmap='viridis_r',
                    cbar_kws={'label': 'Euclidean Distance'},
                    ax=ax1,
                    square=True)
    else:
        # Fallback to matplotlib imshow
        im1 = ax1.imshow(distance_matrix, cmap='viridis_r', aspect='auto')
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.set_xticklabels(glyph_names, rotation=45, ha='right')
        ax1.set_yticklabels(glyph_names)
        plt.colorbar(im1, ax=ax1, label='Euclidean Distance')
        # Add text annotations
        for i in range(n):
            for j in range(n):
                ax1.text(j, i, f'{distance_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=7)
    ax1.set_title('Distance Matrix\n(Lower = More Similar)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Glyph', fontsize=12)
    ax1.set_ylabel('Glyph', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot 2: Confusion Matrix (binary, with highlights)
    ax2 = axes[1]
    
    # Create a custom colormap: white (0), light red (1), dark red (highlighted)
    cmap = plt.cm.Reds
    cmap.set_bad(color='white')
    
    # Plot confusion matrix
    im = ax2.imshow(confusion_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # Overlay highlights for problematic pairs
    if highlight_mask is not None and highlight_mask.sum() > 0:
        # Create a mask for non-highlighted cells
        non_highlight = (confusion_matrix > 0) & (highlight_mask == 0)
        highlighted = (confusion_matrix > 0) & (highlight_mask > 0)
        
        # Draw rectangles for highlighted pairs
        for i in range(n):
            for j in range(n):
                if highlighted[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                        fill=False, edgecolor='darkred', 
                                        linewidth=3, linestyle='--')
                    ax2.add_patch(rect)
    
    # Add text annotations for confusion values
    for i in range(n):
        for j in range(n):
            if i != j and confusion_matrix[i, j] > 0:
                text_color = 'darkred' if (highlight_mask is not None and highlight_mask[i, j] > 0) else 'black'
                ax2.text(j, i, f'{distance_matrix[i, j]:.2f}',
                        ha='center', va='center', 
                        color=text_color, fontweight='bold', fontsize=8)
    
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(glyph_names, rotation=45, ha='right')
    ax2.set_yticklabels(glyph_names)
    ax2.set_title(f'Confusion Matrix (Threshold = {threshold})\n'
                 'Red = Confused Pairs, Dashed Border = Problematic Pairs',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Glyph', fontsize=12)
    ax2.set_ylabel('True Glyph', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {save_path}")
    
    if IN_COLAB:
        plt.show()
    else:
        plt.show()
    
    return fig


def print_confusion_summary(confusion_matrix, distance_matrix, glyph_names, 
                           problematic_pairs, threshold=1.0):
    """
    Print a summary of confusions, especially for problematic pairs.
    
    Args:
        confusion_matrix: Binary confusion matrix
        distance_matrix: Distance matrix
        glyph_names: List of glyph names
        problematic_pairs: List of problematic pairs to highlight
        threshold: Distance threshold
    """
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX SUMMARY")
    print("=" * 70)
    
    # Count total confusions
    total_confusions = confusion_matrix.sum()
    n = len(glyph_names)
    total_possible = n * (n - 1)  # All pairs except diagonal
    
    print(f"\nTotal Confusions: {int(total_confusions)} / {total_possible} pairs")
    print(f"Confusion Rate: {100 * total_confusions / total_possible:.2f}%")
    print(f"Distance Threshold: {threshold}")
    
    # Find most confused pairs
    print("\n" + "-" * 70)
    print("MOST CONFUSED PAIRS (Top 10):")
    print("-" * 70)
    
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and confusion_matrix[i, j] > 0:
                confusions.append((glyph_names[i], glyph_names[j], distance_matrix[i, j]))
    
    # Sort by distance (lowest = most confused)
    confusions.sort(key=lambda x: x[2])
    
    for idx, (name1, name2, dist) in enumerate(confusions[:10], 1):
        is_problematic = (name1, name2) in problematic_pairs or (name2, name1) in problematic_pairs
        marker = " ⚠️ " if is_problematic else "   "
        print(f"{marker}{idx:2d}. {name1:10s} <-> {name2:10s}  (distance: {dist:.3f})")
    
    # Check problematic pairs specifically
    print("\n" + "-" * 70)
    print("PROBLEMATIC PAIRS STATUS:")
    print("-" * 70)
    
    name_to_idx = {name: i for i, name in enumerate(glyph_names)}
    
    for pair in problematic_pairs:
        name1, name2 = pair
        if name1 in name_to_idx and name2 in name_to_idx:
            idx1 = name_to_idx[name1]
            idx2 = name_to_idx[name2]
            dist = distance_matrix[idx1, idx2]
            is_confused = confusion_matrix[idx1, idx2] > 0
            
            status = "❌ CONFUSED" if is_confused else "✅ NOT CONFUSED"
            print(f"{status}: {name1:10s} <-> {name2:10s}  (distance: {dist:.3f}, threshold: {threshold})")
        else:
            print(f"⚠️  WARNING: Pair {pair} not found in dataset")
    
    print("\n" + "=" * 70)


def main():
    """Main function to generate confusion matrix."""
    print("=" * 70)
    print("Safaitic Glyph Confusion Matrix Generator")
    print("=" * 70)
    print()
    
    # Configuration
    if IN_COLAB:
        checkpoint_path = "/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth"
        root_dir = "/content/cleaned_glyphs"
        output_dir = "/content/drive/MyDrive/safaitic_project"
    else:
        checkpoint_path = "safaitic_matcher.pth"
        root_dir = "cleaned_glyphs"
        output_dir = "."
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        root_dir = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Mount Google Drive if in Colab
    if IN_COLAB:
        try:
            drive.mount('/content/drive', force_remount=False)
        except:
            print("Note: Google Drive may already be mounted")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Problematic pairs to monitor
    problematic_pairs = [
        ('s1', 's2'),
        ('h', 'h_dot'),
        ('s_dot', 's1'),
        ('s_dot', 's2'),
    ]
    
    # Distance threshold (same as used in training)
    threshold = 1.0
    
    # Load model
    print("Loading model...")
    try:
        model = load_model(checkpoint_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease provide the correct path to your trained model checkpoint.")
        print("Usage: python generate_confusion_matrix.py [checkpoint_path] [root_dir] [output_dir]")
        return
    
    print()
    
    # Get embeddings for all glyphs
    print("Generating embeddings for all glyphs...")
    embeddings, glyph_names = get_glyph_embeddings(model, root_dir, device, use_ideal=True)
    print()
    
    # Compute distance matrix
    print("Computing pairwise distances...")
    distance_matrix = compute_distance_matrix(embeddings, glyph_names)
    print()
    
    # Generate confusion matrix
    print(f"Generating confusion matrix (threshold = {threshold})...")
    confusion_matrix = generate_confusion_matrix(distance_matrix, glyph_names, threshold=threshold)
    print()
    
    # Create highlight mask for problematic pairs
    highlight_mask = highlight_problematic_pairs(confusion_matrix, glyph_names, problematic_pairs)
    
    # Print summary
    print_confusion_summary(confusion_matrix, distance_matrix, glyph_names, 
                           problematic_pairs, threshold=threshold)
    
    # Plot confusion matrix
    print("\nGenerating visualization...")
    output_path = Path(output_dir) / "confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(distance_matrix, confusion_matrix, glyph_names,
                        highlight_mask=highlight_mask, threshold=threshold,
                        save_path=str(output_path))
    
    print("\n" + "=" * 70)
    print("Confusion Matrix Generation Complete!")
    print(f"Output saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

