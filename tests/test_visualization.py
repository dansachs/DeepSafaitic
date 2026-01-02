#!/usr/bin/env python3
"""
Test script to visualize dataset pairs locally before Colab training.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dataset import SafaiticSiameseDataset

def test_dataset_visualization():
    """Test the dataset and visualize 3 augmented examples per glyph."""
    print("=" * 70)
    print("Testing Safaitic Siamese Dataset Visualization")
    print("=" * 70)
    print()
    
    # Create dataset
    print("Loading dataset...")
    try:
        dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)
        print(f"✓ Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1
    
    # Get unique labels
    unique_labels = sorted(set(dataset.labels))
    num_glyphs = len(unique_labels)
    examples_per_glyph = 3
    
    print(f"Found {num_glyphs} unique glyphs")
    print(f"Showing {examples_per_glyph} augmented examples per glyph...")
    print()
    
    # Create visualization: one row per glyph, (1 anchor + 3 augmented)
    cols = 1 + examples_per_glyph
    fig, axes = plt.subplots(num_glyphs, cols, figsize=(2 * cols, 2 * num_glyphs))
    
    if num_glyphs == 1:
        axes = axes.reshape(1, -1)
    
    print("Generating visualizations...")
    
    # Create visualization-only augmentation (without ToTensorV2)
    import albumentations as A
    vis_augmentation = A.Compose([
        A.SafeRotate(limit=360, p=1.0, border_mode=1, value=255),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.8,
            border_mode=1,
            value=255
        ),
        A.RandomScale(scale_limit=0.1, p=0.7, interpolation=1),
        A.GaussianBlur(blur_limit=(1, 3), p=0.6),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.7),
        A.CoarseDropout(
            max_holes=4,
            max_height=15,
            max_width=15,
            min_holes=0,
            min_height=5,
            min_width=5,
            fill_value=255,
            mask_fill_value=None,
            p=0.5
        ),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
    ])
    
    for glyph_idx, label in enumerate(unique_labels):
        # Find indices for this label
        label_indices = [i for i, l in enumerate(dataset.labels) if l == label]
        if not label_indices:
            continue
        
        # Get anchor image (first occurrence - prefer ideal.png)
        anchor_idx = None
        for idx in label_indices:
            if 'ideal.png' in str(dataset.image_paths[idx]):
                anchor_idx = idx
                break
        if anchor_idx is None:
            anchor_idx = label_indices[0]
        
        # Load anchor image
        anchor_img = Image.open(dataset.image_paths[anchor_idx]).convert('RGB')
        anchor_array = np.array(anchor_img)
        
        # Display anchor (first column)
        anchor_display = anchor_array / 255.0
        axes[glyph_idx, 0].imshow(anchor_display)
        axes[glyph_idx, 0].set_title(f"Anchor\n{label}", fontsize=9, fontweight='bold')
        axes[glyph_idx, 0].axis('off')
        
        # Generate 3 augmented examples
        for aug_idx in range(examples_per_glyph):
            # Use any image from this label for augmentation
            pair_idx = random.choice(label_indices)
            pair_img = Image.open(dataset.image_paths[pair_idx]).convert('RGB')
            pair_array = np.array(pair_img)
            
            # Apply augmentation (visualization version, no tensor conversion)
            try:
                augmented = vis_augmentation(image=pair_array)
                pair_np = augmented['image'] / 255.0
            except Exception as e:
                print(f"  Warning: Augmentation failed for {label}, using original: {e}")
                pair_np = pair_array / 255.0
            
            # Display augmented version
            axes[glyph_idx, aug_idx + 1].imshow(pair_np)
            axes[glyph_idx, aug_idx + 1].set_title(f"Aug #{aug_idx + 1}", fontsize=8)
            axes[glyph_idx, aug_idx + 1].axis('off')
    
    plt.suptitle('Safaitic Dataset: Anchor vs Augmented Examples\n'
                 f'Showing {examples_per_glyph} augmented examples per glyph\n'
                 'Effect of "The Ager" (Aggressive Augmentations)',
                 fontsize=12, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save visualization
    output_file = "test_dataset_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    print(f"  Showing {num_glyphs} glyphs with {examples_per_glyph} examples each")
    
    # Show plot
    plt.show()
    
    # Test dataset info
    print("\n" + "=" * 70)
    print("Dataset Information")
    print("=" * 70)
    print(f"Total images: {len(dataset)}")
    print(f"Unique glyphs: {len(unique_labels)}")
    print(f"Images per glyph: {len(dataset) // len(unique_labels)}")
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(test_dataset_visualization())

