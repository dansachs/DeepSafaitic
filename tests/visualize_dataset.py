#!/usr/bin/env python3
"""
Visualize SafaiticDataset to verify augmentations.
Shows 10 random pairs of anchor vs augmented images side-by-side.
"""

import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from safaitic_dataset import SafaiticDataset


def visualize_dataset(examples_per_glyph=3, output_file="debug_augmentation.png"):
    """
    Load samples from SafaiticDataset and visualize anchor vs augmented pairs.
    Shows multiple examples of each glyph.
    
    Args:
        examples_per_glyph: Number of augmented examples to show per glyph
        output_file: Output filename for the plot
    """
    print("=" * 70)
    print("Safaitic Dataset Visualization")
    print("=" * 70)
    print()
    
    # Create dataset with augmentation enabled
    try:
        dataset = SafaiticDataset(root_dir="cleaned_glyphs", augment=True, variant="ideal")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Showing {examples_per_glyph} examples per glyph...")
    print()
    
    # Get unique labels and their indices
    unique_labels = sorted(set(dataset.labels))
    num_glyphs = len(unique_labels)
    
    # Create figure with subplots: one row per glyph, (1 anchor + examples_per_glyph augmented)
    cols = 1 + examples_per_glyph
    fig, axes = plt.subplots(num_glyphs, cols, figsize=(2 * cols, 2 * num_glyphs))
    
    # Handle case where num_glyphs is 1 (axes would be 1D instead of 2D)
    if num_glyphs == 1:
        axes = axes.reshape(1, -1)
    
    for glyph_idx, label in enumerate(unique_labels):
        # Find indices for this label
        label_indices = [i for i, l in enumerate(dataset.labels) if l == label]
        if not label_indices:
            continue
        
        # Get anchor image (first occurrence of this label)
        anchor_idx = label_indices[0]
        anchor_img = Image.open(dataset.image_paths[anchor_idx]).convert('RGB')
        anchor_array = np.array(anchor_img)
        
        # Display anchor (first column)
        axes[glyph_idx, 0].imshow(anchor_array)
        axes[glyph_idx, 0].set_title(f"Anchor\n{label}", fontsize=9, fontweight='bold')
        axes[glyph_idx, 0].axis('off')
        
        # Generate multiple augmented examples
        for aug_idx in range(examples_per_glyph):
            # Use the same anchor image but apply different random augmentations
            anchor_copy = anchor_array.copy()
            augmented_array = dataset.apply_augmentation(anchor_copy)
            
            # Display augmented version
            axes[glyph_idx, aug_idx + 1].imshow(augmented_array)
            axes[glyph_idx, aug_idx + 1].set_title(f"Aug #{aug_idx + 1}", fontsize=8)
            axes[glyph_idx, aug_idx + 1].axis('off')
    
    # Add overall title
    fig.suptitle('Safaitic Dataset: Anchor vs Augmented Examples\n'
                 f'Showing {examples_per_glyph} augmented examples per glyph\n'
                 'Augmentations: Rotation + Distortion + Noise + Wear + Occlusions',
                 fontsize=12, fontweight='bold', y=0.998)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    print(f"  Showing {num_glyphs} glyphs with {examples_per_glyph} examples each")
    
    # Also show the plot
    plt.show()
    
    return 0


def main():
    """Main function."""
    # Don't set seed - we want different augmentations each time
    # random.seed(42)
    # np.random.seed(42)
    
    return visualize_dataset(examples_per_glyph=3, output_file="debug_augmentation.png")


if __name__ == "__main__":
    exit(main())

