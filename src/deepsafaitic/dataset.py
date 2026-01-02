"""
Siamese Dataset for Safaitic Glyph Similarity Learning.
Creates positive and negative pairs for contrastive learning.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SafaiticSiameseDataset(Dataset):
    """
    Dataset that creates pairs of Safaitic glyphs for Siamese network training.
    
    Positive pairs: Same glyph (ideal + augmented)
    Negative pairs: Different glyphs (ideal + augmented from different letter)
    """
    
    def __init__(self, root_dir="cleaned_glyphs", transform=None, augment=True):
        """
        Initialize the Siamese Dataset.
        
        Args:
            root_dir: Root directory containing glyph subfolders
            transform: Transform to apply to images (default: ToTensor)
            augment: Whether to apply aggressive augmentations
        """
        self.root_dir = Path(root_dir)
        self.augment = augment
        
        if not self.root_dir.exists():
            raise ValueError(f"Root directory '{root_dir}' does not exist!")
        
        # Collect all image paths with their labels
        # Separate ideal and square for anchor integrity
        self.ideal_paths = []  # For anchors (always clean)
        self.square_paths = []  # For augmentation
        self.all_paths = []  # All paths for negative pairs
        self.labels = []
        self.ideal_labels = []
        
        # Get all subdirectories (one per letter)
        for letter_dir in sorted(self.root_dir.iterdir()):
            if not letter_dir.is_dir():
                continue
            
            letter_name = letter_dir.name
            
            # Store ideal.png separately (for anchors)
            ideal_path = letter_dir / "ideal.png"
            square_path = letter_dir / "square.png"
            
            if ideal_path.exists():
                self.ideal_paths.append(ideal_path)
                self.ideal_labels.append(letter_name)
                self.all_paths.append(ideal_path)
                self.labels.append(letter_name)
            
            if square_path.exists():
                self.all_paths.append(square_path)
                self.labels.append(letter_name)
            
            # Also store square for augmentation
            if square_path.exists():
                self.square_paths.append(square_path)
        
        if len(self.ideal_paths) == 0:
            raise ValueError(f"No ideal images found in '{root_dir}'!")
        
        # Get unique labels
        self.unique_labels = sorted(set(self.ideal_labels))
        self.label_to_ideal_indices = {label: [i for i, l in enumerate(self.ideal_labels) if l == label] 
                                       for label in self.unique_labels}
        self.label_to_all_indices = {label: [i for i, l in enumerate(self.labels) if l == label] 
                                     for label in self.unique_labels}
        
        # Identify visually similar glyphs (hard negatives)
        self.similar_glyphs = self._identify_similar_glyphs()
        
        print(f"Loaded {len(self.ideal_paths)} ideal images (anchors) from {len(self.unique_labels)} letters")
        print(f"Loaded {len(self.all_paths)} total images (including square variants)")
        if self.similar_glyphs:
            print(f"Identified {len(self.similar_glyphs)} pairs of similar glyphs for hard negative mining")
        
        # Define transforms
        if transform is None:
            # Default: just convert to tensor and normalize
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        
        # Define aggressive augmentation (The Ager)
        if self.augment:
            self.augmentation = self._create_aggressive_augmentation()
        else:
            self.augmentation = None
    
    def _create_aggressive_augmentation(self):
        """
        Create aggressive augmentation pipeline using Albumentations.
        Includes: 360° rotation, elastic distortion, erosion, stone noise, occlusions.
        """
        return A.Compose([
            # 360-degree rotation (border_mode=1 is cv2.BORDER_CONSTANT, value=255 for white)
            A.SafeRotate(limit=360, p=1.0, border_mode=1, value=255, interpolation=1),  # White background, no black wedges
            
            # Elastic distortion (simulates warping) - border_mode=1 is cv2.BORDER_CONSTANT
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.8,
                border_mode=1,  # cv2.BORDER_CONSTANT
                value=255,  # White fill value
                interpolation=1
            ),
            
            # Random scaling
            A.RandomScale(scale_limit=0.1, p=0.7, interpolation=1),
            
            # Blur (weathering)
            A.GaussianBlur(blur_limit=(1, 3), p=0.6),
            
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            
            # Stone texture noise (GaussNoise)
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.7),
            
            # Random occlusions (CoarseDropout simulates damage/patches)
            A.CoarseDropout(
                max_holes=4,
                max_height=15,
                max_width=15,
                min_holes=0,
                min_height=5,
                min_width=5,
                fill_value=255,  # White patches
                mask_fill_value=None,
                p=0.5
            ),
            
            # Additional noise for stone texture
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            
            # Normalize and convert to tensor
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _identify_similar_glyphs(self):
        """
        Identify visually similar glyph pairs for hard negative mining.
        Based on common visual patterns in Safaitic script.
        """
        # Pairs that are visually similar (vertical lines, similar shapes, etc.)
        # Using actual folder names from validate_safaitic_assets.py
        similar_pairs = [
            ('r', 'z'),      # Similar curved shapes
            ('s1', 's2'),    # Both 's' variants
            ('t', 'th'),     # Similar base shape
            ('h', 'h_dot'),  # Related glyphs
            ('k', 'kh'),     # Similar shapes
            ('g', 'gh'),     # Related
            ('z', 'z_dot'),  # Related
            ('t', 't_dot'),  # Related
            ('d', 'd_dot'),  # Related
            ('s_dot', 's1'),  # Related 's' variants
            ('s_dot', 's2'),  # Related 's' variants
            ('b', 'd'),      # Similar vertical structures
            ('l', 'r'),      # Similar curved shapes
            ('n', 'm'),      # Similar shapes
        ]
        
        # Filter to only include pairs that exist in our dataset
        valid_pairs = []
        for pair in similar_pairs:
            if pair[0] in self.unique_labels and pair[1] in self.unique_labels:
                valid_pairs.append(pair)
        
        return valid_pairs
    
    def __len__(self):
        return len(self.ideal_paths)  # Use ideal paths as base (one per letter)
    
    def __getitem__(self, idx):
        """
        Get a pair of images and label.
        
        CRITICAL: Anchor is ALWAYS ideal.png (clean), pair is ALWAYS augmented.
        
        Returns:
            anchor_tensor: Anchor image (always clean ideal.png)
            pair_tensor: Pair image (augmented version)
            label: 1 for positive pair, 0 for negative pair
            hard_negative: True if this is a hard negative pair (similar glyphs)
        """
        # CRITICAL FIX: Anchor is ALWAYS ideal.png (clean version)
        anchor_path = self.ideal_paths[idx]
        anchor_label = self.ideal_labels[idx]
        
        # Load anchor image (always clean, no augmentation)
        anchor_img = Image.open(anchor_path).convert('RGB')
        anchor_array = np.array(anchor_img)
        
        # Ensure 50/50 balance: use idx to determine positive/negative deterministically
        # Then add randomness for which specific image to use
        is_positive = (idx % 2 == 0)  # Alternating pattern for balance
        
        hard_negative = False
        
        if is_positive:
            # Positive pair: same glyph
            label = 1
            
            # Get another image from the same label (prefer square for variety, but can use ideal)
            same_label_all_indices = self.label_to_all_indices[anchor_label]
            # Remove current ideal index
            same_label_all_indices = [i for i in same_label_all_indices 
                                     if self.all_paths[i] != anchor_path]
            
            if len(same_label_all_indices) > 0:
                pair_idx = random.choice(same_label_all_indices)
                pair_path = self.all_paths[pair_idx]
            else:
                # Fallback: use square if available, otherwise same ideal
                square_path = anchor_path.parent / "square.png"
                if square_path.exists():
                    pair_path = square_path
                else:
                    pair_path = anchor_path  # Last resort: same image
            
            pair_img = Image.open(pair_path).convert('RGB')
            pair_array = np.array(pair_img)
            
        else:
            # Negative pair: different glyph
            label = 0
            
            # 30% chance to use hard negative (similar glyphs)
            use_hard_negative = random.random() < 0.3 and len(self.similar_glyphs) > 0
            
            if use_hard_negative:
                # Hard negative: pick from similar glyphs
                similar_pair = random.choice(self.similar_glyphs)
                if anchor_label == similar_pair[0]:
                    different_label = similar_pair[1]
                elif anchor_label == similar_pair[1]:
                    different_label = similar_pair[0]
                else:
                    # Pick one from the pair
                    different_label = random.choice(similar_pair)
                hard_negative = True
            else:
                # Regular negative: random different label
                different_labels = [l for l in self.unique_labels if l != anchor_label]
                different_label = random.choice(different_labels)
            
            # Get image from different label
            different_indices = self.label_to_all_indices[different_label]
            pair_idx = random.choice(different_indices)
            pair_path = self.all_paths[pair_idx]
            pair_img = Image.open(pair_path).convert('RGB')
            pair_array = np.array(pair_img)
        
        # Apply augmentation to pair image (The Ager) - NEVER augment anchor
        if self.augment and self.augmentation is not None:
            augmented = self.augmentation(image=pair_array)
            pair_tensor = augmented['image']
        else:
            # Just normalize
            transformed = self.transform(image=pair_array)
            pair_tensor = transformed['image']
        
        # Apply transform to anchor (NO AUGMENTATION - always clean)
        anchor_transformed = self.transform(image=anchor_array)
        anchor_tensor = anchor_transformed['image']
        
        return anchor_tensor, pair_tensor, torch.tensor(label, dtype=torch.float32), hard_negative
    
    def get_sample_pairs(self, num_pairs=5):
        """
        Get sample pairs for visualization.
        
        Args:
            num_pairs: Number of pairs to return
        
        Returns:
            List of (anchor_img, pair_img, label, hard_negative) tuples
        """
        samples = []
        indices = random.sample(range(len(self)), min(num_pairs, len(self)))
        
        for idx in indices:
            anchor_tensor, pair_tensor, label, hard_negative = self[idx]
            
            # Convert tensors back to numpy for visualization
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            anchor_np = anchor_tensor.permute(1, 2, 0).numpy()
            anchor_np = anchor_np * std + mean
            anchor_np = np.clip(anchor_np, 0, 1)
            
            pair_np = pair_tensor.permute(1, 2, 0).numpy()
            pair_np = pair_np * std + mean
            pair_np = np.clip(pair_np, 0, 1)
            
            samples.append((anchor_np, pair_np, label.item(), hard_negative))
        
        return samples

