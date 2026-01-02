"""
SafaiticDataset - PyTorch dataset for Safaitic glyphs with augmentation.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Optional PyTorch imports (for DataLoader support)
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy Dataset base class if torch not available
    class Dataset:
        pass


class SafaiticDataset(Dataset):
    """
    Dataset for Safaitic glyphs with augmentation support.
    
    Loads PNG images from cleaned_glyphs/ directory structure.
    Supports rotation and stone-texture noise augmentation.
    """
    
    def __init__(self, root_dir="cleaned_glyphs", augment=True, variant="ideal"):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing glyph subfolders
            augment: Whether to apply augmentations (rotation, stone texture)
            variant: Which variant to load ("ideal", "square", or "both")
        """
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.variant = variant
        
        # Collect all image paths
        self.image_paths = []
        self.labels = []
        
        if not self.root_dir.exists():
            raise ValueError(f"Root directory '{root_dir}' does not exist!")
        
        # Get all subdirectories (one per letter)
        for letter_dir in sorted(self.root_dir.iterdir()):
            if not letter_dir.is_dir():
                continue
            
            letter_name = letter_dir.name
            
            # Load based on variant preference
            if variant == "both":
                # Load both ideal and square
                for img_file in ["ideal.png", "square.png"]:
                    img_path = letter_dir / img_file
                    if img_path.exists():
                        self.image_paths.append(img_path)
                        self.labels.append(letter_name)
            else:
                # Load specific variant
                img_path = letter_dir / f"{variant}.png"
                if img_path.exists():
                    self.image_paths.append(img_path)
                    self.labels.append(letter_name)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in '{root_dir}'!")
        
        print(f"Loaded {len(self.image_paths)} images from {len(set(self.labels))} letters")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Returns:
            If augment=True: (anchor_image, augmented_image, label)
            If augment=False: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        if self.augment:
            # Create anchor (original) and augmented versions
            anchor = img_array.copy()
            augmented = self.apply_augmentation(img_array.copy())
            
            # Convert back to PIL Images
            anchor_img = Image.fromarray(anchor)
            augmented_img = Image.fromarray(augmented)
            
            return anchor_img, augmented_img, label
        else:
            return Image.fromarray(img_array), label
    
    def apply_augmentation(self, image_array):
        """
        Apply aggressive augmentations to simulate heavily degraded stone carvings.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Augmented image array
        """
        pil_img = Image.fromarray(image_array)
        
        # 1. Random rotation (0-360 degrees)
        angle = random.uniform(0, 360)
        rotated = pil_img.rotate(angle, fillcolor='white', resample=Image.BICUBIC)
        
        # 2. Random slight scaling variation (simulates distance/viewing angle)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.92, 1.08)  # Slight zoom in/out
            w, h = rotated.size
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            rotated = rotated.resize((new_w, new_h), Image.Resampling.LANCZOS)
            # Crop or pad to original size
            if new_w != w or new_h != h:
                # Center crop or pad
                left = (new_w - w) // 2 if new_w > w else 0
                top = (new_h - h) // 2 if new_h > h else 0
                if new_w >= w and new_h >= h:
                    rotated = rotated.crop((left, top, left + w, top + h))
                else:
                    # Pad with white
                    padded = Image.new('RGB', (w, h), (255, 255, 255))
                    paste_x = (w - new_w) // 2 if new_w < w else 0
                    paste_y = (h - new_h) // 2 if new_h < h else 0
                    padded.paste(rotated, (paste_x, paste_y))
                    rotated = padded
        
        # 3. Random blur (simulates weathering/wear)
        blur_radius = random.uniform(0.5, 2.5)
        if blur_radius > 0.5:
            from PIL import ImageFilter
            rotated = rotated.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Convert back to array for further processing
        image_array = np.array(rotated)
        
        # 4. Aggressive stone-texture noise
        image_array = self.add_stone_texture(image_array)
        
        # 5. Random brightness and contrast changes (simulates lighting/weathering)
        image_array = self.adjust_brightness_contrast(image_array)
        
        # 6. Geometric distortions (warp the actual lines/shapes)
        image_array = self.apply_geometric_distortion(image_array)
        
        # 7. Random erosion/dilation (simulates wear)
        image_array = self.apply_wear(image_array)
        
        # 8. Line thickness variations
        image_array = self.vary_line_thickness(image_array)
        
        # 9. Random occlusions (simulates damage/patches)
        image_array = self.add_occlusions(image_array)
        
        return image_array
    
    def add_stone_texture(self, image_array):
        """
        Add aggressive stone-texture noise to simulate heavily weathered stone.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with heavy stone texture noise
        """
        # Increased noise strength for more degradation
        noise_strength = random.uniform(0.20, 0.35)  # 20-35% noise strength (was 15%)
        h, w, c = image_array.shape
        
        # Generate multiple noise layers for stone-like texture
        # 1. Fine grain noise (more aggressive)
        fine_noise = np.random.normal(0, noise_strength * 30, (h, w, c)).astype(np.float32)
        
        # 2. Coarse grain noise (larger patches, more variation)
        coarse_scale = random.randint(6, 12)
        coarse_h, coarse_w = max(1, h // coarse_scale), max(1, w // coarse_scale)
        coarse_noise = np.random.normal(0, noise_strength * 25, (coarse_h, coarse_w, c))
        # Upsample coarse noise to match image size exactly
        # Use simple repeat and then resize if needed
        coarse_noise = np.repeat(np.repeat(coarse_noise, coarse_scale, axis=0), coarse_scale, axis=1)
        # Ensure exact size match
        if coarse_noise.shape[0] > h:
            coarse_noise = coarse_noise[:h, :, :]
        elif coarse_noise.shape[0] < h:
            # Pad with last row
            padding = np.repeat(coarse_noise[-1:, :, :], h - coarse_noise.shape[0], axis=0)
            coarse_noise = np.vstack([coarse_noise, padding])
        
        if coarse_noise.shape[1] > w:
            coarse_noise = coarse_noise[:, :w, :]
        elif coarse_noise.shape[1] < w:
            # Pad with last column
            padding = np.repeat(coarse_noise[:, -1:, :], w - coarse_noise.shape[1], axis=1)
            coarse_noise = np.hstack([coarse_noise, padding])
        
        # 3. Add crack-like noise (vertical/horizontal streaks)
        crack_noise = np.zeros((h, w, c), dtype=np.float32)
        num_cracks = random.randint(2, 6)
        for _ in range(num_cracks):
            if random.random() > 0.5:
                # Vertical crack
                x = random.randint(0, w-1)
                width = random.randint(1, 3)
                x_start = max(0, x - width)
                x_end = min(w, x + width)
                crack_width = x_end - x_start
                if crack_width > 0:
                    crack_noise[:, x_start:x_end, :] += np.random.normal(0, noise_strength * 40, (h, crack_width, c))
            else:
                # Horizontal crack
                y = random.randint(0, h-1)
                height = random.randint(1, 3)
                y_start = max(0, y - height)
                y_end = min(h, y + height)
                crack_height = y_end - y_start
                if crack_height > 0:
                    crack_noise[y_start:y_end, :, :] += np.random.normal(0, noise_strength * 40, (crack_height, w, c))
        
        # Combine all noise types
        combined_noise = fine_noise + coarse_noise * 0.7 + crack_noise * 0.3
        
        # Apply noise to image
        img_float = image_array.astype(np.float32)
        noisy_img = img_float + combined_noise
        
        # Clip values to valid range
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return noisy_img
    
    def adjust_brightness_contrast(self, image_array):
        """
        Randomly adjust brightness and contrast to simulate lighting variations.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with adjusted brightness/contrast
        """
        img_float = image_array.astype(np.float32)
        
        # Random brightness adjustment (-30% to +20%)
        brightness_factor = random.uniform(0.70, 1.20)
        img_float = img_float * brightness_factor
        
        # Random contrast adjustment (0.6x to 1.4x)
        contrast_factor = random.uniform(0.6, 1.4)
        mean = img_float.mean()
        img_float = (img_float - mean) * contrast_factor + mean
        
        # Clip values
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
    def apply_geometric_distortion(self, image_array):
        """
        Apply elastic/geometric distortions to warp the actual lines and shapes.
        This creates curves and bends in the glyph lines.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with geometric distortions
        """
        h, w, c = image_array.shape
        
        # Create displacement fields for elastic deformation
        # This will actually warp the image geometry
        alpha = random.uniform(30, 70)  # Strength of distortion (increased from 20-50)
        sigma = random.uniform(5, 10)   # Smoothness of distortion (slightly lower for more variation)
        
        # Generate random displacement fields
        dx = np.random.randn(h, w).astype(np.float32) * alpha
        dy = np.random.randn(h, w).astype(np.float32) * alpha
        
        # Smooth the displacement fields using simple box filter
        kernel_size = max(3, int(sigma))
        if kernel_size > 1:
            # Simple box filter for smoothing
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
            
            # Apply convolution manually (simple approach)
            pad = kernel_size // 2
            dx_padded = np.pad(dx, pad, mode='edge')
            dy_padded = np.pad(dy, pad, mode='edge')
            
            dx_smooth = np.zeros_like(dx)
            dy_smooth = np.zeros_like(dy)
            
            for i in range(h):
                for j in range(w):
                    dx_smooth[i, j] = np.sum(dx_padded[i:i+kernel_size, j:j+kernel_size] * kernel)
                    dy_smooth[i, j] = np.sum(dy_padded[i:i+kernel_size, j:j+kernel_size] * kernel)
            
            dx, dy = dx_smooth, dy_smooth
        
        # Apply displacement to create warped image
        distorted = np.zeros_like(image_array)
        
        for channel in range(c):
            channel_img = image_array[:, :, channel].astype(np.float32)
            
            # Create coordinate grids
            y_coords, x_coords = np.meshgrid(np.arange(h, dtype=np.float32), 
                                            np.arange(w, dtype=np.float32), 
                                            indexing='ij')
            
            # Apply displacement
            new_x = x_coords + dx
            new_y = y_coords + dy
            
            # Clip to valid range
            new_x = np.clip(new_x, 0, w - 1)
            new_y = np.clip(new_y, 0, h - 1)
            
            # Bilinear interpolation
            x0 = np.floor(new_x).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y0 = np.floor(new_y).astype(np.int32)
            y1 = np.clip(y0 + 1, 0, h - 1)
            
            # Get fractional parts
            fx = new_x - x0.astype(np.float32)
            fy = new_y - y0.astype(np.float32)
            
            # Sample from original image
            I00 = channel_img[y0, x0]
            I10 = channel_img[y1, x0]
            I01 = channel_img[y0, x1]
            I11 = channel_img[y1, x1]
            
            # Bilinear interpolation
            distorted_channel = (I00 * (1 - fx) * (1 - fy) +
                                I10 * (1 - fx) * fy +
                                I01 * fx * (1 - fy) +
                                I11 * fx * fy)
            
            distorted[:, :, channel] = np.clip(distorted_channel, 0, 255).astype(np.uint8)
        
        return distorted
    
    def vary_line_thickness(self, image_array):
        """
        Vary line thickness by applying morphological operations.
        This makes lines thicker or thinner randomly, changing their appearance.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with varied line thickness
        """
        # Convert to grayscale for processing
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array
        
        # Create binary mask (glyph vs background)
        threshold = 200
        mask = (gray < threshold).astype(np.uint8)
        
        # Randomly make lines thicker or thinner
        operation = random.choice(['thicken', 'thin', 'both', 'none'])
        if operation == 'none':
            return image_array
        
        kernel_size = random.randint(1, 3)
        result = image_array.copy().astype(np.float32)
        
        if operation == 'thicken' or operation == 'both':
            # Dilate (make lines thicker)
            # Simple dilation: expand dark pixels
            dilated_mask = mask.copy()
            for i in range(kernel_size, mask.shape[0] - kernel_size):
                for j in range(kernel_size, mask.shape[1] - kernel_size):
                    if mask[i, j] == 1:
                        # Expand to neighbors
                        dilated_mask[i-kernel_size:i+kernel_size+1, 
                                    j-kernel_size:j+kernel_size+1] = 1
            
            # Apply darkening to thickened areas
            thickness_diff = (dilated_mask - mask).astype(np.float32)
            for c in range(image_array.shape[2]):
                result[:, :, c] = np.clip(
                    result[:, :, c] - thickness_diff * 120,
                    0, 255
                )
        
        if operation == 'thin' or operation == 'both':
            # Erode (make lines thinner)
            eroded_mask = mask.copy()
            for i in range(kernel_size, mask.shape[0] - kernel_size):
                for j in range(kernel_size, mask.shape[1] - kernel_size):
                    if mask[i, j] == 1:
                        # Check if all neighbors are also dark
                        neighborhood = mask[i-kernel_size:i+kernel_size+1, 
                                          j-kernel_size:j+kernel_size+1]
                        if np.any(neighborhood == 0):
                            eroded_mask[i, j] = 0
            
            # Apply lightening to thinned areas
            thickness_diff = (mask - eroded_mask).astype(np.float32)
            for c in range(image_array.shape[2]):
                result[:, :, c] = np.clip(
                    result[:, :, c] + thickness_diff * 100,
                    0, 255
                )
        
        return result.astype(np.uint8)
    
    def apply_wear(self, image_array):
        """
        Apply erosion/dilation to simulate wear on stone carvings.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with wear effects
        """
        # Convert to grayscale for morphological operations
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array
        
        # Create binary mask (glyph vs background)
        # Threshold to separate glyph from white background
        threshold = 200  # White background is ~255
        mask = (gray < threshold).astype(np.uint8) * 255
        
        # Apply random erosion (makes glyphs thinner/worn)
        if random.random() > 0.3:  # 70% chance
            # Simple erosion: shrink dark areas
            kernel_size = random.randint(1, 2)
            eroded_mask = mask.copy()
            for i in range(kernel_size, mask.shape[0] - kernel_size):
                for j in range(kernel_size, mask.shape[1] - kernel_size):
                    if mask[i, j] == 255:
                        # Check neighbors
                        if np.any(mask[i-kernel_size:i+kernel_size+1, 
                                      j-kernel_size:j+kernel_size+1] == 0):
                            eroded_mask[i, j] = 0
            
            # Blend eroded mask back into image
            mask_diff = (mask.astype(np.float32) - eroded_mask.astype(np.float32)) / 255.0
            for c in range(image_array.shape[2]):
                image_array[:, :, c] = np.clip(
                    image_array[:, :, c].astype(np.float32) + mask_diff * 50,  # Lighten eroded areas
                    0, 255
                ).astype(np.uint8)
        
        return image_array
    
    def apply_geometric_distortion(self, image_array):
        """
        Apply elastic/geometric distortions to warp the actual lines and shapes.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with geometric distortions
        """
        h, w, c = image_array.shape
        
        # Create displacement fields for elastic deformation
        # This will actually warp the image geometry
        alpha = random.uniform(30, 80)  # Strength of distortion
        sigma = random.uniform(8, 15)   # Smoothness of distortion
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth the displacement fields (Gaussian blur approximation)
        # Simple box filter for smoothing
        kernel_size = int(sigma)
        if kernel_size > 1:
            # Simple averaging filter
            from scipy import ndimage
            try:
                dx = ndimage.gaussian_filter(dx, sigma=sigma)
                dy = ndimage.gaussian_filter(dy, sigma=sigma)
            except ImportError:
                # Fallback: simple box filter
                for _ in range(3):
                    dx = np.convolve(dx.flatten(), np.ones(kernel_size)/kernel_size, mode='same').reshape(h, w)
                    dy = np.convolve(dy.flatten(), np.ones(kernel_size)/kernel_size, mode='same').reshape(h, w)
        
        # Apply displacement to each channel
        distorted = np.zeros_like(image_array)
        for channel in range(c):
            channel_img = image_array[:, :, channel]
            
            # Create coordinate grids
            y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # Apply displacement
            new_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
            new_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)
            
            # Interpolate using nearest neighbor (faster) or bilinear
            # Use bilinear interpolation for smoother results
            from scipy.interpolate import griddata
            try:
                points = np.column_stack([y_coords.flatten(), x_coords.flatten()])
                values = channel_img.flatten()
                query_points = np.column_stack([new_y.flatten(), new_x.flatten()])
                distorted_channel = griddata(points, values, query_points, method='linear', fill_value=255)
                distorted[:, :, channel] = distorted_channel.reshape(h, w).astype(np.uint8)
            except ImportError:
                # Fallback: use PIL for distortion
                pil_img = Image.fromarray(channel_img)
                # Create a simple distortion using PIL's transform
                # For now, use a simpler approach with numpy
                distorted[:, :, channel] = self._apply_displacement_numpy(channel_img, dx, dy)
        
        return distorted
    
    def _apply_displacement_numpy(self, img, dx, dy):
        """Apply displacement using numpy (fallback method)."""
        h, w = img.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Calculate new positions
        new_x = (x_coords + dx).astype(np.int32)
        new_y = (y_coords + dy).astype(np.int32)
        
        # Clip to valid range
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        
        # Sample from original image
        distorted = img[new_y, new_x]
        
        return distorted
    
    def vary_line_thickness(self, image_array):
        """
        Vary line thickness by applying morphological operations.
        This makes lines thicker or thinner randomly.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with varied line thickness
        """
        # Convert to grayscale for processing
        if len(image_array.shape) == 3:
            gray = np.mean(image_array, axis=2).astype(np.uint8)
        else:
            gray = image_array
        
        # Create binary mask (glyph vs background)
        threshold = 200
        mask = (gray < threshold).astype(np.uint8)
        
        # Randomly make lines thicker or thinner
        operation = random.choice(['thicken', 'thin', 'both'])
        kernel_size = random.randint(1, 3)
        
        result = image_array.copy()
        
        if operation == 'thicken' or operation == 'both':
            # Dilate (make lines thicker)
            # Simple dilation: expand dark pixels
            dilated_mask = mask.copy()
            for i in range(kernel_size, mask.shape[0] - kernel_size):
                for j in range(kernel_size, mask.shape[1] - kernel_size):
                    if mask[i, j] == 1:
                        # Expand to neighbors
                        dilated_mask[i-kernel_size:i+kernel_size+1, 
                                    j-kernel_size:j+kernel_size+1] = 1
            
            # Apply to all channels
            thickness_diff = (dilated_mask - mask).astype(np.float32)
            for c in range(image_array.shape[2]):
                result[:, :, c] = np.clip(
                    result[:, :, c].astype(np.float32) - thickness_diff * 100,
                    0, 255
                ).astype(np.uint8)
        
        if operation == 'thin' or operation == 'both':
            # Erode (make lines thinner) - but only if we didn't just thicken
            if operation == 'thin':
                eroded_mask = mask.copy()
                for i in range(kernel_size, mask.shape[0] - kernel_size):
                    for j in range(kernel_size, mask.shape[1] - kernel_size):
                        if mask[i, j] == 1:
                            # Check if all neighbors are also dark
                            neighborhood = mask[i-kernel_size:i+kernel_size+1, 
                                              j-kernel_size:j+kernel_size+1]
                            if np.any(neighborhood == 0):
                                eroded_mask[i, j] = 0
                
                # Apply to all channels
                thickness_diff = (mask - eroded_mask).astype(np.float32)
                for c in range(image_array.shape[2]):
                    result[:, :, c] = np.clip(
                        result[:, :, c].astype(np.float32) + thickness_diff * 80,
                        0, 255
                    ).astype(np.uint8)
        
        return result
    
    def add_occlusions(self, image_array):
        """
        Add random occlusions to simulate damage, patches, or debris.
        
        Args:
            image_array: numpy array of the image (H, W, C)
        
        Returns:
            Image array with random occlusions
        """
        h, w, c = image_array.shape
        
        # Random number of occlusions
        num_occlusions = random.randint(0, 4)
        
        for _ in range(num_occlusions):
            # Random occlusion size and position
            occ_size = random.randint(3, 15)
            x = random.randint(0, w - occ_size)
            y = random.randint(0, h - occ_size)
            
            # Random occlusion type
            occ_type = random.choice(['dark', 'light', 'noise'])
            
            if occ_type == 'dark':
                # Dark patch (shadow/damage)
                image_array[y:y+occ_size, x:x+occ_size, :] = np.clip(
                    image_array[y:y+occ_size, x:x+occ_size, :].astype(np.float32) * 0.3,
                    0, 255
                ).astype(np.uint8)
            elif occ_type == 'light':
                # Light patch (reflection/patina)
                image_array[y:y+occ_size, x:x+occ_size, :] = np.clip(
                    image_array[y:y+occ_size, x:x+occ_size, :].astype(np.float32) + 100,
                    0, 255
                ).astype(np.uint8)
            else:
                # Noise patch
                noise = np.random.randint(0, 255, (occ_size, occ_size, c))
                image_array[y:y+occ_size, x:x+occ_size, :] = noise
        
        return image_array


def get_dataloader(root_dir="cleaned_glyphs", batch_size=32, augment=True, variant="ideal"):
    """
    Create a DataLoader for the SafaiticDataset.
    
    Args:
        root_dir: Root directory containing glyph subfolders
        batch_size: Batch size for the DataLoader
        augment: Whether to apply augmentations
        variant: Which variant to load ("ideal", "square", or "both")
    
    Returns:
        DataLoader instance
    
    Note: Requires PyTorch to be installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader. Install with: pip install torch")
    
    dataset = SafaiticDataset(root_dir=root_dir, augment=augment, variant=variant)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility, increase if needed
    )
    return dataloader

