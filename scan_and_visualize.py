#!/usr/bin/env python3
"""
Safaitic Stone Scanner with Visual Debugging

Scans a full stone image and shows step-by-step glyph detection and classification.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from typing import List, Tuple, Optional, Dict
import os

# Import model architecture
from model import SafaiticSiameseNet, euclidean_distance
from model_utils import find_latest_model, list_models


def detect_google_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def load_model_from_drive(
    checkpoint_path: str,
    device: str = None,
    embedding_dim: int = 512,
    colab_drive_path: str = "/content/drive/MyDrive/safaitic_project/safaitic_matcher.pth"
) -> SafaiticSiameseNet:
    """
    Load trained model from Google Drive (Colab) or local path.
    
    Args:
        checkpoint_path: Path to checkpoint (local path or Drive path)
        device: Device to load model on ('cuda' or 'cpu')
        embedding_dim: Embedding dimension (default: 512)
        colab_drive_path: Default Drive path if in Colab
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if we're in Colab and need to mount Drive
    is_colab = detect_google_colab()
    
    if is_colab:
        # Try to mount Drive if not already mounted
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        except Exception as e:
            print(f"Warning: Could not mount Drive: {e}")
        
        # If checkpoint_path is relative or doesn't exist, try Drive path
        if not Path(checkpoint_path).exists():
            checkpoint_path = colab_drive_path
            print(f"Using Drive path: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    
    # If path is a directory, find the latest model
    if checkpoint_path.is_dir():
        print(f"Checkpoint path is a directory, searching for latest model...")
        latest_model = find_latest_model(checkpoint_path)
        if latest_model:
            checkpoint_path = latest_model
            print(f"  Found latest model: {checkpoint_path.name}")
        else:
            raise FileNotFoundError(
                f"No models found in directory: {checkpoint_path}\n"
                f"Expected format: safaitic_matcher_YYYY-MM-DD_HH-MM-SS.pth"
            )
    
    # If path doesn't exist but is a file, try to find latest in parent directory
    if not checkpoint_path.exists():
        parent_dir = checkpoint_path.parent
        if parent_dir.exists() and checkpoint_path.name == "safaitic_matcher.pth":
            print(f"Model not found, searching for latest in: {parent_dir}")
            latest_model = find_latest_model(parent_dir)
            if latest_model:
                checkpoint_path = latest_model
                print(f"  Using latest model: {checkpoint_path.name}")
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at: {checkpoint_path}\n"
                    f"In Colab? Make sure Drive is mounted and path is correct.\n"
                    f"Locally? Download the checkpoint file first.\n"
                    f"Tried to find latest model in: {parent_dir}"
                )
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at: {checkpoint_path}\n"
                f"In Colab? Make sure Drive is mounted and path is correct.\n"
                f"Locally? Download the checkpoint file first."
            )
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = SafaiticSiameseNet(embedding_dim=embedding_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✓ Model loaded successfully on {device}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    if 'timestamp' in checkpoint:
        print(f"  Model timestamp: {checkpoint['timestamp']}")
    
    return model


def load_reference_embeddings(
    model: SafaiticSiameseNet,
    glyphs_dir: str,
    device: str = None,
    colab_drive_path: str = "/content/drive/MyDrive/safaitic_project/cleaned_glyphs"
) -> Dict[str, torch.Tensor]:
    """
    Load all reference glyph embeddings from cleaned_glyphs directory.
    
    Args:
        model: Trained Siamese model
        glyphs_dir: Path to cleaned_glyphs directory
        device: Device to run model on
        colab_drive_path: Default Drive path if in Colab
    
    Returns:
        Dictionary mapping glyph name -> embedding tensor
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Handle Colab Drive path
    is_colab = detect_google_colab()
    if is_colab and not Path(glyphs_dir).exists():
        glyphs_dir = colab_drive_path
        print(f"Using Drive path for glyphs: {glyphs_dir}")
    
    glyphs_dir = Path(glyphs_dir)
    if not glyphs_dir.exists():
        raise FileNotFoundError(f"Glyphs directory not found: {glyphs_dir}")
    
    # Transform for reference images (same as training: resize + normalize)
    transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    embeddings = {}
    
    print("Loading reference embeddings...")
    for glyph_folder in sorted(glyphs_dir.iterdir()):
        if not glyph_folder.is_dir():
            continue
        
        glyph_name = glyph_folder.name
        ideal_path = glyph_folder / "ideal.png"
        
        if not ideal_path.exists():
            print(f"  Warning: No ideal.png found for {glyph_name}, skipping")
            continue
        
        # Load and preprocess image
        img = Image.open(ideal_path).convert('RGB')
        img_array = np.array(img)
        
        # Apply transform
        transformed = transform(image=img_array)
        img_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Get embedding
        with torch.no_grad():
            embedding = model(img_tensor)
        
        # Ensure embedding is on the correct device
        embedding = embedding.squeeze(0).to(device)  # Remove batch dimension and ensure on device
        embeddings[glyph_name] = embedding
        print(f"  ✓ {glyph_name}")
    
    print(f"Loaded {len(embeddings)} reference embeddings")
    return embeddings


def step_a_binary_mask(image: np.ndarray, dilate_iterations: int = 0) -> np.ndarray:
    """
    Step A: Convert stone image to high-contrast black-and-white binary mask.
    Uses Adaptive Thresholding to handle varying lighting conditions.
    Optionally applies dilation to connect broken/stippled letters.
    
    Args:
        image: Input image (numpy array) - can be RGB, RGBA, or grayscale
        dilate_iterations: Number of dilation iterations to connect broken letters (default: 0, disabled)
    
    Returns:
        Binary mask (0 = background, 255 = foreground)
    """
    # Handle different image formats
    if len(image.shape) == 2:
        # Already grayscale
        gray = image
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            # RGBA - convert to RGB first
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:
            # RGB
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            # Other formats - try to convert to grayscale directly
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Apply adaptive thresholding
    # This handles varying lighting better than simple thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,  # Max value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
        cv2.THRESH_BINARY_INV,  # Invert: dark glyphs become white
        blockSize=11,  # Size of neighborhood for threshold calculation
        C=2  # Constant subtracted from mean
    )
    
    # Apply dilation to connect broken/stippled letters
    # This "fattens" the white lines by 2-3 pixels, fusing broken parts together
    if dilate_iterations > 0:
        # Create a small kernel for dilation (3x3 cross shape works well for connecting lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)
    
    return binary


def is_ruler_like(contour, x, y, w, h, img_height, img_width, min_area: int = 100):
    """
    Detect if a contour is likely a ruler or measuring device.
    
    Simple rule: If width > 3x height OR height > 3x width, it's likely a ruler.
    Safaitic letters are generally square-ish or circular, not long thin rectangles.
    
    Args:
        contour: OpenCV contour
        x, y, w, h: Bounding box coordinates
        img_height, img_width: Image dimensions
        min_area: Minimum area threshold
    
    Returns:
        True if contour is likely a ruler
    """
    # Simple aspect ratio check: width > 3x height OR height > 3x width
    aspect_ratio = w / h if h > 0 else 0
    reverse_aspect_ratio = h / w if w > 0 else 0
    
    # If aspect ratio is more than 3:1 in either direction, it's likely a ruler
    if aspect_ratio > 3.0 or reverse_aspect_ratio > 3.0:
        return True
    
    return False


def step_b_find_contours(binary_mask: np.ndarray, min_area: int = 30, 
                         filter_rulers: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Step B: Find all contours and return bounding boxes.
    
    Args:
        binary_mask: Binary mask from Step A
        min_area: Minimum area for a contour to be considered (pixels)
        filter_rulers: If True, filter out ruler-like objects (default: True)
    
    Returns:
        List of (x, y, width, height) bounding boxes
    """
    img_height, img_width = binary_mask.shape
    
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    rulers_filtered = 0
    
    for contour in contours:
        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Filter out rulers
        if filter_rulers and is_ruler_like(contour, x, y, w, h, img_height, img_width, min_area):
            rulers_filtered += 1
            continue
        
        # Filter by aspect ratio (glyphs are roughly square-ish, but some can be wider)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:  # More lenient for horizontal glyphs
            continue
        
        boxes.append((x, y, w, h))
    
    if rulers_filtered > 0:
        print(f"  Filtered out {rulers_filtered} ruler-like objects")
    
    # Sort by reading order (top-to-bottom, left-to-right)
    boxes.sort(key=lambda b: (b[1] // 50, b[0]))  # Group by approximate row
    
    return boxes


def step_c_classify_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    model: SafaiticSiameseNet,
    reference_embeddings: Dict[str, torch.Tensor],
    device: str = None,
    confidence_threshold: float = 1.0
) -> List[Tuple[int, int, int, int, str, float]]:
    """
    Step C: Classify each box using the Siamese model.
    Only keeps boxes with high confidence (low distance to a reference glyph).
    
    Args:
        image: Original RGB image
        boxes: List of (x, y, w, h) bounding boxes
        model: Trained Siamese model
        reference_embeddings: Dictionary of reference embeddings
        device: Device to run model on
        confidence_threshold: Maximum distance to consider a match (default: 1.0)
    
    Returns:
        List of (x, y, w, h, glyph_name, distance) for high-confidence matches
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Transform for candidate images (same as training)
    transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    classified_boxes = []
    
    print(f"Classifying {len(boxes)} candidate boxes...")
    
    for i, (x, y, w, h) in enumerate(boxes):
        # Bounds checking - ensure box coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        
        # Clamp coordinates to image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Skip if box is invalid after clamping
        if w <= 0 or h <= 0:
            print(f"  Warning: Box {i} invalid after bounds checking, skipping")
            continue
        
        # Extract region of interest
        roi = image[y:y+h, x:x+w]
        
        # Skip if ROI is too small
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        
        # Preprocess ROI
        try:
            transformed = transform(image=roi)
            roi_tensor = transformed['image'].unsqueeze(0).to(device)
        except Exception as e:
            print(f"  Warning: Failed to process box {i}: {e}")
            continue
        
        # Get embedding for ROI
        with torch.no_grad():
            roi_embedding = model(roi_tensor).squeeze(0)
        
        # Ensure ROI embedding is on correct device
        roi_embedding = roi_embedding.to(device)
        
        # Compare with all reference embeddings
        best_match = None
        best_distance = float('inf')
        
        for glyph_name, ref_embedding in reference_embeddings.items():
            # Ensure reference embedding is on same device as ROI embedding
            ref_embedding = ref_embedding.to(device)
            
            distance = euclidean_distance(
                roi_embedding.unsqueeze(0),
                ref_embedding.unsqueeze(0)
            ).item()
            
            if distance < best_distance:
                best_distance = distance
                best_match = glyph_name
        
        # Clear GPU cache periodically to prevent memory leaks
        if torch.cuda.is_available() and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
        
        # Only keep if confidence is high (distance is low)
        if best_distance <= confidence_threshold:
            classified_boxes.append((x, y, w, h, best_match, best_distance))
            print(f"  Box {i}: {best_match} (distance: {best_distance:.3f})")
    
    print(f"✓ Classified {len(classified_boxes)} high-confidence glyphs")
    return classified_boxes


def create_visualization(
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    all_boxes: List[Tuple[int, int, int, int]],
    classified_boxes: List[Tuple[int, int, int, int, str, float]],
    ground_truth: Optional[Dict[int, str]] = None,
    output_path: str = "stone_scan_visualization.png"
) -> None:
    """
    Create a 4-panel visualization showing the scanning process.
    
    Args:
        original_image: Original stone image
        binary_mask: Binary mask from Step A
        all_boxes: All candidate boxes from Step B (RED)
        classified_boxes: High-confidence classified boxes from Step C (GREEN)
        ground_truth: Optional dict mapping box index to expected glyph name (for YELLOW highlighting)
        output_path: Path to save visualization
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel 1: Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title("1. Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Binary Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(binary_mask, cmap='gray')
    ax2.set_title("2. Binary Mask (Digital Rubbing)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: All Candidates (RED boxes)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(original_image)
    for x, y, w, h in all_boxes:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
        ax3.add_patch(rect)
    ax3.set_title(f"3. All Candidates ({len(all_boxes)} RED boxes)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Final Transcription (GREEN boxes with labels)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(original_image)
    
    # Create mapping from box coordinates to index for ground truth
    box_to_idx = {}
    for idx, (x, y, w, h) in enumerate(all_boxes):
        box_to_idx[(x, y, w, h)] = idx
    
    for x, y, w, h, glyph_name, distance in classified_boxes:
        # Check if this box disagrees with ground truth
        box_key = (x, y, w, h)
        box_idx = box_to_idx.get(box_key, -1)
        
        if ground_truth and box_idx in ground_truth:
            expected = ground_truth[box_idx]
            if expected != glyph_name:
                # YELLOW box for disagreement
                color = 'yellow'
                label = f"{glyph_name}\n(expected: {expected})"
            else:
                color = 'green'
                label = glyph_name
        else:
            color = 'green'
            label = glyph_name
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax4.add_patch(rect)
        
        # Add label
        ax4.text(x, y - 5, label, fontsize=8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax4.set_title(f"4. Final Transcription ({len(classified_boxes)} GREEN boxes)", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle("Safaitic Stone Scanner - Step-by-Step Process", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def export_transcription(
    classified_boxes: List[Tuple[int, int, int, int, str, float]],
    output_path: str = "transcription.txt"
) -> str:
    """
    Export the reconstructed string to a text file.
    
    Args:
        classified_boxes: List of classified boxes
        output_path: Path to save transcription
    
    Returns:
        Reconstructed string
    """
    # Sort boxes by reading order (top-to-bottom, left-to-right)
    sorted_boxes = sorted(classified_boxes, key=lambda b: (b[1] // 50, b[0]))
    
    # Extract glyph names
    glyph_sequence = [box[4] for box in sorted_boxes]
    transcription = " ".join(glyph_sequence)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Safaitic Stone Transcription\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of glyphs detected: {len(glyph_sequence)}\n\n")
        f.write("Glyph sequence:\n")
        f.write(transcription + "\n\n")
        f.write("Detailed breakdown:\n")
        for i, (x, y, w, h, glyph_name, distance) in enumerate(sorted_boxes, 1):
            f.write(f"  {i}. {glyph_name} (distance: {distance:.3f}, pos: ({x}, {y}))\n")
    
    print(f"✓ Transcription saved to: {output_path}")
    return transcription


def load_ground_truth(csv_path: str) -> Dict[int, str]:
    """
    Load ground truth transcription from CSV file.
    Expected format: CSV with columns 'box_index' and 'glyph_name'
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Dictionary mapping box index to glyph name
    """
    if not Path(csv_path).exists():
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        ground_truth = {}
        for _, row in df.iterrows():
            idx = int(row['box_index'])
            glyph = str(row['glyph_name'])
            ground_truth[idx] = glyph
        print(f"✓ Loaded ground truth from: {csv_path}")
        return ground_truth
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        return {}


def scan_stone(
    image_path: str,
    model_checkpoint: str,
    glyphs_dir: str,
    output_dir: str = "scan_results",
    confidence_threshold: float = 1.0,
    min_contour_area: int = 30,
    ground_truth_csv: Optional[str] = None,
    colab_mode: bool = None,
    filter_rulers: bool = True,
    dilate_iterations: int = 0
) -> Dict:
    """
    Main function to scan a stone image and generate visualization.
    
    Args:
        image_path: Path to stone image
        model_checkpoint: Path to model checkpoint (Drive or local)
        glyphs_dir: Path to cleaned_glyphs directory
        output_dir: Directory to save outputs
        confidence_threshold: Maximum distance for a match (default: 1.0)
        min_contour_area: Minimum contour area in pixels (default: 30, lowered to detect small dots like ayn)
        ground_truth_csv: Optional path to CSV with ground truth
        colab_mode: Whether to use Colab Drive paths (auto-detected if None)
        filter_rulers: If True, filter out ruler-like objects (default: True)
        dilate_iterations: Number of dilation iterations to connect broken letters (default: 0, disabled)
    
    Returns:
        Dictionary with results
    """
    # Auto-detect Colab if not specified
    if colab_mode is None:
        colab_mode = detect_google_colab()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SAFAITIC STONE SCANNER")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Model: {model_checkpoint}")
    print(f"Glyphs: {glyphs_dir}")
    print()
    
    # Load image
    print("Loading stone image...")
    img_pil = Image.open(image_path)
    
    # Handle different image formats
    if img_pil.mode == 'RGBA':
        # Convert RGBA to RGB with white background
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[3])  # Use alpha channel as mask
        img_pil = background
    elif img_pil.mode != 'RGB':
        # Convert other modes (grayscale, palette, etc.) to RGB
        img_pil = img_pil.convert('RGB')
    
    image = np.array(img_pil)
    print(f"✓ Image loaded: {image.shape} (mode: {img_pil.mode})")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    model = load_model_from_drive(model_checkpoint, device=device)
    
    # Load reference embeddings
    print()
    reference_embeddings = load_reference_embeddings(model, glyphs_dir, device=device)
    
    # Step A: Binary Mask
    print("\n" + "=" * 60)
    print("STEP A: Creating Binary Mask")
    print("=" * 60)
    binary_mask = step_a_binary_mask(image, dilate_iterations=dilate_iterations)
    if dilate_iterations > 0:
        print(f"✓ Binary mask created (dilated {dilate_iterations} iterations to connect broken letters)")
    else:
        print("✓ Binary mask created")
    
    # Step B: Find Contours
    print("\n" + "=" * 60)
    print("STEP B: Finding All Candidates")
    print("=" * 60)
    all_boxes = step_b_find_contours(binary_mask, min_area=min_contour_area, filter_rulers=filter_rulers)
    print(f"✓ Found {len(all_boxes)} candidate boxes")
    
    # Step C: Classify Boxes
    print("\n" + "=" * 60)
    print("STEP C: Classifying with Siamese Model")
    print("=" * 60)
    classified_boxes = step_c_classify_boxes(
        image, all_boxes, model, reference_embeddings,
        device=device, confidence_threshold=confidence_threshold
    )
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_csv:
        ground_truth = load_ground_truth(ground_truth_csv)
    
    # Create visualization
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)
    viz_path = output_dir / "stone_scan_visualization.png"
    create_visualization(
        image, binary_mask, all_boxes, classified_boxes,
        ground_truth=ground_truth, output_path=str(viz_path)
    )
    
    # Export transcription
    print("\n" + "=" * 60)
    print("EXPORTING TRANSCRIPTION")
    print("=" * 60)
    transcription_path = output_dir / "transcription.txt"
    transcription = export_transcription(classified_boxes, output_path=str(transcription_path))
    
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Detected {len(classified_boxes)} glyphs")
    print(f"Results saved to: {output_dir}")
    
    return {
        'num_candidates': len(all_boxes),
        'num_detected': len(classified_boxes),
        'transcription': transcription,
        'classified_boxes': classified_boxes
    }


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan Safaitic stone images")
    parser.add_argument("image", help="Path to stone image")
    parser.add_argument("--checkpoint", default="safaitic_matcher.pth",
                       help="Path to model checkpoint (default: safaitic_matcher.pth)")
    parser.add_argument("--glyphs", default="cleaned_glyphs",
                       help="Path to cleaned_glyphs directory (default: cleaned_glyphs)")
    parser.add_argument("--output", default="scan_results",
                       help="Output directory (default: scan_results)")
    parser.add_argument("--confidence", type=float, default=1.0,
                       help="Confidence threshold (max distance, default: 1.0)")
    parser.add_argument("--min-area", type=int, default=30,
                       help="Minimum contour area in pixels (default: 30, lowered to detect small dots like ayn)")
    parser.add_argument("--ground-truth", help="Path to ground truth CSV file")
    parser.add_argument("--no-filter-rulers", action="store_true",
                       help="Disable ruler filtering (default: rulers are filtered)")
    parser.add_argument("--dilate", type=int, default=0,
                       help="Dilation iterations to connect broken letters (default: 0, disabled. Use 2-3 if letters are broken)")
    
    args = parser.parse_args()
    
    scan_stone(
        image_path=args.image,
        model_checkpoint=args.checkpoint,
        glyphs_dir=args.glyphs,
        output_dir=args.output,
        confidence_threshold=args.confidence,
        min_contour_area=args.min_area,
        ground_truth_csv=args.ground_truth,
        filter_rulers=not args.no_filter_rulers,
        dilate_iterations=args.dilate
    )


if __name__ == "__main__":
    main()

