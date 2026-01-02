#!/usr/bin/env python3
"""
Safaitic Stone Scanner - Detection Only Mode

Focuses on detecting glyph locations without classification.
Perfect the detection first, then add text direction, then character recognition.
"""

import cv2
import numpy as np
import sqlite3
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict
from datetime import datetime


def step_a_binary_mask(image: np.ndarray, dilate_iterations: int = 0) -> np.ndarray:
    """
    Step A: Convert stone image to high-contrast black-and-white binary mask.
    Uses Adaptive Thresholding to handle varying lighting conditions.
    
    Args:
        image: Input image (numpy array) - can be RGB, RGBA, or grayscale
        dilate_iterations: Number of dilation iterations (default: 0, disabled)
    
    Returns:
        Binary mask (0 = background, 255 = foreground)
    """
    # Handle different image formats
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11,
        C=2
    )
    
    # Optional dilation
    if dilate_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)
    
    return binary


def is_ruler_like(x, y, w, h):
    """
    Simple ruler detection: if width > 3x height OR height > 3x width, it's likely a ruler.
    """
    aspect_ratio = w / h if h > 0 else 0
    reverse_aspect_ratio = h / w if w > 0 else 0
    
    if aspect_ratio > 3.0 or reverse_aspect_ratio > 3.0:
        return True
    return False


def step_b_find_contours(binary_mask: np.ndarray, min_area: int = 30, 
                         filter_rulers: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Step B: Find all contours and return bounding boxes.
    Focus: Just detect glyph locations, don't classify them.
    
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
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    boxes = []
    rulers_filtered = 0
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Filter out rulers
        if filter_rulers and is_ruler_like(x, y, w, h):
            rulers_filtered += 1
            continue
        
        # Filter by aspect ratio (glyphs are roughly square-ish, but some can be wider)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            continue
        
        boxes.append((x, y, w, h))
    
    if rulers_filtered > 0:
        print(f"  Filtered out {rulers_filtered} ruler-like objects")
    
    # Sort by reading order (top-to-bottom, left-to-right)
    # TODO: Improve reading order detection (text direction)
    boxes.sort(key=lambda b: (b[1] // 50, b[0]))
    
    return boxes


def calculate_glyph_quality(image: np.ndarray, x: int, y: int, w: int, h: int) -> Dict:
    """
    Calculate quality metrics for a detected glyph box.
    Helps assess if it's a good detection.
    
    Args:
        image: Original image
        x, y, w, h: Bounding box coordinates
    
    Returns:
        Dictionary with quality metrics
    """
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {'valid': False}
    
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray_roi = roi
    
    # Calculate metrics
    mean_intensity = np.mean(gray_roi)
    std_intensity = np.std(gray_roi)
    area = w * h
    aspect_ratio = w / h if h > 0 else 0
    
    # Quality score (higher = better)
    # More contrast (higher std) = better
    # Reasonable size = better
    quality_score = std_intensity * (1.0 / (1.0 + abs(aspect_ratio - 1.0)))  # Prefer square-ish
    
    return {
        'valid': True,
        'area': area,
        'aspect_ratio': aspect_ratio,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'quality_score': quality_score
    }


def create_visualization(
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    detected_boxes: List[Tuple[int, int, int, int]],
    output_path: str = "detection_visualization.png"
) -> None:
    """
    Create visualization showing detection results.
    
    Args:
        original_image: Original stone image
        binary_mask: Binary mask from Step A
        detected_boxes: List of detected glyph boxes
        output_path: Path to save visualization
    """
    fig = plt.figure(figsize=(20, 12))
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
    
    # Panel 3: Detected Glyphs (GREEN boxes)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(original_image)
    for i, (x, y, w, h) in enumerate(detected_boxes):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        ax3.add_patch(rect)
        # Add box number
        ax3.text(x, y - 5, str(i + 1), fontsize=8, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax3.set_title(f"3. Detected Glyphs ({len(detected_boxes)} GREEN boxes)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Panel 4: Detection Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    if detected_boxes:
        areas = [w * h for x, y, w, h in detected_boxes]
        aspect_ratios = [w / h if h > 0 else 0 for x, y, w, h in detected_boxes]
        
        stats_text = f"""
Detection Statistics
{'=' * 40}

Total Glyphs Detected: {len(detected_boxes)}

Box Sizes:
  Min area: {min(areas):.0f} px²
  Max area: {max(areas):.0f} px²
  Avg area: {np.mean(areas):.0f} px²

Aspect Ratios:
  Min: {min(aspect_ratios):.2f}
  Max: {max(aspect_ratios):.2f}
  Avg: {np.mean(aspect_ratios):.2f}

Next Steps:
  1. Verify all glyphs are detected
  2. Check for false positives
  3. Improve reading order detection
  4. Add character recognition
"""
    else:
        stats_text = "No glyphs detected.\n\nCheck:\n- Binary mask quality\n- Min area threshold\n- Image quality"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax4.transAxes)
    
    plt.suptitle("Safaitic Stone Scanner - Detection Only Mode", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def get_transliteration_from_db(db_path: str, image_id: str) -> Optional[Dict]:
    """
    Get transliteration from database for a given image ID.
    
    Args:
        db_path: Path to SQLite database
        image_id: ID of the inscription
    
    Returns:
        Dictionary with transliteration data, or None if not found
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, image_url, transliteration, translation, metadata, status
            FROM inscriptions
            WHERE id = ?
        """, (image_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'image_url': result[1],
                'transliteration': result[2],
                'translation': result[3],
                'metadata': result[4],
                'status': result[5]
            }
    except Exception as e:
        print(f"Warning: Could not query database: {e}")
    
    return None


def extract_id_from_filename(filename: str) -> Optional[str]:
    """Extract ID from filename like stone_16820.jpg -> 16820"""
    match = re.search(r'stone_(\d+)', filename)
    return match.group(1) if match else None


def export_detection_results(
    detected_boxes: List[Tuple[int, int, int, int]],
    image_path: str,
    db_path: str = None,
    output_path: str = "detection_results.txt"
) -> None:
    """
    Export detection results to text file, including transliteration from database.
    
    Args:
        detected_boxes: List of detected boxes
        image_path: Path to the stone image
        db_path: Path to SQLite database
        output_path: Path to save results
    """
    # Try to get transliteration from database
    image_id = extract_id_from_filename(Path(image_path).name)
    transliteration_data = None
    
    if image_id:
        transliteration_data = get_transliteration_from_db(db_path, image_id)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Safaitic Glyph Detection Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Add transliteration from database if available
        if transliteration_data:
            f.write("Ground Truth (from database):\n")
            f.write("-" * 50 + "\n")
            f.write(f"ID: {transliteration_data['id']}\n")
            f.write(f"Transliteration: {transliteration_data['transliteration'] or 'N/A'}\n")
            f.write(f"Translation: {transliteration_data['translation'] or 'N/A'}\n")
            f.write(f"Status: {transliteration_data['status'] or 'N/A'}\n")
            f.write("\n")
        
        f.write("Detection Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total glyphs detected: {len(detected_boxes)}\n\n")
        f.write("Detection Details:\n")
        f.write(f"{'Box':<6} {'X':<8} {'Y':<8} {'Width':<8} {'Height':<8} {'Area':<8}\n")
        f.write("-" * 50 + "\n")
        
        for i, (x, y, w, h) in enumerate(detected_boxes, 1):
            area = w * h
            f.write(f"{i:<6} {x:<8} {y:<8} {w:<8} {h:<8} {area:<8}\n")
        
        f.write("\n")
        f.write("Next Steps:\n")
        f.write("  1. Verify all glyphs are detected\n")
        f.write("  2. Check for false positives\n")
        f.write("  3. Improve reading order detection\n")
        f.write("  4. Add character recognition\n")
    
    print(f"✓ Detection results saved to: {output_path}")


def detect_glyphs(
    image_path: str,
    output_dir: str = None,
    min_area: int = 30,
    filter_rulers: bool = True,
    dilate_iterations: int = 0,
    db_path: str = None
) -> Dict:
    """
    Main function to detect glyphs in stone image (no classification).
    
    Args:
        image_path: Path to stone image
        output_dir: Directory to save outputs (if None, creates timestamped folder)
        min_area: Minimum contour area in pixels
        filter_rulers: If True, filter out ruler-like objects
        dilate_iterations: Number of dilation iterations
        db_path: Path to SQLite database
    
    Returns:
        Dictionary with detection results
    """
    # Create timestamped output directory if not specified
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_name = Path(image_path).stem
        output_dir = f"outputs/detection_results/{image_name}_{timestamp}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SAFAITIC GLYPH DETECTION (No Classification)")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load image
    print("Loading stone image...")
    img_pil = Image.open(image_path)
    
    if img_pil.mode == 'RGBA':
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[3])
        img_pil = background
    elif img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    image = np.array(img_pil)
    print(f"✓ Image loaded: {image.shape} (mode: {img_pil.mode})")
    
    # Step A: Binary Mask
    print("\n" + "=" * 60)
    print("STEP A: Creating Binary Mask")
    print("=" * 60)
    binary_mask = step_a_binary_mask(image, dilate_iterations=dilate_iterations)
    if dilate_iterations > 0:
        print(f"✓ Binary mask created (dilated {dilate_iterations} iterations)")
    else:
        print("✓ Binary mask created")
    
    # Step B: Find Contours (Detection Only)
    print("\n" + "=" * 60)
    print("STEP B: Detecting Glyph Locations")
    print("=" * 60)
    detected_boxes = step_b_find_contours(binary_mask, min_area=min_area, filter_rulers=filter_rulers)
    print(f"✓ Detected {len(detected_boxes)} potential glyph locations")
    
    # Create visualization
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)
    viz_path = output_dir / "scanvisualization.png"
    create_visualization(image, binary_mask, detected_boxes, output_path=str(viz_path))
    
    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    results_path = output_dir / "detection_results.txt"
    export_detection_results(
        detected_boxes, 
        image_path=image_path,
        db_path=db_path,
        output_path=str(results_path)
    )
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)
    print(f"Detected {len(detected_boxes)} glyph locations")
    print(f"Results saved to: {output_dir}")
    print()
    print("Next Steps:")
    print("  1. Review visualization to verify detections")
    print("  2. Adjust min_area if needed (currently: {min_area})")
    print("  3. Improve reading order detection")
    print("  4. Add character recognition later")
    
    return {
        'num_detected': len(detected_boxes),
        'boxes': detected_boxes,
        'image_shape': image.shape
    }


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect Safaitic glyphs (no classification)")
    parser.add_argument("image", help="Path to stone image")
    parser.add_argument("--output", default=None,
                       help="Output directory (default: auto-generated with timestamp)")
    parser.add_argument("--min-area", type=int, default=30,
                       help="Minimum contour area in pixels (default: 30)")
    parser.add_argument("--no-filter-rulers", action="store_true",
                       help="Disable ruler filtering")
    parser.add_argument("--dilate", type=int, default=0,
                       help="Dilation iterations (default: 0, disabled)")
    
    args = parser.parse_args()
    
    detect_glyphs(
        image_path=args.image,
        output_dir=args.output,
        min_area=args.min_area,
        filter_rulers=not args.no_filter_rulers,
        dilate_iterations=args.dilate
    )


if __name__ == "__main__":
    main()

