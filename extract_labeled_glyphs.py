#!/usr/bin/env python3
"""
Extract labeled glyph ROIs from stone images using ground truth transliterations.

This creates a training dataset of real stone glyphs that can be used to fine-tune the model.
"""

import sqlite3
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import re
import json

def get_transliteration(db_path, image_id):
    """Get transliteration from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT transliteration FROM inscriptions WHERE id = ?", (image_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def extract_id_from_filename(filename):
    """Extract ID from filename."""
    match = re.search(r'stone_(\d+)', filename)
    return match.group(1) if match else None

def safaitic_to_glyph_names(transliteration):
    """
    Convert Safaitic transliteration to glyph names.
    This is a mapping - you may need to adjust based on your transliteration system.
    """
    # Basic mapping - expand this based on your transliteration system
    mapping = {
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'b': 'b',
        't': 't',
        'r': 'r',
        'h': 'h',
        'ʿ': 'ayn',  # ayin
        'ʾ': 'alif',  # aleph
        'ḥ': 'h_dot',
        'ḫ': 'kh',
        'ġ': 'gh',
        'š': 's2',
        'ṣ': 's_dot',
        'ḍ': 'd_dot',
        'ṭ': 't_dot',
        'ẓ': 'z_dot',
        # Add more mappings as needed
    }
    
    glyphs = []
    for char in transliteration:
        if char == ' ':
            continue
        # Try direct mapping first
        if char in mapping:
            glyphs.append(mapping[char])
        # Try lowercase
        elif char.lower() in mapping:
            glyphs.append(mapping[char.lower()])
        # Try to match common patterns
        elif char in 'abcdefghijklmnopqrstuvwxyz':
            glyphs.append(char)
        # Skip unknown characters for now
        else:
            print(f"  Warning: Unknown character '{char}' in transliteration")
    
    return glyphs

def extract_glyph_rois(image_path, boxes, labels, output_dir):
    """
    Extract glyph ROIs and save them with labels.
    
    Args:
        image_path: Path to stone image
        boxes: List of (x, y, w, h) bounding boxes
        labels: List of glyph names corresponding to boxes
        output_dir: Directory to save extracted glyphs
    """
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.array(Image.open(image_path).convert('RGB'))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted = []
    
    for i, ((x, y, w, h), label) in enumerate(zip(boxes, labels)):
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            continue
        
        # Save ROI
        glyph_dir = output_dir / label
        glyph_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{Path(image_path).stem}_glyph_{i}.png"
        filepath = glyph_dir / filename
        
        cv2.imwrite(str(filepath), roi)
        extracted.append({
            'label': label,
            'filepath': str(filepath),
            'box': (x, y, w, h)
        })
    
    return extracted

def process_stone_image(image_path, db_path, output_base_dir="stone_glyph_dataset"):
    """
    Process a stone image: detect boxes, get ground truth, extract labeled ROIs.
    """
    image_path = Path(image_path)
    image_id = extract_id_from_filename(image_path.name)
    
    if not image_id:
        print(f"Could not extract ID from {image_path.name}")
        return None
    
    # Get ground truth
    transliteration = get_transliteration(db_path, image_id)
    if not transliteration:
        print(f"No transliteration found for ID {image_id}")
        return None
    
    print(f"Processing {image_path.name}")
    print(f"  ID: {image_id}")
    print(f"  Transliteration: {transliteration}")
    
    # Convert to glyph names
    glyph_labels = safaitic_to_glyph_names(transliteration)
    print(f"  Glyphs: {glyph_labels}")
    
    # Load image and detect boxes (using existing detection)
    from scan_and_visualize import step_a_binary_mask, step_b_find_contours
    
    image = np.array(Image.open(image_path).convert('RGB'))
    binary_mask = step_a_binary_mask(image)
    boxes = step_b_find_contours(binary_mask, min_area=50)
    
    print(f"  Detected {len(boxes)} boxes, need {len(glyph_labels)} glyphs")
    
    # For now, just take first N boxes (you'll need to manually align or use better matching)
    # TODO: Implement better box-to-glyph matching
    if len(boxes) >= len(glyph_labels):
        # Take first N boxes
        selected_boxes = boxes[:len(glyph_labels)]
        selected_labels = glyph_labels
    else:
        # Not enough boxes - skip or pad
        print(f"  Warning: Not enough boxes ({len(boxes)} < {len(glyph_labels)})")
        selected_boxes = boxes
        selected_labels = glyph_labels[:len(boxes)]
    
    # Extract ROIs
    output_dir = Path(output_base_dir) / image_path.stem
    extracted = extract_glyph_rois(image_path, selected_boxes, selected_labels, output_dir)
    
    print(f"  ✓ Extracted {len(extracted)} labeled glyphs")
    print()
    
    return {
        'image_id': image_id,
        'transliteration': transliteration,
        'glyphs': glyph_labels,
        'extracted': extracted
    }

def main():
    """Main function."""
    import sys
    
    db_path = "/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db"
    output_dir = "stone_glyph_dataset"
    
    if len(sys.argv) > 1:
        image_files = sys.argv[1:]
    else:
        # Process all test images
        image_files = list(Path("stone_images").glob("stone_*.jpg"))[:4]
    
    print("=" * 70)
    print("Extracting Labeled Glyph ROIs from Stone Images")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Output: {output_dir}")
    print()
    
    all_results = []
    
    for image_file in image_files:
        result = process_stone_image(image_file, db_path, output_dir)
        if result:
            all_results.append(result)
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Processed: {len(all_results)} images")
    
    total_glyphs = sum(len(r['extracted']) for r in all_results)
    print(f"Extracted: {total_glyphs} labeled glyph ROIs")
    print(f"Saved to: {output_dir}/")
    print()
    
    # Save metadata
    metadata_file = Path(output_dir) / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")
    
    print()
    print("Next steps:")
    print("1. Review extracted glyphs in stone_glyph_dataset/")
    print("2. Manually verify/correct labels if needed")
    print("3. Use this data to fine-tune the model")

if __name__ == "__main__":
    main()

