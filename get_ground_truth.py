#!/usr/bin/env python3
"""
Get ground truth transliterations from database for stone images.
"""

import sqlite3
import sys
from pathlib import Path
import re

def get_transliteration_from_db(db_path, image_id):
    """
    Get transliteration for a specific image ID from database.
    
    Args:
        db_path: Path to SQLite database
        image_id: ID of the inscription (can be extracted from filename)
    
    Returns:
        Dictionary with transliteration and metadata
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Try to find by ID
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
    return None

def extract_id_from_filename(filename):
    """Extract ID from filename like stone_16820.jpg -> 16820"""
    match = re.search(r'stone_(\d+)', filename)
    if match:
        return match.group(1)
    return None

def get_all_transliterations(db_path, image_files):
    """
    Get transliterations for multiple image files.
    
    Args:
        db_path: Path to database
        image_files: List of image file paths
    
    Returns:
        Dictionary mapping filename -> transliteration data
    """
    results = {}
    
    for img_file in image_files:
        filename = Path(img_file).name
        image_id = extract_id_from_filename(filename)
        
        if image_id:
            data = get_transliteration_from_db(db_path, image_id)
            if data:
                results[filename] = data
            else:
                results[filename] = {'error': f'ID {image_id} not found in database'}
        else:
            results[filename] = {'error': 'Could not extract ID from filename'}
    
    return results

def main():
    """Main function."""
    db_path = "/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db"
    
    if len(sys.argv) > 1:
        # Get transliterations for specified files
        image_files = sys.argv[1:]
    else:
        # Get transliterations for all test images
        test_dir = Path("test_results")
        image_files = []
        for result_dir in test_dir.iterdir():
            if result_dir.is_dir() and result_dir.name.startswith("stone_"):
                # Find corresponding image
                img_file = Path("stone_images") / f"{result_dir.name}.jpg"
                if img_file.exists():
                    image_files.append(str(img_file))
    
    if not image_files:
        print("No images found. Usage: python get_ground_truth.py [image_files...]")
        return 1
    
    print("=" * 70)
    print("Fetching Ground Truth Transliterations")
    print("=" * 70)
    print(f"Database: {db_path}")
    print(f"Images: {len(image_files)}")
    print()
    
    results = get_all_transliterations(db_path, image_files)
    
    for filename, data in results.items():
        print(f"File: {filename}")
        print("-" * 70)
        
        if 'error' in data:
            print(f"  ✗ {data['error']}")
        else:
            print(f"  ID: {data['id']}")
            print(f"  Transliteration: {data['transliteration'] or 'N/A'}")
            print(f"  Translation: {data['translation'] or 'N/A'}")
            print(f"  Status: {data['status'] or 'N/A'}")
            
            # Save to file
            output_file = Path("test_results") / filename.replace('.jpg', '') / "ground_truth.txt"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Ground Truth for {filename}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"ID: {data['id']}\n")
                f.write(f"Transliteration: {data['transliteration'] or 'N/A'}\n")
                f.write(f"Translation: {data['translation'] or 'N/A'}\n")
                f.write(f"Status: {data['status'] or 'N/A'}\n")
                if data['metadata']:
                    f.write(f"\nMetadata:\n{data['metadata']}\n")
            
            print(f"  ✓ Saved to: {output_file}")
        
        print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

