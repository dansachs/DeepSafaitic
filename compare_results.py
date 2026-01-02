#!/usr/bin/env python3
"""
Compare model predictions with ground truth from database.
"""

import sys
from pathlib import Path
import re

def parse_transcription_file(transcription_path):
    """Parse transcription.txt file to get predicted sequence."""
    try:
        with open(transcription_path, 'r') as f:
            content = f.read()
        
        # Extract glyph sequence
        match = re.search(r'Glyph sequence:\s*(.+)', content)
        if match:
            sequence = match.group(1).strip()
            glyphs = sequence.split()
            return glyphs
    except:
        pass
    return []

def parse_ground_truth_file(gt_path):
    """Parse ground_truth.txt file to get actual transliteration."""
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        match = re.search(r'Transliteration:\s*(.+)', content)
        if match:
            transliteration = match.group(1).strip()
            if transliteration != 'N/A':
                # Split into words/glyphs (Safaitic transliteration uses spaces)
                glyphs = transliteration.split()
                return glyphs
    except:
        pass
    return []

def compare_results(test_dir="test_results"):
    """Compare all results with ground truth."""
    test_dir = Path(test_dir)
    
    print("=" * 70)
    print("Model Performance Analysis")
    print("=" * 70)
    print()
    
    results = []
    
    for result_folder in sorted(test_dir.iterdir()):
        if not result_folder.is_dir() or not result_folder.name.startswith("stone_"):
            continue
        
        stone_id = result_folder.name
        transcription_file = result_folder / "transcription.txt"
        ground_truth_file = result_folder / "ground_truth.txt"
        
        if not transcription_file.exists() or not ground_truth_file.exists():
            continue
        
        predicted = parse_transcription_file(transcription_file)
        actual = parse_ground_truth_file(ground_truth_file)
        
        results.append({
            'stone_id': stone_id,
            'predicted': predicted,
            'actual': actual,
            'predicted_str': ' '.join(predicted),
            'actual_str': ' '.join(actual)
        })
    
    # Print comparison
    for r in results:
        print(f"Stone: {r['stone_id']}")
        print("-" * 70)
        print(f"  Ground Truth: {r['actual_str']}")
        print(f"  Model Output: {r['predicted_str']}")
        print(f"  Predicted: {len(r['predicted'])} glyphs")
        print(f"  Actual: {len(r['actual'])} glyphs")
        
        # Simple accuracy (exact match)
        if r['predicted'] == r['actual']:
            print("  ✓ Perfect match!")
        else:
            print("  ✗ Mismatch")
        
        print()
    
    # Summary statistics
    total = len(results)
    if total > 0:
        exact_matches = sum(1 for r in results if r['predicted'] == r['actual'])
        avg_predicted = sum(len(r['predicted']) for r in results) / total
        avg_actual = sum(len(r['actual']) for r in results) / total
        
        print("=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Total images: {total}")
        print(f"Exact matches: {exact_matches} ({exact_matches/total*100:.1f}%)")
        print(f"Average predicted glyphs: {avg_predicted:.1f}")
        print(f"Average actual glyphs: {avg_actual:.1f}")
        print()
    
    return results

if __name__ == "__main__":
    compare_results()

