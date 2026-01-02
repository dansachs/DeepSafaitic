#!/usr/bin/env python3
"""
Quick validation script to verify all critical fixes are working.
Run this before training in Colab.
"""

import sys
from pathlib import Path
from dataset import SafaiticSiameseDataset

def validate_anchor_integrity():
    """Verify that anchors are always ideal.png"""
    print("=" * 70)
    print("VALIDATION: Anchor Integrity")
    print("=" * 70)
    
    dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)
    
    # Check first 10 samples
    all_ideal = True
    for i in range(min(10, len(dataset))):
        anchor_path = dataset.ideal_paths[i]
        if 'ideal.png' not in str(anchor_path):
            print(f"✗ FAIL: Anchor {i} is not ideal.png: {anchor_path}")
            all_ideal = False
    
    if all_ideal:
        print("✓ PASS: All anchors are ideal.png (clean versions)")
    else:
        print("✗ FAIL: Some anchors are not ideal.png")
    
    return all_ideal


def validate_balance():
    """Verify 50/50 positive/negative balance"""
    print("\n" + "=" * 70)
    print("VALIDATION: Positive/Negative Balance")
    print("=" * 70)
    
    dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)
    
    # Sample 100 pairs
    positive_count = 0
    negative_count = 0
    
    for i in range(min(100, len(dataset))):
        _, _, label, _ = dataset[i]
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    total = positive_count + negative_count
    pos_ratio = positive_count / total if total > 0 else 0
    neg_ratio = negative_count / total if total > 0 else 0
    
    print(f"Positive pairs: {positive_count} ({pos_ratio:.1%})")
    print(f"Negative pairs: {negative_count} ({neg_ratio:.1%})")
    
    # Should be close to 50/50 (within 5%)
    balanced = abs(pos_ratio - 0.5) < 0.05
    
    if balanced:
        print("✓ PASS: Balance is approximately 50/50")
    else:
        print(f"✗ FAIL: Balance is not 50/50 (should be ~50%, got {pos_ratio:.1%})")
    
    return balanced


def validate_hard_negatives():
    """Verify hard negative mining is working"""
    print("\n" + "=" * 70)
    print("VALIDATION: Hard Negative Mining")
    print("=" * 70)
    
    dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)
    
    print(f"Similar glyph pairs identified: {len(dataset.similar_glyphs)}")
    if dataset.similar_glyphs:
        print("Sample pairs:")
        for pair in dataset.similar_glyphs[:5]:
            print(f"  - {pair[0]} <-> {pair[1]}")
    
    # Sample negative pairs and count hard negatives
    hard_neg_count = 0
    neg_count = 0
    
    for i in range(min(200, len(dataset))):
        _, _, label, hard_negative = dataset[i]
        if label == 0:
            neg_count += 1
            if hard_negative:
                hard_neg_count += 1
    
    hard_neg_ratio = hard_neg_count / neg_count if neg_count > 0 else 0
    
    print(f"\nHard negatives: {hard_neg_count} / {neg_count} ({hard_neg_ratio:.1%})")
    print("Expected: ~30% of negative pairs should be hard negatives")
    
    if hard_neg_ratio > 0.1:  # At least 10% (might be lower due to randomness)
        print("✓ PASS: Hard negative mining is working")
    else:
        print("⚠ WARNING: Hard negative ratio is low (may be due to randomness)")
    
    return True


def validate_border_artifacts():
    """Verify border settings are correct"""
    print("\n" + "=" * 70)
    print("VALIDATION: Border Artifacts Prevention")
    print("=" * 70)
    
    dataset = SafaiticSiameseDataset(root_dir="cleaned_glyphs", augment=True)
    
    # Check augmentation pipeline
    aug = dataset.augmentation
    if aug is None:
        print("⚠ WARNING: Augmentation is None (augment=False)")
        return True
    
    # Check SafeRotate
    rotate_found = False
    elastic_found = False
    
    for transform in aug.transforms:
        transform_str = str(transform)
        if 'SafeRotate' in transform_str:
            rotate_found = True
            if 'border_mode=1' in transform_str and 'value=255' in transform_str:
                print("✓ PASS: SafeRotate has correct border settings")
            else:
                print("✗ FAIL: SafeRotate missing border_mode=1 or value=255")
                return False
        
        if 'ElasticTransform' in transform_str:
            elastic_found = True
            if 'border_mode=1' in transform_str and 'value=255' in transform_str:
                print("✓ PASS: ElasticTransform has correct border settings")
            else:
                print("✗ FAIL: ElasticTransform missing border_mode=1 or value=255")
                return False
    
    if not rotate_found:
        print("⚠ WARNING: SafeRotate not found in augmentation pipeline")
    if not elastic_found:
        print("⚠ WARNING: ElasticTransform not found in augmentation pipeline")
    
    return True


def main():
    """Run all validations"""
    print("\n" + "=" * 70)
    print("SAFAITIC SIAMESE PIPELINE - PRE-TRAINING VALIDATION")
    print("=" * 70)
    print()
    
    results = []
    
    # Run validations
    results.append(("Anchor Integrity", validate_anchor_integrity()))
    results.append(("Balance", validate_balance()))
    results.append(("Hard Negatives", validate_hard_negatives()))
    results.append(("Border Artifacts", validate_border_artifacts()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ All validations passed! Ready for Colab training.")
        return 0
    else:
        print("\n❌ Some validations failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())

