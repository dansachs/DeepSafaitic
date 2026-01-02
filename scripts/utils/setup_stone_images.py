#!/usr/bin/env python3
"""
Helper script to set up stone images directory from SQL dataset.
"""

import sys
from pathlib import Path
import shutil


def setup_stone_images(sql_path, method='symlink'):
    """
    Set up stone_images directory from SQL dataset.
    
    Args:
        sql_path: Path to SQL directory containing stone images
        method: 'symlink', 'copy', or 'check'
    """
    sql_path = Path(sql_path)
    stone_images_dir = Path("stone_images")
    
    if not sql_path.exists():
        print(f"✗ Error: SQL path does not exist: {sql_path}")
        return False
    
    # Find image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(sql_path.glob(f"*{ext}"))
    
    if len(image_files) == 0:
        print(f"✗ No image files found in: {sql_path}")
        print(f"  Looking for: {', '.join(image_extensions)}")
        return False
    
    print(f"Found {len(image_files)} image files in SQL directory")
    
    if method == 'check':
        print(f"\nSQL directory: {sql_path}")
        print(f"Images found: {len(image_files)}")
        print("\nFirst 5 images:")
        for img in image_files[:5]:
            print(f"  - {img.name}")
        if len(image_files) > 5:
            print(f"  ... and {len(image_files) - 5} more")
        return True
    
    # Create stone_images directory
    stone_images_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created: {stone_images_dir}")
    
    if method == 'symlink':
        print(f"Creating symlinks...")
        created = 0
        for img_file in image_files:
            link_path = stone_images_dir / img_file.name
            if link_path.exists():
                if link_path.is_symlink():
                    print(f"  Already linked: {img_file.name}")
                    continue
                else:
                    print(f"  Warning: {img_file.name} already exists (not a symlink), skipping")
                    continue
            
            try:
                link_path.symlink_to(img_file)
                created += 1
            except Exception as e:
                print(f"  Error linking {img_file.name}: {e}")
        
        print(f"✓ Created {created} symlinks")
        return True
    
    elif method == 'copy':
        print(f"Copying files...")
        copied = 0
        for img_file in image_files:
            dest_path = stone_images_dir / img_file.name
            if dest_path.exists():
                print(f"  Already exists: {img_file.name}, skipping")
                continue
            
            try:
                shutil.copy2(img_file, dest_path)
                copied += 1
            except Exception as e:
                print(f"  Error copying {img_file.name}: {e}")
        
        print(f"✓ Copied {copied} files")
        return True
    
    return False


def main():
    """Interactive setup."""
    print("=" * 60)
    print("Stone Images Setup Helper")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        sql_path = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else 'symlink'
    else:
        sql_path = input("Enter path to SQL stone images directory: ").strip()
        if not sql_path:
            print("No path provided. Exiting.")
            return
        
        print("\nChoose method:")
        print("  1. Symlink (recommended - saves space, always current)")
        print("  2. Copy (independent copy, uses disk space)")
        print("  3. Just check (don't set up, just verify)")
        choice = input("\nChoice (1/2/3): ").strip()
        
        method_map = {'1': 'symlink', '2': 'copy', '3': 'check'}
        method = method_map.get(choice, 'symlink')
    
    print()
    success = setup_stone_images(sql_path, method)
    
    if success:
        print()
        print("=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        if method != 'check':
            print(f"\nStone images ready in: stone_images/")
            print("\nTest with:")
            print(f"  python scan_and_visualize.py stone_images/<image_name>.jpg \\")
            print(f"      --checkpoint models/safaitic_matcher.pth \\")
            print(f"      --glyphs cleaned_glyphs")
    else:
        print("\nSetup failed. Check the path and try again.")


if __name__ == "__main__":
    main()

