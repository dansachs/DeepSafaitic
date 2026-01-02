#!/usr/bin/env python3
"""
Validation script to check that all Safaitic alphabet assets are present
and that directory names match the expected folder name list.
"""

import os
from pathlib import Path

# Configuration
OUTPUT_DIR = "safaitic_assets"

# Expected folder names (from the mapping table)
EXPECTED_FOLDER_NAMES = [
    'alif',
    'b',
    't',
    'th',
    'g',
    'h_dot',
    'kh',
    'd',
    'dh',
    'r',
    'z',
    's1',
    's2',
    's_dot',
    'd_dot',
    't_dot',
    'z_dot',
    'ayn',
    'gh',
    'f',
    'q',
    'k',
    'l',
    'm',
    'n',
    'h',
    'w',
    'y',
]

# Mapping from common scraper-generated names to expected folder names
# This maps what the scraper might create to what we actually want
FOLDER_NAME_MAPPING = {
    # Already correct - no mapping needed:
    # 'alif': 'alif',
    # 'ayn': 'ayn',
    # 'b': 'b',
    # 'd': 'd',
    # 'dh': 'dh',
    # 'f': 'f',
    # 'g': 'g',
    # 'gh': 'gh',
    # 'h': 'h',
    # 'h_dot': 'h_dot',
    # 'k': 'k',
    # 'kh': 'kh',
    # 'l': 'l',
    # 'm': 'm',
    # 'n': 'n',
    # 'q': 'q',
    # 'r': 'r',
    # 's1': 's1',
    # 's2': 's2',
    # 's_dot': 's_dot',
    # 't': 't',
    # 't_dot': 't_dot',
    # 'th': 'th',
    # 'w': 'w',
    # 'y': 'y',
    # 'z': 'z',
    # 'z_dot': 'z_dot',
    
    # Mappings needed:
    'ha': 'h',              # ha -> h
    'sin': 's1',            # sin -> s1
    'shin': 's2',           # shin -> s2
    'ba': 'b',              # ba -> b
    'ta': 't',              # ta -> t
    'tha': 'th',            # tha -> th
    'gim': 'g',             # gim -> g
    'hha': 'h_dot',         # hha -> h_dot
    'kha': 'kh',            # kha -> kh
    'dal': 'd',             # dal -> d
    'dhal': 'dh',           # dhal -> dh
    'ra': 'r',              # ra -> r
    'za': 'z',              # za -> z
    'sad': 's_dot',         # sad -> s_dot
    's_sad': 's_dot',       # s_sad -> s_dot
    'dad': 'd_dot',         # dad -> d_dot
    'd_dad': 'd_dot',       # d_dad -> d_dot (if d_dad exists, rename to d_dot)
    'tta': 't_dot',         # tta -> t_dot
    'zayn': 'z_dot',        # zayn -> z_dot
    'ghayn': 'gh',          # ghayn -> gh
    'fa': 'f',              # fa -> f
    'qaf': 'q',             # qaf -> q
    'kaf': 'k',             # kaf -> k
    'lam': 'l',             # lam -> l
    'mim': 'm',             # mim -> m
    'nun': 'n',             # nun -> n
    'waw': 'w',             # waw -> w
    'ya': 'y',              # ya -> y
}


def check_asset_directory(directory_path, expected_files=['ideal.svg', 'square.svg']):
    """
    Check if a directory exists and contains the expected files.
    Returns (exists, has_all_files, missing_files, extra_files)
    """
    dir_path = Path(directory_path)
    exists = dir_path.exists() and dir_path.is_dir()
    
    if not exists:
        return False, False, expected_files.copy(), []
    
    # Check for expected files
    missing_files = []
    for filename in expected_files:
        file_path = dir_path / filename
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(filename)
    
    # Find extra files (not expected)
    extra_files = []
    if exists:
        for item in dir_path.iterdir():
            if item.is_file() and item.name not in expected_files:
                extra_files.append(item.name)
    
    has_all_files = len(missing_files) == 0
    return exists, has_all_files, missing_files, extra_files


def rename_directory(old_path, new_path):
    """
    Rename a directory from old_path to new_path.
    Returns True if successful, False otherwise.
    """
    try:
        old_dir = Path(old_path)
        new_dir = Path(new_path)
        
        if not old_dir.exists():
            return False
        
        if new_dir.exists():
            print(f"    Warning: Target directory already exists: {new_dir}")
            return False
        
        old_dir.rename(new_dir)
        return True
    except Exception as e:
        print(f"    Error renaming {old_path} to {new_path}: {e}")
        return False


def main():
    """Main validation function."""
    print("=" * 70)
    print("Safaitic Assets Validation Script")
    print("=" * 70)
    print()
    
    print(f"Checking {len(EXPECTED_FOLDER_NAMES)} expected folder names...")
    print()
    
    # First, rename any directories that don't match expected names
    print("Step 1: Renaming directories to match expected names...")
    print("-" * 70)
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"ERROR: Output directory '{OUTPUT_DIR}' does not exist!")
        return 1
    
    actual_dirs = {d for d in os.listdir(OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(OUTPUT_DIR, d))}
    expected_dirs = set(EXPECTED_FOLDER_NAMES)
    
    renamed_count = 0
    merged_count = 0
    for actual_dir in sorted(actual_dirs):
        if actual_dir not in expected_dirs:
            # Check if this directory should be renamed
            if actual_dir in FOLDER_NAME_MAPPING:
                expected_name = FOLDER_NAME_MAPPING[actual_dir]
                old_path = os.path.join(OUTPUT_DIR, actual_dir)
                new_path = os.path.join(OUTPUT_DIR, expected_name)
                
                # Check if target already exists
                if os.path.exists(new_path):
                    # Try to merge: move files from old directory to new directory
                    old_dir = Path(old_path)
                    new_dir = Path(new_path)
                    files_moved = 0
                    
                    if old_dir.exists() and old_dir.is_dir():
                        for item in old_dir.iterdir():
                            if item.is_file():
                                target_file = new_dir / item.name
                                if not target_file.exists():
                                    try:
                                        item.rename(target_file)
                                        files_moved += 1
                                    except Exception as e:
                                        print(f"    Error moving {item.name}: {e}")
                    
                    # Remove old directory if empty
                    try:
                        if old_dir.exists() and not any(old_dir.iterdir()):
                            old_dir.rmdir()
                            print(f"  ✓ {actual_dir:20s} -> {expected_name:20s} [MERGED & REMOVED - {files_moved} files moved]")
                            merged_count += 1
                        elif files_moved > 0:
                            print(f"  ⚠ {actual_dir:20s} -> {expected_name:20s} [MERGED - {files_moved} files moved, directory not empty]")
                        else:
                            print(f"  ⚠ {actual_dir:20s} -> {expected_name:20s} [TARGET EXISTS - NO FILES TO MERGE]")
                    except Exception as e:
                        if files_moved > 0:
                            print(f"  ⚠ {actual_dir:20s} -> {expected_name:20s} [MERGED - {files_moved} files moved, could not remove old dir]")
                else:
                    if rename_directory(old_path, new_path):
                        print(f"  ✓ {actual_dir:20s} -> {expected_name:20s} [RENAMED]")
                        renamed_count += 1
                    else:
                        print(f"  ✗ {actual_dir:20s} -> {expected_name:20s} [RENAME FAILED]")
            else:
                # Directory not in mapping - will be reported as unexpected later
                pass
    
    if renamed_count > 0 or merged_count > 0:
        print(f"\nRenamed {renamed_count} directory(ies), merged {merged_count} directory(ies).")
    else:
        print("No directories needed renaming.")
    
    # Re-check actual directories after renaming
    actual_dirs = {d for d in os.listdir(OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(OUTPUT_DIR, d))}
    
    # Check each expected folder
    print("\nStep 2: Validating expected folders...")
    print("-" * 70)
    
    results = {
        'complete': [],           # Has directory and both files
        'missing_directory': [],  # Directory doesn't exist
        'missing_files': [],      # Directory exists but missing files
        'extra_directories': []   # Directories that aren't expected
    }
    
    for folder_name in EXPECTED_FOLDER_NAMES:
        dir_path = os.path.join(OUTPUT_DIR, folder_name)
        exists, has_all_files, missing_files, extra_files = check_asset_directory(dir_path)
        
        status_icon = "✓" if (exists and has_all_files) else "✗"
        print(f"{status_icon} {folder_name:20s}", end="")
        
        if not exists:
            print(" [MISSING DIRECTORY]")
            results['missing_directory'].append(folder_name)
        elif not has_all_files:
            print(f" [MISSING FILES: {', '.join(missing_files)}]")
            results['missing_files'].append((folder_name, missing_files))
        else:
            print(" [OK]")
            results['complete'].append(folder_name)
        
        if extra_files:
            print(f"    Warning: Extra files found: {', '.join(extra_files)}")
    
    # Check for unexpected directories
    print("\n" + "-" * 70)
    print("Step 3: Checking for unexpected directories...")
    
    expected_dirs = set(EXPECTED_FOLDER_NAMES)
    unexpected_dirs = actual_dirs - expected_dirs
    
    if unexpected_dirs:
        print(f"Found {len(unexpected_dirs)} unexpected directory(ies):")
        removed_empty = 0
        for dir_name in sorted(unexpected_dirs):
            dir_path = Path(OUTPUT_DIR) / dir_name
            # Check if directory is empty
            if dir_path.exists() and dir_path.is_dir():
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        print(f"  - {dir_name} [EMPTY - REMOVED]")
                        removed_empty += 1
                    else:
                        print(f"  - {dir_name} [NOT EMPTY - KEEPING]")
                        results['extra_directories'].append(dir_name)
                except Exception as e:
                    print(f"  - {dir_name} [ERROR CHECKING: {e}]")
                    results['extra_directories'].append(dir_name)
            else:
                results['extra_directories'].append(dir_name)
        
        if removed_empty > 0:
            print(f"\nRemoved {removed_empty} empty unexpected directory(ies).")
    else:
        print("No unexpected directories found.")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total expected folders: {len(EXPECTED_FOLDER_NAMES)}")
    print(f"  ✓ Complete (directory + both files): {len(results['complete'])}")
    print(f"  ✗ Missing directory: {len(results['missing_directory'])}")
    print(f"  ✗ Missing files: {len(results['missing_files'])}")
    print(f"  ? Unexpected directories: {len(results['extra_directories'])}")
    print()
    
    # Detailed reports
    if results['missing_directory']:
        print("MISSING DIRECTORIES:")
        for folder_name in results['missing_directory']:
            print(f"  - {folder_name}")
        print()
    
    if results['missing_files']:
        print("DIRECTORIES WITH MISSING FILES:")
        for folder_name, missing in results['missing_files']:
            print(f"  - {folder_name}: missing {', '.join(missing)}")
        print()
    
    if results['extra_directories']:
        print("UNEXPECTED DIRECTORIES:")
        for dir_name in results['extra_directories']:
            print(f"  - {dir_name}")
        print()
    
    # Final status
    all_complete = (len(results['missing_directory']) == 0 and 
                   len(results['missing_files']) == 0)
    
    if all_complete:
        print("✓ VALIDATION PASSED: All expected folders have complete assets!")
        return 0
    else:
        print("✗ VALIDATION FAILED: Some folders are missing directories or files.")
        return 1


if __name__ == "__main__":
    exit(main())

