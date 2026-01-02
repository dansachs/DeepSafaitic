#!/usr/bin/env python3
"""
Check and fix dependencies for Safaitic scanner.
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def get_package_version(package_name):
    """Get version of installed package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        pass
    return None

def main():
    """Check all dependencies."""
    print("=" * 60)
    print("Dependency Checker for Safaitic Scanner")
    print("=" * 60)
    print()
    
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'albumentations': 'albumentations',
        'pandas': 'pandas',
    }
    
    issues = []
    warnings = []
    
    print("Checking required packages...")
    print("-" * 60)
    
    for package_name, import_name in required_packages.items():
        installed = check_package(package_name, import_name)
        version = get_package_version(package_name)
        
        if installed:
            status = "✓"
            print(f"{status} {package_name:20s} {version or 'installed'}")
        else:
            status = "✗"
            print(f"{status} {package_name:20s} NOT INSTALLED")
            issues.append(package_name)
    
    print()
    
    # Check NumPy version specifically
    if check_package('numpy'):
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        
        if major_version >= 2:
            warnings.append(f"NumPy {numpy_version} detected - may cause compatibility issues with PyTorch")
            print("⚠️  WARNING: NumPy 2.x detected")
            print("   PyTorch may need NumPy < 2.0")
            print("   Consider: pip install 'numpy<2'")
            print()
    
    # Check PyTorch version
    if check_package('torch'):
        import torch
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        print()
    
    # Summary
    print("=" * 60)
    if issues:
        print("✗ Missing packages:")
        for pkg in issues:
            print(f"  - {pkg}")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(issues)}")
        print()
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    if not issues and not warnings:
        print("✓ All dependencies are installed and compatible!")
        return 0
    elif issues:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())

