# Fix Dependency Issues

You have two issues:
1. **NumPy 2.0.2** - Incompatible with your PyTorch version
2. **Missing torchvision** - Not installed

## Quick Fix (Recommended)

Run this command to fix everything:

```bash
bash fix_dependencies.sh
```

This will:
- Downgrade NumPy to 1.x (compatible with PyTorch)
- Install all missing packages (torchvision, etc.)

## Manual Fix

If you prefer to fix manually:

### Step 1: Fix NumPy Version

```bash
pip3 install 'numpy<2' --upgrade
```

### Step 2: Install Missing Packages

```bash
pip3 install torch torchvision opencv-python pillow matplotlib albumentations pandas
```

## Verify Fix

After fixing, check dependencies:

```bash
python3 check_dependencies.py
```

Should show all packages as ✓ installed.

## Then Test Again

```bash
bash quick_test.sh stone_images/stone_16820.jpg
```

## Alternative: Use Virtual Environment (Recommended for Production)

If you want to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install 'numpy<2' torch torchvision opencv-python pillow matplotlib albumentations pandas

# Then run tests
bash quick_test.sh stone_images/stone_16820.jpg
```

