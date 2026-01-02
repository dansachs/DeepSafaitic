#!/bin/bash
# Fix corrupted package metadata and install dependencies

echo "=========================================="
echo "Fixing Corrupted Packages"
echo "=========================================="
echo ""

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo ""

# Step 1: Fix corrupted NumPy installation
echo "Step 1: Fixing NumPy installation..."
echo "----------------------------------------"
$PIP_CMD uninstall numpy -y 2>/dev/null
$PIP_CMD install --force-reinstall 'numpy<2'
echo ""

# Step 2: Install torchvision
echo "Step 2: Installing torchvision..."
echo "----------------------------------------"
$PIP_CMD install torchvision
echo ""

# Step 3: Install other missing packages
echo "Step 3: Installing other dependencies..."
echo "----------------------------------------"
$PIP_CMD install opencv-python albumentations pandas
echo ""

# Step 4: Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

$PYTHON_CMD check_dependencies.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All dependencies fixed!"
    echo "=========================================="
    echo ""
    echo "You can now run:"
    echo "  bash quick_test.sh stone_images/stone_16820.jpg"
else
    echo ""
    echo "=========================================="
    echo "⚠️  Some issues remain"
    echo "=========================================="
    echo "Try running with --user flag:"
    echo "  pip3 install --user torchvision opencv-python albumentations"
fi

