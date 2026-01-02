#!/bin/bash
# Fix dependency issues for Safaitic scanner

echo "=========================================="
echo "Fixing Dependencies"
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

# Check current NumPy version
echo "Checking NumPy version..."
NUMPY_VERSION=$($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Current NumPy: $NUMPY_VERSION"
    MAJOR_VERSION=$(echo $NUMPY_VERSION | cut -d. -f1)
    if [ "$MAJOR_VERSION" -ge 2 ]; then
        echo "⚠️  NumPy 2.x detected - will downgrade to 1.x for PyTorch compatibility"
        echo ""
        echo "Downgrading NumPy..."
        $PIP_CMD install 'numpy<2' --upgrade
    else
        echo "✓ NumPy version is compatible"
    fi
else
    echo "NumPy not installed"
fi

echo ""

# Install missing packages
echo "Installing required packages..."
echo ""

$PIP_CMD install torch torchvision opencv-python pillow matplotlib albumentations pandas

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

$PYTHON_CMD check_dependencies.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Dependencies fixed!"
    echo "=========================================="
    echo ""
    echo "You can now run:"
    echo "  bash quick_test.sh stone_images/stone_16820.jpg"
else
    echo ""
    echo "=========================================="
    echo "⚠️  Some issues remain"
    echo "=========================================="
    echo "Check the output above for details"
fi

