#!/bin/bash
# Force fix NumPy version issue

echo "=========================================="
echo "Force Fixing NumPy Compatibility"
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
echo "Current NumPy version:"
$PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>&1 || echo "NumPy not importable"
echo ""

# Step 1: Force uninstall NumPy
echo "Step 1: Force uninstalling NumPy..."
echo "----------------------------------------"
$PIP_CMD uninstall numpy -y 2>&1 | grep -v "WARNING" || true
echo ""

# Step 2: Clear pip cache
echo "Step 2: Clearing pip cache..."
echo "----------------------------------------"
$PIP_CMD cache purge 2>&1 | head -5 || true
echo ""

# Step 3: Install NumPy 1.x
echo "Step 3: Installing NumPy 1.x..."
echo "----------------------------------------"
$PIP_CMD install --no-cache-dir 'numpy<2.0' --force-reinstall
echo ""

# Step 4: Verify installation
echo "Step 4: Verifying NumPy installation..."
echo "----------------------------------------"
$PYTHON_CMD -c "import numpy; print('✓ NumPy version:', numpy.__version__); import numpy as np; print('✓ NumPy is working')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ NumPy fixed!"
    echo "=========================================="
    echo ""
    echo "Now try running the test again:"
    echo "  bash quick_test.sh stone_images/stone_16820.jpg"
else
    echo ""
    echo "=========================================="
    echo "✗ NumPy fix failed"
    echo "=========================================="
    echo "Try with --user flag:"
    echo "  pip3 install --user --force-reinstall 'numpy<2.0'"
fi

