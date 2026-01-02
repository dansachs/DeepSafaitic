#!/bin/bash
# Quick test script for scan_and_visualize.py

IMAGE=$1
if [ -z "$IMAGE" ]; then
    # Try to find first image in stone_images
    if [ -d "stone_images" ] && [ -n "$(ls -A stone_images/*.jpg stone_images/*.png 2>/dev/null)" ]; then
        IMAGE=$(ls stone_images/*.jpg stone_images/*.png 2>/dev/null | head -1)
        echo "Using first image found: $IMAGE"
    else
        echo "Error: No image specified and no images found in stone_images/"
        echo "Usage: $0 <image_path>"
        exit 1
    fi
fi

if [ ! -f "$IMAGE" ]; then
    echo "Error: Image not found: $IMAGE"
    exit 1
fi

echo "=========================================="
echo "Safaitic Scanner - Quick Test"
echo "=========================================="
echo "Image: $IMAGE"
echo ""

# Detect Python command first
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Neither python3 nor python found in PATH"
    exit 1
fi

# Check prerequisites
echo "Checking prerequisites..."
if [ ! -f "models/safaitic_matcher.pth" ]; then
    echo "✗ Error: Model checkpoint not found: models/safaitic_matcher.pth"
    exit 1
fi
echo "✓ Model checkpoint found"

if [ ! -d "cleaned_glyphs" ]; then
    echo "✗ Error: Reference glyphs not found: cleaned_glyphs/"
    exit 1
fi
echo "✓ Reference glyphs found"

# Check dependencies
echo ""
echo "Checking dependencies..."
if ! $PYTHON_CMD -c "import torch, torchvision, cv2, albumentations" 2>/dev/null; then
    echo "⚠️  Warning: Some dependencies may be missing"
    echo "   Run: bash fix_dependencies.sh"
    echo "   Or: pip3 install torch torchvision opencv-python albumentations"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Running scanner..."
echo "=========================================="

$PYTHON_CMD scan_and_visualize.py "$IMAGE" \
    --checkpoint models/safaitic_matcher.pth \
    --glyphs cleaned_glyphs \
    --output test_results \
    --confidence 1.0 \
    --min-area 30

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test completed successfully!"
    echo "=========================================="
    echo "Results saved to: test_results/"
    echo ""
    echo "Check these files:"
    echo "  - test_results/stone_scan_visualization.png"
    echo "  - test_results/transcription.txt"
    echo ""
    echo "To view visualization:"
    echo "  open test_results/stone_scan_visualization.png"
else
    echo ""
    echo "=========================================="
    echo "✗ Test failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo "Check the error messages above."
    exit $EXIT_CODE
fi

