#!/bin/bash
# Quick test script for detection-only mode

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
echo "Safaitic Detection - Quick Test"
echo "=========================================="
echo "Image: $IMAGE"
echo ""

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python not found"
    exit 1
fi

# Create output directory based on image name
IMAGE_NAME=$(basename "$IMAGE" .jpg)
IMAGE_NAME=$(basename "$IMAGE_NAME" .png)
OUTPUT_DIR="detection_results/$IMAGE_NAME"

echo "Output directory: $OUTPUT_DIR"
echo ""

echo "Running detection..."
echo "=========================================="

$PYTHON_CMD scan_detection_only.py "$IMAGE" \
    --output "$OUTPUT_DIR" \
    --min-area 30

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test completed successfully!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "Files created:"
    echo "  - $OUTPUT_DIR/scanvisualization.png"
    echo "  - $OUTPUT_DIR/detection_results.txt"
    echo ""
    echo "To view visualization:"
    echo "  open $OUTPUT_DIR/scanvisualization.png"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_DIR/detection_results.txt"
else
    echo ""
    echo "=========================================="
    echo "✗ Test failed with exit code: $EXIT_CODE"
    echo "=========================================="
    exit $EXIT_CODE
fi

