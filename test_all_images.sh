#!/bin/bash
# Test all JPG images in stone_images directory

STONE_IMAGES_DIR="stone_images"

if [ ! -d "$STONE_IMAGES_DIR" ]; then
    echo "Error: Directory not found: $STONE_IMAGES_DIR"
    exit 1
fi

# Find all JPG files
JPG_FILES=$(find "$STONE_IMAGES_DIR" -name "*.jpg" -o -name "*.JPG" | sort)

if [ -z "$JPG_FILES" ]; then
    echo "Error: No JPG files found in $STONE_IMAGES_DIR"
    exit 1
fi

# Count files
TOTAL=$(echo "$JPG_FILES" | wc -l | tr -d ' ')
echo "=========================================="
echo "Testing All Stone Images"
echo "=========================================="
echo "Found $TOTAL JPG file(s)"
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

# Process each image
COUNT=0
SUCCESS=0
FAILED=0

while IFS= read -r IMAGE; do
    COUNT=$((COUNT + 1))
    IMAGE_NAME=$(basename "$IMAGE")
    
    echo "=========================================="
    echo "[$COUNT/$TOTAL] Processing: $IMAGE_NAME"
    echo "=========================================="
    
    # Run detection (auto-generates timestamped folder)
    if $PYTHON_CMD scan_detection_only.py "$IMAGE" --min-area 30; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ Success: $IMAGE_NAME"
    else
        FAILED=$((FAILED + 1))
        echo "✗ Failed: $IMAGE_NAME"
    fi
    
    echo ""
done <<< "$JPG_FILES"

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total images: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""
echo "Results saved in: detection_results/"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi

