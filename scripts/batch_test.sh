#!/bin/bash
# Batch test multiple stone images

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python not found"
    exit 1
fi

# Get number of images to process (default: 4)
NUM_IMAGES=${1:-4}

echo "=========================================="
echo "Batch Testing Safaitic Scanner"
echo "=========================================="
echo "Processing $NUM_IMAGES images..."
echo ""

# Get list of images
IMAGES=($(ls -1 stone_images/*.jpg 2>/dev/null | head -$NUM_IMAGES))

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "Error: No images found in stone_images/"
    exit 1
fi

echo "Found ${#IMAGES[@]} images to process:"
for img in "${IMAGES[@]}"; do
    echo "  - $(basename $img)"
done
echo ""

# Process each image
SUCCESS=0
FAILED=0

for i in "${!IMAGES[@]}"; do
    IMAGE="${IMAGES[$i]}"
    IMAGE_NAME=$(basename "$IMAGE" .jpg)
    
    echo "=========================================="
    echo "[$((i+1))/${#IMAGES[@]}] Processing: $IMAGE_NAME"
    echo "=========================================="
    
    OUTPUT_DIR="test_results/$IMAGE_NAME"
    
    $PYTHON_CMD scan_and_visualize.py "$IMAGE" \
        --checkpoint models/safaitic_matcher.pth \
        --glyphs cleaned_glyphs \
        --output "$OUTPUT_DIR" \
        --confidence 1.0 \
        --min-area 100
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success: $IMAGE_NAME"
        SUCCESS=$((SUCCESS + 1))
        echo "  Results: $OUTPUT_DIR/"
        echo ""
    else
        echo "✗ Failed: $IMAGE_NAME (exit code: $EXIT_CODE)"
        FAILED=$((FAILED + 1))
        echo ""
    fi
done

# Summary
echo "=========================================="
echo "Batch Processing Complete"
echo "=========================================="
echo "Success: $SUCCESS"
echo "Failed:  $FAILED"
echo "Total:   ${#IMAGES[@]}"
echo ""
echo "Results saved to: test_results/"
echo ""
echo "View results:"
for img in "${IMAGES[@]}"; do
    IMAGE_NAME=$(basename "$img" .jpg)
    echo "  open test_results/$IMAGE_NAME/stone_scan_visualization.png"
done

