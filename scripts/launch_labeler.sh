#!/bin/bash
# Quick launcher for the interactive labeler

IMAGE=$1
if [ -z "$IMAGE" ]; then
    echo "Usage: $0 <image_path>"
    echo ""
    echo "Example:"
    echo "  $0 stone_images/stone_16820.jpg"
    echo ""
    echo "The script will automatically:"
    echo "  - Find the most recent detection results"
    echo "  - Load ground truth from database"
    echo "  - Launch the interactive labeler"
    exit 1
fi

if [ ! -f "$IMAGE" ]; then
    echo "Error: Image not found: $IMAGE"
    exit 1
fi

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python not found"
    exit 1
fi

echo "Launching Interactive Safaitic Labeler..."
echo "Image: $IMAGE"
echo ""

# Try to find detection results automatically
IMAGE_NAME=$(basename "$IMAGE" .jpg)
IMAGE_NAME=$(basename "$IMAGE_NAME" .png)
IMAGE_DIR=$(dirname "$IMAGE")

# Look for most recent detection results
RESULTS_FILE=""
# Check in current directory first
if [ -d "detection_results" ]; then
    LATEST_DIR=$(ls -td "detection_results/${IMAGE_NAME}_"* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/detection_results.txt" ]; then
        RESULTS_FILE="$LATEST_DIR/detection_results.txt"
        echo "Found detection results: $RESULTS_FILE"
    fi
fi
# Also check in image directory
if [ -z "$RESULTS_FILE" ] && [ -d "$IMAGE_DIR/detection_results" ]; then
    LATEST_DIR=$(ls -td "$IMAGE_DIR/detection_results/${IMAGE_NAME}_"* 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/detection_results.txt" ]; then
        RESULTS_FILE="$LATEST_DIR/detection_results.txt"
        echo "Found detection results: $RESULTS_FILE"
    fi
fi

# Launch labeler
if [ -n "$RESULTS_FILE" ]; then
    $PYTHON_CMD interactive_labeler.py "$IMAGE" --results "$RESULTS_FILE"
else
    echo "Warning: No detection results found. Launching without boxes."
    $PYTHON_CMD interactive_labeler.py "$IMAGE"
fi

