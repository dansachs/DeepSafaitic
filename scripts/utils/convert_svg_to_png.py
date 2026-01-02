#!/usr/bin/env python3
"""
Convert Safaitic SVG glyphs to normalized PNG images for machine learning.
Uses CairoSVG for high-quality rendering and Pillow for centering/padding.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import cairosvg
import io

# Configuration
SOURCE_DIR = "safaitic_assets"
OUTPUT_DIR = "cleaned_glyphs"
TARGET_SIZE = 128
PADDING_PERCENT = 0.10  # 10% padding
PADDING_PIXELS = int(TARGET_SIZE * PADDING_PERCENT)  # 12 pixels
CONTENT_SIZE = TARGET_SIZE - (2 * PADDING_PIXELS)  # 104 pixels for content
BACKGROUND_COLOR = "white"  # White background for better stone-blending


def convert_svg_to_png(svg_path, output_path):
    """
    Convert an SVG file to a normalized PNG image.
    
    Args:
        svg_path: Path to the input SVG file
        output_path: Path where the output PNG will be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the SVG file
        with open(svg_path, 'rb') as svg_file:
            svg_data = svg_file.read()
        
        if len(svg_data) == 0:
            print(f"  Warning: Empty SVG file: {svg_path}")
            return False
        
        # Convert SVG to PNG using CairoSVG
        # Render at a higher resolution first for better quality
        scale_factor = 2  # Render at 2x for better quality
        render_size = CONTENT_SIZE * scale_factor
        
        png_data = cairosvg.svg2png(
            bytestring=svg_data,
            output_width=render_size,
            output_height=render_size,
            background_color=BACKGROUND_COLOR
        )
        
        if not png_data:
            print(f"  Warning: Failed to render SVG: {svg_path}")
            return False
        
        # Open the rendered PNG with Pillow
        img = Image.open(io.BytesIO(png_data))
        
        # Convert to RGBA if needed (for transparency handling)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # If background is white, we can convert to RGB for smaller file size
        # Otherwise keep RGBA for transparency
        if BACKGROUND_COLOR == "white":
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            else:
                background.paste(img)
            img = background
        else:
            img = img.convert('RGBA')
        
        # Resize to content size (with padding already accounted for)
        img = img.resize((CONTENT_SIZE, CONTENT_SIZE), Image.Resampling.LANCZOS)
        
        # Create final 128x128 image with padding
        final_img = Image.new(img.mode, (TARGET_SIZE, TARGET_SIZE), 
                             (255, 255, 255) if BACKGROUND_COLOR == "white" else (0, 0, 0, 0))
        
        # Calculate center position
        x_offset = (TARGET_SIZE - CONTENT_SIZE) // 2
        y_offset = (TARGET_SIZE - CONTENT_SIZE) // 2
        
        # Paste the resized image in the center
        final_img.paste(img, (x_offset, y_offset))
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the final image
        final_img.save(output_path, 'PNG', optimize=True)
        
        return True
        
    except Exception as e:
        print(f"  Error processing {svg_path}: {e}")
        return False


def process_directory(source_dir, output_dir):
    """
    Process all SVG files in the source directory structure.
    
    Args:
        source_dir: Root directory containing letter subfolders
        output_dir: Root directory for output PNG files
    
    Returns:
        Tuple of (successful_count, failed_count, processed_files)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return 0, 0, []
    
    successful_count = 0
    failed_count = 0
    processed_files = []
    
    # Find all SVG files
    svg_files = list(source_path.rglob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in '{source_dir}'")
        return 0, 0, []
    
    print(f"Found {len(svg_files)} SVG file(s) to process...")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE} pixels")
    print(f"Padding: {PADDING_PIXELS} pixels ({PADDING_PERCENT*100}%)")
    print(f"Content area: {CONTENT_SIZE}x{CONTENT_SIZE} pixels")
    print(f"Background: {BACKGROUND_COLOR}")
    print()
    
    # Process each SVG file
    for svg_file in sorted(svg_files):
        # Calculate relative path from source directory
        relative_path = svg_file.relative_to(source_path)
        
        # Create corresponding output path (change .svg to .png)
        output_file = output_path / relative_path.with_suffix('.png')
        
        # Print progress
        print(f"Processing: {relative_path} -> {output_file.relative_to(output_path)}")
        
        # Convert the file
        if convert_svg_to_png(svg_file, output_file):
            successful_count += 1
            processed_files.append(str(output_file))
        else:
            failed_count += 1
    
    return successful_count, failed_count, processed_files


def main():
    """Main function."""
    print("=" * 70)
    print("SVG to PNG Converter for Safaitic Glyphs")
    print("=" * 70)
    print()
    
    # Check if required libraries are available
    try:
        import cairosvg
    except ImportError:
        print("Error: cairosvg is not installed.")
        print("Please install it with: pip install cairosvg")
        return 1
    
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow (PIL) is not installed.")
        print("Please install it with: pip install pillow")
        return 1
    
    # Process all SVG files
    successful, failed, processed = process_directory(SOURCE_DIR, OUTPUT_DIR)
    
    # Print summary
    print()
    print("=" * 70)
    print("CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total PNGs created: {successful}")
    print(f"Failed conversions: {failed}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("=" * 70)
    
    if successful > 0:
        print(f"\n✓ Successfully converted {successful} SVG file(s) to PNG")
        print(f"  All files saved in: {os.path.abspath(OUTPUT_DIR)}")
    
    if failed > 0:
        print(f"\n⚠ {failed} file(s) failed to convert (see warnings above)")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

