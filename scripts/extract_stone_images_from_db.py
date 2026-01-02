#!/usr/bin/env python3
"""
Extract stone images from SQLite database.

Connects to safaitic.db and extracts stone images to stone_images/ directory.
Handles both image URLs and local file paths.
"""

import sqlite3
import sys
from pathlib import Path
from PIL import Image
import io
import urllib.request
import urllib.parse
import os


def inspect_database(db_path):
    """Inspect database structure to find image tables/columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 60)
    print("Database Inspection")
    print("=" * 60)
    print(f"Database: {db_path}\n")
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"Tables found: {len(tables)}")
    for table in tables:
        table_name = table[0]
        print(f"\n  Table: {table_name}")
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print(f"    Columns:")
        for col in columns:
            col_name, col_type = col[1], col[2]
            print(f"      - {col_name} ({col_type})")
        
        # Check for image-related columns
        image_columns = [col[1] for col in columns if any(keyword in col[1].lower() 
                          for keyword in ['image', 'img', 'photo', 'picture', 'blob', 'data', 'url'])]
        if image_columns:
            print(f"    ⚠️  Potential image columns: {', '.join(image_columns)}")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"    Rows: {count}")
    
    conn.close()
    return tables


def is_url(path_string):
    """Check if string is a URL."""
    if not isinstance(path_string, str):
        return False
    return path_string.startswith(('http://', 'https://', 'file://'))


def download_image(url, output_path, timeout=10):
    """Download image from URL."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"    Error downloading {url}: {e}")
        return False


def extract_images_from_db(db_path, output_dir="stone_images", table_name="inscriptions",
                           image_column="image_url", id_column="id", 
                           local_image_base=None, download_urls=False, limit=None):
    """
    Extract images from SQLite database.
    
    Args:
        db_path: Path to SQLite database
        output_dir: Directory to save extracted images
        table_name: Name of table containing images
        image_column: Name of column with image URL/path
        id_column: Name of ID column for naming files
        local_image_base: Base directory if image URLs are relative paths
        download_urls: If True, download images from URLs (default: False, just check local)
    """
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not db_path.exists():
        print(f"✗ Error: Database not found: {db_path}")
        return False
    
    print("=" * 60)
    print("Extracting Stone Images from Database")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Table: {table_name}")
    print(f"Image column: {image_column}")
    print(f"Output: {output_dir}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query for images
    print(f"Querying images from {table_name}...")
    query = f"SELECT {id_column}, {image_column} FROM {table_name} WHERE {image_column} IS NOT NULL AND {image_column} != ''"
    if limit:
        query += f" LIMIT {limit}"
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if len(rows) == 0:
        print("✗ No images found in database")
        conn.close()
        return False
    
    print(f"Found {len(rows)} rows with image references\n")
    
    # Extract images
    extracted = 0
    skipped = 0
    errors = 0
    urls_found = 0
    local_found = 0
    
    for row_id, image_ref in rows:
        try:
            if not image_ref or not isinstance(image_ref, str):
                skipped += 1
                continue
            
            # Determine if it's a URL or local path
            if is_url(image_ref):
                urls_found += 1
                if download_urls:
                    # Download from URL
                    ext = Path(urllib.parse.urlparse(image_ref).path).suffix or '.jpg'
                    output_path = output_dir / f"stone_{row_id}{ext}"
                    
                    if download_image(image_ref, output_path):
                        extracted += 1
                    else:
                        errors += 1
                else:
                    # Skip URLs if not downloading
                    skipped += 1
                    continue
            else:
                # Local file path
                local_found += 1
                
                # Try to resolve path
                if local_image_base:
                    # If base directory provided, join with it
                    image_path = Path(local_image_base) / image_ref
                else:
                    # Try as absolute path first
                    image_path = Path(image_ref)
                    if not image_path.exists():
                        # Try relative to database directory
                        db_dir = db_path.parent
                        image_path = db_dir / image_ref
                    if not image_path.exists():
                        # Try relative to project root
                        project_root = Path(__file__).parent
                        image_path = project_root / image_ref
                
                if image_path.exists() and image_path.is_file():
                    # Copy or symlink the file
                    ext = image_path.suffix or '.jpg'
                    output_path = output_dir / f"stone_{row_id}{ext}"
                    
                    if output_path.exists():
                        # Skip if already exists
                        continue
                    
                    try:
                        # Try to open as image to verify
                        img = Image.open(image_path)
                        img.verify()  # Verify it's a valid image
                        
                        # Copy file
                        import shutil
                        shutil.copy2(image_path, output_path)
                        extracted += 1
                        
                        if extracted % 10 == 0:
                            print(f"  Extracted {extracted} images...")
                    except Exception as e:
                        print(f"  Warning: Row {row_id} - Invalid image: {e}")
                        errors += 1
                else:
                    skipped += 1
                    if skipped <= 5:  # Only show first few warnings
                        print(f"  Warning: Row {row_id} - File not found: {image_ref}")
        
        except Exception as e:
            print(f"  Error processing row {row_id}: {e}")
            errors += 1
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"✓ Extracted: {extracted} images")
    print(f"  - Local files: {local_found}")
    print(f"  - URLs: {urls_found}")
    if skipped > 0:
        print(f"⚠ Skipped: {skipped} rows")
    if errors > 0:
        print(f"✗ Errors: {errors} rows")
    print(f"\nImages saved to: {output_dir}")
    
    if urls_found > 0 and not download_urls:
        print(f"\n⚠️  Found {urls_found} URLs but download_urls=False")
        print("   To download URLs, run with --download-urls flag")
    
    return extracted > 0


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract stone images from SQLite database")
    parser.add_argument("db_path", nargs='?', 
                       default="/Users/dansachs/Desktop/Safaitic Inscription Reader/data/safaitic.db",
                       help="Path to SQLite database")
    parser.add_argument("--table", default="inscriptions",
                       help="Table name (default: inscriptions)")
    parser.add_argument("--image-column", default="image_url",
                       help="Image column name (default: image_url)")
    parser.add_argument("--id-column", default="id",
                       help="ID column name (default: id)")
    parser.add_argument("--output", default="data/stone_images",
                       help="Output directory (default: data/stone_images)")
    parser.add_argument("--local-base", 
                       help="Base directory for relative image paths")
    parser.add_argument("--download-urls", action="store_true",
                       help="Download images from URLs (default: False)")
    parser.add_argument("--limit", type=int,
                       help="Limit number of images to extract (for testing)")
    parser.add_argument("--inspect-only", action="store_true",
                       help="Only inspect database, don't extract")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Safaitic Database Image Extractor")
    print("=" * 60)
    print()
    
    # Inspect database
    print("Inspecting database structure...\n")
    tables = inspect_database(args.db_path)
    
    if args.inspect_only:
        return
    
    # Extract images
    print("\n" + "=" * 60)
    print("Extracting images...\n")
    
    success = extract_images_from_db(
        db_path=args.db_path,
        output_dir=args.output,
        table_name=args.table,
        image_column=args.image_column,
        id_column=args.id_column,
        local_image_base=args.local_base,
        download_urls=args.download_urls,
        limit=args.limit
    )
    
    if success:
        print("\n✓ Setup complete!")
        print("\nTest with:")
        print(f"  python scan_and_visualize.py {args.output}/stone_1.jpg \\")
        print("      --checkpoint models/safaitic_matcher.pth \\")
        print("      --glyphs cleaned_glyphs")
    else:
        print("\n✗ Extraction failed or no images found.")
        print("\nTips:")
        print("  - Check if image URLs point to local files")
        print("  - Use --local-base to specify base directory for relative paths")
        print("  - Use --download-urls to download from URLs")


if __name__ == "__main__":
    main()
