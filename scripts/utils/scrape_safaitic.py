#!/usr/bin/env python3
"""
Script to scrape Safaitic alphabet SVG assets from Wikipedia.
"""

import requests
from bs4 import BeautifulSoup
import os
import re
import time
import random
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:
    # Fallback if urllib3 is not available (shouldn't happen with requests)
    print("Warning: urllib3 not available. Retry functionality may be limited.")
    Retry = None

# Configuration
BASE_URL = "https://en.wikipedia.org/wiki/Safaitic"
OUTPUT_DIR = "safaitic_assets"
MIN_DELAY_SECONDS = 2  # Minimum delay between requests
MAX_DELAY_SECONDS = 5  # Maximum delay between requests
MAX_RETRIES = 5  # Maximum number of retries for failed requests
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# Multiple user agents to rotate through
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Edge/120.0.0.0",
]


def sanitize_name(name):
    """
    Sanitize phoneme name to be filesystem-safe.
    Replaces special characters with safe alternatives.
    """
    # Replace common special characters
    replacements = {
        'ḥ': 'h_dot',
        '’': 'alist',
        'ʾ': 'alist',
        'ʿ': 'ayn',
        ' ': '_',
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
    }
    
    sanitized = name
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    
    # Remove any remaining non-alphanumeric characters except underscores and hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    
    # Remove leading/trailing underscores and hyphens
    sanitized = sanitized.strip('_-')
    
    return sanitized if sanitized else "unknown"


def convert_thumbnail_to_svg(thumbnail_url):
    """
    Convert Wikipedia thumbnail URL to original SVG URL.
    
    Example:
    Input:  https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Safaitic_sad.svg/20px-Safaitic_sad.svg.png
    Output: https://upload.wikimedia.org/wikipedia/commons/d/de/Safaitic_sad.svg
    """
    if not thumbnail_url:
        return None
    
    # Ensure it starts with https:
    if thumbnail_url.startswith('//'):
        thumbnail_url = 'https:' + thumbnail_url
    elif thumbnail_url.startswith('/'):
        thumbnail_url = 'https://en.wikipedia.org' + thumbnail_url
    
    # Remove /thumb/ from the path
    if '/thumb/' in thumbnail_url:
        thumbnail_url = thumbnail_url.replace('/thumb/', '/')
    
    # Remove the trailing /{width}px-Filename.svg.png
    # Pattern: /{number}px-{filename}.{ext}
    thumbnail_url = re.sub(r'/\d+px-[^/]+\.(png|jpg|jpeg|gif)$', '', thumbnail_url)
    
    return thumbnail_url


def get_random_user_agent():
    """Return a random user agent from the list."""
    return random.choice(USER_AGENTS)


def create_session_with_retries():
    """
    Create a requests session with retry strategy and random user agent.
    """
    session = requests.Session()
    
    # Configure retry strategy if urllib3 is available
    if Retry is not None:
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    
    return session


def download_svg(session, url, filepath, retry_count=0):
    """
    Download an SVG file from URL and save to filepath.
    Uses session for connection reuse and retry logic.
    """
    # Note: File existence check is done before calling this function
    # This allows for better progress reporting
    
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Verify it's actually an SVG (check content or extension)
        if response.content and len(response.content) > 0:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"  Warning: Empty response for {url}")
            return False
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # Rate limited - wait longer before retrying
            wait_time = RETRY_BACKOFF_FACTOR ** (retry_count + 1) + random.uniform(1, 3)
            print(f"  Rate limited (429). Waiting {wait_time:.1f} seconds before retry...")
            time.sleep(wait_time)
            if retry_count < MAX_RETRIES:
                return download_svg(session, url, filepath, retry_count + 1)
            else:
                print(f"  Max retries reached for {url}")
                return False
        else:
            print(f"  HTTP Error {e.response.status_code} downloading {url}: {e}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  Request error downloading {url}: {e}")
        if retry_count < MAX_RETRIES:
            wait_time = RETRY_BACKOFF_FACTOR ** (retry_count + 1) + random.uniform(0.5, 1.5)
            print(f"  Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            return download_svg(session, url, filepath, retry_count + 1)
        return False
    except Exception as e:
        print(f"  Unexpected error downloading {url}: {e}")
        return False


def random_delay():
    """Add a random delay between requests to be less predictable."""
    delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
    time.sleep(delay)


def main():
    """Main scraping function."""
    print(f"Fetching Wikipedia page: {BASE_URL}")
    print(f"Using {len(USER_AGENTS)} different user agents")
    print(f"Delay range: {MIN_DELAY_SECONDS}-{MAX_DELAY_SECONDS} seconds between requests")
    print(f"Max retries: {MAX_RETRIES} with exponential backoff\n")
    
    # Create session with retry strategy
    session = create_session_with_retries()
    
    # Fetch the Wikipedia page
    try:
        headers = {'User-Agent': get_random_user_agent()}
        response = session.get(BASE_URL, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching Wikipedia page: {e}")
        return
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table with caption "Script chart of the Safaitic letter forms"
    # Try multiple strategies to find the table
    table = None
    
    # Strategy 1: Look for caption with exact or partial text
    for caption in soup.find_all('caption'):
        caption_text = caption.get_text()
        if 'Safaitic' in caption_text and ('letter' in caption_text.lower() or 'script chart' in caption_text.lower()):
            table = caption.find_parent('table')
            if table:
                print(f"Found table with caption: {caption_text.strip()}")
                break
    
    # Strategy 2: Look for wikitable class and check if it has Safaitic-related content
    if not table:
        for table_elem in soup.find_all('table', class_='wikitable'):
            # Check if table has images with 'Safaitic' in the src
            if table_elem.find('img', src=re.compile('Safaitic', re.I)):
                table = table_elem
                print("Found table by searching for Safaitic images")
                break
    
    # Strategy 3: Look for any table with Safaitic images in the first few rows
    if not table:
        for table_elem in soup.find_all('table'):
            rows = table_elem.find_all('tr')
            if len(rows) > 1:
                # Check first data row for Safaitic images
                first_data_row = rows[1] if len(rows) > 1 else None
                if first_data_row:
                    imgs = first_data_row.find_all('img')
                    if any('Safaitic' in img.get('src', '') for img in imgs):
                        table = table_elem
                        print("Found table by checking for Safaitic images in rows")
                        break
    
    if not table:
        print("Error: Could not find the target table with Safaitic letter forms")
        print("Attempting to list all tables with captions for debugging...")
        for caption in soup.find_all('caption'):
            print(f"  - {caption.get_text().strip()}")
        return
    
    print("Found target table. Processing rows...")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process table rows
    rows = table.find_all('tr')
    downloaded_count = 0
    skipped_count = 0
    already_existed_count = 0
    total_rows = 0
    
    # Count total data rows first
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 3:
            total_rows += 1
    
    print(f"Found {total_rows} data rows to process\n")
    
    current_row = 0
    for i, row in enumerate(rows):
        # Only process rows with td elements (data rows), skip header rows with th
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        
        # Try to find the name column - it should have text but no image
        # And find columns with images (Idealized and Square)
        name = None
        name_col_idx = None
        idealized_col_idx = None
        square_col_idx = None
        
        for j, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            img = cell.find('img')
            
            # Name column: has text, typically short, no image
            if not name and text and not img and len(text) < 50 and j < 3:
                # Check if it looks like a phoneme name (not a header)
                if text.lower() not in ['name', 'pronunciation', 'ipa', 'idealized', 'square', 'ociana']:
                    name = text
                    name_col_idx = j
            
            # Image columns: have images with Safaitic in src
            if img:
                src = img.get('src') or img.get('data-src') or ''
                if 'Safaitic' in src:
                    if idealized_col_idx is None:
                        idealized_col_idx = j
                    elif square_col_idx is None:
                        square_col_idx = j
        
        if not name:
            continue
        
        current_row += 1
        sanitized_name = sanitize_name(name)
        print(f"\n[{current_row}/{total_rows}] Processing: {name} -> {sanitized_name}")
        
        # Create subfolder
        glyph_dir = os.path.join(OUTPUT_DIR, sanitized_name)
        os.makedirs(glyph_dir, exist_ok=True)
        
        # Get Idealized image
        idealized_svg_url = None
        if idealized_col_idx is not None and idealized_col_idx < len(cells):
            idealized_cell = cells[idealized_col_idx]
            idealized_img = idealized_cell.find('img')
            if idealized_img:
                thumbnail_url = (idealized_img.get('src') or 
                               idealized_img.get('data-src') or 
                               (idealized_img.get('srcset', '').split()[0] if idealized_img.get('srcset') else None))
                if thumbnail_url:
                    idealized_svg_url = convert_thumbnail_to_svg(thumbnail_url)
        
        # Get Square image
        square_svg_url = None
        if square_col_idx is not None and square_col_idx < len(cells):
            square_cell = cells[square_col_idx]
            square_img = square_cell.find('img')
            if square_img:
                thumbnail_url = (square_img.get('src') or 
                               square_img.get('data-src') or 
                               (square_img.get('srcset', '').split()[0] if square_img.get('srcset') else None))
                if thumbnail_url:
                    square_svg_url = convert_thumbnail_to_svg(thumbnail_url)
        
        # Download Idealized SVG
        if idealized_svg_url:
            idealized_path = os.path.join(glyph_dir, 'ideal.svg')
            if os.path.exists(idealized_path) and os.path.getsize(idealized_path) > 0:
                print(f"  Idealized already exists, skipping")
                already_existed_count += 1
            else:
                print(f"  Downloading Idealized: {idealized_svg_url}")
                if download_svg(session, idealized_svg_url, idealized_path):
                    downloaded_count += 1
            random_delay()
        else:
            print(f"  Warning: No Idealized image found for {name}")
            skipped_count += 1
        
        # Download Square SVG
        if square_svg_url:
            square_path = os.path.join(glyph_dir, 'square.svg')
            if os.path.exists(square_path) and os.path.getsize(square_path) > 0:
                print(f"  Square already exists, skipping")
                already_existed_count += 1
            else:
                print(f"  Downloading Square: {square_svg_url}")
                if download_svg(session, square_svg_url, square_path):
                    downloaded_count += 1
            random_delay()
        else:
            print(f"  Warning: No Square image found for {name}")
            skipped_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Scraping Summary:")
    print(f"  Total SVG files downloaded: {downloaded_count}")
    print(f"  Files that already existed: {already_existed_count}")
    print(f"  Files skipped (missing images): {skipped_count}")
    print(f"  Total rows processed: {current_row}/{total_rows}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

