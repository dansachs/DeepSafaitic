#!/usr/bin/env python3
"""
Print out the complete folder name mapping for review.
"""

# Expected folder names (final target names)
EXPECTED_FOLDER_NAMES = [
    'alif',
    'b',
    't',
    'th',
    'g',
    'h_dot',
    'kh',
    'd',
    'dh',
    'r',
    'z',
    's1',
    's2',
    's_dot',
    'd_dot',
    't_dot',
    'z_dot',
    'ayn',
    'gh',
    'f',
    'q',
    'k',
    'l',
    'm',
    'n',
    'h',
    'w',
    'y',
]

# Mapping from scraper-generated names to expected folder names
FOLDER_NAME_MAPPING = {
    'ha': 'h',
    'sin': 's1',
    'shin': 's2',
    'ba': 'b',
    'ta': 't',
    'tha': 'th',
    'gim': 'g',
    'hha': 'h_dot',
    'kha': 'kh',
    'dal': 'd',
    'dhal': 'dh',
    'ra': 'r',
    'za': 'z',
    'sad': 's_dot',
    's_sad': 's_dot',
    'dad': 'd_dot',
    'd_dad': 'd_dot',
    'tta': 't_dot',
    'zayn': 'z_dot',
    'ghayn': 'gh',
    'fa': 'f',
    'qaf': 'q',
    'kaf': 'k',
    'lam': 'l',
    'mim': 'm',
    'nun': 'n',
    'waw': 'w',
    'ya': 'y',
}

print("=" * 80)
print("SAFAITIC ALPHABET FOLDER NAME MAPPING")
print("=" * 80)
print()
print("EXPECTED FOLDER NAMES (28 total):")
print("-" * 80)
for i, name in enumerate(EXPECTED_FOLDER_NAMES, 1):
    print(f"  {i:2d}. {name}")
print()

print("MAPPING TABLE (Scraper Name -> Expected Name):")
print("-" * 80)
print(f"{'Scraper Name':<20} -> {'Expected Name':<20} {'Notes':<30}")
print("-" * 80)

# Sort mappings for easier review
sorted_mappings = sorted(FOLDER_NAME_MAPPING.items())

for scraper_name, expected_name in sorted_mappings:
    # Check if this is a correction needed
    if scraper_name != expected_name:
        notes = "Needs renaming"
    else:
        notes = ""
    print(f"{scraper_name:<20} -> {expected_name:<20} {notes:<30}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total expected folders: {len(EXPECTED_FOLDER_NAMES)}")
print(f"Total mappings defined: {len(FOLDER_NAME_MAPPING)}")
print()
print("Note: Folders that already match expected names don't need mapping.")
print("      Only folders that need renaming are listed in the mapping table.")
print("=" * 80)

