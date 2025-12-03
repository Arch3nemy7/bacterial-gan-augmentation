"""
Organize bacterial images into gram_positive and gram_negative folders.

This script:
1. Reads the Excel file with bacterial classification
2. Maps image file prefixes to Gram stain type
3. Copies images to appropriate folders (gram_positive or gram_negative)
4. Excludes fungus samples
"""
import openpyxl
from pathlib import Path
import shutil
from collections import defaultdict
import sys

# Paths
EXCEL_PATH = Path("data/01_raw/datasets/PBCs_microorgansim_information.xlsx")
IMAGES_DIR = Path("data/01_raw/datasets/PBCs_microorgansim_image/PBCs_microorgansim_image_review")
OUTPUT_DIR = Path("data/01_raw")

def load_classification_mapping():
    """Load bacterial ID to Gram stain mapping from Excel."""
    wb = openpyxl.load_workbook(EXCEL_PATH)
    sheet = wb.active

    mapping = {}
    stats = defaultdict(int)

    # Skip header row
    for row in sheet.iter_rows(min_row=2, values_only=True):
        bacterial_id = row[0]  # ID column
        species = row[1]  # Species name
        gram_stain = row[2]  # Gram stain column

        if gram_stain in ['positive', 'negative']:
            mapping[bacterial_id] = {
                'gram_stain': gram_stain,
                'species': species
            }
            stats[gram_stain] += 1
        elif gram_stain == 'fungus':
            stats['fungus'] += 1

    return mapping, stats

def organize_images(mapping, dry_run=False):
    """
    Organize images into gram_positive and gram_negative folders.

    Args:
        mapping: Dictionary mapping bacterial IDs to classification
        dry_run: If True, only print what would be done without copying
    """
    # Create output directories
    gram_positive_dir = OUTPUT_DIR / "gram_positive"
    gram_negative_dir = OUTPUT_DIR / "gram_negative"

    if not dry_run:
        gram_positive_dir.mkdir(exist_ok=True)
        gram_negative_dir.mkdir(exist_ok=True)

    # Statistics
    stats = {
        'gram_positive': 0,
        'gram_negative': 0,
        'fungus': 0,
        'unknown': 0
    }

    unknown_prefixes = set()

    # Process all images
    for image_path in sorted(IMAGES_DIR.glob("*.jpg")):
        # Extract bacterial ID from filename (prefix before _XX.jpg)
        filename = image_path.name
        prefix = filename.split('_')[0]

        if prefix in mapping:
            gram_type = mapping[prefix]['gram_stain']
            species = mapping[prefix]['species']

            # Determine destination
            if gram_type == 'positive':
                dest_dir = gram_positive_dir
                stats['gram_positive'] += 1
            elif gram_type == 'negative':
                dest_dir = gram_negative_dir
                stats['gram_negative'] += 1

            dest_path = dest_dir / filename

            if dry_run:
                print(f"Would copy: {filename} -> {gram_type}/ ({species})")
            else:
                shutil.copy2(image_path, dest_path)
        else:
            stats['unknown'] += 1
            unknown_prefixes.add(prefix)
            if dry_run:
                print(f"Unknown prefix: {prefix} ({filename})")

    return stats, unknown_prefixes

def main():
    """Main function."""
    print("=" * 80)
    print("BACTERIAL IMAGE DATASET ORGANIZER")
    print("=" * 80)

    print(f"\nSource: {IMAGES_DIR}")
    print(f"Destination: {OUTPUT_DIR}")

    # Load classification mapping
    print("\nLoading bacterial classification from Excel...")
    mapping, excel_stats = load_classification_mapping()

    print(f"\n✅ Loaded {len(mapping)} bacterial species:")
    print(f"   - Gram-positive: {excel_stats['positive']} species")
    print(f"   - Gram-negative: {excel_stats['negative']} species")
    print(f"   - Fungus (excluded): {excel_stats['fungus']} species")

    # Count available images
    total_images = len(list(IMAGES_DIR.glob("*.jpg")))
    print(f"\n📊 Found {total_images} images to process")

    # Dry run first
    print("\n" + "=" * 80)
    print("DRY RUN - Preview of operations")
    print("=" * 80)
    stats, unknown_prefixes = organize_images(mapping, dry_run=True)

    print("\n" + "=" * 80)
    print("DRY RUN SUMMARY")
    print("=" * 80)
    print(f"Gram-positive images: {stats['gram_positive']}")
    print(f"Gram-negative images: {stats['gram_negative']}")
    print(f"Unknown images: {stats['unknown']}")

    if unknown_prefixes:
        print(f"\n⚠️  Unknown prefixes: {', '.join(sorted(unknown_prefixes))}")

    # Ask for confirmation
    print("\n" + "=" * 80)

    # Check for --yes flag
    auto_yes = '--yes' in sys.argv or '-y' in sys.argv

    if auto_yes:
        print("Auto-confirm enabled (--yes flag)")
        response = 'y'
    else:
        response = input("Proceed with copying images? (y/N): ")

    if response.lower() == 'y':
        print("\n" + "=" * 80)
        print("ORGANIZING IMAGES")
        print("=" * 80)

        stats, _ = organize_images(mapping, dry_run=False)

        print("\n" + "=" * 80)
        print("✅ ORGANIZATION COMPLETE!")
        print("=" * 80)
        print(f"Gram-positive images: {stats['gram_positive']}")
        print(f"  Location: {OUTPUT_DIR / 'gram_positive'}")
        print(f"Gram-negative images: {stats['gram_negative']}")
        print(f"  Location: {OUTPUT_DIR / 'gram_negative'}")

        print("\n📝 Next steps:")
        print("  1. Run preprocessing: poetry run python scripts/prepare_data.py")
        print("  2. Start training: bacterial-gan train")
    else:
        print("\n❌ Operation cancelled")

if __name__ == "__main__":
    main()
