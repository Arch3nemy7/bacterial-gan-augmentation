"""
Organize 640DataSet bacterial images by Gram stain type.

This script:
1. Reads JSON annotation files from 640DataSet
2. Extracts Gram stain labels (G+ or G)
3. Copies images to gram_positive or gram_negative folders
4. Provides dry-run mode with confirmation
"""
import json
from pathlib import Path
import shutil
import sys
from collections import defaultdict

# Paths
DATASET_DIR = Path("data/01_raw/DeepDataSet/640DataSet")
JSON_DIR = DATASET_DIR / "json"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_DIR = Path("data/01_raw")


def parse_json_labels():
    """
    Parse all JSON files and create image-to-label mapping.

    Returns:
        tuple: (mapping dict, statistics dict)
            - mapping: {image_filename: 'gram_positive' or 'gram_negative'}
            - statistics: counts of each label type
    """
    mapping = {}
    stats = defaultdict(int)
    errors = []

    print(f"\nParsing JSON annotations from: {JSON_DIR}")

    json_files = list(JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Extract image filename
            image_filename = data.get("imagePath", "")
            if not image_filename:
                errors.append(f"No imagePath in {json_path.name}")
                stats['no_image_path'] += 1
                continue

            # Extract label from first shape
            shapes = data.get("shapes", [])
            if not shapes:
                errors.append(f"No shapes in {json_path.name}")
                stats['no_shapes'] += 1
                continue

            label = shapes[0].get("label", "")

            # Map label to gram stain type
            if label == "G+":
                gram_type = "gram_positive"
                stats['gram_positive'] += 1
            elif label == "G":
                gram_type = "gram_negative"
                stats['gram_negative'] += 1
            else:
                errors.append(f"Unknown label '{label}' in {json_path.name}")
                stats['unknown_label'] += 1
                continue

            mapping[image_filename] = gram_type

        except json.JSONDecodeError:
            errors.append(f"Invalid JSON: {json_path.name}")
            stats['invalid_json'] += 1
        except Exception as e:
            errors.append(f"Error processing {json_path.name}: {e}")
            stats['other_errors'] += 1

    return mapping, stats, errors


def organize_images(mapping, dry_run=False):
    """
    Copy images to gram_positive or gram_negative folders.

    Args:
        mapping: Dictionary mapping image filenames to gram stain type
        dry_run: If True, only print what would be done without copying

    Returns:
        dict: Statistics of copied/skipped images
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
        'missing_image': 0,
        'already_exists': 0
    }

    missing_images = []

    # Process images according to mapping
    for image_filename, gram_type in mapping.items():
        source_path = IMAGES_DIR / image_filename

        # Check if image exists
        if not source_path.exists():
            stats['missing_image'] += 1
            missing_images.append(image_filename)
            continue

        # Determine destination
        if gram_type == "gram_positive":
            dest_dir = gram_positive_dir
        else:  # gram_negative
            dest_dir = gram_negative_dir

        dest_path = dest_dir / image_filename

        # Check if already exists
        if dest_path.exists() and not dry_run:
            stats['already_exists'] += 1
            continue

        if dry_run:
            if not dest_path.exists():
                stats[gram_type] += 1
        else:
            shutil.copy2(source_path, dest_path)
            stats[gram_type] += 1

    return stats, missing_images


def count_existing_images():
    """Count images currently in gram_positive and gram_negative folders."""
    gram_positive_dir = OUTPUT_DIR / "gram_positive"
    gram_negative_dir = OUTPUT_DIR / "gram_negative"

    pos_count = len(list(gram_positive_dir.glob("*.jpg"))) if gram_positive_dir.exists() else 0
    neg_count = len(list(gram_negative_dir.glob("*.jpg"))) if gram_negative_dir.exists() else 0

    return pos_count, neg_count


def main():
    """Main function."""
    print("=" * 80)
    print("640DATASET BACTERIAL IMAGE ORGANIZER")
    print("=" * 80)

    print(f"\nSource: {DATASET_DIR}")
    print(f"Destination: {OUTPUT_DIR}")

    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"\n❌ Error: Dataset directory not found: {DATASET_DIR}")
        return

    if not JSON_DIR.exists():
        print(f"\n❌ Error: JSON directory not found: {JSON_DIR}")
        return

    if not IMAGES_DIR.exists():
        print(f"\n❌ Error: Images directory not found: {IMAGES_DIR}")
        return

    # Count existing images
    existing_pos, existing_neg = count_existing_images()
    print(f"\n📊 Current dataset:")
    print(f"   Gram-positive: {existing_pos} images")
    print(f"   Gram-negative: {existing_neg} images")
    print(f"   Total: {existing_pos + existing_neg} images")

    # Parse JSON labels
    print("\n" + "=" * 80)
    print("PARSING JSON ANNOTATIONS")
    print("=" * 80)

    mapping, parse_stats, errors = parse_json_labels()

    print(f"\n✅ Parsed {len(mapping)} image annotations:")
    print(f"   Gram-positive (G+): {parse_stats['gram_positive']} images")
    print(f"   Gram-negative (G): {parse_stats['gram_negative']} images")

    if parse_stats.get('no_shapes', 0) > 0:
        print(f"\n⚠️  Skipped {parse_stats['no_shapes']} files with no shapes")
    if parse_stats.get('unknown_label', 0) > 0:
        print(f"⚠️  Skipped {parse_stats['unknown_label']} files with unknown labels")
    if parse_stats.get('invalid_json', 0) > 0:
        print(f"⚠️  Skipped {parse_stats['invalid_json']} invalid JSON files")

    if errors and '--verbose' in sys.argv:
        print("\nErrors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Dry run
    print("\n" + "=" * 80)
    print("DRY RUN - Preview of operations")
    print("=" * 80)

    dry_stats, missing_images = organize_images(mapping, dry_run=True)

    print(f"\nImages to be copied:")
    print(f"   Gram-positive: {dry_stats['gram_positive']}")
    print(f"   Gram-negative: {dry_stats['gram_negative']}")
    print(f"   Total new images: {dry_stats['gram_positive'] + dry_stats['gram_negative']}")

    if dry_stats['missing_image'] > 0:
        print(f"\n⚠️  Warning: {dry_stats['missing_image']} images not found in {IMAGES_DIR}")
        if '--verbose' in sys.argv and missing_images:
            print("Missing images:")
            for img in missing_images[:10]:
                print(f"  - {img}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")

    # Calculate final counts
    final_pos = existing_pos + dry_stats['gram_positive']
    final_neg = existing_neg + dry_stats['gram_negative']

    print(f"\n📊 After organization:")
    print(f"   Gram-positive: {final_pos} images ({existing_pos} existing + {dry_stats['gram_positive']} new)")
    print(f"   Gram-negative: {final_neg} images ({existing_neg} existing + {dry_stats['gram_negative']} new)")
    print(f"   Total: {final_pos + final_neg} images")

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

        copy_stats, _ = organize_images(mapping, dry_run=False)

        print("\n" + "=" * 80)
        print("✅ ORGANIZATION COMPLETE!")
        print("=" * 80)
        print(f"\nImages copied:")
        print(f"   Gram-positive: {copy_stats['gram_positive']}")
        print(f"   Location: {OUTPUT_DIR / 'gram_positive'}")
        print(f"   Gram-negative: {copy_stats['gram_negative']}")
        print(f"   Location: {OUTPUT_DIR / 'gram_negative'}")

        if copy_stats['already_exists'] > 0:
            print(f"\n⚠️  Skipped {copy_stats['already_exists']} images (already exist)")

        # Verify final counts
        final_pos_count, final_neg_count = count_existing_images()
        print(f"\n📊 Final dataset:")
        print(f"   Gram-positive: {final_pos_count} images")
        print(f"   Gram-negative: {final_neg_count} images")
        print(f"   Total: {final_pos_count + final_neg_count} images")

        print("\n📝 Next steps:")
        print("   1. Verify images: ls data/01_raw/gram_positive/ | wc -l")
        print("   2. Run preprocessing: poetry run python scripts/prepare_data.py")
        print("   3. Start training: bacterial-gan train")
    else:
        print("\n❌ Operation cancelled")


if __name__ == "__main__":
    main()
