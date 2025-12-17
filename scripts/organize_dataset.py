"""
Organize DeepDataSet by bacterial type based on JSON annotations.

This script:
1. Reads JSON annotations for each image
2. Determines Gram type from labels (G+ = positive, G = negative)
3. Copies/moves images to data/01_raw/gram_positive/ and gram_negative/

Usage:
    poetry run python scripts/organize_dataset.py [--copy | --move] [--dry-run]
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_gram_type_from_json(json_path: Path) -> str | None:
    """
    Determine Gram type from JSON annotation.

    Returns:
        'gram_positive' for G+ labels
        'gram_negative' for G labels
        None if no valid label found
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        labels = set()
        for shape in data.get("shapes", []):
            label = shape.get("label", "").upper()
            labels.add(label)

        if "G+" in labels:
            return "gram_positive"
        elif "G" in labels:
            return "gram_negative"
        else:
            return None

    except Exception as e:
        print(f"  ⚠️  Error reading {json_path.name}: {e}")
        return None


def organize_deepdataset(
    dataset_dir: Path,
    output_dir: Path,
    copy_files: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Organize DeepDataSet images by Gram type.

    Args:
        dataset_dir: Path to 640DataSet directory
        output_dir: Output directory (data/01_raw/)
        copy_files: If True, copy files. If False, move files.
        dry_run: If True, only show what would be done

    Returns:
        Dictionary with statistics
    """
    images_dir = dataset_dir / "images"
    json_dir = dataset_dir / "json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    gram_positive_dir = output_dir / "gram_positive"
    gram_negative_dir = output_dir / "gram_negative"

    if not dry_run:
        gram_positive_dir.mkdir(parents=True, exist_ok=True)
        gram_negative_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "gram_positive": 0,
        "gram_negative": 0,
        "no_label": 0,
        "no_json": 0,
        "no_image": 0,
        "errors": [],
    }

    json_files = sorted(json_dir.glob("*.json"))
    print(f"\n📁 Found {len(json_files)} JSON annotation files")

    for idx, json_path in enumerate(json_files, 1):
        image_name = json_path.stem + ".jpg"
        image_path = images_dir / image_name

        if not image_path.exists():
            for ext in [".png", ".jpeg", ".JPG", ".PNG"]:
                alt_path = images_dir / (json_path.stem + ext)
                if alt_path.exists():
                    image_path = alt_path
                    break

        if not image_path.exists():
            stats["no_image"] += 1
            continue

        gram_type = get_gram_type_from_json(json_path)

        if gram_type is None:
            stats["no_label"] += 1
            continue

        if gram_type == "gram_positive":
            dest_dir = gram_positive_dir
            stats["gram_positive"] += 1
        else:
            dest_dir = gram_negative_dir
            stats["gram_negative"] += 1

        dest_path = dest_dir / image_path.name

        if idx % 500 == 0 or idx == len(json_files):
            print(f"  [{idx}/{len(json_files)}] Processing...")

        if not dry_run:
            try:
                if copy_files:
                    shutil.copy2(image_path, dest_path)
                else:
                    shutil.move(str(image_path), str(dest_path))
            except Exception as e:
                stats["errors"].append(f"{image_path.name}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Organize DeepDataSet by Gram type")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/01_raw/DeepDataSet/640DataSet",
        help="Path to 640DataSet directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/01_raw",
        help="Output directory for organized images",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ORGANIZE DEEPDATASET")
    print("=" * 80)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {'MOVE' if args.move else 'COPY'}")
    print(f"  Dry run: {args.dry_run}")

    print("\nLabel mapping:")
    print("  G+ → gram_positive")
    print("  G  → gram_negative")

    if args.dry_run:
        print("\n⚠️  DRY RUN - No files will be copied/moved")

    try:
        stats = organize_deepdataset(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            copy_files=not args.move,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"  ✅ Gram-positive: {stats['gram_positive']}")
        print(f"  ✅ Gram-negative: {stats['gram_negative']}")
        print(f"  ⚠️  No label: {stats['no_label']}")
        print(f"  ⚠️  No image: {stats['no_image']}")
        if stats["errors"]:
            print(f"  ❌ Errors: {len(stats['errors'])}")

        print(f"\n📂 Output directories:")
        print(f"  {output_dir}/gram_positive/")
        print(f"  {output_dir}/gram_negative/")

        if not args.dry_run:
            print("\n✅ Dataset organized successfully!")
            print("\n📝 Next step: Run prepare_data.py to preprocess")
            print("   poetry run python scripts/prepare_data.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
