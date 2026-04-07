"""
Removes labels.csv entries for image files that no longer exist on disk.

Useful after manually deleting bad/noisy crops via inspecionar_crops.py.

Usage:
    python sync_labels_csv.py [--data-dir PATH]

Default data dir: ../../../data/coco_cropped
"""

import os
import csv
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(BASE_DIR / "../../../data/coco_cropped")


def sync_split(split_dir):
    csv_path = os.path.join(split_dir, 'labels.csv')
    if not os.path.exists(csv_path):
        print(f"  {split_dir}: no labels.csv, skipping")
        return

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    original_count = len(rows)
    valid_rows = []
    removed = []

    for row in rows:
        img_path = os.path.join(split_dir, row['filename'])
        if os.path.exists(img_path):
            valid_rows.append(row)
        else:
            removed.append(row['filename'])

    if removed:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_rows)

        print(f"  {os.path.basename(split_dir)}: removed {len(removed)}/{original_count} entries")
        for name in removed[:10]:
            print(f"    - {name}")
        if len(removed) > 10:
            print(f"    ... and {len(removed)-10} more")
    else:
        print(f"  {os.path.basename(split_dir)}: all {original_count} files present, no changes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Syncing labels in: {data_dir}")

    for split in ('train', 'valid', 'test'):
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            sync_split(split_dir)
        else:
            print(f"  {split}: directory not found, skipping")

    print("\nSync complete!")


if __name__ == '__main__':
    main()
