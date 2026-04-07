"""
Merges the fullHD (FHD) training set with IAUG augmented training set.

Files from each source are prefixed (fhd_ / iaug_) to avoid name collisions.
The merged dataset inherits the FHD validation and test splits unchanged.

Output:
    coco_cropped_combined/
        train/   — fhd_* + iaug_* images + labels.csv
        valid/   — symlink/copy from FHD
        test/    — symlink/copy from FHD

Usage:
    python merge_datasets.py

Environment variables:
    FHD_DATA_PATH  — path to coco_cropped (default: ../../../../data/coco_cropped)
    IAUG_OUT_DIR   — path to coco_cropped_iaug (default: ../../../../data/coco_cropped_iaug)
    COMBINED_DIR   — output path (default: ../../../../data/coco_cropped_combined)
"""

import os
import shutil
import csv
from pathlib import Path
from tqdm import tqdm

BASE_DIR      = Path(__file__).resolve().parent
FHD_DATA_PATH = os.environ.get("FHD_DATA_PATH", str(BASE_DIR / "../../../../data/coco_cropped"))
IAUG_OUT_DIR  = os.environ.get("IAUG_OUT_DIR",  str(BASE_DIR / "../../../../data/coco_cropped_iaug"))
COMBINED_DIR  = os.environ.get("COMBINED_DIR",  str(BASE_DIR / "../../../../data/coco_cropped_combined"))

CSV_FIELDS = ['filename', 'class_id', 'class_name',
              'cy_norm', 'aspect_ratio', 'bbox_area_norm',
              'bbox_w_norm', 'bbox_h_norm']


def read_csv(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def copy_with_prefix(src_dir, dst_dir, prefix, rows):
    """Copy images with prefix and return updated rows."""
    updated = []
    for row in tqdm(rows, desc=f"Copying {prefix}", leave=False):
        src_file = os.path.join(src_dir, row['filename'])
        if not os.path.exists(src_file):
            continue
        new_filename = prefix + row['filename']
        dst_file = os.path.join(dst_dir, new_filename)
        shutil.copy2(src_file, dst_file)
        new_row = dict(row)
        new_row['filename'] = new_filename
        updated.append(new_row)
    return updated


def main():
    # Prepare combined train directory
    combined_train = os.path.join(COMBINED_DIR, 'train')
    Path(combined_train).mkdir(parents=True, exist_ok=True)

    # Load FHD train rows
    fhd_train_dir = os.path.join(FHD_DATA_PATH, 'train')
    fhd_rows = read_csv(os.path.join(fhd_train_dir, 'labels.csv'))
    print(f"FHD train: {len(fhd_rows)} samples")

    # Load IAUG train rows
    iaug_train_dir = os.path.join(IAUG_OUT_DIR, 'train')
    iaug_rows = read_csv(os.path.join(iaug_train_dir, 'labels.csv'))
    print(f"IAUG train: {len(iaug_rows)} samples")

    # Copy with prefixes
    fhd_updated  = copy_with_prefix(fhd_train_dir,  combined_train, 'fhd_',  fhd_rows)
    iaug_updated = copy_with_prefix(iaug_train_dir, combined_train, 'iaug_', iaug_rows)

    # Write combined labels.csv
    combined_rows = fhd_updated + iaug_updated
    csv_out = os.path.join(combined_train, 'labels.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Combined train: {len(combined_rows)} samples")

    # Copy valid and test from FHD (no prefix needed — used as-is)
    for split in ('valid', 'test'):
        src = os.path.join(FHD_DATA_PATH, split)
        dst = os.path.join(COMBINED_DIR, split)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied {split} from FHD.")

    print(f"\nMerged dataset saved to: {COMBINED_DIR}")


if __name__ == '__main__':
    main()
