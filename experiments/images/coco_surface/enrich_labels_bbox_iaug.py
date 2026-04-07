"""
Enriches IAUG labels.csv files with bbox metadata.

Handles both fhd_* and iaug_* prefixed files in the combined dataset:
  - fhd_BOAT_2291.jpg → look up ann_id=2291 in FHD COCO JSON
  - iaug_BOAT_9999.jpg → look up ann_id=9999 in IAUG COCO JSON

Updates:
  - coco_cropped_iaug/train/labels.csv
  - coco_cropped_combined/train/labels.csv

Usage:
    python enrich_labels_bbox_iaug.py

Environment variables:
    FHD_JSON_PATH  — fullHD633 _annotations.coco.json
    IAUG_JSON_PATH — IAUG _annotations.coco.json
    IAUG_DATA_DIR  — coco_cropped_iaug directory
    COMBINED_DIR   — coco_cropped_combined directory
"""

import os
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FHD_JSON_PATH = os.environ.get(
    "FHD_JSON_PATH",
    str(BASE_DIR / "../../../data/seadev_fullHD633/train/_annotations.coco.json")
)
IAUG_JSON_PATH = os.environ.get(
    "IAUG_JSON_PATH",
    str(BASE_DIR / "../../../data/seadev_2_IAUG.v1-com-iaug.coco/train/_annotations.coco.json")
)
IAUG_DATA_DIR = os.environ.get(
    "IAUG_DATA_DIR",
    str(BASE_DIR / "../../../data/coco_cropped_iaug")
)
COMBINED_DIR = os.environ.get(
    "COMBINED_DIR",
    str(BASE_DIR / "../../../data/coco_cropped_combined")
)


def build_bbox_map(coco_json_path):
    with open(coco_json_path) as f:
        coco = json.load(f)

    id_to_size = {img['id']: (img['width'], img['height'])
                  for img in coco['images']}

    bbox_map = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        ann_id = ann['id']
        x, y, w, h = ann['bbox']
        img_w, img_h = id_to_size[img_id]
        if img_w == 0 or img_h == 0:
            continue

        bbox_map[ann_id] = {
            'cy_norm':        round((y + h / 2) / img_h, 4),
            'aspect_ratio':   round(w / h, 4) if h > 0 else 1.0,
            'bbox_area_norm': round((w * h) / (img_w * img_h), 4),
            'bbox_w_norm':    round(w / img_w, 4),
            'bbox_h_norm':    round(h / img_h, 4),
        }

    return bbox_map


def enrich_csv_combined(csv_path, fhd_map, iaug_map):
    rows = []
    updated = 0
    skipped = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        extra_cols = ['cy_norm', 'aspect_ratio', 'bbox_area_norm', 'bbox_w_norm', 'bbox_h_norm']
        new_fields = list(fieldnames)
        for col in extra_cols:
            if col not in new_fields:
                new_fields.append(col)

        for row in reader:
            filename = row['filename']
            stem = Path(filename).stem

            try:
                ann_id = int(stem.split('_')[-1])
            except (ValueError, IndexError):
                skipped += 1
                rows.append(row)
                continue

            # Route by prefix
            if filename.startswith('fhd_'):
                bbox_info = fhd_map.get(ann_id)
            elif filename.startswith('iaug_'):
                bbox_info = iaug_map.get(ann_id)
            else:
                bbox_info = fhd_map.get(ann_id) or iaug_map.get(ann_id)

            if bbox_info:
                row.update(bbox_info)
                updated += 1
            else:
                skipped += 1
            rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {csv_path}: {updated} updated, {skipped} skipped")


def enrich_csv_simple(csv_path, bbox_map):
    rows = []
    updated = 0
    skipped = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        extra_cols = ['cy_norm', 'aspect_ratio', 'bbox_area_norm', 'bbox_w_norm', 'bbox_h_norm']
        new_fields = list(fieldnames)
        for col in extra_cols:
            if col not in new_fields:
                new_fields.append(col)

        for row in reader:
            filename = row['filename']
            try:
                ann_id = int(Path(filename).stem.split('_')[-1])
            except (ValueError, IndexError):
                skipped += 1
                rows.append(row)
                continue

            if ann_id in bbox_map:
                row.update(bbox_map[ann_id])
                updated += 1
            else:
                skipped += 1
            rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {csv_path}: {updated} updated, {skipped} skipped")


def main():
    print(f"Building FHD bbox map: {FHD_JSON_PATH}")
    fhd_map = build_bbox_map(FHD_JSON_PATH)
    print(f"  {len(fhd_map)} FHD annotations")

    print(f"Building IAUG bbox map: {IAUG_JSON_PATH}")
    iaug_map = build_bbox_map(IAUG_JSON_PATH)
    print(f"  {len(iaug_map)} IAUG annotations")

    # Enrich IAUG standalone dataset
    iaug_csv = os.path.join(IAUG_DATA_DIR, 'train', 'labels.csv')
    if os.path.exists(iaug_csv):
        print(f"\nEnriching IAUG dataset:")
        enrich_csv_simple(iaug_csv, iaug_map)

    # Enrich combined dataset (handles both prefixes)
    combined_csv = os.path.join(COMBINED_DIR, 'train', 'labels.csv')
    if os.path.exists(combined_csv):
        print(f"\nEnriching combined dataset:")
        enrich_csv_combined(combined_csv, fhd_map, iaug_map)

    print("\nEnrichment complete!")


if __name__ == '__main__':
    main()
