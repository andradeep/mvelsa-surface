"""
Enriches the fullHD coco_cropped labels.csv files with bbox metadata.

Reads bbox dimensions from the original COCO annotation JSON and adds:
  aspect_ratio, bbox_area_norm, bbox_w_norm, bbox_h_norm

Matches via ann_id embedded in crop filename: e.g. BOAT_2291.jpg → ann_id=2291

Usage:
    python enrich_labels_bbox.py

Environment variables:
    COCO_JSON_PATH — fullHD633 _annotations.coco.json
    DATA_DIR       — coco_cropped directory (default: ../../../data/coco_cropped)
"""

import os
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

COCO_JSON_PATH = os.environ.get(
    "COCO_JSON_PATH",
    str(BASE_DIR / "../../../data/seadev_fullHD633/train/_annotations.coco.json")
)
DATA_DIR = os.environ.get(
    "DATA_DIR",
    str(BASE_DIR / "../../../data/coco_cropped")
)


def build_bbox_map(coco_json_path):
    """Build ann_id → bbox metadata dict from COCO JSON."""
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

        bbox_map[ann_id] = {
            'cy_norm':        round((y + h / 2) / img_h, 4),
            'aspect_ratio':   round(w / h, 4) if h > 0 else 1.0,
            'bbox_area_norm': round((w * h) / (img_w * img_h), 4),
            'bbox_w_norm':    round(w / img_w, 4),
            'bbox_h_norm':    round(h / img_h, 4),
        }

    return bbox_map


def enrich_csv(csv_path, bbox_map):
    rows = []
    updated = 0
    skipped = 0

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Add new columns if missing
        extra_cols = ['cy_norm', 'aspect_ratio', 'bbox_area_norm', 'bbox_w_norm', 'bbox_h_norm']
        new_fields = list(fieldnames)
        for col in extra_cols:
            if col not in new_fields:
                new_fields.append(col)

        for row in reader:
            # Extract ann_id from filename: CLASS_ANNID.jpg
            filename = row['filename']
            try:
                stem = Path(filename).stem
                ann_id = int(stem.split('_')[-1])
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
    print(f"Loading COCO annotations: {COCO_JSON_PATH}")
    bbox_map = build_bbox_map(COCO_JSON_PATH)
    print(f"Loaded {len(bbox_map)} annotations")

    for split in ('train', 'valid', 'test'):
        csv_path = os.path.join(DATA_DIR, split, 'labels.csv')
        if os.path.exists(csv_path):
            enrich_csv(csv_path, bbox_map)
        else:
            print(f"  {csv_path}: not found, skipping")

    print("\nEnrichment complete!")


if __name__ == '__main__':
    main()
