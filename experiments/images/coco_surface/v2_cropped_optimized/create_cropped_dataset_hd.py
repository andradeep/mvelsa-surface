"""
Creates the fullHD cropped dataset from COCO annotations.

Reads the COCO JSON annotation file, crops objects with square padding,
and saves them with labels.csv containing bbox metadata.

Usage:
    python create_cropped_dataset_hd.py

Environment variables (optional overrides):
    COCO_JSON_PATH   — path to _annotations.coco.json
    COCO_IMG_DIR     — directory with source images
    OUT_DIR          — output directory for cropped dataset
"""

import os
import json
import shutil
import random
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent

COCO_JSON_PATH = os.environ.get(
    "COCO_JSON_PATH",
    str(BASE_DIR / "../../../../data/seadev_fullHD633/train/_annotations.coco.json")
)
COCO_IMG_DIR = os.environ.get(
    "COCO_IMG_DIR",
    str(BASE_DIR / "../../../../data/seadev_fullHD633/train")
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    str(BASE_DIR / "../../../../data/coco_cropped")
)

# Classes to process (COCO category IDs)
FOCUS_CLASSES = {1, 3, 4, 5, 6}

# Class names mapped from COCO category IDs
CLASS_NAMES = {
    1: 'BOAT',
    3: 'BUOY',
    4: 'LAND',
    5: 'SHIP',
    6: 'SKY',
}

# Per-class sample caps (None = no cap)
MAX_SAMPLES_PER_CLASS = {
    3: 400,   # BUOY — overrepresented
    6: 200,   # SKY — overrepresented
}

CROP_SIZE = 128
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
# TEST_RATIO  = 0.1 (remainder)

MARGIN = 0.05  # fractional padding around tight crop


def crop_square(img, x, y, w, h, class_id):
    """Crop a square region around the bbox with margin."""
    img_w, img_h = img.size

    # For SHIP: preserve aspect with a taller side
    if class_id == 5:
        side = min(w, max(h, int(h * 2)))
    else:
        side = min(w, h)

    # Add margin
    side = int(side * (1 + MARGIN))

    cx = x + w // 2
    cy = y + h // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)

    # Adjust if clipped
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)

    return img.crop((x1, y1, x2, y2))


def main():
    print(f"Loading annotations: {COCO_JSON_PATH}")
    with open(COCO_JSON_PATH) as f:
        coco = json.load(f)

    # Build image_id → filename map
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

    # Filter annotations to focus classes
    anns = [a for a in coco['annotations'] if a['category_id'] in FOCUS_CLASSES]

    # Group by class
    class_anns = {c: [] for c in FOCUS_CLASSES}
    for ann in anns:
        class_anns[ann['category_id']].append(ann)

    # Apply per-class caps
    for cls_id, cap in MAX_SAMPLES_PER_CLASS.items():
        if cls_id in class_anns and len(class_anns[cls_id]) > cap:
            random.shuffle(class_anns[cls_id])
            class_anns[cls_id] = class_anns[cls_id][:cap]

    # Prepare output directories
    for split in ('train', 'valid', 'test'):
        Path(OUT_DIR, split).mkdir(parents=True, exist_ok=True)

    all_samples = []  # (ann_id, class_id, filename_on_disk)

    print("Cropping images...")
    for cls_id in FOCUS_CLASSES:
        cls_name = CLASS_NAMES[cls_id]
        anns_for_class = class_anns[cls_id]
        print(f"  {cls_name}: {len(anns_for_class)} annotations")

        for ann in tqdm(anns_for_class, desc=cls_name, leave=False):
            img_id = ann['image_id']
            ann_id = ann['id']
            x, y, w, h = [int(v) for v in ann['bbox']]
            img_file = id_to_file[img_id]
            img_w, img_h = id_to_size[img_id]

            src_path = os.path.join(COCO_IMG_DIR, img_file)
            if not os.path.exists(src_path):
                continue

            try:
                img = Image.open(src_path).convert('RGB')
                crop = crop_square(img, x, y, w, h, cls_id)
                crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
            except Exception as e:
                print(f"  Skip {img_file}: {e}")
                continue

            # cy_norm: normalized vertical center of bbox
            cy_norm = round((y + h / 2) / img_h, 4)
            aspect_ratio = round(w / h, 4) if h > 0 else 1.0
            bbox_area_norm = round((w * h) / (img_w * img_h), 4)
            bbox_w_norm = round(w / img_w, 4)
            bbox_h_norm = round(h / img_h, 4)

            out_name = f"{cls_name}_{ann_id}.jpg"
            all_samples.append({
                'ann_id': ann_id,
                'class_id': cls_id,
                'class_name': cls_name,
                'filename': out_name,
                'cy_norm': cy_norm,
                'aspect_ratio': aspect_ratio,
                'bbox_area_norm': bbox_area_norm,
                'bbox_w_norm': bbox_w_norm,
                'bbox_h_norm': bbox_h_norm,
                '_crop': crop,
            })

    # Split into train/valid/test
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        'train': all_samples[:n_train],
        'valid': all_samples[n_train:n_train + n_val],
        'test':  all_samples[n_train + n_val:],
    }

    csv_fields = ['filename', 'class_id', 'class_name',
                  'cy_norm', 'aspect_ratio', 'bbox_area_norm',
                  'bbox_w_norm', 'bbox_h_norm']

    for split, samples in splits.items():
        out_split_dir = os.path.join(OUT_DIR, split)
        csv_path = os.path.join(out_split_dir, 'labels.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

            for s in tqdm(samples, desc=f"Saving {split}"):
                crop = s.pop('_crop')
                crop.save(os.path.join(out_split_dir, s['filename']), quality=95)
                writer.writerow({k: s[k] for k in csv_fields})

    # Print summary
    for split, samples in splits.items():
        print(f"  {split}: {len(samples)} images")

    print(f"\nDataset saved to: {OUT_DIR}")


if __name__ == '__main__':
    random.seed(42)
    main()
