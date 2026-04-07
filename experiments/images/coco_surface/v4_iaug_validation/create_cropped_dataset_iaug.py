"""
Creates the IAUG cropped dataset from the seadev_2_IAUG COCO annotations.

Key differences from fullHD version:
  - Only BOAT, BUOY, SKY from IAUG (LAND and SHIP excluded due to domain shift)
  - BUOY capped at 3000 samples
  - Category ID remapping: SHIP(6→5), SKY(7→6) in IAUG annotations

Usage:
    python create_cropped_dataset_iaug.py

Environment variables:
    IAUG_JSON_PATH — path to IAUG _annotations.coco.json
    IAUG_IMG_DIR   — IAUG train images directory
    IAUG_OUT_DIR   — output directory (default: ../../../../data/coco_cropped_iaug)
"""

import os
import json
import random
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

BASE_DIR      = Path(__file__).resolve().parent

IAUG_JSON_PATH = os.environ.get(
    "IAUG_JSON_PATH",
    str(BASE_DIR / "../../../../data/seadev_2_IAUG.v1-com-iaug.coco/train/_annotations.coco.json")
)
IAUG_IMG_DIR = os.environ.get(
    "IAUG_IMG_DIR",
    str(BASE_DIR / "../../../../data/seadev_2_IAUG.v1-com-iaug.coco/train")
)
IAUG_OUT_DIR = os.environ.get(
    "IAUG_OUT_DIR",
    str(BASE_DIR / "../../../../data/coco_cropped_iaug")
)

# IAUG category IDs (after remapping)
# Original IAUG: BOAT=1, BUOY=3, LAND=4, SHIP=6→5, SKY=7→6
# IAUG_REMAP: maps IAUG category_id → standard category_id
IAUG_REMAP = {
    1: 1,  # BOAT
    3: 3,  # BUOY
    4: 4,  # LAND — excluded
    6: 5,  # SHIP → remap to 5 — excluded
    7: 6,  # SKY  → remap to 6
}

CLASS_NAMES = {
    1: 'BOAT',
    3: 'BUOY',
    4: 'LAND',
    5: 'SHIP',
    6: 'SKY',
}

# Only these standard class IDs included from IAUG
FOCUS_CLASSES_BASE = {1, 3, 4, 5, 6}
EXCLUDED_CLASSES   = {4, 6}  # LAND and SHIP excluded from IAUG
FOCUS_CLASSES      = FOCUS_CLASSES_BASE - EXCLUDED_CLASSES

# Per-class caps for IAUG
MAX_SAMPLES_PER_CLASS = {
    3: 3000,  # BUOY only — still overrepresented
}

CROP_SIZE   = 128
MARGIN      = 0.05
TRAIN_RATIO = 1.0  # IAUG data used only for training (validation stays FHD)


def crop_square(img, x, y, w, h, class_id):
    img_w, img_h = img.size
    if class_id == 5:
        side = min(w, max(h, int(h * 2)))
    else:
        side = min(w, h)
    side = int(side * (1 + MARGIN))
    cx = x + w // 2
    cy = y + h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_w, x1 + side)
    y2 = min(img_h, y1 + side)
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)
    return img.crop((x1, y1, x2, y2))


def main():
    print(f"Loading IAUG annotations: {IAUG_JSON_PATH}")
    with open(IAUG_JSON_PATH) as f:
        coco = json.load(f)

    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

    # Filter and remap category IDs
    valid_iaug_ids = {k for k, v in IAUG_REMAP.items() if v in FOCUS_CLASSES}
    anns = [a for a in coco['annotations'] if a['category_id'] in valid_iaug_ids]

    # Group by standard class
    class_anns = {c: [] for c in FOCUS_CLASSES}
    for ann in anns:
        std_id = IAUG_REMAP[ann['category_id']]
        if std_id in FOCUS_CLASSES:
            class_anns[std_id].append(ann)

    print("IAUG class counts (before cap):")
    for cls_id, anns_list in class_anns.items():
        print(f"  {CLASS_NAMES[cls_id]}: {len(anns_list)}")

    # Apply caps
    for cls_id, cap in MAX_SAMPLES_PER_CLASS.items():
        if cls_id in class_anns and len(class_anns[cls_id]) > cap:
            random.shuffle(class_anns[cls_id])
            class_anns[cls_id] = class_anns[cls_id][:cap]

    out_train_dir = os.path.join(IAUG_OUT_DIR, 'train')
    Path(out_train_dir).mkdir(parents=True, exist_ok=True)

    csv_fields = ['filename', 'class_id', 'class_name',
                  'cy_norm', 'aspect_ratio', 'bbox_area_norm',
                  'bbox_w_norm', 'bbox_h_norm']

    all_samples = []
    print("\nCropping IAUG images...")

    for cls_id in FOCUS_CLASSES:
        cls_name = CLASS_NAMES[cls_id]
        anns_for_class = class_anns.get(cls_id, [])
        print(f"  {cls_name}: {len(anns_for_class)} samples")

        for ann in tqdm(anns_for_class, desc=cls_name, leave=False):
            img_id = ann['image_id']
            ann_id = ann['id']
            x, y, w, h = [int(v) for v in ann['bbox']]
            img_file = id_to_file[img_id]
            img_w, img_h = id_to_size[img_id]

            src_path = os.path.join(IAUG_IMG_DIR, img_file)
            if not os.path.exists(src_path):
                continue

            try:
                img = Image.open(src_path).convert('RGB')
                crop = crop_square(img, x, y, w, h, cls_id)
                crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
            except Exception as e:
                continue

            cy_norm        = round((y + h / 2) / img_h, 4)
            aspect_ratio   = round(w / h, 4) if h > 0 else 1.0
            bbox_area_norm = round((w * h) / (img_w * img_h), 4)
            bbox_w_norm    = round(w / img_w, 4)
            bbox_h_norm    = round(h / img_h, 4)

            out_name = f"{cls_name}_{ann_id}.jpg"
            all_samples.append({
                'filename':       out_name,
                'class_id':       cls_id,
                'class_name':     cls_name,
                'cy_norm':        cy_norm,
                'aspect_ratio':   aspect_ratio,
                'bbox_area_norm': bbox_area_norm,
                'bbox_w_norm':    bbox_w_norm,
                'bbox_h_norm':    bbox_h_norm,
                '_crop':          crop,
            })

    csv_path = os.path.join(out_train_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in tqdm(all_samples, desc="Saving"):
            crop = s.pop('_crop')
            crop.save(os.path.join(out_train_dir, s['filename']), quality=95)
            writer.writerow({k: s[k] for k in csv_fields})

    print(f"\nIAUG dataset saved: {len(all_samples)} images → {IAUG_OUT_DIR}")


if __name__ == '__main__':
    random.seed(42)
    main()
