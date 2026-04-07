"""
Generates MVELSA encoded features for the combined FHD+IAUG dataset.

Uses the IAUG-trained specialists to encode all images.
For validation/test, uses FHD data only (IAUG was training only).

Output:
    ENCODED_DATA_IAUG/
        train_encoded.pt
        valid_encoded.pt
        test_encoded.pt

Usage:
    python gen_encoded_iaug.py

Environment variables:
    COMBINED_DATA_PATH — path to coco_cropped_combined
    FHD_DATA_PATH      — path to coco_cropped (for valid/test)
    MODEL_DIR          — ELSA_MODEL_IAUG directory
    ENCODED_DIR        — output directory (ENCODED_DATA_IAUG)
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM

BASE_DIR      = Path(__file__).resolve().parent
COMBINED_DATA = os.environ.get("COMBINED_DATA_PATH", str(BASE_DIR / "../../../../data/coco_cropped_combined"))
FHD_DATA      = os.environ.get("FHD_DATA_PATH",      str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR     = os.environ.get("MODEL_DIR",           str(BASE_DIR / "ELSA_MODEL_IAUG"))
ENCODED_DIR   = os.environ.get("ENCODED_DIR",         str(BASE_DIR / "ENCODED_DATA_IAUG"))

FOCUS_CLASSES = {1, 3, 4, 5, 6}
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE    = 128


def load_specialists():
    specialists = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]
        model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
        ckpt = os.path.join(MODEL_DIR, f"ae_{cls_name}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        specialists[cls_name] = model
    return specialists


def encode_split(split, data_dir, specialists):
    ds = CroppedSurfaceDataset(os.path.join(data_dir, split),
                                focus_classes=FOCUS_CLASSES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(DEVICE)
            latents = []
            for cls_name in sorted(specialists.keys()):
                z = specialists[cls_name].encode(imgs)
                latents.append(z.cpu())
            concat = torch.cat(latents, dim=1)
            all_features.append(concat)
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels   = torch.cat(all_labels,   dim=0)
    print(f"  {split}: {features.shape[0]} samples, {features.shape[1]}D")
    return {'features': features, 'labels': labels}


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(ENCODED_DIR, exist_ok=True)

    print("Loading IAUG specialists...")
    specialists = load_specialists()

    # Train: combined data
    data = encode_split('train', COMBINED_DATA, specialists)
    torch.save(data, os.path.join(ENCODED_DIR, 'train_encoded.pt'))

    # Valid / Test: FHD only
    for split in ('valid', 'test'):
        data = encode_split(split, FHD_DATA, specialists)
        torch.save(data, os.path.join(ENCODED_DIR, f"{split}_encoded.pt"))

    print(f"\nEncoded data saved to: {ENCODED_DIR}")


if __name__ == '__main__':
    main()
