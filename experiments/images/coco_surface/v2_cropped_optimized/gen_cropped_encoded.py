"""
Generates encoded latent vectors for all images using all MVELSA specialists.

For each image, passes it through all 5 specialist autoencoders and
concatenates the latent vectors → 5×256=1280D representation.

Output:
    ENCODED_DATA_CROPPED_SURFACE/
        train_encoded.pt  — {'features': Tensor[N,1280], 'labels': Tensor[N]}
        valid_encoded.pt
        test_encoded.pt

Usage:
    python gen_cropped_encoded.py

Environment variables:
    DATA_DIR    — coco_cropped directory
    MODEL_DIR   — ELSA_MODEL_CROPPED_SURFACE directory
    ENCODED_DIR — output directory (ENCODED_DATA_CROPPED_SURFACE)
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM, FOCUS_CLASSES

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = os.environ.get("DATA_DIR",     str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR    = os.environ.get("MODEL_DIR",    str(BASE_DIR / "ELSA_MODEL_CROPPED_SURFACE"))
ENCODED_DIR  = os.environ.get("ENCODED_DIR",  str(BASE_DIR / "ENCODED_DATA_CROPPED_SURFACE"))

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE   = 128


def load_specialists():
    specialists = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]
        model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
        ckpt = os.path.join(MODEL_DIR, f"ae_{cls_name}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        specialists[cls_name] = model
        print(f"  Loaded specialist: {cls_name}")
    return specialists


def encode_split(split, data_dir, specialists):
    ds = CroppedSurfaceDataset(
        os.path.join(data_dir, split),
        focus_classes=FOCUS_CLASSES
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_features = []
    all_labels   = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding {split}"):
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(DEVICE)

            # Concatenate latents from all specialists
            latents = []
            for cls_name in sorted(specialists.keys()):
                z = specialists[cls_name].encode(imgs)
                latents.append(z.cpu())
            concat = torch.cat(latents, dim=1)  # [B, 5*256]

            all_features.append(concat)
            all_labels.append(labels)

    features = torch.cat(all_features, dim=0)
    labels   = torch.cat(all_labels,   dim=0)

    print(f"  {split}: {features.shape[0]} samples, {features.shape[1]}D")
    return {'features': features, 'labels': labels}


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(ENCODED_DIR, exist_ok=True)

    print("Loading specialists...")
    specialists = load_specialists()

    for split in ('train', 'valid', 'test'):
        data = encode_split(split, DATA_DIR, specialists)
        out_path = os.path.join(ENCODED_DIR, f"{split}_encoded.pt")
        torch.save(data, out_path)
        print(f"  Saved: {out_path}")

    print("\nEncoding complete!")


if __name__ == '__main__':
    main()
