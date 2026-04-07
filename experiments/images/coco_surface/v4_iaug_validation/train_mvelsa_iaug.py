"""
Trains MVELSA specialists on the combined FHD+IAUG dataset.

Same architecture as V3, but training data includes augmented samples.
Specialists for LAND and SHIP are trained only on FHD data (IAUG excluded
those classes due to domain shift degradation).

Usage:
    python train_mvelsa_iaug.py

Environment variables:
    COMBINED_DATA_PATH — path to coco_cropped_combined (train split)
    FHD_DATA_PATH      — path to coco_cropped (for LAND/SHIP specialists)
    MODEL_DIR          — where to save ELSA_MODEL_IAUG (default: ./ELSA_MODEL_IAUG)
    AE_TIMES           — autoencoder training passes (default: 2)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM

BASE_DIR      = Path(__file__).resolve().parent
COMBINED_DATA = os.environ.get("COMBINED_DATA_PATH", str(BASE_DIR / "../../../../data/coco_cropped_combined"))
FHD_DATA      = os.environ.get("FHD_DATA_PATH",      str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR     = os.environ.get("MODEL_DIR",           str(BASE_DIR / "ELSA_MODEL_IAUG"))
AE_TIMES      = int(os.environ.get("AE_TIMES", "2"))

FOCUS_CLASSES = {1, 3, 4, 5, 6}
# Classes trained only on FHD data (IAUG domain shift degrades these)
FHD_ONLY_CLASSES = {4, 6}  # LAND, SHIP

EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_specialist(class_id, class_name, train_dir, model_save_dir):
    print(f"\n=== Training specialist: {class_name} ===")

    ds_own = CroppedSurfaceDataset(train_dir, focus_classes=FOCUS_CLASSES,
                                    single_class=class_id)
    ds_train = ConcatDataset([ds_own] * AE_TIMES)
    loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    print(f"  Samples: {len(ds_own)} × {AE_TIMES} = {len(ds_train)}")

    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            imgs = batch[0].to(DEVICE)
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(ds_train)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}")

    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, f"ae_{class_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")
    return model


def main():
    print(f"Device:        {DEVICE}")
    print(f"Combined data: {COMBINED_DATA}")
    print(f"FHD data:      {FHD_DATA}")
    print(f"Model dir:     {MODEL_DIR}")
    print(f"AE times:      {AE_TIMES}")

    all_losses = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]

        # LAND and SHIP: use FHD only
        if cls_id in FHD_ONLY_CLASSES:
            train_dir = os.path.join(FHD_DATA, 'train')
            print(f"\n  {cls_name}: using FHD-only data (domain shift issue)")
        else:
            train_dir = os.path.join(COMBINED_DATA, 'train')

        train_specialist(cls_id, cls_name, train_dir, MODEL_DIR)

    print("\nTraining complete!")

    # Save loss graph if matplotlib available
    try:
        import matplotlib.pyplot as plt
        # (losses not collected in this simplified version — skip graph)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
