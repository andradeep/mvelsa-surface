"""
Extracts 55D REP profiles for the V4 IAUG experiment.

Train profiles use combined data; val profiles use FHD only.

Output:
    rep_profiles_train_iaug.pt
    rep_profiles_val_iaug.pt

Usage:
    python extract_rep_profiles_iaug.py

Environment variables:
    COMBINED_DATA_PATH — coco_cropped_combined
    FHD_DATA_PATH      — coco_cropped
    MODEL_DIR          — ELSA_MODEL_IAUG directory
    OUT_DIR            — where to save .pt files (default: ./)
"""

import os
import sys
import torch
import torch.nn as nn
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
OUT_DIR       = os.environ.get("OUT_DIR",             str(BASE_DIR))

FOCUS_CLASSES = {1, 3, 4, 5, 6}
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE    = 64
CRITERION     = nn.MSELoss(reduction='none')
N_PATCHES     = 5


def load_specialists():
    specialists = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]
        model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
        ckpt = os.path.join(MODEL_DIR, f"ae_{cls_name}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        specialists[cls_name] = model
        print(f"  Loaded: {cls_name}")
    return specialists


def compute_rep_profiles_batch(imgs, specialists):
    batch_size = imgs.shape[0]
    all_feats = []
    with torch.no_grad():
        for cls_name in sorted(specialists.keys()):
            model = specialists[cls_name]
            recon, _ = model(imgs)
            err = CRITERION(recon, imgs).mean(dim=1)  # [B, H, W]
            h, w = err.shape[1], err.shape[2]
            ph = h // N_PATCHES
            patch_means, patch_stds = [], []
            for pi in range(N_PATCHES):
                p = err[:, pi*ph:(pi+1)*ph, :]
                patch_means.append(p.mean(dim=[1, 2]))
                patch_stds.append(p.std(dim=[1, 2]))
            means = torch.stack(patch_means, dim=1).cpu()
            stds  = torch.stack(patch_stds,  dim=1).cpu()
            interleaved = torch.stack([means, stds], dim=2).reshape(batch_size, -1)
            all_feats.append(interleaved)
    return torch.cat(all_feats, dim=1)  # [B, 50]


def extract_split(split, data_dir, specialists, suffix=''):
    ds = CroppedSurfaceDataset(
        os.path.join(data_dir, split),
        focus_classes=FOCUS_CLASSES,
        return_meta=True
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_profiles, all_labels = [], []
    for imgs, labels, meta in tqdm(loader, desc=f"Extracting {split}{suffix}"):
        imgs = imgs.to(DEVICE)
        rec_feats    = compute_rep_profiles_batch(imgs, specialists)
        global_feats = meta.cpu()
        profile      = torch.cat([rec_feats, global_feats], dim=1)
        all_profiles.append(profile)
        all_labels.append(labels)

    profiles = torch.cat(all_profiles, dim=0)
    labels   = torch.cat(all_labels,   dim=0)
    print(f"  {split}{suffix}: {profiles.shape[0]} samples, {profiles.shape[1]}D")
    return profiles, labels


def main():
    print(f"Device: {DEVICE}")
    print("Loading IAUG specialists...")
    specialists = load_specialists()

    # Train: combined data
    profiles, labels = extract_split('train', COMBINED_DATA, specialists, suffix=' (combined)')
    torch.save({'profiles': profiles, 'labels': labels, 'global_features': 5},
               os.path.join(OUT_DIR, 'rep_profiles_train_iaug.pt'))

    # Val: FHD only
    profiles, labels = extract_split('valid', FHD_DATA, specialists, suffix=' (FHD)')
    torch.save({'profiles': profiles, 'labels': labels, 'global_features': 5},
               os.path.join(OUT_DIR, 'rep_profiles_val_iaug.pt'))

    print("\nExtraction complete!")


if __name__ == '__main__':
    main()
