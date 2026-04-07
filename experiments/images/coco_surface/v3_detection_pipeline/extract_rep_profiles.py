"""
Extracts REP (Reconstruction Error Profile) feature vectors for all images.

For each image, computes reconstruction error against all 5 specialists
(10 errors per specialist: mean+std for 5 patches) plus global bbox features.

Profile = [50 reconstruction features] + [5 bbox features] = 55D

Output:
    rep_profiles_train.pt  — {'profiles': Tensor[N,55], 'labels': Tensor[N], 'global_features': 5}
    rep_profiles_val.pt

Usage:
    python extract_rep_profiles.py

Environment variables:
    DATA_DIR   — coco_cropped directory
    MODEL_DIR  — ELSA_MODEL_CROPPED_SURFACE directory
    OUT_DIR    — where to save .pt files (default: ./)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM, FOCUS_CLASSES

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = os.environ.get("DATA_DIR",  str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR.parent / "v2_cropped_optimized/ELSA_MODEL_CROPPED_SURFACE"))
OUT_DIR   = os.environ.get("OUT_DIR",   str(BASE_DIR))

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
CRITERION  = nn.MSELoss(reduction='none')

# Number of spatial patches to compute reconstruction stats over
N_PATCHES = 5


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


def compute_rep_features(imgs, specialists):
    """
    Computes 50 reconstruction error features per image:
    For each of 5 specialists: 10 values = mean+std across 5 spatial patches
    """
    batch_size = imgs.shape[0]
    features = []

    with torch.no_grad():
        for cls_name in sorted(specialists.keys()):
            model = specialists[cls_name]
            recon, _ = model(imgs)
            err = CRITERION(recon, imgs)  # [B, C, H, W]

            # Divide spatial dims into N_PATCHES patches
            err_mean_all = err.mean(dim=1)  # [B, H, W]
            h, w = err_mean_all.shape[1], err_mean_all.shape[2]
            ph, pw = h // N_PATCHES, max(1, w // N_PATCHES)

            patch_vals = []
            for pi in range(N_PATCHES):
                patch = err_mean_all[:, pi*ph:(pi+1)*ph, :]
                patch_vals.append(patch.mean(dim=[1, 2]))  # [B]

            patch_tensor = torch.stack(patch_vals, dim=1)  # [B, N_PATCHES]
            spec_mean = patch_tensor.mean(dim=1, keepdim=True)
            spec_std  = patch_tensor.std(dim=1, keepdim=True)
            # [mean_p1..p5, global_mean, global_std] = 7 values
            spec_feats = torch.cat([patch_tensor, spec_mean, spec_std], dim=1)  # [B, 7]
            features.append(spec_feats.cpu())

    return torch.cat(features, dim=1)  # [B, 5*7=35] — but we use 10 per specialist below


def compute_rep_profiles_batch(imgs, specialists):
    """
    10 features per specialist: reconstruction error at 5 patches (mean, std each).
    Total = 5 specialists × 10 = 50 features.
    """
    batch_size = imgs.shape[0]
    all_feats = []

    with torch.no_grad():
        for cls_name in sorted(specialists.keys()):
            model = specialists[cls_name]
            recon, _ = model(imgs)
            err = CRITERION(recon, imgs).mean(dim=1)  # [B, H, W]
            h, w = err.shape[1], err.shape[2]

            ph = h // N_PATCHES
            patch_means = []
            patch_stds  = []
            for pi in range(N_PATCHES):
                patch = err[:, pi*ph:(pi+1)*ph, :]
                patch_means.append(patch.mean(dim=[1, 2]))
                patch_stds.append(patch.std(dim=[1, 2]))

            means = torch.stack(patch_means, dim=1)  # [B, 5]
            stds  = torch.stack(patch_stds,  dim=1)  # [B, 5]
            # interleave: [m1,s1,m2,s2,...] = 10
            interleaved = torch.stack([means, stds], dim=2).reshape(batch_size, -1)
            all_feats.append(interleaved.cpu())

    return torch.cat(all_feats, dim=1)  # [B, 50]


def extract_split(split, data_dir, specialists):
    ds = CroppedSurfaceDataset(
        os.path.join(data_dir, split),
        focus_classes=FOCUS_CLASSES,
        return_meta=True
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_profiles = []
    all_labels   = []

    for imgs, labels, meta in tqdm(loader, desc=f"Extracting {split}"):
        imgs = imgs.to(DEVICE)

        # 50 reconstruction features
        rec_feats = compute_rep_profiles_batch(imgs, specialists)  # [B, 50]

        # 5 bbox global features from meta
        global_feats = meta.cpu()  # [B, 5]

        # Full 55D profile
        profile = torch.cat([rec_feats, global_feats], dim=1)  # [B, 55]

        all_profiles.append(profile)
        all_labels.append(labels)

    profiles = torch.cat(all_profiles, dim=0)
    labels   = torch.cat(all_labels,   dim=0)
    print(f"  {split}: {profiles.shape[0]} samples, {profiles.shape[1]}D profile")
    return profiles, labels


def main():
    print(f"Device: {DEVICE}")
    print("Loading specialists...")
    specialists = load_specialists()

    for split in ('train', 'valid'):
        profiles, labels = extract_split(split, DATA_DIR, specialists)
        out_path = os.path.join(OUT_DIR, f"rep_profiles_{split}.pt")
        torch.save({
            'profiles': profiles,
            'labels': labels,
            'global_features': 5,
        }, out_path)
        print(f"  Saved: {out_path}")

    print("\nExtraction complete!")


if __name__ == '__main__':
    main()
