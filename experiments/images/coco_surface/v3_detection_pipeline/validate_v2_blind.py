"""
Blind validation of V3 MVELSA on the held-out test split.

Evaluates all three strategies:
  A — Reconstruction error argmin (which specialist reconstructs best)
  B — MLP on 1280D latent concatenation
  C — MLP on 1280D latent (same as B, from encoded data)
  D — Random Forest on 55D REP profile

Usage:
    python validate_v2_blind.py

Environment variables:
    DATA_DIR      — coco_cropped directory
    MODEL_DIR     — ELSA_MODEL_CROPPED_SURFACE directory
    ENCODED_DIR   — ENCODED_DATA_CROPPED_SURFACE directory
    CLASSIFIER_DIR— directory with MVELSA_CLASSIFIER.pth and rep_meta_classifier.pkl
"""

import os
import sys
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM, FOCUS_CLASSES
from train_cropped_classifier import MLPClassifier, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES

BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = os.environ.get("DATA_DIR",       str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR      = os.environ.get("MODEL_DIR",      str(BASE_DIR.parent / "v2_cropped_optimized/ELSA_MODEL_CROPPED_SURFACE"))
ENCODED_DIR    = os.environ.get("ENCODED_DIR",    str(BASE_DIR.parent / "v2_cropped_optimized/ENCODED_DATA_CROPPED_SURFACE"))
CLASSIFIER_DIR = os.environ.get("CLASSIFIER_DIR", str(BASE_DIR.parent / "v2_cropped_optimized"))

GLOBAL_FEATURES = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 64
CRITERION       = nn.MSELoss(reduction='none')
N_PATCHES       = 5

IDX_TO_CLASS = {v: BASE_CLASSES[k] for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES  = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]


def load_specialists():
    specialists = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]
        model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
        ckpt = os.path.join(MODEL_DIR, f"ae_{cls_name}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        specialists[cls_id] = (cls_name, model)
    return specialists


def strategy_a_b(specialists, data_dir):
    """Strategy A: argmin reconstruction error. Also collects latents for B/C."""
    ds = CroppedSurfaceDataset(
        os.path.join(data_dir, 'test'),
        focus_classes=FOCUS_CLASSES,
        return_meta=True
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    y_true_a, y_pred_a = [], []
    all_latents, all_labels = [], []
    all_rep_profiles = []

    sorted_ids = sorted(FOCUS_CLASSES)

    for inputs, labels, meta in tqdm(loader, desc="Strategy A+D"):
        inputs = inputs.to(DEVICE)
        errors  = []
        latents = []

        for cls_id in sorted_ids:
            cls_name, model = specialists[cls_id]
            with torch.no_grad():
                recon, z = model(inputs)
            err = CRITERION(recon, inputs).mean(dim=[1, 2, 3]).cpu()
            errors.append(err)
            latents.append(z.cpu())

        errors_stack = torch.stack(errors, dim=1)  # [B, 5]
        pred_idx = errors_stack.argmin(dim=1)

        # Map pred_idx back to class_id index in CLASS_TO_IDX
        pred_cls_ids = [sorted_ids[i] for i in pred_idx.tolist()]
        pred_labels  = [CLASS_TO_IDX[c] for c in pred_cls_ids]

        y_true_a.extend(labels.tolist())
        y_pred_a.extend(pred_labels)

        # Latent concatenation
        latent_cat = torch.cat(latents, dim=1)
        all_latents.append(latent_cat)
        all_labels.append(labels)

        # REP profile: 50 reconstruction features + 5 global
        rec_feats = []
        for cls_id in sorted_ids:
            cls_name, model = specialists[cls_id]
            with torch.no_grad():
                recon, _ = model(inputs)
            err = CRITERION(recon, inputs).mean(dim=1)  # [B, H, W]
            h, w = err.shape[1], err.shape[2]
            ph = h // N_PATCHES
            patch_means, patch_stds = [], []
            for pi in range(N_PATCHES):
                p = err[:, pi*ph:(pi+1)*ph, :]
                patch_means.append(p.mean(dim=[1, 2]))
                patch_stds.append(p.std(dim=[1, 2]))
            means = torch.stack(patch_means, dim=1).cpu()
            stds  = torch.stack(patch_stds,  dim=1).cpu()
            interleaved = torch.stack([means, stds], dim=2).reshape(inputs.size(0), -1)
            rec_feats.append(interleaved)

        rec_tensor = torch.cat(rec_feats, dim=1)
        global_tensor = meta.cpu()
        full_profile = torch.cat([rec_tensor, global_tensor], dim=1)
        all_rep_profiles.append(full_profile)

    all_latents = torch.cat(all_latents, dim=0)
    all_labels  = torch.cat(all_labels,  dim=0)
    all_rep_profiles = torch.cat(all_rep_profiles, dim=0)

    acc_a = accuracy_score(y_true_a, y_pred_a)
    return acc_a, y_true_a, y_pred_a, all_latents, all_labels, all_rep_profiles


def strategy_d(rep_profiles, labels, classifier_dir):
    """Strategy D: Random Forest on 55D REP profile."""
    clf_path = os.path.join(classifier_dir, 'rep_meta_classifier.pkl')
    with open(clf_path, 'rb') as f:
        clf_data = pickle.load(f)
    rf = clf_data['classifier']

    X = rep_profiles.numpy()
    y = labels.numpy()
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc, y.tolist(), y_pred.tolist()


def strategy_bc(latents, labels, classifier_dir):
    """Strategy B/C: MLP on 1280D encoded features."""
    clf_path = os.path.join(classifier_dir, 'MVELSA_CLASSIFIER.pth')
    model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(clf_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logits = model(latents.to(DEVICE))
        preds  = logits.argmax(dim=1).cpu()

    y_true = labels.numpy()
    y_pred = preds.numpy()
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true.tolist(), y_pred.tolist()


def main():
    print(f"Device: {DEVICE}")
    print("Loading specialists...")
    specialists = load_specialists()

    print("\n--- Strategy A + collecting data for B/C/D ---")
    acc_a, y_true, y_pred_a, latents, labels, rep_profiles = strategy_a_b(specialists, DATA_DIR)

    print("\n--- Strategy B/C (MLP on 1280D) ---")
    acc_bc, _, y_pred_bc = strategy_bc(latents, labels, CLASSIFIER_DIR)

    print("\n--- Strategy D (RF on 55D REP) ---")
    acc_d, _, y_pred_d = strategy_d(rep_profiles, labels, BASE_DIR)

    print("\n" + "="*60)
    print("FINAL RESULTS (V3 Blind Test)")
    print("="*60)
    print(f"  Strategy A (argmin recon error): {acc_a*100:.2f}%")
    print(f"  Strategy B/C (MLP 1280D):        {acc_bc*100:.2f}%")
    print(f"  Strategy D (RF 55D REP):         {acc_d*100:.2f}%")

    print("\nStrategy D — Classification Report:")
    print(classification_report(y_true, y_pred_d, target_names=CLASS_NAMES))

    # Confusion matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred_d)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'V3 Confusion Matrix (Strategy D) — {acc_d*100:.2f}%')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'Cropped_ConfusionMatrix.png'))
        print("Confusion matrix saved.")
    except Exception:
        pass


if __name__ == '__main__':
    main()
