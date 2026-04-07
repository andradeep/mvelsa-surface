"""
Blind validation of V4 IAUG MVELSA on the FHD held-out test split.

Compares all strategies against V3 reference results.

Usage:
    python validate_iaug_blind.py

Environment variables:
    FHD_DATA_PATH  — coco_cropped directory (test split)
    MODEL_DIR      — ELSA_MODEL_IAUG directory
    ENCODED_DIR    — ENCODED_DATA_IAUG directory
    CLASSIFIER_DIR — directory with MVELSA_CLASSIFIER_IAUG.pth and rep_meta_classifier_iaug.pkl
"""

import os
import sys
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
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM
from train_cropped_classifier import MLPClassifier, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES

BASE_DIR       = Path(__file__).resolve().parent
FHD_DATA       = os.environ.get("FHD_DATA_PATH",    str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR      = os.environ.get("MODEL_DIR",         str(BASE_DIR / "ELSA_MODEL_IAUG"))
ENCODED_DIR    = os.environ.get("ENCODED_DIR",       str(BASE_DIR / "ENCODED_DATA_IAUG"))
CLASSIFIER_DIR = os.environ.get("CLASSIFIER_DIR",    str(BASE_DIR))

GLOBAL_FEATURES = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE      = 64
CRITERION       = nn.MSELoss(reduction='none')
N_PATCHES       = 5

FOCUS_CLASSES = {1, 3, 4, 5, 6}
IDX_TO_CLASS  = {v: BASE_CLASSES[k] for k, v in CLASS_TO_IDX.items()}
CLASS_NAMES   = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]

# V3 reference results (55D with bbox features)
V3_REFERENCE = {
    "B": 0.6447,
    "C": 0.7005,
    "D": 0.8832,
}


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


def collect_all(specialists, data_dir):
    ds = CroppedSurfaceDataset(
        os.path.join(data_dir, 'test'),
        focus_classes=FOCUS_CLASSES,
        return_meta=True
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    sorted_ids = sorted(FOCUS_CLASSES)
    y_true_a, y_pred_a = [], []
    all_latents, all_labels, all_rep = [], [], []

    for inputs, labels, meta in tqdm(loader, desc="Evaluating"):
        inputs = inputs.to(DEVICE)
        errors, latents = [], []

        for cls_id in sorted_ids:
            _, model = specialists[cls_id]
            with torch.no_grad():
                recon, z = model(inputs)
            err = CRITERION(recon, inputs).mean(dim=[1, 2, 3]).cpu()
            errors.append(err)
            latents.append(z.cpu())

        # Strategy A
        errors_stack = torch.stack(errors, dim=1)
        pred_idx     = errors_stack.argmin(dim=1)
        pred_cls_ids = [sorted_ids[i] for i in pred_idx.tolist()]
        pred_labels  = [CLASS_TO_IDX[c] for c in pred_cls_ids]
        y_true_a.extend(labels.tolist())
        y_pred_a.extend(pred_labels)

        # Latents for B/C
        all_latents.append(torch.cat(latents, dim=1))
        all_labels.append(labels)

        # REP profile (50 + 5)
        rec_feats = []
        for cls_id in sorted_ids:
            _, model = specialists[cls_id]
            with torch.no_grad():
                recon, _ = model(inputs)
            err = CRITERION(recon, inputs).mean(dim=1)
            h, w = err.shape[1], err.shape[2]
            ph   = h // N_PATCHES
            pms, pss = [], []
            for pi in range(N_PATCHES):
                p = err[:, pi*ph:(pi+1)*ph, :]
                pms.append(p.mean(dim=[1, 2]))
                pss.append(p.std(dim=[1, 2]))
            means = torch.stack(pms, dim=1).cpu()
            stds  = torch.stack(pss, dim=1).cpu()
            interleaved = torch.stack([means, stds], dim=2).reshape(inputs.size(0), -1)
            rec_feats.append(interleaved)
        rec_tensor  = torch.cat(rec_feats, dim=1)
        full_profile = torch.cat([rec_tensor, meta.cpu()], dim=1)
        all_rep.append(full_profile)

    latents    = torch.cat(all_latents, dim=0)
    labels_t   = torch.cat(all_labels,  dim=0)
    rep_tensor = torch.cat(all_rep,     dim=0)

    acc_a = accuracy_score(y_true_a, y_pred_a)
    return acc_a, y_true_a, y_pred_a, latents, labels_t, rep_tensor


def main():
    print(f"Device: {DEVICE}")
    specialists = load_specialists()

    acc_a, y_true, y_pred_a, latents, labels, rep_profiles = collect_all(specialists, FHD_DATA)

    # Strategy B/C
    clf_path = os.path.join(CLASSIFIER_DIR, 'MVELSA_CLASSIFIER_IAUG.pth')
    mlp = MLPClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    mlp.load_state_dict(torch.load(clf_path, map_location=DEVICE))
    mlp.eval()
    with torch.no_grad():
        logits = mlp(latents.to(DEVICE))
        preds  = logits.argmax(1).cpu().numpy()
    acc_bc = accuracy_score(labels.numpy(), preds)

    # Strategy D
    pkl_path = os.path.join(CLASSIFIER_DIR, 'rep_meta_classifier_iaug.pkl')
    with open(pkl_path, 'rb') as f:
        clf_data = pickle.load(f)
    rf = clf_data['classifier']
    y_pred_d = rf.predict(rep_profiles.numpy())
    acc_d = accuracy_score(labels.numpy(), y_pred_d)

    print("\n" + "="*65)
    print("FINAL RESULTS — V4 IAUG vs V3 Reference")
    print("="*65)
    print(f"{'Strategy':<30} {'V3 Ref':>10} {'V4 IAUG':>10} {'Delta':>8}")
    print("-"*65)
    print(f"{'A — Argmin recon error':<30} {'—':>10} {acc_a*100:>9.2f}% {'':>8}")
    print(f"{'B/C — MLP 1280D':<30} {V3_REFERENCE['B']*100:>9.2f}% {acc_bc*100:>9.2f}% {(acc_bc-V3_REFERENCE['B'])*100:>+7.2f}%")
    print(f"{'D — RF 55D REP':<30} {V3_REFERENCE['D']*100:>9.2f}% {acc_d*100:>9.2f}% {(acc_d-V3_REFERENCE['D'])*100:>+7.2f}%")

    print("\nStrategy D — Classification Report:")
    print(classification_report(labels.numpy(), y_pred_d, target_names=CLASS_NAMES))

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(labels.numpy(), y_pred_d)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'V4 IAUG Confusion Matrix (Strategy D) — {acc_d*100:.2f}%')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'IAUG_ConfusionMatrix.png'))
    except Exception:
        pass


if __name__ == '__main__':
    main()
