"""
Calibrates IAUG-trained MVELSA specialists on the FHD validation split.

Output:
    expert_calibration_iaug.json

Usage:
    python calibrate_experts_iaug.py

Environment variables:
    FHD_DATA_PATH — coco_cropped directory
    MODEL_DIR     — ELSA_MODEL_IAUG directory
    OUT_DIR       — where to save JSON (default: ./)
"""

import os
import sys
import json
import statistics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v2_cropped_optimized"))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX
from train_cropped_mvelsa import ConvAutoencoder, LATENT_DIM

BASE_DIR  = Path(__file__).resolve().parent
FHD_DATA  = os.environ.get("FHD_DATA_PATH", str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR = os.environ.get("MODEL_DIR",     str(BASE_DIR / "ELSA_MODEL_IAUG"))
OUT_DIR   = os.environ.get("OUT_DIR",       str(BASE_DIR))

FOCUS_CLASSES = {1, 3, 4, 5, 6}
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE    = 128
CRITERION     = nn.MSELoss(reduction='none')


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


def compute_errors_on_val(specialist_model, fhd_data_path):
    ds = CroppedSurfaceDataset(os.path.join(fhd_data_path, 'valid'),
                                focus_classes=FOCUS_CLASSES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    errors_by_class = {cls_id: [] for cls_id in FOCUS_CLASSES}
    idx_to_cls = {v: k for k, v in CLASS_TO_IDX.items()}

    specialist_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Errors", leave=False):
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(DEVICE)
            recon, _ = specialist_model(imgs)
            err = CRITERION(recon, imgs).mean(dim=[1, 2, 3]).cpu()
            for i, lbl in enumerate(labels):
                cls_id = idx_to_cls[lbl.item()]
                errors_by_class[cls_id].append(err[i].item())

    return errors_by_class


def main():
    print(f"Device: {DEVICE}")
    specialists = load_specialists()
    calibration = {}

    for spec_cls_id, (spec_name, model) in specialists.items():
        print(f"\nCalibrating: {spec_name}")
        errors = compute_errors_on_val(model, FHD_DATA)

        in_errors  = errors[spec_cls_id]
        out_errors = [e for cid, errs in errors.items() if cid != spec_cls_id for e in errs]

        in_mean  = statistics.mean(in_errors)  if in_errors  else 0.0
        out_mean = statistics.mean(out_errors) if out_errors else 0.0
        threshold = (in_mean + out_mean) / 2.0

        print(f"  In-class:     {in_mean:.6f}")
        print(f"  Out-of-class: {out_mean:.6f}")
        print(f"  Threshold:    {threshold:.6f}")

        calibration[spec_name] = {
            'class_id':  spec_cls_id,
            'in_mean':   in_mean,
            'out_mean':  out_mean,
            'threshold': threshold,
            'n_in':      len(in_errors),
            'n_out':     len(out_errors),
        }

    out_path = os.path.join(OUT_DIR, 'expert_calibration_iaug.json')
    with open(out_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
