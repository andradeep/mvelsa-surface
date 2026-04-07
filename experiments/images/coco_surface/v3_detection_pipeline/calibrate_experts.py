"""
Calibrates each MVELSA specialist by computing per-class reconstruction error
statistics on the validation split.

For each specialist (trained on class C), we measure:
  - mean reconstruction error on class C images (in-distribution)
  - mean reconstruction error on all other classes (out-of-distribution)

Output:
    expert_calibration.json — thresholds and statistics per specialist

Usage:
    python calibrate_experts.py

Environment variables:
    DATA_DIR   — coco_cropped directory (contains valid/)
    MODEL_DIR  — ELSA_MODEL_CROPPED_SURFACE directory
    OUT_DIR    — where to save expert_calibration.json (default: ./)
"""

import os
import sys
import json
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
BATCH_SIZE = 128
CRITERION  = nn.MSELoss(reduction='none')


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


def compute_errors(specialist_model, data_dir, split='valid'):
    ds = CroppedSurfaceDataset(os.path.join(data_dir, split),
                                focus_classes=FOCUS_CLASSES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

    errors_by_class = {cls_id: [] for cls_id in FOCUS_CLASSES}
    idx_to_cls = {v: k for k, v in CLASS_TO_IDX.items()}

    specialist_model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing errors", leave=False):
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(DEVICE)
            recon, _ = specialist_model(imgs)
            # Per-sample MSE
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
        print(f"\nCalibrating specialist: {spec_name}")
        errors = compute_errors(model, DATA_DIR, split='valid')

        in_errors  = errors[spec_cls_id]
        out_errors = []
        for cls_id, errs in errors.items():
            if cls_id != spec_cls_id:
                out_errors.extend(errs)

        import statistics
        in_mean  = statistics.mean(in_errors)  if in_errors  else 0.0
        out_mean = statistics.mean(out_errors) if out_errors else 0.0
        # Threshold: midpoint between in and out mean
        threshold = (in_mean + out_mean) / 2.0

        print(f"  In-class MSE:     {in_mean:.6f}")
        print(f"  Out-of-class MSE: {out_mean:.6f}")
        print(f"  Threshold:        {threshold:.6f}")

        calibration[spec_name] = {
            'class_id':   spec_cls_id,
            'in_mean':    in_mean,
            'out_mean':   out_mean,
            'threshold':  threshold,
            'n_in':       len(in_errors),
            'n_out':      len(out_errors),
        }

    out_path = os.path.join(OUT_DIR, 'expert_calibration.json')
    with open(out_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration saved to: {out_path}")


if __name__ == '__main__':
    main()
