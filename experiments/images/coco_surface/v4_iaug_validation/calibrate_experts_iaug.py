"""
MVELSA V4 — Calibração dos Especialistas (IAUG)
================================================
Mede o erro de reconstrução baseline de cada especialista em sua própria classe.
Usa o split de validação fullHD633 (mesma base da V3) para calibração comparável.
"""
import sys
import os
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA, RMSELoss
from elsanet.elsa import ELSA
from cropped_data_generator import CroppedDataset, PadToSquare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELSA_PATH        = "ELSA_MODEL_IAUG"
CALIBRATION_FILE = "expert_calibration_iaug.json"

transform_val = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(f"Carregando modelo: {ELSA_PATH}...")
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

# Calibra no fullHD633 val (mesma base da V3 — garante comparabilidade)
print("Carregando dataset de calibração: fullHD633 val...")
_fhd_path = os.environ.get("FHD_DATA_PATH", "../../../../data/coco_cropped")
dataset_val = CroppedDataset(_fhd_path, train=False, transform=transform_val)

base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
calibration_data = {}

print("\nCalibrando especialistas...")
for idx, label in enumerate(mvelsa.labels):
    class_name = base_classes.get(label, f"Class_{label}")
    indices = [i for i, t in enumerate(dataset_val.targets) if t == label]

    if len(indices) == 0:
        print(f"  [{class_name}] AVISO: nenhuma amostra no val. Usando erro padrão=0.1")
        calibration_data[class_name] = {"label_id": int(label), "avg_reconstruction_error": 0.1, "weight": 1.0}
        continue

    indices = indices[:100]
    val_subset = Subset(dataset_val, indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    elsa_expert = mvelsa.mvelsa[idx]
    total_error, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            input_flat = inputs.view(inputs.shape[0], 1, -1).to(dtype=torch.float)
            _, dec_avg = elsa_expert.image_forward(input_flat)
            mse = F.mse_loss(dec_avg, input_flat, reduction='mean').item()
            total_error  += mse * inputs.shape[0]
            total_samples += inputs.shape[0]

    avg_error = total_error / total_samples if total_samples > 0 else 0.1
    calibration_data[class_name] = {
        "label_id":                int(label),
        "avg_reconstruction_error": float(avg_error),
        "weight":                   float(1.0 / (avg_error + 1e-6))
    }
    print(f"  [{class_name}] Erro médio: {avg_error:.6f} ({total_samples} amostras)")

with open(CALIBRATION_FILE, "w") as f:
    json.dump(calibration_data, f, indent=4)

print(f"\n✅ Calibração salva em: {CALIBRATION_FILE}")
print("Próximo passo: python extract_rep_profiles_iaug.py")
