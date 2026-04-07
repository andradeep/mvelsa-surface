import sys
import os
import torch
import torch.nn.functional as F
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Adicionar root do projeto
sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from elsanet.elsa import ELSA
from elsanet.mvelsa import RMSELoss
from cropped_data_generator import CroppedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_V2_DIR = "../v2_cropped_optimized"
ELSA_PATH = os.path.join(MODEL_V2_DIR, "ELSA_MODEL_CROPPED_SURFACE")
CALIBRATION_FILE = "expert_calibration.json"

# --- 1. CONFIGURAÇÃO DE DADOS ---
x_resolution, y_resolution = 64, 64
resolution = (x_resolution, y_resolution)

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Carregar mvelsa
print(f"Carregando modelo: {ELSA_PATH}...")
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

# Carregar o Dataset diretamente
print("Carregando dataset de validação...")
dataset_val = CroppedDataset("../../../../data/coco_cropped", train=False, transform=transform_val)

# --- 2. CALIBRATION ---
calibration_data = {}
base_classes = {1: 'BOAT', 2: 'BUILDING', 3: 'BUOY', 4: 'LAND', 10: 'SHIP', 11: 'SKY', 12: 'WATER'}

print("\nIniciando calibração de especialistas (Baseline Error)...")

for idx, label in enumerate(mvelsa.labels):
    class_name = base_classes.get(label, f"Class_{label}")
    print(f"Calibrando Especialista {idx}: {class_name} (ID {label})")
    
    # Filtrar índices da classe no dataset de validação
    indices = [i for i, targ in enumerate(dataset_val.targets) if targ == label]
    
    if len(indices) == 0:
        print(f" -> AVISO: Nenhuma imagem encontrada para {class_name} no set de validação. Usando erro padrão.")
        calibration_data[class_name] = {"label_id": int(label), "avg_reconstruction_error": 0.1, "weight": 1.0}
        continue
        
    # Limitar a calibração a no máximo 100 imagens por classe para ser rápido
    indices = indices[:100]
    val_subset = Subset(dataset_val, indices)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    elsa_expert = mvelsa.mvelsa[idx]
    total_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs, _ = batch
            inputs = inputs.to(device)
            input_flat = inputs.view(inputs.shape[0], 1, -1).to(dtype=torch.float)
            
            _, dec_avg = elsa_expert.image_forward(input_flat)
            
            # MSE por amostra
            mse = F.mse_loss(dec_avg, input_flat, reduction='mean').item()
            
            total_error += mse * inputs.shape[0]
            total_samples += inputs.shape[0]
            
    avg_error = total_error / total_samples if total_samples > 0 else 0.1
    calibration_data[class_name] = {
        "label_id": int(label),
        "avg_reconstruction_error": float(avg_error),
        "weight": float(1.0 / (avg_error + 1e-6))
    }
    print(f" -> Erro Médio: {avg_error:.6f}")

# --- 3. SALVAR ---
with open(CALIBRATION_FILE, "w") as f:
    json.dump(calibration_data, f, indent=4)

print(f"\nCalibração concluída! Arquivo salvo em: {CALIBRATION_FILE}")
