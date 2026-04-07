import sys
import os
import torch
import cv2
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Adicionar root do projeto para imports do elsanet e data
sys.path.append("../../../../")
# Adicionar v2 para carregar classes customizadas do dataset (pickling)
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from elsanet.classifier import MultiVariableClassifier

# --- CONFIGURAÇÃO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_V2_DIR = "../v2_cropped_optimized"
ELSA_PATH = os.path.join(MODEL_V2_DIR, "ELSA_MODEL_CROPPED_SURFACE")
CLFS_PATH = os.path.join(MODEL_V2_DIR, "MVELSA_CLASSIFIER.pth")
TRIAL_DIR = "trial"
LOG_FILE = "results.txt"

# Mapping de classes original (COCO IDs)
base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}

# --- CARREGAMENTO ---
print("Carregando modelos treinados v2...")
if not os.path.exists(CLFS_PATH):
    print(f"ERRO: O classificador {CLFS_PATH} não foi encontrado. Certifique-se de rodar o treino da v2 primeiro!")
    sys.exit(1)

# Carregar mvelsa (autoencoders especialistas)
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

# Carregar o checkpoint leve do classificador (state_dict)
checkpoint = torch.load(CLFS_PATH, map_location=device, weights_only=False)
model_params = checkpoint['model_parameters']
labels_list = checkpoint['labels_list']

classifier_obj = MultiVariableClassifier(model_params)
classifier_obj.load_state_dict(checkpoint['state_dict'])
classifier_obj.to(device)
classifier_obj.eval()

# Identificar a ordem das classes
class_names = [base_classes.get(l, f"Class_{l}") for l in labels_list]
print(f"Classes suportadas: {class_names}")

# Transformação idêntica ao treino
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- CALIBRAÇÃO DE ESPECIALISTAS ---
CALIBRATION_FILE = "expert_calibration.json"
calibration_weights = {}
if os.path.exists(CALIBRATION_FILE):
    import json
    with open(CALIBRATION_FILE, "r") as f:
        calib_data = json.load(f)
        for class_name, info in calib_data.items():
            calibration_weights[class_name] = info["avg_reconstruction_error"]
    print(f"Calibração carregada para: {list(calibration_weights.keys())}")
else:
    print("AVISO: Arquivo de calibração não encontrado. Usando score bruto.")

def predict_single_object(img_path):
    """
    Realiza a inferência em uma imagem contendo um único objeto.
    Usa a lógica de Consenso MVELSA: Combina a probabilidade do classificador
    com a qualidade da reconstrução do especialista, NORMALIZADA pelo erro base da classe.
    """
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        return None, 0.0
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    # Redimensionar para o formato que o ELSA espera (batch, 1, input_dim)
    input_dim = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
    input_flat = input_tensor.view(1, 1, input_dim)

    best_combined_score = -1
    best_label = "UNKNOWN"
    best_prob = 0.0
    
    import torch.nn.functional as F

    with torch.no_grad():
        # 1. Passar por TODOS os especialistas e coletar erros + latentes
        all_encs = []
        recon_errors = []
        for idx, elsa_model in enumerate(mvelsa.mvelsa):
            encs, dec_avg = elsa_model.image_forward(input_flat)
            recon_error = F.mse_loss(dec_avg, input_flat).item()
            recon_errors.append(recon_error)
            all_encs.append(encs.detach().cpu())

        # 2. Concatenar latentes de todos os especialistas (consistente com o treino)
        # Shape: (1, 1, n_specialists * ae_times * latent_dim)
        full_latent = torch.cat(all_encs, dim=-1).to(device)
        latent_flat = full_latent.view(full_latent.shape[0], -1)

        # 3. Classificar com o vetor latente completo
        output = classifier_obj(latent_flat)
        probs = torch.exp(output)  # shape: (1, n_classes)

        # 4. Score final: combina probabilidade do classificador com qualidade de reconstrução
        for idx in range(len(class_names)):
            class_name = class_names[idx]
            baseline = calibration_weights.get(class_name, 0.1)
            recon_quality = 1.0 / ((recon_errors[idx] / baseline) + 1e-6)
            clf_prob = probs[0, idx].item()
            combined_score = clf_prob * recon_quality

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_label = class_name
                best_prob = clf_prob

    return best_label, best_prob

def run_trial():
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(TRIAL_DIR, ext)))
    
    if not images:
        print(f"Nenhuma imagem encontrada na pasta '{TRIAL_DIR}'.")
        return

    print(f"Processando {len(images)} imagens em '{TRIAL_DIR}'...")
    
    results = []
    for img_path in sorted(images):
        img_name = os.path.basename(img_path)
        label, score = predict_single_object(img_path)
        
        if label:
            result_str = f"{img_name}: {label} (Confiança: {score:.2f})"
            print(result_str)
            results.append(result_str)
        else:
            print(f"{img_name}: Erro ao carregar imagem.")

    # Salvar resultados
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(results))
    
    print(f"\nResultados salvos em: {LOG_FILE}")

if __name__ == "__main__":
    run_trial()
