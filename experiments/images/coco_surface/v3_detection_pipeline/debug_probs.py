import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# Adicionar root do projeto para imports do elsanet e data
sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from elsanet.classifier import MultiVariableClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_V2_DIR = "../v2_cropped_optimized"
ELSA_PATH = os.path.join(MODEL_V2_DIR, "ELSA_MODEL_CROPPED_SURFACE")
CLFS_PATH = os.path.join(MODEL_V2_DIR, "MVELSA_CLASSIFIER.pth")

base_classes = {1: 'BOAT', 2: 'BUILDING', 3: 'BUOY', 4: 'LAND', 10: 'SHIP', 11: 'SKY', 12: 'WATER'}

# Carregar mvelsa
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

# Carregar o classificador
checkpoint = torch.load(CLFS_PATH, map_location=device, weights_only=False)
model_params = checkpoint['model_parameters']
labels_list = checkpoint['labels_list']
classifier_obj = MultiVariableClassifier(model_params)
classifier_obj.load_state_dict(checkpoint['state_dict'])
classifier_obj.to(device)
classifier_obj.eval()

class_names = [base_classes.get(l, f"Class_{l}") for l in labels_list]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def debug_image(img_path):
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"Erro ao ler {img_path}")
        return
        
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)
    input_dim = input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3]
    input_flat = input_tensor.view(1, 1, input_dim)

    print(f"\n--- DEBUG: {os.path.basename(img_path)} ---")
    
    with torch.no_grad():
        for idx, elsa_model in enumerate(mvelsa.mvelsa):
            # 1. Forward pelo especialista
            encs, dec_avg = elsa_model.image_forward(input_flat)
            
            # 2. Erro de Reconstrução
            recon_error = F.mse_loss(dec_avg, input_flat).item()
            
            # 3. Classificação
            latent = encs.detach().view(encs.shape[0], -1) 
            output = classifier_obj(latent)
            probs = torch.exp(output)
            clf_prob = probs[0, idx].item()
            
            # 4. Score Combinado
            recon_quality = 1.0 / (recon_error + 1e-6)
            combined_score = clf_prob * recon_quality
            
            print(f"Expert [{class_names[idx]}]: Prob={clf_prob:.4f}, Error={recon_error:.6f}, Score={combined_score:.4f}")

if __name__ == "__main__":
    test_img = "trial/BUOY_16.jpg" # Uma das que deu BUOY
    if os.path.exists(test_img):
        debug_image(test_img)
    
    test_img2 = "trial/BOAT_2951.jpg" # Uma das que deu SKY
    if os.path.exists(test_img2):
        debug_image(test_img2)
