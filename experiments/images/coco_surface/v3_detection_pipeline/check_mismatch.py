import sys
import os
import torch

# Adicionar root do projeto
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
print(f"MVELSA Experts (Labels): {mvelsa.labels}")

# Carregar o classificador
checkpoint = torch.load(CLFS_PATH, map_location=device, weights_only=False)
labels_list = checkpoint['labels_list']
print(f"Classifier Checkpoint Labels: {labels_list}")

if mvelsa.labels == labels_list:
    print("MATCH: As listas de labels são idênticas e na mesma ordem.")
else:
    print("MISMATCH: As listas de labels estão diferentes!")
    # Criar um mapping para corrigir se necessário
    mapping = {label: i for i, label in enumerate(mvelsa.labels)}
    reordered_classifier_indices = [mapping[label] for label in labels_list]
    print(f"Indices do classificador mapeados para o MVELSA: {reordered_classifier_indices}")

# Verificar os nomes das classes para o usuário
print("\nOrdem resolvida (MVELSA Index -> Class Name):")
for i, label in enumerate(mvelsa.labels):
    print(f"{i}: {base_classes.get(label, f'ID_{label}')}")
