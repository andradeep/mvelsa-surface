"""
MVELSA V4 — Treinamento de Especialistas (dataset IAUG)
========================================================
Treino: dataset IAUG augmentado (coco_cropped_iaug/train/)
Validação: dataset fullHD633 val (coco_cropped/valid/)  ← MESMA base da V3

Comparação direta com V3: diferença isolada é APENAS o dataset de treino.
"""
import sys
import os
sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA, RMSELoss
import torchvision.transforms as transforms
import torch
import json
import collections

from data.data_preparation import DataPreparation
from cropped_data_generator import CroppedDataset, PadToSquare

x_resolution = 64
y_resolution  = 64
channels      = 3
resolution    = (x_resolution, y_resolution)

transform_train = transforms.Compose([
    PadToSquare(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Treino: fullHD633 + IAUG combinados | Validação: fullHD633 puro
COMBINED_DATA = os.environ.get("COMBINED_DATA_PATH", "../../../../data/coco_cropped_combined")
FHD_DATA      = os.environ.get("FHD_DATA_PATH",      "../../../../data/coco_cropped")

dataset_train = CroppedDataset(COMBINED_DATA, train=True,  transform=transform_train)
dataset_val   = CroppedDataset(FHD_DATA,  train=False, transform=transform_val)

train_labels = set(dataset_train.targets)
val_labels   = set(dataset_val.targets)

focus_classes = {1, 3, 4, 5, 6}  # BOAT, BUOY, LAND, SHIP, SKY (IDs fullHD633/saída)
category_ids  = [c for c in train_labels.intersection(val_labels) if c in focus_classes]
print(f"Classes comuns treino∩val (filtradas): {category_ids}")

# Sanity check
train_counts = collections.Counter(dataset_train.targets)
val_counts   = collections.Counter(dataset_val.targets)
print("\n[SANITY CHECK] Amostras por classe:")
base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
for cid in sorted(category_ids):
    print(f"  {base_classes.get(cid, cid):6s} (id {cid}): "
          f"train(IAUG)={train_counts.get(cid, 0)}, val(fullHD633)={val_counts.get(cid, 0)}")

data_parameters = {
    "data_type":         "image",
    "file_path":         FHD_DATA + "/",   # usa fullHD633 para que gen() encontre valid/
    "dataset_name":      CroppedDataset,
    "transform":         transform_val,
    "batch_size":        64,
    "data_train_lenght": None,
    "data_val_lenght":   50,
}

data_instance = DataPreparation(data_parameters).gen()
data_instance.train = dataset_train  # fullHD633 + IAUG combinados
data_instance.test  = dataset_val    # fullHD633 val
data_instance.get_labels(category_ids)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDispositivo: {device}")

initial_ae_layer = x_resolution * y_resolution * channels  # 12288

model_hyperparameters = {
    "ae_architecture": [
        initial_ae_layer,
        1024,
        256,
        128,
    ],
    "ae_times":      int(os.environ.get("AE_TIMES", "2")),  # Colab: exportar AE_TIMES=3
    "activation":    "ReLU",
    "epochs":        100,
    "learning_rate": 0.001,
    "loss_function": RMSELoss().to(device),
    "seed":          42,
    "device":        device,
}

mvelsa = MVELSA(model_hyperparameters)

print("\nINICIANDO TREINAMENTO MVELSA V4 (IAUG)")
mvelsa.fit(data_instance)
print("TREINAMENTO FINALIZADO")

mvelsa.save(file_name="ELSA_MODEL_IAUG")
print("Modelo salvo como ELSA_MODEL_IAUG")
print("Próximo passo: python gen_encoded_iaug.py")
