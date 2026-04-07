"""
MVELSA V4 — Geração de Representações Latentes (IAUG)
=====================================================
Passa o dataset IAUG train pelo modelo treinado e gera os vetores latentes.
Usa fullHD633 val para validação (consistência com V3).
"""
import sys
import os
sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

import torch
import torchvision.transforms as transforms

from data.data_preparation import DataPreparation
from elsanet.mvelsa import MVELSA, RMSELoss
from elsanet.elsa import ELSA
from cropped_data_generator import CroppedDataset, PadToSquare

x_resolution = 64
y_resolution  = 64
channels      = 3
resolution    = (x_resolution, y_resolution)

transform = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

COMBINED_DATA = os.environ.get("COMBINED_DATA_PATH", "../../../../data/coco_cropped_combined")
FHD_DATA      = os.environ.get("FHD_DATA_PATH",      "../../../../data/coco_cropped")

dataset_train = CroppedDataset(COMBINED_DATA, train=True,  transform=transform)
dataset_val   = CroppedDataset(FHD_DATA,      train=False, transform=transform)

train_labels = set(dataset_train.targets)
val_labels   = set(dataset_val.targets)

focus_classes = {1, 3, 4, 5, 6}
category_ids  = [c for c in train_labels.intersection(val_labels) if c in focus_classes]
print(f"Classes para encoding: {category_ids}")

data_parameters = {
    "data_type":         "image",
    "file_path":         FHD_DATA + "/",   # usa fullHD633 para que gen() encontre valid/
    # train será substituído pelo combined logo abaixo
    "dataset_name":      CroppedDataset,
    "transform":         transform,
    "batch_size":        128,
    "data_train_lenght": None,
    "data_val_lenght":   50,
}

data_instance = DataPreparation(data_parameters).gen()
data_instance.train = dataset_train  # substitui pelo IAUG train
data_instance.test  = dataset_val    # fullHD633 val
data_instance.get_labels(category_ids)

model_file = "ELSA_MODEL_IAUG"
print(f"Carregando modelo: {model_file}...")
try:
    with torch.serialization.safe_globals([MVELSA, ELSA, RMSELoss]):
        mvelsa = torch.load(model_file, weights_only=False)
except AttributeError:
    mvelsa = torch.load(model_file)

print("Gerando representações latentes (IAUG train + fullHD633 val)...")
data_instance.mvelsa = mvelsa
data_instance.gen_encoded_data()

data_instance.save(file_name="ENCODED_DATA_IAUG")
print("Salvo como ENCODED_DATA_IAUG")
print("Próximo passo: python train_classifier_iaug.py")
