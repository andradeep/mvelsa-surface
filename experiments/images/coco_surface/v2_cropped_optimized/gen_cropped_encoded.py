import sys
import os
sys.path.append("../../../../")

import torch
import torchvision.transforms as transforms

from data.data_preparation import DataPreparation
from elsanet.mvelsa import MVELSA
from elsanet.elsa import ELSA
from cropped_data_generator import CroppedDataset, PadToSquare

x_resolution = 64
y_resolution = 64
channels = 3
resolution = (x_resolution, y_resolution)

transform = transforms.Compose([
    PadToSquare(),                                           # Preserva aspect ratio
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # CRÍTICO: Deve bater com o treino
])

dataset_train = CroppedDataset("../../../../data/coco_cropped", train=True, transform=transform)
dataset_val = CroppedDataset("../../../../data/coco_cropped", train=False, transform=transform)

train_labels = set(dataset_train.targets)
val_labels = set(dataset_val.targets)

category_ids = list(train_labels.intersection(val_labels))
# OTIMIZAÇÃO DE TESE: Expansão para ambiente marinho completo
# 1 = BOAT, 3 = BUOY, 4 = LAND, 10 = SHIP, 11 = SKY, 12 = WATER
focus_classes = {1, 3, 4, 5, 6}  # BOAT, BUOY, LAND, SHIP, SKY
category_ids = [c for c in category_ids if c in focus_classes]
print(f"Categories to encode (Filtered to Focus Classes): {category_ids}")
data_parameters = {
    "data_type": "image",
    "file_path": "../../../../data/coco_cropped/",
    "dataset_name": CroppedDataset,
    "transform": transform,
    "batch_size": 128,          
    "data_train_lenght": None,
    "data_val_lenght": 50,
}

data_instance = DataPreparation(data_parameters).gen()
data_instance.get_labels(category_ids)

from elsanet.mvelsa import RMSELoss

model_file = "ELSA_MODEL_CROPPED_SURFACE"

print(f"Buscando modelo: {model_file}...")
try:
    with torch.serialization.safe_globals([MVELSA, ELSA, RMSELoss]):
        mvelsa = torch.load(model_file, weights_only=False)
except AttributeError:
    # Fallback for older PyTorch versions
    mvelsa = torch.load(model_file)

print("Modelo MVELSA carregado. Gerando variavéis latentes para as imagens recortadas...")
data_instance.mvelsa = mvelsa
data_instance.gen_encoded_data()

data_instance.save(file_name="ENCODED_DATA_CROPPED_SURFACE")
print("Saved encoded features as ENCODED_DATA_CROPPED_SURFACE")
