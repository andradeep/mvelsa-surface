import sys
import os
sys.path.append("../../../../")

from elsanet.mvelsa import MVELSA, RMSELoss
import torchvision.transforms as transforms
import torch
import json

# Setup parameters
from data.data_preparation import DataPreparation
from cropped_data_generator import CroppedDataset, PadToSquare

x_resolution = 64
y_resolution = 64
channels = 3 # OTIMIZAÇÃO: RGB (12288 neurônios)
resolution = (x_resolution, y_resolution)

transform_train = transforms.Compose([
    PadToSquare(),                                           # Preserva aspect ratio antes do resize
    transforms.RandomHorizontalFlip(p=0.5), # Geometric augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Contrast augmentation
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Regularização padrão
])

transform_val = transforms.Compose([
    PadToSquare(),                                           # Idem
    transforms.ToTensor(),
    transforms.Resize(size=resolution),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_train = CroppedDataset("../../../../data/coco_cropped", train=True, transform=transform_train)
dataset_val = CroppedDataset("../../../../data/coco_cropped", train=False, transform=transform_val)

train_labels = set(dataset_train.targets)
val_labels = set(dataset_val.targets)

# Encontra a interseção para garantir que a rede ELSA possa validar a classe durante o treino
category_ids = list(train_labels.intersection(val_labels))

# Classes do fullHD633: BOAT(1), BUOY(3), LAND(4), SHIP(5), SKY(6)
focus_classes = {1, 3, 4, 5, 6}
category_ids = [c for c in category_ids if c in focus_classes]
print(f"Categories to train (Filtered to Focus Classes): {category_ids}")

# Verificar quantos samples temos por classe (sanity check antes de treinar)
import collections
train_counts = collections.Counter(dataset_train.targets)
val_counts = collections.Counter(dataset_val.targets)
print("[SANITY CHECK] Amostras de treino por classe (IDs filtrados):")
for cid in sorted(category_ids):
    print(f"  Class {cid}: train={train_counts.get(cid,0)}, val={val_counts.get(cid,0)}")

data_parameters = {
    "data_type": "image",
    "file_path": "../../../../data/coco_cropped/",
    "dataset_name": CroppedDataset,
    "transform": transform_val, # Use val transforms for standard loading testing fallback
    "batch_size": 64,          
    "data_train_lenght": None,  # Use all data
    "data_val_lenght": 50,      # Number of samples for validation
}

data_instance = DataPreparation(data_parameters).gen()
# Injecting our heavily augmented dataset instances manually to preserve the hooks
data_instance.train = dataset_train
data_instance.test = dataset_val

# Ensure we only use labels that have both train and test samples.
data_instance.get_labels(category_ids)

# OTIMIZAÇÃO 3: AE Profundo (Múltiplas Camadas para 3 Canais RGB de Entrada)
# 64 * 64 * 3 (RGB) = 12288 Neurônios na Base Linear (O antigo dava erro numérico porque era *1)
initial_ae_layer = x_resolution * y_resolution * channels

# Descobrir dispositivo (GPU para treinar mais rápido)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando specialistas em: {device}")

model_hyperparameters = {
    "ae_architecture": [
        initial_ae_layer,
        1024,   # Compressão 1
        256,    # Compressão 2
        128,    # Latent space final robusto (128)
    ],
    "ae_times": 2,
    "activation": "ReLU",
    "epochs": 100, # Aumentado de 35 para 100 para maior poder de discriminação
    "learning_rate": 0.001, # LR levemente menor para convergência mais fina
    "loss_function": RMSELoss().to(device),
    "seed": 42,
    "device": device, # CRÍTICO: garante que especialistas usem GPU
}

mvelsa = MVELSA(model_hyperparameters)

print("INICIANDO TREINAMENTO MVELSA (CROPPED DATASET)")
mvelsa.fit(data_instance)
print("TREINAMENTO FINALIZADO")

mvelsa.save(file_name="ELSA_MODEL_CROPPED_SURFACE")
print("Modelo salvo como ELSA_MODEL_CROPPED_SURFACE")
