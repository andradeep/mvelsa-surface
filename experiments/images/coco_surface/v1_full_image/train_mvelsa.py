import sys
import os

# Assuming running from mvelsa/experiments/images/coco_surface/
sys.path.append("../../../../")

from data.data_preparation import DataPreparation
from elsanet.mvelsa import MVELSA, RMSELoss
import torchvision.transforms as transforms
from coco_data_generator import COCODataset
import json

# Setup parameters
x_resolution = 64
y_resolution = 64
channels = 1
resolution = (x_resolution, y_resolution)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=resolution)
])

dataset_train = COCODataset("../../../data/coco640", train=True, transform=transform)
dataset_val = COCODataset("../../../data/coco640", train=False, transform=transform)

train_labels = set(dataset_train.targets)
val_labels = set(dataset_val.targets)

# Classes disponíveis na anotação COCO original (coco640):
# 1: 'BOAT', 2: 'BUILDING', 3: 'BUOY', 4: 'LAND', 10: 'SHIP', 11: 'SKY', 12: 'WATER'
# A interseção garante que só usemos classes com imagens presentes TANTO no treino QUANTO na validação
# Para o seu dataset, as classes válidas que passaram nesse filtro foram: [2, 11, 12] (BUILDING, SKY, WATER)
category_ids = list(train_labels.intersection(val_labels))
print(f"Categories to train (present in train & val): {category_ids}")

data_parameters = {
    "data_type": "image",
    "file_path": "../../../data/coco640/",
    "dataset_name": COCODataset,
    "transform": transform,
    "batch_size": 128,          
    "data_train_lenght": None,  # Use all data
    "data_val_lenght": 100,        # Number of samples for validation
}

data_instance = DataPreparation(data_parameters).gen()

# Ensure we only use labels that have both train and test samples.
# In MVELSA, each label trains a standalone autoencoder.
data_instance.get_labels(category_ids)

initial_ae_layer = x_resolution * y_resolution * channels

model_hyperparameters = {
    "ae_architecture": [
        initial_ae_layer,
        128,
        64,     # Latent space = 64
    ],
    "ae_times": 2,
    "activation": "ReLU",
    "epochs": 20,
    "learning_rate": 0.004,
    "loss_function": RMSELoss(),
    "seed": 42,
}

mvelsa = MVELSA(model_hyperparameters)

print("INICIANDO TREINAMENTO MVELSA")
mvelsa.fit(data_instance)
print("TREINAMENTO FINALIZADO")

mvelsa.save(file_name="ELSA_MODEL_COCO_SURFACE")
print("Modelo salvo como ELSA_MODEL_COCO_SURFACE")
