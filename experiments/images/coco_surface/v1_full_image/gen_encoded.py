import sys
import torch
import torchvision.transforms as transforms
import json

sys.path.append("../../../../")

from data.data_preparation import DataPreparation
from elsanet.mvelsa import MVELSA, RMSELoss
from elsanet.elsa import ELSA
from coco_data_generator import COCODataset

# Load the saved model with safe_globals for PyTorch
with torch.serialization.safe_globals([MVELSA, ELSA, RMSELoss]):
    mvelsa = torch.load("ELSA_MODEL_COCO_SURFACE", weights_only=False)

x_resolution = 64
y_resolution = 64
resolution = (x_resolution, y_resolution)
channels = 1
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize(size=resolution)
])

dataset_train = COCODataset("../../../data/coco640", train=True, transform=transform)
dataset_val = COCODataset("../../../data/coco640", train=False, transform=transform)

train_labels = set(dataset_train.targets)
val_labels = set(dataset_val.targets)

# Da mesma forma que no treinamento, as classes usadas serão as completas [2, 11, 12] (BUILDING, SKY, WATER)
category_ids = list(train_labels.intersection(val_labels))

data_parameters = {
    "data_type": "image",
    "file_path": "../../../data/coco640/",
    "dataset_name": COCODataset,
    "transform": transform,
    "batch_size": 128,
    "data_train_lenght": None, 
    "data_val_lenght": 100, 
    "mvelsa": mvelsa,
}

data_instance = DataPreparation(data_parameters).gen()

# Ensure we limit to categories just like during training
data_instance.get_labels(category_ids)

# Extract encoded variables
data_instance.gen_encoded_data()

inputs, targets = data_instance.data_train.dataset.tensors
print("Inputs shape:", inputs.shape)
print("Targets shape:", targets.shape)

data_instance.save("ENCODED_DATA_COCO_SURFACE")
print("Saved encoded features as ENCODED_DATA_COCO_SURFACE")
