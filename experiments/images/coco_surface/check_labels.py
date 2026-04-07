import json
import torch
from coco_data_generator import COCODataset
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(64,64))])
dataset_train = COCODataset("../../../data/coco640", train=True, transform=transform)
dataset_val = COCODataset("../../../data/coco640", train=False, transform=transform)

train_labels = set(dataset_train.targets)
val_labels = set(dataset_val.targets)

print("Train labels:", train_labels)
print("Val labels:", val_labels)

valid_labels = train_labels.intersection(val_labels)
print("Valid labels for MVELSA (in both sets):", valid_labels)
