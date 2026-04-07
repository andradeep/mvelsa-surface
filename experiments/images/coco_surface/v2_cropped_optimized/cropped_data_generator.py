import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class PadToSquare:
    """Pad a PIL image to a square with black borders (preserves aspect ratio before resize)."""
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        # center the image on a black square canvas
        padded = Image.new(img.mode, (max_side, max_side), 0)
        padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
        return padded


# Classes do fullHD633: BOAT(1), BUOY(3), LAND(4), SHIP(5), SKY(6)
FOCUS_CLASSES = {1, 3, 4, 5, 6}

import warnings

warnings.filterwarnings("ignore")

class CroppedDataset(Dataset):
    """Simple CSV-based dataset loader for the cropped COCO instances."""

    def __init__(self, root, train=True, transform=None, return_meta=False):
        self.root_dir = root
        self.folder = "train" if train else "valid"
        self.return_meta = return_meta

        self.csv_path = os.path.join(self.root_dir, self.folder, "labels.csv")

        if os.path.exists(self.csv_path):
            self.data_info = pd.read_csv(self.csv_path)
            self.targets = self.data_info['class_id'].tolist()
        else:
            print(f"File not found: {self.csv_path}")
            self.data_info = pd.DataFrame(columns=['filename', 'class_id'])
            self.targets = []

        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_info.iloc[idx]
        img_path = os.path.join(self.root_dir, self.folder, row['filename'])

        # OTIMIZAÇÃO DE TESE 1: Preservar características RGB vitais
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(row['class_id']))

        if self.transform:
            image = self.transform(image)

        if self.return_meta:
            cy_norm = torch.tensor(float(row.get('cy_norm', 0.5)))
            return image, label, cy_norm
        return image, label
