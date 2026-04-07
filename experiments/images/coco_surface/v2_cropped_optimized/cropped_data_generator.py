"""
Dataset generator for MVELSA cropped surface experiment.
Reads images and labels from a directory with labels.csv.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

BASE_CLASSES = {
    1: 'BOAT',
    3: 'BUOY',
    4: 'LAND',
    5: 'SHIP',
    6: 'SKY',
}

CLASS_TO_IDX = {cls_id: idx for idx, cls_id in enumerate(sorted(BASE_CLASSES.keys()))}
IDX_TO_CLASS = {idx: BASE_CLASSES[cls_id] for cls_id, idx in CLASS_TO_IDX.items()}

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class CroppedSurfaceDataset(Dataset):
    """
    Dataset of cropped surface objects for MVELSA training.

    Args:
        data_dir: Directory containing images and labels.csv
        transform: Optional torchvision transform
        return_meta: If True, returns (image, label, meta_tensor) where
                     meta_tensor is [cy_norm, aspect_ratio, bbox_area_norm,
                                     bbox_w_norm, bbox_h_norm]
        focus_classes: Set of class_ids to include (default: all BASE_CLASSES)
        single_class: If set, only load images of this class_id
    """

    def __init__(self, data_dir, transform=None, return_meta=False,
                 focus_classes=None, single_class=None):
        self.data_dir = data_dir
        self.transform = transform or DEFAULT_TRANSFORM
        self.return_meta = return_meta

        if focus_classes is None:
            focus_classes = set(BASE_CLASSES.keys())
        self.focus_classes = focus_classes

        csv_path = os.path.join(data_dir, 'labels.csv')
        df = pd.read_csv(csv_path)

        # Filter to focus classes
        df = df[df['class_id'].isin(focus_classes)].reset_index(drop=True)

        # Filter to single class if requested
        if single_class is not None:
            df = df[df['class_id'] == single_class].reset_index(drop=True)

        # Keep only files that exist on disk
        df = df[df['filename'].apply(
            lambda f: os.path.exists(os.path.join(data_dir, f))
        )].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Map class_id to sequential index
        label = CLASS_TO_IDX[row['class_id']]

        if self.return_meta:
            cy_norm        = float(row.get('cy_norm', 0.5))
            aspect_ratio   = float(row.get('aspect_ratio', 1.0))
            bbox_area_norm = float(row.get('bbox_area_norm', 0.05))
            bbox_w_norm    = float(row.get('bbox_w_norm', 0.05))
            bbox_h_norm    = float(row.get('bbox_h_norm', 0.05))
            meta = torch.tensor(
                [cy_norm, aspect_ratio, bbox_area_norm, bbox_w_norm, bbox_h_norm],
                dtype=torch.float32
            )
            return image, label, meta

        return image, label
