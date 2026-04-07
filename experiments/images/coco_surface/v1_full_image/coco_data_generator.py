import os
import json
import torch
from torch.utils.data import Dataset
from skimage import io
import warnings

warnings.filterwarnings("ignore")

class COCODataset(Dataset):
    """COCO format dataset loader for MVELSA. Assumes single dominant object class per image for classification."""

    def __init__(self, root, train=True, transform=None):
        self.root_dir = root
        # Valid options for coco640: train, valid, test
        # We map True to train, and False to valid (or test)
        # Assuming DataPreparation passes train=True or False
        if train:
            self.folder = "train"
        else:
            self.folder = "valid"

        self.json_file = os.path.join(self.root_dir, self.folder, "_annotations.coco.json")

        with open(self.json_file, 'r') as f:
            self.coco_data = json.load(f)

        self.transform = transform
        
        # Build image mapping: image_id -> file_name
        self.images_info = {img['id']: img['file_name'] for img in self.coco_data['images']}

        # Build mapping of image_id to its annotations
        self.img_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_annotations:
                self.img_annotations[img_id] = []
            self.img_annotations[img_id].append(ann)

        # Do not preload all images into memory as MVELSA dataset structure usually relies on self.targets 
        # for label checking and splits
        self.targets = []
        for i in range(len(self)):
            img_id = list(self.images_info.keys())[i]
            annotations = self.img_annotations.get(img_id, [])
            category_id = 0
            # Ordem de importância (ID): SHIP (10), BOAT (1), BUOY (3), BUILDING (2), LAND (4), WATER (12), SKY (11)
            # Queremos garantir que se houver um Barco e Água na foto, a imagem seja classificada como Barco, não Água.
            priority_order = [10, 1, 3, 2, 4, 12, 11]
            
            best_ann = None
            best_priority = 999
            
            for a in annotations:
                cat = a['category_id']
                prio = priority_order.index(cat) if cat in priority_order else 999
                if prio < best_priority:
                    best_priority = prio
                    best_ann = a
            
            if best_ann:
                category_id = best_ann['category_id']
            self.targets.append(category_id)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = list(self.images_info.keys())[idx]
        img_filename = self.images_info[img_id]

        img_path = os.path.join(self.root_dir, self.folder, img_filename)
        img_path = img_path.replace("\\", "/")
        
        # Assuming grayscale processing based on MVELSA structure
        image = io.imread(img_path, as_gray=True)

        annotations = self.img_annotations.get(img_id, [])
        category_id = 0 # Default if no annotations
        if len(annotations) > 0:
            # We take the category of the largest bounding box or just the first if multiple exist
            # For classification, assuming one main object per image
            best_ann = max(annotations, key=lambda a: a.get('area', 0))
            category_id = best_ann['category_id']

        label = torch.tensor(int(category_id))

        if self.transform:
            image = self.transform(image)

        return image, label


