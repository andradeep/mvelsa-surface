"""
Trains MVELSA autoencoders on the cropped surface dataset.

One autoencoder specialist per class. Each specialist is trained on
its own class data with ae_times passes to reinforce specialization.

Usage:
    python train_cropped_mvelsa.py

Environment variables:
    DATA_DIR     — path to coco_cropped directory (default: ../../../../data/coco_cropped)
    MODEL_DIR    — where to save ELSA_MODEL_CROPPED_SURFACE (default: ./ELSA_MODEL_CROPPED_SURFACE)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cropped_data_generator import CroppedSurfaceDataset, BASE_CLASSES, CLASS_TO_IDX

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = os.environ.get("DATA_DIR",  str(BASE_DIR / "../../../../data/coco_cropped"))
MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "ELSA_MODEL_CROPPED_SURFACE"))

FOCUS_CLASSES = {1, 3, 4, 5, 6}
LATENT_DIM    = 256
EPOCHS        = 50
AE_TIMES      = 2        # passes through own-class data per epoch
BATCH_SIZE    = 64
LR            = 1e-3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_specialist(class_id, class_name, train_dir, model_save_path):
    print(f"\n=== Training specialist for {class_name} ===")

    ds_own = CroppedSurfaceDataset(train_dir, focus_classes=FOCUS_CLASSES,
                                    single_class=class_id)
    # Repeat own-class data ae_times
    ds_train = ConcatDataset([ds_own] * AE_TIMES)

    loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    model = ConvAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            imgs, _ = batch[0], batch[1]
            imgs = imgs.to(DEVICE)
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(ds_train)
        loss_history.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.6f}")

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(model_save_path, f"ae_{class_name}.pth"))
    print(f"  Saved: ae_{class_name}.pth")
    return loss_history


def main():
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_DIR}")
    print(f"Models: {MODEL_DIR}")

    train_dir = os.path.join(DATA_DIR, 'train')

    all_losses = {}
    for cls_id in sorted(FOCUS_CLASSES):
        cls_name = BASE_CLASSES[cls_id]
        losses = train_specialist(cls_id, cls_name, train_dir, MODEL_DIR)
        all_losses[cls_name] = losses

    # Save loss plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for name, losses in all_losses.items():
            plt.plot(losses, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('MVELSA Autoencoder Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, 'Cropped_Loss_Graph.png'))
        print("\nLoss graph saved.")
    except ImportError:
        pass

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
