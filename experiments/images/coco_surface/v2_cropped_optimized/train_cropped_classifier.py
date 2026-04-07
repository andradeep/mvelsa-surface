"""
Trains an MLP classifier on top of the MVELSA 1280D encoded features.

Strategy B: MLP on concatenated latent space.

Usage:
    python train_cropped_classifier.py

Environment variables:
    ENCODED_DIR  — ENCODED_DATA_CROPPED_SURFACE directory
    MODEL_DIR    — where to save MVELSA_CLASSIFIER.pth (default: ./)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
ENCODED_DIR  = os.environ.get("ENCODED_DIR",  str(BASE_DIR / "ENCODED_DATA_CROPPED_SURFACE"))
MODEL_DIR    = os.environ.get("MODEL_DIR",    str(BASE_DIR))

NUM_CLASSES  = 5
LATENT_DIM   = 256
INPUT_DIM    = NUM_CLASSES * LATENT_DIM  # 1280
HIDDEN_DIM   = 512
EPOCHS       = 100
BATCH_SIZE   = 64
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_CLASSES = {
    1: 'BOAT',
    3: 'BUOY',
    4: 'LAND',
    5: 'SHIP',
    6: 'SKY',
}


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_split(split):
    path = os.path.join(ENCODED_DIR, f"{split}_encoded.pt")
    data = torch.load(path, map_location='cpu')
    return data['features'], data['labels']


def main():
    print(f"Device: {DEVICE}")
    print(f"Encoded dir: {ENCODED_DIR}")

    X_train, y_train = load_split('train')
    X_val,   y_val   = load_split('valid')
    X_test,  y_test  = load_split('test')

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False)

    model = MLPClassifier(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            correct = 0
            total   = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    preds = model(X_batch).argmax(1)
                    correct += (preds == y_batch).sum().item()
                    total   += y_batch.size(0)
            val_acc = correct / total
            print(f"  Epoch {epoch:3d}/{EPOCHS}  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join(MODEL_DIR, "MVELSA_CLASSIFIER.pth")
                torch.save(model.state_dict(), save_path)

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "MVELSA_CLASSIFIER.pth"),
                                     map_location=DEVICE))
    model.eval()
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    correct = 0
    total   = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch).argmax(1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

    test_acc = correct / total
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(f"Model saved to: {MODEL_DIR}/MVELSA_CLASSIFIER.pth")


if __name__ == '__main__':
    main()
