"""
MVELSA-REP: Extraction de Reconstruction Error Profiles (40D)
=============================================================
Para cada imagem, passa por TODOS os especialistas e extrai um perfil
multi-dimensional de como cada especialista falha na reconstrução.

Resultado: vetor de 40 dimensões (10 features × 4 especialistas)
"""
import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from cropped_data_generator import CroppedDataset, PadToSquare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_V2_DIR = "../v2_cropped_optimized"
ELSA_PATH = os.path.join(MODEL_V2_DIR, "ELSA_MODEL_CROPPED_SURFACE")

# --- 1. CARREGAR MVELSA ---
print(f"Carregando MVELSA de: {ELSA_PATH}")
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

labels_list = mvelsa.labels
n_experts = len(labels_list)
print(f"Especialistas: {n_experts} | Labels: {labels_list}")

# --- 2. DATASET ---
transform = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def compute_ssim_simple(img1, img2):
    """SSIM simplificado entre duas imagens 1D (flatten). Usa estatísticas de janela."""
    # Reshape para (C, H, W) para calcular SSIM por canal
    c, h, w = 3, 64, 64
    try:
        i1 = img1.view(c, h, w)
        i2 = img2.view(c, h, w)
    except RuntimeError:
        return 0.5  # fallback se dimensões não batem

    mu1 = i1.mean(dim=(1, 2))
    mu2 = i2.mean(dim=(1, 2))
    sigma1_sq = ((i1 - mu1.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    sigma2_sq = ((i2 - mu2.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    sigma12 = ((i1 - mu1.view(c, 1, 1)) * (i2 - mu2.view(c, 1, 1))).mean(dim=(1, 2))

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_per_channel = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_per_channel.mean().item()


def extract_rep_profile(input_flat, expert, centroid=None):
    """
    Extrai perfil REP de 10 dimensões para um especialista:
      [0]    MSE global
      [1-3]  MSE por canal (R, G, B)
      [4-7]  MSE por quadrante (TL, TR, BL, BR)
      [8]    SSIM
      [9]    Distância latente ao centroide (se disponível)
    """
    encs, dec_avg = expert.image_forward(input_flat)
    inp = input_flat.squeeze()   # (12288,)
    rec = dec_avg.squeeze()      # (12288,)

    # Feature 0: MSE global
    mse_global = F.mse_loss(rec, inp).item()

    # Features 1-3: MSE por canal RGB
    c, h, w = 3, 64, 64
    try:
        inp_img = inp.view(c, h, w)
        rec_img = rec.view(c, h, w)
        mse_r = F.mse_loss(rec_img[0], inp_img[0]).item()
        mse_g = F.mse_loss(rec_img[1], inp_img[1]).item()
        mse_b = F.mse_loss(rec_img[2], inp_img[2]).item()
    except RuntimeError:
        mse_r = mse_g = mse_b = mse_global

    # Features 4-7: MSE por quadrante espacial
    try:
        mid_h, mid_w = h // 2, w // 2
        mse_tl = F.mse_loss(rec_img[:, :mid_h, :mid_w], inp_img[:, :mid_h, :mid_w]).item()
        mse_tr = F.mse_loss(rec_img[:, :mid_h, mid_w:], inp_img[:, :mid_h, mid_w:]).item()
        mse_bl = F.mse_loss(rec_img[:, mid_h:, :mid_w], inp_img[:, mid_h:, :mid_w]).item()
        mse_br = F.mse_loss(rec_img[:, mid_h:, mid_w:], inp_img[:, mid_h:, mid_w:]).item()
    except RuntimeError:
        mse_tl = mse_tr = mse_bl = mse_br = mse_global

    # Feature 8: SSIM
    ssim_val = compute_ssim_simple(inp, rec)

    # Feature 9: Distância latente ao centroide
    latent = encs.detach().view(-1)
    if centroid is not None:
        latent_dist = torch.norm(latent - centroid).item()
    else:
        latent_dist = 0.0

    return [mse_global, mse_r, mse_g, mse_b, mse_tl, mse_tr, mse_bl, mse_br, ssim_val, latent_dist], encs


def extract_profiles_from_dataset(dataset_path, train=True, precomputed_centroids=None):
    """Extrai perfis REP para todas as imagens de um split.

    Retorna perfis de (n_experts * 10 + 1) features:
      - 10 features por especialista (MSE global/canal/quadrante, SSIM, dist latente)
      - +1 feature global: cy_norm (posição vertical do bbox na imagem original)
    """
    dataset = CroppedDataset(dataset_path, train=train, transform=transform, return_meta=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    split_name = "TRAIN" if train else "VALID"
    print(f"\n--- Extraindo perfis REP ({split_name}, {len(dataset)} imagens) ---")

    # Fase 1: Calcular centroides latentes por especialista (só no train; val usa os do train)
    centroids = precomputed_centroids if precomputed_centroids is not None else [None] * n_experts
    if train and precomputed_centroids is None:
        print("Calculando centroides latentes por especialista...")
        # Usa dataset sem return_meta para o subset (compatível com Subset)
        dataset_plain = CroppedDataset(dataset_path, train=True, transform=transform)
        for idx, expert in enumerate(mvelsa.mvelsa):
            label = labels_list[idx]
            class_indices = [i for i, t in enumerate(dataset_plain.targets) if t == label]
            if len(class_indices) == 0:
                continue

            latent_vecs = []
            subset_loader = DataLoader(
                torch.utils.data.Subset(dataset_plain, class_indices[:200]),
                batch_size=1, shuffle=False
            )
            with torch.no_grad():
                for imgs, _ in subset_loader:
                    inp_flat = imgs.view(1, 1, -1).to(device, dtype=torch.float)
                    encs, _ = expert.image_forward(inp_flat)
                    latent_vecs.append(encs.detach().view(-1).cpu())

            centroids[idx] = torch.stack(latent_vecs).mean(dim=0).to(device)
            print(f"  Especialista {idx} (label {label}): centroide de {len(latent_vecs)} amostras")

    # Fase 2: Extrair perfis REP
    all_profiles = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, cy_norm in tqdm(loader, desc=f"REP {split_name}"):
            lbl = labels.item()
            if lbl not in labels_list:
                continue

            input_flat = imgs.view(1, 1, -1).to(device, dtype=torch.float)

            # 50 features de reconstrução (10 por especialista)
            image_profile = []
            for idx, expert in enumerate(mvelsa.mvelsa):
                profile, _ = extract_rep_profile(input_flat, expert, centroids[idx])
                image_profile.extend(profile)

            # +1 feature global: posição vertical normalizada
            image_profile.append(cy_norm.item())

            all_profiles.append(image_profile)
            all_labels.append(lbl)

    profiles_tensor = torch.tensor(all_profiles, dtype=torch.float32)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    print(f"  Perfis extraídos: {profiles_tensor.shape} | Labels: {labels_tensor.shape}")
    return profiles_tensor, labels_tensor, centroids


if __name__ == "__main__":
    DATA_PATH = "../../../../data/coco_cropped"

    # Extrair perfis do TRAIN (com cálculo de centroides)
    train_profiles, train_labels, centroids = extract_profiles_from_dataset(DATA_PATH, train=True)

    # Extrair perfis do VALID usando centroides calculados no train
    val_profiles, val_labels, _ = extract_profiles_from_dataset(
        DATA_PATH, train=False, precomputed_centroids=centroids)

    # Salvar
    torch.save({
        'profiles': train_profiles,
        'labels': train_labels,
        'centroids': [c.cpu() if c is not None else None for c in centroids],
        'labels_list': labels_list,
        'n_experts': n_experts,
        'features_per_expert': 10,
        'global_features': 1,  # cy_norm
    }, 'rep_profiles_train.pt')

    torch.save({
        'profiles': val_profiles,
        'labels': val_labels,
    }, 'rep_profiles_val.pt')

    print(f"\n✅ Perfis salvos:")
    print(f"  rep_profiles_train.pt: {train_profiles.shape}")
    print(f"  rep_profiles_val.pt:   {val_profiles.shape}")
