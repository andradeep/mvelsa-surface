"""
MVELSA V4 — Extração de Perfis REP (IAUG)
==========================================
Extrai perfis de reconstrução de 51 dimensões (10/especialista × 5 + cy_norm).

Treino REP: IAUG augmentado (coco_cropped_iaug/train/)
Val   REP: fullHD633 val  (coco_cropped/valid/)  ← MESMA da V3

Centroides latentes calculados no IAUG train e reusados na extração do val.
"""
import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from cropped_data_generator import CroppedDataset, PadToSquare

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ELSA_PATH     = "ELSA_MODEL_IAUG"
COMBINED_DATA = os.environ.get("COMBINED_DATA_PATH", "../../../../data/coco_cropped_combined")
FHD_DATA      = os.environ.get("FHD_DATA_PATH",      "../../../../data/coco_cropped")

print(f"Carregando MVELSA: {ELSA_PATH}")
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

labels_list = mvelsa.labels
n_experts   = len(labels_list)
print(f"Especialistas: {n_experts} | Labels: {labels_list}")

transform = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def compute_ssim_simple(img1, img2):
    c, h, w = 3, 64, 64
    try:
        i1 = img1.view(c, h, w)
        i2 = img2.view(c, h, w)
    except RuntimeError:
        return 0.5
    mu1 = i1.mean(dim=(1, 2))
    mu2 = i2.mean(dim=(1, 2))
    sigma1_sq = ((i1 - mu1.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    sigma2_sq = ((i2 - mu2.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    sigma12   = ((i1 - mu1.view(c, 1, 1)) * (i2 - mu2.view(c, 1, 1))).mean(dim=(1, 2))
    C1, C2   = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.mean().item()


def extract_rep_profile(input_flat, expert, centroid=None):
    """10 features por especialista: MSE global/R/G/B/TL/TR/BL/BR + SSIM + dist latente."""
    encs, dec_avg = expert.image_forward(input_flat)
    inp = input_flat.squeeze()
    rec = dec_avg.squeeze()

    mse_global = F.mse_loss(rec, inp).item()
    c, h, w    = 3, 64, 64
    try:
        inp_img = inp.view(c, h, w)
        rec_img = rec.view(c, h, w)
        mse_r   = F.mse_loss(rec_img[0], inp_img[0]).item()
        mse_g   = F.mse_loss(rec_img[1], inp_img[1]).item()
        mse_b   = F.mse_loss(rec_img[2], inp_img[2]).item()
        mid_h, mid_w = h // 2, w // 2
        mse_tl  = F.mse_loss(rec_img[:, :mid_h, :mid_w], inp_img[:, :mid_h, :mid_w]).item()
        mse_tr  = F.mse_loss(rec_img[:, :mid_h, mid_w:], inp_img[:, :mid_h, mid_w:]).item()
        mse_bl  = F.mse_loss(rec_img[:, mid_h:, :mid_w], inp_img[:, mid_h:, :mid_w]).item()
        mse_br  = F.mse_loss(rec_img[:, mid_h:, mid_w:], inp_img[:, mid_h:, mid_w:]).item()
    except RuntimeError:
        mse_r = mse_g = mse_b = mse_global
        mse_tl = mse_tr = mse_bl = mse_br = mse_global

    ssim_val    = compute_ssim_simple(inp, rec)
    latent      = encs.detach().view(-1)
    latent_dist = torch.norm(latent - centroid).item() if centroid is not None else 0.0

    return [mse_global, mse_r, mse_g, mse_b, mse_tl, mse_tr, mse_bl, mse_br, ssim_val, latent_dist], encs


def extract_profiles(data_path, train=True, precomputed_centroids=None):
    """Extrai perfis REP (51D) de um split de dataset."""
    dataset     = CroppedDataset(data_path, train=train, transform=transform, return_meta=True)
    loader      = DataLoader(dataset, batch_size=1, shuffle=False)
    split_name  = "TRAIN(IAUG)" if train else "VAL(fullHD633)"
    print(f"\n--- Extraindo perfis REP [{split_name}] — {len(dataset)} imagens ---")

    centroids = precomputed_centroids if precomputed_centroids is not None else [None] * n_experts

    # Centroides: calculados no IAUG train, reutilizados no val
    if train and precomputed_centroids is None:
        print("Calculando centroides latentes por especialista (IAUG train)...")
        dataset_plain = CroppedDataset(COMBINED_DATA, train=True, transform=transform)
        for idx, expert in enumerate(mvelsa.mvelsa):
            label         = labels_list[idx]
            class_indices = [i for i, t in enumerate(dataset_plain.targets) if t == label]
            if not class_indices:
                continue
            latent_vecs = []
            subset_loader = DataLoader(
                torch.utils.data.Subset(dataset_plain, class_indices[:200]),
                batch_size=1, shuffle=False
            )
            with torch.no_grad():
                for imgs, _ in subset_loader:
                    inp_flat = imgs.view(1, 1, -1).to(device, dtype=torch.float)
                    encs, _  = expert.image_forward(inp_flat)
                    latent_vecs.append(encs.detach().view(-1).cpu())
            centroids[idx] = torch.stack(latent_vecs).mean(dim=0).to(device)
            print(f"  Especialista {idx} (label {label}): centroide de {len(latent_vecs)} amostras")

    all_profiles, all_labels = [], []

    with torch.no_grad():
        for imgs, labels, cy_norm in tqdm(loader, desc=f"REP {split_name}"):
            lbl = labels.item()
            if lbl not in labels_list:
                continue

            input_flat    = imgs.view(1, 1, -1).to(device, dtype=torch.float)
            image_profile = []
            for idx, expert in enumerate(mvelsa.mvelsa):
                profile, _ = extract_rep_profile(input_flat, expert, centroids[idx])
                image_profile.extend(profile)

            image_profile.append(cy_norm.item())  # 51ª feature: cy_norm

            all_profiles.append(image_profile)
            all_labels.append(lbl)

    profiles_tensor = torch.tensor(all_profiles, dtype=torch.float32)
    labels_tensor   = torch.tensor(all_labels, dtype=torch.long)
    print(f"  Perfis: {profiles_tensor.shape} | Labels: {labels_tensor.shape}")
    return profiles_tensor, labels_tensor, centroids


if __name__ == "__main__":
    # Train: fullHD633 + IAUG combinados (calcula centroides)
    train_profiles, train_labels, centroids = extract_profiles(COMBINED_DATA, train=True)

    # Val: fullHD633 (usa centroides do IAUG train)
    val_profiles, val_labels, _ = extract_profiles(FHD_DATA, train=False, precomputed_centroids=centroids)

    torch.save({
        'profiles':           train_profiles,
        'labels':             train_labels,
        'centroids':          [c.cpu() if c is not None else None for c in centroids],
        'labels_list':        labels_list,
        'n_experts':          n_experts,
        'features_per_expert': 10,
        'global_features':    1,  # cy_norm
    }, 'rep_profiles_train_iaug.pt')

    torch.save({
        'profiles': val_profiles,
        'labels':   val_labels,
    }, 'rep_profiles_val_iaug.pt')

    print(f"\n✅ Perfis salvos:")
    print(f"  rep_profiles_train_iaug.pt: {train_profiles.shape}")
    print(f"  rep_profiles_val_iaug.pt:   {val_profiles.shape}")
    print("Próximo passo: python train_meta_classifier_iaug.py")
