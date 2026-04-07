"""
MVELSA V4 — Validação Final Cega (IAUG → fullHD633 val)
=========================================================
Compara 3 estratégias com o modelo treinado no IAUG:
  B) Z-Score de Reconstrução
  C) prob × quality (calibrada)
  D) MVELSA-REP Random Forest (perfil 51D)

Validação feita sobre os 199 crops do fullHD633 val — MESMA base da V3.
Comparação direta: V3 (fullHD633 treino) vs V4 (IAUG treino).
"""
import sys
import os
import torch
import torch.nn.functional as F
import json
import pickle
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append("../../../../")
sys.path.append("../v2_cropped_optimized")

from elsanet.mvelsa import MVELSA
from elsanet.classifier import MultiVariableClassifier
from cropped_data_generator import CroppedDataset, PadToSquare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ELSA_PATH        = "ELSA_MODEL_IAUG"
CLFS_PATH        = "MVELSA_CLASSIFIER_IAUG.pth"
CALIBRATION_FILE = "expert_calibration_iaug.json"
REP_MODEL_FILE   = "rep_meta_classifier_iaug.pkl"
REP_CENTROIDS_FILE = "rep_profiles_train_iaug.pt"
FHD_DATA         = os.environ.get("FHD_DATA_PATH", "../../../../data/coco_cropped")

base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
EVAL_IDS     = sorted(base_classes.keys())
EVAL_NAMES   = [base_classes[k] for k in EVAL_IDS]

# --- 1. Carregar modelos ---
print("Carregando modelos V4 (IAUG)...")
mvelsa = torch.load(ELSA_PATH, map_location=device, weights_only=False)
mvelsa.to(device)
mvelsa.eval()

checkpoint     = torch.load(CLFS_PATH, map_location=device, weights_only=False)
classifier_obj = MultiVariableClassifier(checkpoint['model_parameters'])
classifier_obj.load_state_dict(checkpoint['state_dict'])
classifier_obj.to(device)
classifier_obj.eval()

calibration_weights = {}
if os.path.exists(CALIBRATION_FILE):
    with open(CALIBRATION_FILE) as f:
        calib_data = json.load(f)
        for class_name, info in calib_data.items():
            calibration_weights[class_name] = info["avg_reconstruction_error"]

FEATURES_PER_EXPERT = 10
GLOBAL_FEATURES     = 5   # cy_norm, aspect_ratio, bbox_area_norm, bbox_w_norm, bbox_h_norm
n_experts_current   = len(mvelsa.mvelsa)

rep_model    = None
rep_centroids = None
if os.path.exists(REP_MODEL_FILE):
    with open(REP_MODEL_FILE, 'rb') as f:
        rep_data = pickle.load(f)
    expected_features = rep_data['model'].n_features_in_
    total_expected    = n_experts_current * FEATURES_PER_EXPERT + GLOBAL_FEATURES
    if expected_features != total_expected:
        print(f"⚠️  REP ignorado: treinado com {expected_features} features, "
              f"modelo atual usa {total_expected}.")
    else:
        rep_model = rep_data['model']
        print(f"✅ Meta-classificador REP carregado (acc val: {rep_data.get('accuracy', '?'):.4f})")
        if os.path.exists(REP_CENTROIDS_FILE):
            train_data    = torch.load(REP_CENTROIDS_FILE, weights_only=False)
            rep_centroids = [c.to(device) if c is not None else None
                             for c in train_data['centroids']]
else:
    print("⚠️  REP não encontrado. Execute extract_rep_profiles_iaug.py e train_meta_classifier_iaug.py.")

# --- 2. Dataset de validação ---
transform_val = transforms.Compose([
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Resize(size=(64, 64)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset_val = CroppedDataset(FHD_DATA, train=False, transform=transform_val, return_meta=True)
val_loader  = DataLoader(dataset_val, batch_size=1, shuffle=False)

labels_list  = mvelsa.labels
class_names  = [base_classes.get(l, f"ID_{l}") for l in labels_list]
n_experts    = len(labels_list)

checkpoint_labels  = checkpoint['labels_list']
expert_to_clf_idx  = {label: checkpoint_labels.index(label)
                      for label in labels_list if label in checkpoint_labels}
print(f"Mapeamento Expert→Classificador: {expert_to_clf_idx}")


# --- Funções REP ---
def compute_ssim_simple(img1, img2):
    c, h, w = 3, 64, 64
    try:
        i1 = img1.view(c, h, w)
        i2 = img2.view(c, h, w)
    except RuntimeError:
        return 0.5
    mu1 = i1.mean(dim=(1, 2))
    mu2 = i2.mean(dim=(1, 2))
    s1  = ((i1 - mu1.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    s2  = ((i2 - mu2.view(c, 1, 1)) ** 2).mean(dim=(1, 2))
    s12 = ((i1 - mu1.view(c, 1, 1)) * (i2 - mu2.view(c, 1, 1))).mean(dim=(1, 2))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return ssim.mean().item()


def extract_rep_profile(input_flat, expert, centroid=None):
    encs, dec_avg = expert.image_forward(input_flat)
    inp = input_flat.squeeze()
    rec = dec_avg.squeeze()
    mse_global = F.mse_loss(rec, inp).item()
    c, h, w = 3, 64, 64
    try:
        inp_img = inp.view(c, h, w)
        rec_img = rec.view(c, h, w)
        mse_r   = F.mse_loss(rec_img[0], inp_img[0]).item()
        mse_g   = F.mse_loss(rec_img[1], inp_img[1]).item()
        mse_b   = F.mse_loss(rec_img[2], inp_img[2]).item()
        mh, mw  = h // 2, w // 2
        mse_tl  = F.mse_loss(rec_img[:, :mh, :mw], inp_img[:, :mh, :mw]).item()
        mse_tr  = F.mse_loss(rec_img[:, :mh, mw:], inp_img[:, :mh, mw:]).item()
        mse_bl  = F.mse_loss(rec_img[:, mh:, :mw], inp_img[:, mh:, :mw]).item()
        mse_br  = F.mse_loss(rec_img[:, mh:, mw:], inp_img[:, mh:, mw:]).item()
    except RuntimeError:
        mse_r = mse_g = mse_b = mse_global
        mse_tl = mse_tr = mse_bl = mse_br = mse_global

    ssim_val    = compute_ssim_simple(inp, rec)
    latent      = encs.detach().view(-1)
    latent_dist = torch.norm(latent - centroid).item() if centroid is not None else 0.0
    return [mse_global, mse_r, mse_g, mse_b, mse_tl, mse_tr, mse_bl, mse_br, ssim_val, latent_dist], encs


# --- 3. Validação ---
y_true          = []
y_pred_recon    = []
y_pred_combined = []
y_pred_rep      = []

print(f"\nValidando sobre fullHD633 val ({len(dataset_val)} imagens, {n_experts} especialistas)...")

with torch.no_grad():
    for inputs, labels, cy_norm in tqdm(val_loader):
        lbl = labels.item()
        if lbl not in labels_list:
            continue
        y_true.append(lbl)

        input_flat     = inputs.view(1, 1, -1).to(device, dtype=torch.float)
        recon_errors   = []
        combined_scores = []
        rep_profile    = []
        all_encs       = []

        for idx, expert in enumerate(mvelsa.mvelsa):
            encs, dec_avg = expert.image_forward(input_flat)
            recon_error   = F.mse_loss(dec_avg, input_flat).item()
            recon_errors.append(recon_error)
            all_encs.append(encs.detach().cpu())

            if rep_model is not None:
                centroid = rep_centroids[idx] if rep_centroids else None
                profile, _ = extract_rep_profile(input_flat, expert, centroid)
                rep_profile.extend(profile)

        # Classificador MLP
        full_latent = torch.cat(all_encs, dim=-1).to(device)
        latent_flat = full_latent.view(full_latent.shape[0], -1)
        output      = classifier_obj(latent_flat)
        probs_all   = torch.exp(output)[0]

        # Scores combinados (estratégia C)
        for idx in range(len(class_names)):
            class_name = class_names[idx]
            baseline   = calibration_weights.get(class_name, 0.1)
            quality    = 1.0 / ((recon_errors[idx] / baseline) + 1e-6)
            clf_idx    = expert_to_clf_idx.get(labels_list[idx], idx)
            combined_scores.append(probs_all[clf_idx].item() * quality)

        # Estratégia B: Z-Score
        arr = recon_errors
        mu  = sum(arr) / len(arr)
        var = sum((x - mu) ** 2 for x in arr) / len(arr)
        sig = var ** 0.5
        z_scores = [(x - mu) / sig for x in arr] if sig > 0 else [0.0] * len(arr)
        y_pred_recon.append(labels_list[min(range(len(z_scores)), key=lambda i: z_scores[i])])

        # Estratégia C
        y_pred_combined.append(labels_list[max(range(len(combined_scores)), key=lambda i: combined_scores[i])])

        # Estratégia D: MVELSA-REP
        rep_total = n_experts * FEATURES_PER_EXPERT + GLOBAL_FEATURES
        if rep_model is not None and len(rep_profile) == n_experts * FEATURES_PER_EXPERT:
            meta = cy_norm  # tensor of 5 features
            if hasattr(meta, 'tolist'):
                rep_profile.extend(meta.squeeze().tolist())
            else:
                rep_profile.append(float(meta))
            if len(rep_profile) == rep_total:
                y_pred_rep.append(rep_model.predict([rep_profile])[0])


# --- 4. Métricas ---
V3_REFERENCE = {"B": 0.6447, "C": 0.7157, "D": 0.8832}


def print_report(y_true_f, y_pred_f, title, v3_ref=None):
    if not y_true_f:
        return None
    print(f"\n{'='*65}")
    print(title)
    print('='*65)
    print(classification_report(y_true_f, y_pred_f, labels=EVAL_IDS,
                                 target_names=EVAL_NAMES, zero_division=0))
    cm  = confusion_matrix(y_true_f, y_pred_f, labels=EVAL_IDS)
    print("Matriz de Confusão (BOAT | BUOY | LAND | SHIP | SKY):")
    print(cm)
    acc = accuracy_score(y_true_f, y_pred_f)
    print(f"Acurácia V4: {acc:.4f}")
    if v3_ref:
        delta = acc - v3_ref
        print(f"Referência V3: {v3_ref:.4f} | Delta: {delta:+.4f} "
              f"({'✅ melhora' if delta > 0 else '❌ piora'})")
    return acc


def filter_eval(yt, yp):
    pairs = [(t, p) for t, p in zip(yt, yp) if t in EVAL_IDS]
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]


yt_b, yp_b = filter_eval(y_true, y_pred_recon)
yt_c, yp_c = filter_eval(y_true, y_pred_combined)
yt_d, yp_d = filter_eval(y_true, y_pred_rep)

acc_b = print_report(yt_b, yp_b, "📊 ESTRATÉGIA B — Z-Score de Reconstrução",      V3_REFERENCE["B"])
acc_c = print_report(yt_c, yp_c, "📊 ESTRATÉGIA C — prob × quality (calibrada)",   V3_REFERENCE["C"])
acc_d = print_report(yt_d, yp_d, "✅ ESTRATÉGIA D — MVELSA-REP (RF em 51D)", V3_REFERENCE["D"])

print(f"\n{'='*65}")
print("RESUMO COMPARATIVO V3 vs V4")
print('='*65)
print(f"{'Estratégia':<30} {'V3 (fullHD633)':>15} {'V4 (IAUG)':>12} {'Delta':>8}")
print("-"*65)
if acc_b:
    print(f"  B - Z-Score Reconstrução      {V3_REFERENCE['B']:>15.4f} {acc_b:>12.4f} {acc_b - V3_REFERENCE['B']:>+8.4f}")
if acc_c:
    print(f"  C - prob × quality            {V3_REFERENCE['C']:>15.4f} {acc_c:>12.4f} {acc_c - V3_REFERENCE['C']:>+8.4f}")
if acc_d:
    print(f"  D - MVELSA-REP (51D+RF)       {V3_REFERENCE['D']:>15.4f} {acc_d:>12.4f} {acc_d - V3_REFERENCE['D']:>+8.4f}")
print(f"  Baseline aleatório            {1/len(EVAL_IDS):>15.4f}")

if acc_d:
    print(f"\n  {'✅ Hipótese CONFIRMADA' if acc_d > V3_REFERENCE['D'] else '❌ Hipótese NÃO confirmada'}: "
          f"augmentação generativa {'melhorou' if acc_d > V3_REFERENCE['D'] else 'não melhorou'} a acurácia "
          f"({acc_d:.4f} vs {V3_REFERENCE['D']:.4f})")
