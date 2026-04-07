# MVELSA — Maritime Surface Object Classification

MVELSA (Multi-Variate ELSA) applies per-class autoencoder specialists to classify maritime surface objects: **BOAT, BUOY, LAND, SHIP, SKY**.

## Results Summary

| Version | Strategy | Accuracy |
|---------|----------|----------|
| V3 (fullHD633) | A — Argmin reconstruction error | ~64% |
| V3 (fullHD633) | B/C — MLP on 1280D latent | 64.47% |
| V3 (fullHD633) | C — MLP on 1280D latent | 70.05% |
| **V3 (fullHD633)** | **D — Random Forest on 55D REP** | **88.32%** |
| V4 (fullHD633 + IAUG) | D — Random Forest on 55D REP | 84.26% |

> REP = Reconstruction Error Profile (50D reconstruction features + 5D bbox geometry)

---

## Architecture

```
Input Image (128×128)
       ↓
┌─────────────────────────────────────────┐
│  5 Specialist Autoencoders (one/class)  │
│  Each: Conv encoder → 256D latent       │
└─────────────────────────────────────────┘
       ↓ (concatenate all 5 latents)
  1280D representation
       ↓
  Strategy B/C: MLP classifier
       OR
  55D REP Profile:
    50D: reconstruction error statistics per specialist (5 patches × mean+std × 5 specialists)
     5D: bbox geometry (cy_norm, aspect_ratio, bbox_area_norm, bbox_w_norm, bbox_h_norm)
       ↓
  Strategy D: Random Forest → class prediction
```

---

## Dataset Structure

```
data/
├── coco_cropped/            ← fullHD633 dataset (V3)
│   ├── train/
│   │   ├── BOAT_2291.jpg
│   │   └── labels.csv       ← filename, class_id, class_name, cy_norm, aspect_ratio, ...
│   ├── valid/
│   └── test/
├── coco_cropped_iaug/       ← IAUG-only crops (intermediate)
│   └── train/
└── coco_cropped_combined/   ← FHD + IAUG merged (V4)
    ├── train/               ← fhd_* and iaug_* prefixed
    ├── valid/               ← FHD only
    └── test/                ← FHD only
```

---

## Reproducing V3 Results (fullHD633 baseline)

### 1. Create the dataset

```bash
cd experiments/images/coco_surface
python enrich_labels_bbox.py    # add bbox features to existing labels.csv
```

If starting from scratch:
```bash
python v2_cropped_optimized/create_cropped_dataset_hd.py
```

### 2. Train MVELSA specialists

```bash
python v2_cropped_optimized/train_cropped_mvelsa.py
```

Saves 5 autoencoder models to `v2_cropped_optimized/ELSA_MODEL_CROPPED_SURFACE/`.

### 3. Generate encoded features (Strategy B/C)

```bash
python v2_cropped_optimized/gen_cropped_encoded.py
```

Saves 1280D feature tensors to `v2_cropped_optimized/ENCODED_DATA_CROPPED_SURFACE/`.

### 4. Train MLP classifier (Strategy B/C)

```bash
python v2_cropped_optimized/train_cropped_classifier.py
```

### 5. Calibrate specialists

```bash
python v3_detection_pipeline/calibrate_experts.py
```

### 6. Extract REP profiles (Strategy D)

```bash
python v3_detection_pipeline/extract_rep_profiles.py
```

Saves 55D profile tensors to `v3_detection_pipeline/`.

### 7. Train Random Forest (Strategy D)

```bash
python v3_detection_pipeline/train_meta_classifier.py
```

### 8. Blind validation

```bash
python v3_detection_pipeline/validate_v2_blind.py
```

Expected output: **Strategy D ≈ 88.32%**

---

## Reproducing V4 Results (IAUG augmentation)

### 1. Create IAUG crops

```bash
python v4_iaug_validation/create_cropped_dataset_iaug.py
```

> Only BOAT, BUOY, SKY from IAUG (LAND and SHIP excluded — domain shift degrades these specialists).

### 2. Enrich IAUG labels

```bash
python enrich_labels_bbox_iaug.py
```

### 3. Merge FHD + IAUG

```bash
python v4_iaug_validation/merge_datasets.py
```

### 4. Train IAUG specialists

```bash
python v4_iaug_validation/train_mvelsa_iaug.py
```

> LAND and SHIP specialists are trained on FHD-only data.

### 5. Generate encoded features

```bash
python v4_iaug_validation/gen_encoded_iaug.py
```

### 6. Calibrate IAUG specialists

```bash
python v4_iaug_validation/calibrate_experts_iaug.py
```

### 7. Extract IAUG REP profiles

```bash
python v4_iaug_validation/extract_rep_profiles_iaug.py
```

### 8. Train IAUG Random Forest

```bash
python v4_iaug_validation/train_meta_classifier_iaug.py
```

### 9. Blind validation

```bash
python v4_iaug_validation/validate_iaug_blind.py
```

Expected output: **Strategy D ≈ 84.26%**

---

## Utility Scripts

### Inspect and clean dataset

```bash
python inspecionar_crops.py --split train --class BOAT
```

Click images to mark for deletion, press `[d]` to delete and update CSV.

### Sync labels.csv after manual deletion

```bash
python sync_labels_csv.py
python sync_labels_csv.py --data-dir /path/to/coco_cropped_combined
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Square crop with margin=5% | Preserves aspect context; margin reduces border artifacts |
| SHIP crop: side=min(w, max(h, h×2)) | Ships are elongated horizontally; square crop loses shape |
| LAND/SHIP excluded from IAUG | IAUG augmentation (blur/noise) degrades these specialists by reducing inter-class error differences |
| BUOY capped at 3000 in IAUG | Overrepresented; more samples degraded REP discriminability |
| 5D bbox features | cy_norm alone couldn't separate BOAT vs BUOY; area_norm separates them (BOAT median=0.032 vs BUOY=0.016) |
| Random Forest over MLP | Handles non-linear feature interactions; no retraining of autoencoders needed |

---

## Directory Structure

```
experiments/images/coco_surface/
├── README.md
├── RELATORIO_PROGRESSO.md
├── ARTIGO_DRAFT.md
├── enrich_labels_bbox.py
├── enrich_labels_bbox_iaug.py
├── sync_labels_csv.py
├── inspecionar_crops.py
│
├── v2_cropped_optimized/
│   ├── cropped_data_generator.py   ← Dataset class (shared by all scripts)
│   ├── create_cropped_dataset_hd.py
│   ├── train_cropped_mvelsa.py
│   ├── gen_cropped_encoded.py
│   ├── train_cropped_classifier.py
│   ├── ELSA_MODEL_CROPPED_SURFACE/ ← Trained autoencoder weights
│   ├── ENCODED_DATA_CROPPED_SURFACE/
│   └── MVELSA_CLASSIFIER.pth
│
├── v3_detection_pipeline/
│   ├── calibrate_experts.py
│   ├── extract_rep_profiles.py
│   ├── train_meta_classifier.py
│   ├── validate_v2_blind.py
│   ├── rep_meta_classifier.pkl
│   ├── rep_profiles_train.pt
│   └── rep_profiles_val.pt
│
└── v4_iaug_validation/
    ├── create_cropped_dataset_iaug.py
    ├── merge_datasets.py
    ├── train_mvelsa_iaug.py
    ├── gen_encoded_iaug.py
    ├── calibrate_experts_iaug.py
    ├── extract_rep_profiles_iaug.py
    ├── train_meta_classifier_iaug.py
    ├── validate_iaug_blind.py
    ├── ELSA_MODEL_IAUG/
    ├── ENCODED_DATA_IAUG/
    ├── rep_meta_classifier_iaug.pkl
    └── expert_calibration_iaug.json
```

---

## Dependencies

```
torch torchvision
scikit-learn
pandas
pillow
matplotlib
seaborn
tqdm
```

Install: `pip install torch torchvision scikit-learn pandas pillow matplotlib seaborn tqdm`
