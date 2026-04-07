# V4 — Validação com Dataset Generativamente Aumentado (IAUG)

## Objetivo

Testar a hipótese: *"treinar o MVELSA com dados generativamente aumentados melhora a acurácia sobre o conjunto de validação real?"*

A comparação é justa porque:
- **V3**: treinou no fullHD633 train, validou no fullHD633 val (77.89% na estratégia D)
- **V4**: treina no IAUG augmentado, valida no **mesmo** fullHD633 val (199 imagens)
- Pipeline, arquitetura e hiperparâmetros são idênticos

---

## Dataset IAUG

- **Fonte**: `seadev_2_IAUG.v1-com-iaug.coco/train/`
- **Augmentações**: flip horizontal, crop aleatório (0–20%), variação de brilho (±25%), blur gaussiano, ruído salt-and-pepper
- **Remapeamento de IDs**: SHIP(6→5), SKY(7→6) — alinha com fullHD633
- **Removidos**: WATER(8), BUILDING(2), PLATE(5)
- **Cap**: BUOY capado em 500 (era 20.264), demais classes capadas em 300–400

---

## Execução (ordem obrigatória)

Execute cada script **a partir da pasta `v4_iaug_validation/`**:

```bash
cd experiments/images/coco_surface/v4_iaug_validation

# 1. Gerar crops do IAUG train → data/coco_cropped_iaug/train/
python create_cropped_dataset_iaug.py

# 2. Treinar especialistas MVELSA no IAUG (~2h na GPU local)
python train_mvelsa_iaug.py

# 3. Gerar representações latentes
python gen_encoded_iaug.py

# 4. Treinar classificador MLP
python train_classifier_iaug.py

# 5. Calibrar especialistas (usa fullHD633 val)
python calibrate_experts_iaug.py

# 6. Extrair perfis REP (51D) — train=IAUG, val=fullHD633
python extract_rep_profiles_iaug.py

# 7. Treinar Random Forest nos perfis REP
python train_meta_classifier_iaug.py

# 8. Validação final comparativa V3 vs V4
python validate_iaug_blind.py
```

---

## Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `ELSA_MODEL_IAUG` | Modelo MVELSA treinado no IAUG |
| `ENCODED_DATA_IAUG` | Representações latentes |
| `MVELSA_CLASSIFIER_IAUG.pth` | Classificador MLP |
| `expert_calibration_iaug.json` | Calibração dos especialistas |
| `rep_profiles_train_iaug.pt` | Perfis REP do IAUG train |
| `rep_profiles_val_iaug.pt` | Perfis REP do fullHD633 val |
| `rep_meta_classifier_iaug.pkl` | Random Forest treinado |
| `IAUG_REP_Feature_Importance.png` | Feature importance do RF |

---

## Referências V3 (baseline de comparação)

| Estratégia | V3 (fullHD633 treino) |
|---|---|
| B — Z-Score | 64.3% |
| C — prob × quality | 69.4% |
| **D — MVELSA-REP** | **77.9%** |
| Baseline aleatório | 20.0% |
