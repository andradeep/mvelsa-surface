# MVELSA — Maritime Visual ELement Supervised Autoencoder

Pipeline de classificação de objetos marítimos baseado em autoencoders MLP especializados por classe, desenvolvido como parte da minha tese de doutorado em visão computacional.

A ideia central é simples: em vez de um único modelo tentando aprender tudo, cada classe tem seu próprio autoencoder especialista. Na inferência, a imagem que um especialista **não consegue reconstruir bem** provavelmente não é da sua classe — esse erro de reconstrução vira a principal pista para a classificação.

---

## Resultados

| Estratégia | V3 — treino fullHD633 | V4 — treino IAUG | Ganho vs Baseline |
|---|:---:|:---:|:---:|
| B — Z-Score de Reconstrução | 64.47% | 64.47% | — |
| C — prob × quality (calibrada) | 71.57% | ~72% | — |
| **D — MVELSA-REP (Random Forest 55D)** | **88.32%** | **84.26%** | **+37%** |
| Baseline aleatório (5 classes) | 20.00% | 20.00% | — |

**Classes:** BOAT (1), BUOY (3), LAND (4), SHIP (5), SKY (6)  
**Validação:** fullHD633 val — 197 imagens (mesma base para V3 e V4, comparação direta)

---

## Arquitetura

### Especialistas MVELSA

Cada classe tem um autoencoder MLP independente treinado exclusivamente com amostras daquela classe:

```
Entrada: 64×64×3 = 12.288 neurônios (imagem RGB achatada)

Encoder: 12288 → 1024 → 256 → 128  (espaço latente)
Decoder: 128 → 256 → 1024 → 12288

Ativação: ReLU | Loss: RMSE | Épocas: 100 | LR: 0.001 | Batch: 64
```

### Classificador MLP

Os vetores latentes dos 5 especialistas são concatenados (5×128 = 640D) e passam por um `MultiVariableClassifier` com pesos balanceados por classe.

### MVELSA-REP — Estratégia D (principal contribuição)

Em vez de classificar só pelo latente, extraio um perfil de 55 features por imagem:

**50 features de reconstrução** (10 por especialista):
- MSE global
- MSE por canal: R, G, B
- MSE por quadrante: TL, TR, BL, BR
- SSIM entre entrada e reconstrução
- Distância euclidiana do vetor latente ao centroide da classe (treinamento)

**5 features geométricas** do bounding box:
- `cy_norm` — posição vertical do centro (normalizada pela altura da imagem)
- `aspect_ratio` — razão largura/altura do crop
- `bbox_area_norm` — área do bbox relativa à imagem original
- `bbox_w_norm` e `bbox_h_norm` — largura e altura normalizadas

Esse vetor 55D é classificado por um **Random Forest** (200 árvores, `class_weight='balanced'`, `min_samples_leaf=2`).

---

## Requisitos

### Ambiente

```bash
Python 3.10+
```

### Dependências

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn numpy pandas Pillow tqdm matplotlib
```

Ou:

```bash
pip install -r requirements.txt
```

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

> Para treinar com GPU, substitua `--index-url .../cpu` pela versão CUDA correspondente ao seu driver.

---

## Datasets

Os dados **não estão no repositório**. São necessários:

| Dataset | Onde é usado | Pasta esperada |
|---|---|---|
| fullHD633 (COCO anotado) | Treino V2/V3, validação de tudo | `data/coco_cropped/` |
| IAUG (`seadev_2_IAUG.v1`) | Treino V4 | configurável em `create_cropped_dataset_iaug.py` |

O `labels.csv` gerado precisa ter as colunas:
`filename`, `class_id`, `cy_norm`, `aspect_ratio`, `bbox_area_norm`, `bbox_w_norm`, `bbox_h_norm`

---

## Estrutura do Repositório

```
mvelsa/
├── elsanet/                          # Framework MVELSA
│   ├── autoencoder.py                # AutoEncoder MLP (encoder + decoder)
│   ├── elsa.py                       # Especialista ELSA (treino por classe)
│   ├── mvelsa.py                     # Orquestrador multi-especialista
│   └── classifier.py                 # MultiVariableClassifier
├── data/                             # Datasets — não versionados (.gitignore)
│   └── coco_cropped/
│       ├── train/labels.csv + *.png
│       └── valid/labels.csv + *.png
└── experiments/images/coco_surface/
    ├── v2_cropped_optimized/         # Treino base (fullHD633)
    ├── v3_detection_pipeline/        # MVELSA-REP V3 — 88.32%
    └── v4_iaug_validation/           # Treino IAUG, validação comparativa
```

---

## Como Reproduzir

> Todos os scripts devem ser executados **de dentro da pasta onde estão**. Os caminhos relativos dependem disso.

---

### V2 — Treino Base (fullHD633)

```bash
cd experiments/images/coco_surface/v2_cropped_optimized
```

**1. Gerar os crops do dataset**

```bash
python create_cropped_dataset.py
```

Lê as anotações COCO JSON, gera um crop por bounding box (mínimo 400 px²), remove redundâncias via IoU, faz o scene-level split 70/30 para evitar data leakage entre imagens da mesma cena.

**2. Treinar os especialistas MVELSA**

```bash
python train_cropped_mvelsa.py
```

Treina um autoencoder por classe. Salva o modelo em `ELSA_MODEL_CROPPED_SURFACE`.

**3. Extrair representações latentes**

```bash
python gen_cropped_encoded.py
```

Passa todo o dataset pelo MVELSA e salva os vetores em `ENCODED_DATA_CROPPED_SURFACE`.

**4. Treinar o classificador MLP**

```bash
python train_cropped_classifier.py
```

Treina o `MultiVariableClassifier` com os latentes 640D. Gera as curvas de avaliação (PR, ROC, confusão).

---

### V3 — MVELSA-REP e Validação Cega

```bash
cd experiments/images/coco_surface/v3_detection_pipeline
```

**5. Calibrar os especialistas**

```bash
python calibrate_experts.py
```

Calcula o erro médio de reconstrução por especialista no conjunto de validação. Salva `expert_calibration.json` — usado na Estratégia C.

**6. Extrair perfis REP 55D**

```bash
python extract_rep_profiles.py
```

Extrai as 50 features de reconstrução + 5 features geométricas por imagem. Calcula os centroides por especialista. Salva `rep_profiles_train.pt` e `rep_profiles_val.pt`.

**7. Treinar o Random Forest**

```bash
python train_meta_classifier.py
```

Treina o RF sobre os perfis 55D. Salva `rep_meta_classifier.pkl`.

**8. Validação final**

```bash
python validate_v2_blind.py
```

Avalia as 3 estratégias no conjunto de validação fullHD633.

Resultado esperado:
```
B — Z-Score de Reconstrução :  64.47%
C — prob × quality          :  71.57%
D — MVELSA-REP (55D + RF)  :  88.32%
Ganho REP vs Z-Score        : +37.0%
```

---

### V4 — Treino com Augmentação Generativa (IAUG)

Teste da hipótese: *treinar com dados gerados por augmentação generativa melhora a acurácia sobre dados reais?*

A comparação é justa: V3 e V4 usam **exatamente o mesmo conjunto de validação** (fullHD633 val, 197 imagens).

```bash
cd experiments/images/coco_surface/v4_iaug_validation
```

```bash
# 1. Gerar crops do IAUG (remapeia IDs: SHIP 6→5, SKY 7→6)
python create_cropped_dataset_iaug.py

# 2. Treinar especialistas no IAUG
python train_mvelsa_iaug.py

# 3. Extrair representações latentes
python gen_encoded_iaug.py

# 4. Treinar classificador MLP
python train_classifier_iaug.py

# 5. Calibrar especialistas (usa fullHD633 val como referência)
python calibrate_experts_iaug.py

# 6. Extrair perfis REP 55D (train=IAUG, val=fullHD633)
python extract_rep_profiles_iaug.py

# 7. Treinar Random Forest
python train_meta_classifier_iaug.py

# 8. Validação comparativa V3 vs V4
python validate_iaug_blind.py
```

Resultado esperado:
```
D — MVELSA-REP V4 (IAUG) :  84.26%
Referência V3             :  88.32%
Delta                     :  -0.0406  (hipótese não confirmada)
```

A augmentação generativa não superou o treino no dataset real. O modelo treinado com dados sintéticos perdeu especificidade discriminativa na validação com imagens reais.

---

## Variável de Ambiente

Por padrão, os scripts de validação buscam o dataset em `../../../../data/coco_cropped`. Para usar outro caminho:

```bash
FHD_DATA_PATH=/caminho/para/coco_cropped python validate_v2_blind.py
```

---

## Mapeamento de Classes

| ID | Classe | Observação |
|---|---|---|
| 1 | BOAT | |
| 3 | BUOY | |
| 4 | LAND | Filtro IoU aplicado (faixas horizontais) |
| 5 | SHIP | |
| 6 | SKY | Limitada a 120 amostras (V2/V3) |

No IAUG, os IDs originais são diferentes e remapeados automaticamente por `create_cropped_dataset_iaug.py`.
