import os
import json
import random
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Classes de fundo com tendência a gerar crops redundantes por sobreposição
BACKGROUND_CLASSES = {4, 11}  # LAND, SKY
IOU_THRESHOLD      = 0.50     # Sobreposição máxima aceita entre dois crops da mesma classe/imagem
MAX_CROPS_PER_IMAGE_CLASS = 4 # Limite de crops por classe por imagem (só para BACKGROUND_CLASSES)
MIN_BBOX_AREA      = 400      # Área mínima do bbox original (px²) — filtra anotações ruidosas

# Cap global por classe para evitar dominância de SKY no dataset
# None = sem limite; defina um inteiro para limitar (ex: {11: 120})
MAX_SAMPLES_PER_CLASS = {11: 120}  # SKY limitada a 120 amostras


def _iou(b1, b2):
    """IoU entre dois bboxes no formato COCO [x, y, w, h]."""
    ax1, ay1 = b1[0], b1[1]
    ax2, ay2 = ax1 + b1[2], ay1 + b1[3]
    bx1, by1 = b2[0], b2[1]
    bx2, by2 = bx1 + b2[2], by1 + b2[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def _safe_zone(bbox, other_bboxes, min_side=20):
    """
    Para LAND e SKY (faixas horizontais), encontra a sub-região do bbox
    que não se sobrepõe com bboxes de outras classes na mesma imagem.

    Estratégia: ajuste vertical (eixo y), pois SKY/LAND são bandas horizontais.
      - SKY: toma a faixa SUPERIOR do bbox, acima das outras anotações.
      - LAND: toma a faixa INFERIOR do bbox, abaixo das outras anotações.

    Retorna bbox ajustado [x, y, w, h] ou None se a zona pura for muito pequena.
    """
    if not other_bboxes:
        return bbox

    x, y, w, h = [float(v) for v in bbox]
    x2, y2 = x + w, y + h

    # y-ranges de outros bboxes que se sobrepõem horizontalmente com o nosso
    forbidden = []
    for ob in other_bboxes:
        ox, oy, ow, oh = [float(v) for v in ob]
        ox2, oy2 = ox + ow, oy + oh
        if ox2 > x and ox < x2:          # sobreposição horizontal existe
            forbidden.append((oy, oy2))

    if not forbidden:
        return bbox

    min_forbidden_y = min(fy  for fy, _  in forbidden)  # topo do conjunto proibido
    max_forbidden_y = max(fy2 for _, fy2 in forbidden)  # base do conjunto proibido

    # Zona pura ACIMA do conjunto proibido (boa para SKY)
    top_h = min_forbidden_y - y
    # Zona pura ABAIXO do conjunto proibido (boa para LAND)
    bot_h = y2 - max_forbidden_y

    if top_h >= bot_h and top_h >= min_side:
        return [x, y, w, top_h]
    elif bot_h >= min_side:
        return [x, max_forbidden_y, w, bot_h]
    elif top_h >= min_side:
        return [x, y, w, top_h]
    else:
        return None  # zona pura muito pequena — descartar este crop


def _deduplicate(df):
    """Remove crops sobrepostos e limita quantidade por imagem/classe para classes de fundo."""
    keep = []
    for (img_path, cat_id), group in df.groupby(['abs_path', 'cat_id']):
        if cat_id not in BACKGROUND_CLASSES:
            keep.extend(group.index.tolist())
            continue

        # Ordena por área decrescente: preserva o maior bbox quando há sobreposição
        group = group.copy()
        group['_area'] = group['bbox'].apply(lambda b: b[2] * b[3])
        group = group.sort_values('_area', ascending=False)

        accepted = []
        for idx, row in group.iterrows():
            # Verifica sobreposição com todos os já aceitos
            overlap = any(_iou(row['bbox'], group.loc[k, 'bbox']) > IOU_THRESHOLD
                          for k in accepted)
            if not overlap:
                accepted.append(idx)
            if len(accepted) >= MAX_CROPS_PER_IMAGE_CLASS:
                break

        keep.extend(accepted)

    return df.loc[keep].drop(columns=['_area'], errors='ignore')


def create_cropped_dataset(base_dir, out_dir):
    print(f"Gerando dataset recortado ESTRATIFICADO em: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    splits = ["train", "valid", "test"]
    all_samples = []
    
    # 1. Coletar TODAS as bboxes de todos os splits originais (DEDUP GLOBAL)
    samples_dict = {} # Keyed by ann_id to prevent redundant entries from base dataset
    
    unique_images_map = {} # path -> image_id (to handle base dataset inconsistencies)

    for split in splits:
        json_path = os.path.join(base_dir, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            continue
            
        print(f"Coletando dados de {split}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        categories = {cat["id"]: cat["name"] for cat in data["categories"]}
        img_info = {img["id"]: img["file_name"] for img in data["images"]}
        
        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            # Manter apenas as 4 classes de interesse para a tese
            # 1=BOAT, 3=BUOY, 4=LAND, 11=SKY
            if cat_id not in {1, 3, 4, 11}: continue

            bbox = ann["bbox"]
            if bbox[2] <= 1 or bbox[3] <= 1: continue
            if bbox[2] * bbox[3] < MIN_BBOX_AREA: continue  # descarta anotações minúsculas
                
            img_filename = img_info[ann["image_id"]]
            img_path = os.path.abspath(os.path.join(base_dir, split, img_filename))
            
            # Só adicionar se for único (evita a contaminação massiva do coco640)
            ann_id = ann["id"]
            if ann_id not in samples_dict:
                samples_dict[ann_id] = {
                    "abs_path": img_path,
                    "bbox": bbox,
                    "cat_id": cat_id,
                    "cat_name": categories[cat_id],
                    "ann_id": ann_id
                }

    df_total = pd.DataFrame(list(samples_dict.values()))
    print(f"Total de amostras ÚNICAS encontradas: {len(df_total)}")

    # --- DEDUPLICAÇÃO: remove crops sobrepostos de LAND e SKY ---
    before = len(df_total)
    df_total = _deduplicate(df_total)
    print(f"Após deduplicação IoU (LAND/SKY): {len(df_total)} amostras "
          f"({before - len(df_total)} removidas)")

    # --- CAP GLOBAL: limita amostras por classe para evitar dominância ---
    if MAX_SAMPLES_PER_CLASS:
        frames = []
        for cat_id, group in df_total.groupby('cat_id'):
            cap = MAX_SAMPLES_PER_CLASS.get(cat_id)
            if cap and len(group) > cap:
                group = group.sample(n=cap, random_state=42)
                print(f"  [cap] cat_id {cat_id}: {len(group)} amostras (limitado a {cap})")
            frames.append(group)
        df_total = pd.concat(frames).reset_index(drop=True)

    print("Distribuição por classe após deduplicação + cap:")
    for cat_id, count in df_total['cat_id'].value_counts().sort_index().items():
        print(f"  cat_id {cat_id}: {count} amostras")

    # --- OTIMIZAÇÃO DE RIGOR CIENTÍFICO: Split por IMAGEM (não por objeto) ---
    unique_images = df_total[['abs_path']].drop_duplicates()
    img_main_class = df_total.groupby('abs_path')['cat_id'].first()
    unique_images = unique_images.merge(img_main_class, on='abs_path')

    # Fallback para estratificação: se houver classes com menos de 2 membros no pool atual, 
    # não estratificamos especificamente esse split.
    def safe_strat_split(data, test_size, stratify_col):
        counts = data[stratify_col].value_counts()
        rare_classes = counts[counts < 2].index.tolist()
        if len(rare_classes) > 0:
            print(f"  [!] Alerta: Classes {rare_classes} são muito raras para estratificação perfeita. Usando split aleatório simples.")
            return train_test_split(data, test_size=test_size, random_state=42)
        return train_test_split(data, test_size=test_size, stratify=data[stratify_col], random_state=42)

    train_imgs, temp_imgs = safe_strat_split(unique_images, 0.30, 'cat_id')
    
    # Para o segundo split (val/test), a chance de ter classe única na metade é ALTA.
    # Vamos garantir que val e test tenham pelo menos um pouco de tudo se possível, ou split aleatório.
    val_imgs, test_imgs = safe_strat_split(temp_imgs, 0.50, 'cat_id')

    split_map = {
        "train": df_total[df_total['abs_path'].isin(train_imgs['abs_path'])],
        "valid": df_total[df_total['abs_path'].isin(val_imgs['abs_path'])],
        "test": df_total[df_total['abs_path'].isin(test_imgs['abs_path'])]
    }

    # Índice global: para cada imagem, lista de bboxes de OUTRAS classes
    # Usado para calcular a zona pura de LAND e SKY
    img_other_bboxes = {}
    for _, row in df_total.iterrows():
        key = row['abs_path']
        if key not in img_other_bboxes:
            img_other_bboxes[key] = {}
        cat = row['cat_id']
        if cat not in img_other_bboxes[key]:
            img_other_bboxes[key][cat] = []
        img_other_bboxes[key][cat].append(row['bbox'])

    # 4. Salvar recortes e gerar CSVs
    for split_name, df_split in split_map.items():
        print(f"Processando split NEW_{split_name} ({len(df_split)} objetos de {len(df_split['abs_path'].unique())} imagens)...")
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        csv_data = []
        skipped_pure = 0
        for _, row in df_split.iterrows():
            try:
                cat_id  = row['cat_id']
                img_path = row['abs_path']

                # Para LAND e SKY: calcular a zona pura antes de recortar
                bbox = row['bbox']
                if cat_id in BACKGROUND_CLASSES:
                    other = [b for cid, bboxes in img_other_bboxes.get(img_path, {}).items()
                               if cid != cat_id for b in bboxes]
                    bbox = _safe_zone(bbox, other)
                    if bbox is None:
                        skipped_pure += 1
                        continue  # zona pura muito pequena, descarta

                img = Image.open(img_path).convert('RGB')
                img_w, img_h = img.size
                x, y, w, h = [int(v) for v in bbox]

                # --- Crop Quadrado com Contexto Real ---
                cx, cy = x + w // 2, y + h // 2
                side   = max(w, h)
                x1 = cx - side // 2
                y1 = cy - side // 2
                x2 = x1 + side
                y2 = y1 + side
                if x1 < 0:  x2 -= x1;  x1 = 0
                if y1 < 0:  y2 -= y1;  y1 = 0
                if x2 > img_w: x1 -= (x2 - img_w); x2 = img_w
                if y2 > img_h: y1 -= (y2 - img_h); y2 = img_h
                x1, y1 = max(0, x1), max(0, y1)

                cropped_img = img.crop((x1, y1, x2, y2))
                if cropped_img.size[0] < 5 or cropped_img.size[1] < 5: continue

                cropped_img = cropped_img.resize((64, 64), Image.LANCZOS)

                new_filename = f"{row['cat_name']}_{row['ann_id']}.jpg"
                new_path     = os.path.join(split_dir, new_filename)
                cropped_img.save(new_path)
                csv_data.append({"filename": new_filename, "class_id": cat_id, "class_name": row['cat_name']})
            except Exception as e:
                print(f"Erro em {row['abs_path']}: {e}")
        if skipped_pure:
            print(f"  [!] {skipped_pure} crops de LAND/SKY descartados por zona pura insuficiente.")
        pd.DataFrame(csv_data).to_csv(os.path.join(split_dir, "labels.csv"), index=False)
        print(f"  -> Concluído {split_name}.")
if __name__ == "__main__":
    base = "../../../../data/coco640"
    out = "../../../../data/coco_cropped"
    create_cropped_dataset(base, out)
    print("\nDataset ESTRATIFICADO gerado com sucesso!")
