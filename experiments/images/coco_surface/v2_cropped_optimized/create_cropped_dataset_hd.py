"""
Gerador de dataset recortado a partir do fullHD633.
Uma única pasta de imagens com _annotations.coco.json.

Classes: BOAT(1), BUOY(3), LAND(4), SHIP(5), SKY(6)
Saída:   ../../../../data/coco_cropped  (mesmos caminhos dos scripts de treino)
"""
import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÃO ---
SOURCE_DIR   = "/home/andradearthurb/Documentos/DOC/reaugment/fullHD633/fullHD"
ANNOT_FILE   = os.path.join(SOURCE_DIR, "_annotations.coco.json")
OUT_DIR      = "../../../../data/coco_cropped"

FOCUS_CLASSES    = {1, 3, 4, 5, 6}      # BOAT, BUOY, LAND, SHIP, SKY
BACKGROUND_CLASSES = {4, 6}             # LAND, SKY — zonas puras + deduplicação
MIN_BBOX_AREA    = 1600                 # 40×40px mínimo na imagem original
IOU_THRESHOLD    = 0.50
MAX_CROPS_PER_IMAGE_CLASS = 4           # só para BACKGROUND_CLASSES
MAX_SAMPLES_PER_CLASS = {3: 400, 6: 200}  # cap BUOY e SKY para equilíbrio


# --- FUNÇÕES AUXILIARES ---

def _iou(b1, b2):
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


def _deduplicate(df):
    """Remove sobreposição IoU e limita crops por imagem/classe para BACKGROUND_CLASSES."""
    keep = []
    for (img_path, cat_id), group in df.groupby(['abs_path', 'cat_id']):
        if cat_id not in BACKGROUND_CLASSES:
            keep.extend(group.index.tolist())
            continue
        group = group.copy()
        group['_area'] = group['bbox'].apply(lambda b: b[2] * b[3])
        group = group.sort_values('_area', ascending=False)
        accepted = []
        for idx, row in group.iterrows():
            overlap = any(_iou(row['bbox'], group.loc[k, 'bbox']) > IOU_THRESHOLD
                          for k in accepted)
            if not overlap:
                accepted.append(idx)
            if len(accepted) >= MAX_CROPS_PER_IMAGE_CLASS:
                break
        keep.extend(accepted)
    return df.loc[keep].drop(columns=['_area'], errors='ignore')


def _safe_zone(bbox, other_bboxes, min_side=40):
    """Retorna sub-região do bbox sem sobreposição com outras classes (eixo Y)."""
    if not other_bboxes:
        return bbox
    x, y, w, h = [float(v) for v in bbox]
    x2, y2 = x + w, y + h
    forbidden = []
    for ob in other_bboxes:
        ox, oy, ow, oh = [float(v) for v in ob]
        if (ox + ow) > x and ox < x2:
            forbidden.append((oy, oy + oh))
    if not forbidden:
        return bbox
    min_fy = min(fy  for fy, _  in forbidden)
    max_fy = max(fy2 for _, fy2 in forbidden)
    top_h  = min_fy - y
    bot_h  = y2 - max_fy
    if top_h >= bot_h and top_h >= min_side:
        return [x, y, w, top_h]
    elif bot_h >= min_side:
        return [x, max_fy, w, bot_h]
    elif top_h >= min_side:
        return [x, y, w, top_h]
    return None


# --- PIPELINE PRINCIPAL ---

def create_cropped_dataset():
    print(f"Lendo anotações de: {ANNOT_FILE}")
    with open(ANNOT_FILE) as f:
        data = json.load(f)

    cat_name = {c['id']: c['name'] for c in data['categories']}
    img_info  = {img['id']: img['file_name'] for img in data['images']}

    # 1. Coletar anotações viáveis
    samples = {}
    for ann in data['annotations']:
        cid  = ann['category_id']
        if cid not in FOCUS_CLASSES:
            continue
        bbox = ann['bbox']
        if bbox[2] * bbox[3] < MIN_BBOX_AREA:
            continue
        aid = ann['id']
        if aid not in samples:
            samples[aid] = {
                'abs_path': os.path.join(SOURCE_DIR, img_info[ann['image_id']]),
                'bbox':     bbox,
                'cat_id':   cid,
                'cat_name': cat_name[cid],
                'ann_id':   aid,
            }

    df = pd.DataFrame(list(samples.values()))
    print(f"Anotações viáveis (>= {MIN_BBOX_AREA}px²): {len(df)}")

    # 2. Deduplicação IoU para LAND/SKY
    before = len(df)
    df = _deduplicate(df)
    print(f"Após deduplicação IoU: {len(df)} ({before - len(df)} removidas)")

    # 3. Cap global por classe
    frames = []
    for cid, group in df.groupby('cat_id'):
        cap = MAX_SAMPLES_PER_CLASS.get(cid)
        if cap and len(group) > cap:
            group = group.sample(n=cap, random_state=42)
            print(f"  [cap] {cat_name[cid]}: {len(group)} amostras (limitado a {cap})")
        frames.append(group)
    df = pd.concat(frames).reset_index(drop=True)

    print("\nDistribuição por classe:")
    for cid, cnt in df['cat_id'].value_counts().sort_index().items():
        print(f"  {cat_name[cid]:10s} (id {cid}): {cnt}")

    # 4. Split por IMAGEM (evita data leakage)
    unique_imgs = df[['abs_path']].drop_duplicates()
    img_cls     = df.groupby('abs_path')['cat_id'].first().reset_index()
    unique_imgs = unique_imgs.merge(img_cls, on='abs_path')

    def safe_split(data, test_size, col):
        counts = data[col].value_counts()
        if (counts < 2).any():
            return train_test_split(data, test_size=test_size, random_state=42)
        return train_test_split(data, test_size=test_size, stratify=data[col], random_state=42)

    train_imgs, temp  = safe_split(unique_imgs, 0.30, 'cat_id')
    val_imgs,   test_imgs = safe_split(temp, 0.50, 'cat_id')

    split_map = {
        'train': df[df['abs_path'].isin(train_imgs['abs_path'])],
        'valid': df[df['abs_path'].isin(val_imgs['abs_path'])],
        'test':  df[df['abs_path'].isin(test_imgs['abs_path'])],
    }

    # Índice de bboxes de outras classes por imagem (para zona pura)
    img_other = {}
    for _, row in df.iterrows():
        key = row['abs_path']
        img_other.setdefault(key, {})
        img_other[key].setdefault(row['cat_id'], []).append(row['bbox'])

    # 5. Salvar crops
    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name, df_split in split_map.items():
        split_dir = os.path.join(OUT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        csv_data, skipped = [], 0

        print(f"\nProcessando {split_name}: {len(df_split)} objetos de "
              f"{df_split['abs_path'].nunique()} imagens...")

        for _, row in df_split.iterrows():
            try:
                cid      = row['cat_id']
                img_path = row['abs_path']
                bbox     = row['bbox']

                if cid in BACKGROUND_CLASSES:
                    others = [b for c, bboxes in img_other.get(img_path, {}).items()
                              if c != cid for b in bboxes]
                    bbox = _safe_zone(bbox, others)
                    if bbox is None:
                        skipped += 1
                        continue

                img = Image.open(img_path).convert('RGB')
                iw, ih = img.size
                x, y, w, h = [int(v) for v in bbox]

                # Posição vertical normalizada no original (feature REP)
                orig_cy_norm = round((row['bbox'][1] + row['bbox'][3] / 2) / ih, 4)

                # Crop quadrado — estratégia por classe:
                # SHIP (5): side = min(w, 2h) — captura silhueta horizontal
                # Demais: side = min(w, h) — inscrito, 100% da classe
                if cid == 5:
                    side = min(w, max(h, h * 2))
                else:
                    side = min(w, h)
                cx = x + w // 2
                cy = y + h // 2
                x1 = max(0, cx - side // 2)
                y1 = max(0, cy - side // 2)
                x2 = min(iw, x1 + side)
                y2 = min(ih, y1 + side)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                crop = img.crop((x1, y1, x2, y2)).resize((64, 64), Image.LANCZOS)
                fname = f"{row['cat_name']}_{row['ann_id']}.jpg"
                crop.save(os.path.join(split_dir, fname))
                csv_data.append({'filename': fname, 'class_id': cid,
                                 'class_name': row['cat_name'], 'cy_norm': orig_cy_norm})
            except Exception as e:
                print(f"  Erro: {e}")

        if skipped:
            print(f"  [!] {skipped} crops LAND/SKY descartados (zona pura insuficiente)")

        df_out = pd.DataFrame(csv_data)
        df_out.to_csv(os.path.join(split_dir, 'labels.csv'), index=False)

        print(f"  Salvos: {len(df_out)} crops")
        for cid, cnt in df_out['class_id'].value_counts().sort_index().items():
            print(f"    {cat_name.get(cid, cid):10s}: {cnt}")


if __name__ == '__main__':
    create_cropped_dataset()
    print("\nDataset fullHD633 gerado com sucesso!")
