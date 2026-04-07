"""
Gerador de dataset recortado a partir do IAUG (seadev_2_IAUG).
==============================================================
Usa apenas o split de TREINO do IAUG.
A validação permanece com o split fullHD633 existente (coco_cropped/valid/).

Remapeamento de IDs (IAUG → fullHD633):
  BOAT(1)     → 1   (sem mudança)
  BUOY(3)     → 3   (sem mudança)
  LAND(4)     → 4   (sem mudança)
  SHIP(6)     → 5   (remapeado)
  SKY(7)      → 6   (remapeado)
  BUILDING(2) → removida
  PLATE(5)    → removida
  WATER(8)    → removida

Saída: ../../../../data/coco_cropped_iaug/train/
"""
import os
import json
import pandas as pd
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÃO ---
# Suporta override via variável de ambiente (útil no Google Colab)
SOURCE_DIR  = os.environ.get(
    "IAUG_SOURCE_DIR",
    "/home/andradearthurb/Documentos/DOC/reaugment/seadev_2_IAUG.v1-com-iaug.coco/train"
)
ANNOT_FILE  = os.path.join(SOURCE_DIR, "_annotations.coco.json")
OUT_DIR     = os.environ.get("IAUG_OUT_DIR", "../../../../data/coco_cropped_iaug")

# IDs no dataset IAUG original
FOCUS_CLASSES      = {1, 3, 4, 6, 7}   # BOAT, BUOY, LAND, SHIP, SKY (IAUG IDs)
BACKGROUND_CLASSES = {4, 7}            # LAND, SKY
ID_REMAP           = {6: 5, 7: 6}     # SHIP 6→5, SKY 7→6 (alinha com fullHD633)
NAME_REMAP         = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 6: 'SHIP', 7: 'SKY'}

MIN_BBOX_AREA          = 900             # 30×30px mínimo (imagens 1200×1200)
IOU_THRESHOLD          = 0.50
MAX_CROPS_PER_IMG_CLS  = 4              # deduplicação LAND/SKY por imagem

# Cap por classe (usando IDs IAUG originais)
# Usamos todo o dado disponível; apenas BUOY recebe cap para evitar razão >10:1 com SHIP.
# MLP e RF já usam class_weight='balanced' — imbalance moderado é tolerado.
MAX_SAMPLES_PER_CLASS = {
    3: 3000,  # BUOY (20.264 → 3.000)
}

# LAND e SHIP não recebem IAUG — augmentação degrada os especialistas dessas classes.
# LAND: specialist confunde com BOAT quando treinado em versões augmentadas.
# SHIP: era 100% recall com dados reais puros — augmentação introduz ruído.
# Apenas BOAT, BUOY e SKY se beneficiam da variação de tempo/iluminação.
FOCUS_CLASSES = FOCUS_CLASSES - {4, 6}  # remove LAND(4) e SHIP(6) do IAUG


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
            if len(accepted) >= MAX_CROPS_PER_IMG_CLS:
                break
        keep.extend(accepted)
    return df.loc[keep].drop(columns=['_area'], errors='ignore')


def _safe_zone(bbox, other_bboxes, min_side=30):
    """Sub-região do bbox sem sobreposição com outras classes (eixo Y)."""
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

def create_cropped_dataset_iaug():
    print(f"Lendo anotações IAUG: {ANNOT_FILE}")
    with open(ANNOT_FILE) as f:
        data = json.load(f)

    cat_name = {c['id']: c['name'] for c in data['categories']}
    img_info  = {img['id']: img['file_name'] for img in data['images']}

    print(f"Total de imagens no IAUG train: {len(img_info)}")

    # 1. Coletar anotações viáveis
    samples = {}
    for ann in data['annotations']:
        cid = ann['category_id']
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
                'cat_name': NAME_REMAP.get(cid, cat_name[cid]),
                'ann_id':   aid,
            }

    df = pd.DataFrame(list(samples.values()))
    print(f"\nAnotações viáveis (>= {MIN_BBOX_AREA}px²): {len(df)}")
    print("Distribuição por classe (IDs IAUG originais):")
    for cid, cnt in df['cat_id'].value_counts().sort_index().items():
        print(f"  {NAME_REMAP.get(cid, cat_name.get(cid, cid)):10s} (IAUG id {cid}): {cnt}")

    # 2. Deduplicação IoU para LAND/SKY
    before = len(df)
    df = _deduplicate(df)
    print(f"\nApós deduplicação IoU: {len(df)} ({before - len(df)} removidas)")

    # 3. Cap por classe
    frames = []
    for cid, group in df.groupby('cat_id'):
        cap = MAX_SAMPLES_PER_CLASS.get(cid)
        if cap and len(group) > cap:
            group = group.sample(n=cap, random_state=42)
            print(f"  [cap] {NAME_REMAP.get(cid, cid)}: {len(group)} amostras (cap={cap})")
        frames.append(group)
    df = pd.concat(frames).reset_index(drop=True)

    print("\nDistribuição final (após cap):")
    for cid, cnt in df['cat_id'].value_counts().sort_index().items():
        print(f"  {NAME_REMAP.get(cid, cid):10s} (IAUG id {cid} → out id {ID_REMAP.get(cid, cid)}): {cnt}")

    # Índice de bboxes de outras classes por imagem (para zona pura em LAND/SKY)
    img_other = {}
    for _, row in df.iterrows():
        key = row['abs_path']
        img_other.setdefault(key, {})
        img_other[key].setdefault(row['cat_id'], []).append(row['bbox'])

    # 4. Salvar crops em OUT_DIR/train/
    split_dir = os.path.join(OUT_DIR, "train")
    os.makedirs(split_dir, exist_ok=True)
    csv_data, skipped = [], 0

    print(f"\nProcessando {len(df)} objetos...")

    for _, row in df.iterrows():
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

            # cy_norm calculado sobre o bbox original (não truncado)
            orig_cy_norm = round((row['bbox'][1] + row['bbox'][3] / 2) / ih, 4)

            # Estratégia de crop por classe:
            # SHIP (IAUG id=6): side = min(w, 2h) — captura silhueta horizontal
            # LAND/SKY/BOAT/BUOY: side = min(w, h) — inscrito
            if cid == 6:  # SHIP (IAUG ID)
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

            # Remapear IDs para alinhar com fullHD633
            out_cid  = ID_REMAP.get(cid, cid)
            out_name = NAME_REMAP.get(cid, row['cat_name'])
            fname    = f"{out_name}_{row['ann_id']}.jpg"
            crop.save(os.path.join(split_dir, fname))
            csv_data.append({
                'filename':   fname,
                'class_id':   out_cid,
                'class_name': out_name,
                'cy_norm':    orig_cy_norm
            })
        except Exception as e:
            print(f"  Erro: {e}")

    if skipped:
        print(f"  [!] {skipped} crops LAND/SKY descartados (zona pura insuficiente)")

    df_out = pd.DataFrame(csv_data)
    df_out.to_csv(os.path.join(split_dir, 'labels.csv'), index=False)

    print(f"\n✅ Crops salvos em: {split_dir}")
    print(f"   Total: {len(df_out)} crops")
    print("   Distribuição final (IDs remapeados):")
    for cid, cnt in df_out['class_id'].value_counts().sort_index().items():
        out_name = df_out[df_out['class_id'] == cid]['class_name'].iloc[0]
        print(f"    {out_name:10s} (id {cid}): {cnt}")
    print("\nPróximo passo: python train_mvelsa_iaug.py")


if __name__ == '__main__':
    create_cropped_dataset_iaug()
