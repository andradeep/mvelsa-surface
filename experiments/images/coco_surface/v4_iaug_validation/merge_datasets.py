"""
MVELSA V4 — Merge fullHD633 train + IAUG train
===============================================
Combina os crops reais (fullHD633) com os crops augmentados (IAUG)
em um único diretório de treino.

Saída: ../../../../data/coco_cropped_combined/train/
Val:   permanece em ../../../../data/coco_cropped/valid/  (fullHD633 puro)
"""
import os
import shutil
import pandas as pd

FHD_TRAIN  = os.path.join(os.environ.get("FHD_DATA_PATH",      "../../../../data/coco_cropped"),          "train")
IAUG_TRAIN = os.path.join(os.environ.get("IAUG_OUT_DIR",       "../../../../data/coco_cropped_iaug"),       "train")
OUT_TRAIN  = os.path.join(os.environ.get("COMBINED_DATA_PATH", "../../../../data/coco_cropped_combined"),   "train")

os.makedirs(OUT_TRAIN, exist_ok=True)

dfs = []
total_copied = 0

for src_dir, prefix in [(FHD_TRAIN, "fhd_"), (IAUG_TRAIN, "iaug_")]:
    csv_path = os.path.join(src_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"[AVISO] {csv_path} não encontrado — pulando.")
        continue

    df = pd.read_csv(csv_path)
    df['filename'] = prefix + df['filename']

    for orig_name, new_name in zip(pd.read_csv(csv_path)['filename'], df['filename']):
        src = os.path.join(src_dir, orig_name)
        dst = os.path.join(OUT_TRAIN, new_name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        total_copied += 1

    dfs.append(df)
    print(f"  {prefix}: {len(df)} crops de {src_dir}")

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(os.path.join(OUT_TRAIN, "labels.csv"), index=False)

print(f"\n✅ Dataset combinado: {len(merged)} crops em {OUT_TRAIN}")
print("Distribuição final:")
base_classes = {1: 'BOAT', 3: 'BUOY', 4: 'LAND', 5: 'SHIP', 6: 'SKY'}
for cid, cnt in merged['class_id'].value_counts().sort_index().items():
    name = base_classes.get(cid, str(cid))
    print(f"  {name:8s} (id {cid}): {cnt}")
print("\nPróximo passo: python train_mvelsa_iaug.py")
