import json
import os

base_dir = "../../../data/coco640"
splits = ["train", "valid", "test"]

counts = {s: {} for s in splits}
categories = {}

for split in splits:
    json_path = os.path.join(base_dir, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for cat in data["categories"]:
        categories[cat["id"]] = cat["name"]
        
    img_ann = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_ann: img_ann[img_id] = []
        img_ann[img_id].append(ann)
        
    for img in data["images"]:
        img_id = img["id"]
        anns = img_ann.get(img_id, [])
        if not anns: continue
        best_ann = max(anns, key=lambda a: a.get("area", 0))
        cat_id = best_ann["category_id"]
        counts[split][cat_id] = counts[split].get(cat_id, 0) + 1

artifact_path = "/home/andradearthurb/.gemini/antigravity/brain/4de79fbe-b7da-4907-8e39-ffdaae9a40d3/dataset_composition.md"
with open(artifact_path, "w") as f:
    f.write("# Composição do Dataset COCO640\n\n")
    f.write("Abaixo está a distribuição de imagens (considerando a classe dominante ou de maior área em cada imagem) para cada divisão do dataset.\n\n")
    f.write("| ID | Classe | Train | Valid | Test |\n")
    f.write("|---|---|---|---|---|\n")
    for cat_id, cat_name in sorted(categories.items()):
        tr = counts["train"].get(cat_id, 0)
        va = counts["valid"].get(cat_id, 0)
        te = counts["test"].get(cat_id, 0)
        f.write(f"| {cat_id} | {cat_name} | {tr} | {va} | {te} |\n")

print(f"Generated artifact at {artifact_path}")
