import json

def get_counts(json_path):
    with open(json_path) as f:
        data = json.load(f)
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    img_ann = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_ann: img_ann[img_id] = []
        img_ann[img_id].append(ann)
    
    counts = {}
    for img in data["images"]:
        img_id = img["id"]
        anns = img_ann.get(img_id, [])
        if not anns: continue
        best_ann = max(anns, key=lambda a: a.get("area", 0))
        cat_id = best_ann["category_id"]
        counts[cat_id] = counts.get(cat_id, 0) + 1
        
    return counts, categories

train_path = "../../../data/coco640/train/_annotations.coco.json"
val_path = "../../../data/coco640/valid/_annotations.coco.json"

train_counts, categories = get_counts(train_path)
val_counts, _ = get_counts(val_path)

print("--- TREINO ---")
for k, v in train_counts.items(): print(f"{categories.get(k, k)} (ID {k}): {v}")

print("\n--- VALIDACAO ---")
for k, v in val_counts.items(): print(f"{categories.get(k, k)} (ID {k}): {v}")
