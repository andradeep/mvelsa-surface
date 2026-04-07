import json

base_dir = "../../../data/coco640"
splits = ["train", "valid", "test"]
counts = {s: {} for s in splits}
with open(f"{base_dir}/train/_annotations.coco.json") as f:
    categories = {c["id"]: c["name"] for c in json.load(f)["categories"]}

for split in splits:
    try:
        with open(f"{base_dir}/{split}/_annotations.coco.json") as f:
            data = json.load(f)
            for ann in data["annotations"]:
                cat_id = ann["category_id"]
                counts[split][cat_id] = counts[split].get(cat_id, 0) + 1
    except: pass

print("--- TODAS AS BBOXES ---")
for cat_id, name in sorted(categories.items()):
    t, v, te = counts['train'].get(cat_id, 0), counts['valid'].get(cat_id, 0), counts['test'].get(cat_id, 0)
    print(f"[{name}]: Train={t}, Valid={v}, Test={te}")
