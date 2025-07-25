import json
from pathlib import Path

# === CONFIGURATION ===
split = "val"  # Change to "val" or "test" if needed
dataset_root = Path("C:/Users/Nadine/ESA/Phi-lab Onboard AI - Onboard tip and cue - Onboard tip and cue/3_Software/1_Datasets/whales/whales_from_space_matched/0_dataset")
annotations_path = dataset_root / "annotations" / f"instances_{split}.json"
output_path = dataset_root / "annotations" / f"instances_{split}_mapped.json"

# === LOAD ANNOTATIONS ===
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# === FIX CATEGORIES: Ensure a single whale category with id 0 ===
valid_category = {"id": 0, "name": "whale", "supercategory": "whale"}
coco_data["categories"] = [valid_category]

# === REMAP ANNOTATION CATEGORY IDs TO 0 ===
for ann in coco_data["annotations"]:
    ann["category_id"] = 0

# === SAVE UPDATED ANNOTATIONS ===
with open(output_path, "w") as f:
    json.dump(coco_data, f)

print(f"\n✔ Saved fixed annotations to: {output_path}")
