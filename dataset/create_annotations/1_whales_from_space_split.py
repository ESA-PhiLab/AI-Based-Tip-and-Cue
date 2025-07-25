import os
import shutil
import json
import pandas as pd
from pathlib import Path

os.chdir("Whales from Satellite")

extension = 'train'         # test, train, valid

# === CONFIG ===
BASE_DIR = os.getcwd()
TRAIN_FOLDER = os.path.join(BASE_DIR, extension)
EXCEL_PATH = os.path.join(BASE_DIR, "WhaleFromSpaceDB_Whales.csv")

OUT_WHALES = os.path.join(BASE_DIR, "0_whales_from_space_" + extension)
OUT_NOT_WHALES = os.path.join(BASE_DIR, "0_not_whales_from_space_" + extension)
OUT_WHALES_JSON = os.path.join(OUT_WHALES, "annotations.json")
OUT_NOT_WHALES_JSON = os.path.join(OUT_NOT_WHALES, "annotations.json")

# === PREP OUTPUT ===
os.makedirs(OUT_WHALES, exist_ok=True)
os.makedirs(OUT_NOT_WHALES, exist_ok=True)

# === LOAD CSV ===
df = pd.read_csv(EXCEL_PATH)
csv_names = set(df['BoxID/ImageChip'].astype(str))
print(f"Loaded {len(csv_names)} BoxID/ImageChip names from CSV")

# === LOAD COCO ===
COCO_PATH = os.path.join(TRAIN_FOLDER, "_annotations.coco.json")
with open(COCO_PATH, "r") as f:
    coco = json.load(f)

# === Prepare new COCO dicts ===
whales_coco = {k: [] if isinstance(v, list) else v for k, v in coco.items()}
not_whales_coco = {k: [] if isinstance(v, list) else v for k, v in coco.items()}

# Keep categories & licenses
whales_coco["categories"] = coco["categories"]
whales_coco["licenses"] = coco["licenses"]
not_whales_coco["categories"] = coco["categories"]
not_whales_coco["licenses"] = coco["licenses"]

# Build image ID to annotations
image_id_to_annots = {}
for ann in coco["annotations"]:
    image_id_to_annots.setdefault(ann["image_id"], []).append(ann)

# === Process all COCO images ===
for img in coco["images"]:
    file_name = img["file_name"]
    src_path = os.path.join(TRAIN_FOLDER, file_name)

    # If any BoxID/ImageChip is a substring in the file name -> whales_from_space
    belongs_to_whales = any(name in file_name for name in csv_names)

    if belongs_to_whales:
        dst = os.path.join(OUT_WHALES, file_name)
        shutil.copy2(src_path, dst)
        whales_coco["images"].append(img)
        whales_coco["annotations"].extend(image_id_to_annots.get(img["id"], []))
    else:
        dst = os.path.join(OUT_NOT_WHALES, file_name)
        shutil.copy2(src_path, dst)
        not_whales_coco["images"].append(img)
        not_whales_coco["annotations"].extend(image_id_to_annots.get(img["id"], []))

print(f"Copied {len(whales_coco['images'])} whale images")
print(f"Copied {len(not_whales_coco['images'])} non-whale images")

# === Save new COCO JSONs ===
with open(OUT_WHALES_JSON, "w") as f:
    json.dump(whales_coco, f)

with open(OUT_NOT_WHALES_JSON, "w") as f:
    json.dump(not_whales_coco, f)

print(f"New COCO saved:\n - {OUT_WHALES_JSON}\n - {OUT_NOT_WHALES_JSON}")
