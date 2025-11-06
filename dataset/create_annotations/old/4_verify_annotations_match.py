import json
from pathlib import Path
from PIL import Image
from collections import Counter

# === CONFIGURATION ===
split = "train"  # or "train", "test"
dataset_root = Path("C:/Users/Nadine/ESA/Phi-lab Onboard AI - Onboard tip and cue - Onboard tip and cue/3_Software/1_Datasets/whales/whales_from_space_matched/0_dataset")
images_dir = dataset_root / "images" / split
annotations_path = dataset_root / "annotations" / f"instances_{split}.json"



# === LOAD ANNOTATIONS ===
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

image_filenames_in_json = [img["file_name"] for img in coco_data["images"]]
image_filenames_on_disk = set(p.name for p in images_dir.glob("*"))

# === CHECK FOR MISSING/EXTRA FILES ===
missing_files = [f for f in image_filenames_in_json if f not in image_filenames_on_disk]
extra_files = [f for f in image_filenames_on_disk if f not in image_filenames_in_json]

# === REPORT ===
print(f"\n--- Dataset verification for split: {split} ---")
print(f"Total images in JSON: {len(image_filenames_in_json)}")
print(f"Total images on disk: {len(image_filenames_on_disk)}")
print(f"Missing image files: {len(missing_files)}")
if missing_files:
    print("\nMissing files:")
    for f in missing_files:
        print(f"  {f}")

print(f"\nExtra image files not in JSON: {len(extra_files)}")
if extra_files:
    print("\nExtra files:")
    for f in extra_files:
        print(f"  {f}")

# === IMAGE SIZE CHECK ===
min_size = None
max_size = None
min_image = max_image = None
size_counter = Counter()

print("\nScanning image sizes...")
for image_path in images_dir.glob("*"):
    try:
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            size_counter[size] += 1
            if min_size is None or size[0]*size[1] < min_size[0]*min_size[1]:
                min_size = size
                min_image = image_path.name
            if max_size is None or size[0]*size[1] > max_size[0]*max_size[1]:
                max_size = size
                max_image = image_path.name
    except Exception as e:
        print(f"Failed to read image {image_path.name}: {e}")

most_common_size, count = size_counter.most_common(1)[0] if size_counter else ((0, 0), 0)

print(f"\nSmallest image: {min_image} with size {min_size}")
print(f"Largest image: {max_image} with size {max_size}")
print(f"Most common image size: {most_common_size} occurred {count} times")

# === CATEGORY ID CHECK ===
category_ids = [ann["category_id"] for ann in coco_data["annotations"]]
if 0 in category_ids:
    print("\nDetected category_id=0 in annotations. This may cause index errors during training.")
else:
    print("\nAll category_ids are valid (no zero IDs found).")

# === COUNT NUMBER OF CATEGORIES ===
category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
unique_categories = set(category_ids)

print(f"\nTotal unique categories: {len(unique_categories)}")
print("Categories found:")
for cid in sorted(unique_categories):
    name = category_id_to_name.get(cid, "Unknown")
    print(f"  ID {cid}: {name}")
