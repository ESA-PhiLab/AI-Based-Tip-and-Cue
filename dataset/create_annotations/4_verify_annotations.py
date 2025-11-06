import json
from pathlib import Path
from PIL import Image
from collections import Counter


# === CONFIG ===
annotations_path = Path("merged_annotations.json")  # path to your annotation file
base_dir = Path("../whales_from_space")          # base dir containing <LocationYear> folders
split = "verification"


# === LOAD ANNOTATIONS ===
with open(annotations_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

# Collect all image file paths listed in JSON
image_filenames_in_json = [img["file_name"] for img in coco_data["images"] if "file_name" in img]

# Collect all actual image files on disk (recursively)
valid_exts = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}
image_filenames_on_disk = set(p.relative_to(base_dir).as_posix() for p in base_dir.rglob("*") if p.suffix in valid_exts)

# === CHECK FOR MISSING/EXTRA FILES ===
missing_files = [f for f in image_filenames_in_json if f not in image_filenames_on_disk]
extra_files = [f for f in image_filenames_on_disk if f not in image_filenames_in_json]

# === REPORT ===
print(f"\n--- Dataset verification for split: {split} ---")
print(f"Annotations file: {annotations_path}")
print(f"Base directory:   {base_dir}")
print(f"Total images in JSON: {len(image_filenames_in_json)}")
print(f"Total images on disk: {len(image_filenames_on_disk)}")

print(f"\nMissing image files: {len(missing_files)}")
if missing_files:
    print("Missing examples:")
    for f in missing_files[:20]:
        print(f"  {f}")
    if len(missing_files) > 20:
        print(f"  ... and {len(missing_files) - 20} more")

print(f"\nExtra image files not in JSON: {len(extra_files)}")
if extra_files:
    print("Extra examples:")
    for f in list(extra_files)[:20]:
        print(f"  {f}")
    if len(extra_files) > 20:
        print(f"  ... and {len(extra_files) - 20} more")


# === IMAGE SIZE CHECK ===
min_size = None
max_size = None
min_image = max_image = None
size_counter = Counter()

existing_files = [f for f in image_filenames_in_json if f in image_filenames_on_disk]

print("\nScanning image sizes...")
for rel_path in existing_files:
    image_path = base_dir / rel_path
    try:
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            size_counter[size] += 1
            if min_size is None or size[0]*size[1] < min_size[0]*min_size[1]:
                min_size = size
                min_image = rel_path
            if max_size is None or size[0]*size[1] > max_size[0]*max_size[1]:
                max_size = size
                max_image = rel_path
    except Exception as e:
        print(f"Failed to read image {rel_path}: {e}")

most_common_size, count = size_counter.most_common(1)[0] if size_counter else ((0, 0), 0)

print(f"\nSmallest image: {min_image} with size {min_size}")
print(f"Largest image:  {max_image} with size {max_size}")
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

print("\nVerification complete.")
