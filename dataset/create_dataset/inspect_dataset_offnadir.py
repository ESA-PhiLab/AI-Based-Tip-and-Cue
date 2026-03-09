import json
import os
import re
from collections import defaultdict
from pathlib import Path

# =========================
# Config
# =========================

# Assuming this script is located in the 'dataset/create_dataset' folder
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

mode = "reflection_offnadir_glint_255"
DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "create_dataset" / "0_merged" / mode
ANNOTATIONS_PATH = BASE_DIR / "final_annotations_repaired.json"

# Dictionary to store counts of images per off-nadir angle
ocean_images_count = defaultdict(int)
whale_images_count = defaultdict(int)
combined_images_count = defaultdict(int)

# Regular expressions to match filenames for ocean and whale images
ocean_pattern = re.compile(r'_O_(\d+deg)')
whale_pattern = re.compile(r'_F_(\d+deg)')

# =========================
# Load COCO Annotations
# =========================
def load_json(path: Path) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Count Ocean and Whale Images
# =========================
def count_images_in_subfolders(main_dir: Path):
    """Count images in subfolders based on off-nadir angle for ocean and whale."""
    ocean_images_count = defaultdict(int)
    whale_images_count = defaultdict(int)
    combined_images_count = defaultdict(int)

    # Load COCO annotations (just to confirm structure, not used for grid)
    coco = load_json(ANNOTATIONS_PATH)
    anns = coco.get("annotations", [])

    for subfolder in os.listdir(main_dir):
        subfolder_path = os.path.join(main_dir, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it is a directory
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                if os.path.isfile(file_path):
                    # Check if the file is an ocean image
                    ocean_match = ocean_pattern.search(filename)
                    if ocean_match:
                        off_nadir_angle = ocean_match.group(1)
                        ocean_images_count[off_nadir_angle] += 1

                    # Check if the file is a whale image
                    whale_match = whale_pattern.search(filename)
                    if whale_match:
                        off_nadir_angle = whale_match.group(1)
                        whale_images_count[off_nadir_angle] += 1

    # Combine counts of ocean and whale images for each off-nadir angle
    for angle in set(ocean_images_count.keys()).union(set(whale_images_count.keys())):
        combined_images_count[angle] = ocean_images_count[angle] + whale_images_count[angle]

    return ocean_images_count, whale_images_count, combined_images_count

# =========================
# Print the Results
# =========================
def print_counts(ocean_images_count, whale_images_count, combined_images_count):
    """Print the counts for ocean, whale, and combined images per off-nadir angle."""
    print("\nOcean Image Counts per Off-Nadir Angle:")
    for angle, count in ocean_images_count.items():
        print(f"Angle {angle}: {count} ocean images")

    print("\nWhale Image Counts per Off-Nadir Angle:")
    for angle, count in whale_images_count.items():
        print(f"Angle {angle}: {count} whale images")

    print("\nCombined Image Counts per Off-Nadir Angle:")
    for angle, count in combined_images_count.items():
        print(f"Angle {angle}: {count} combined images")


# =========================
# Main Function
# =========================
def main():
    ocean_images_count, whale_images_count, combined_images_count = count_images_in_subfolders(BASE_DIR)
    print_counts(ocean_images_count, whale_images_count, combined_images_count)


if __name__ == "__main__":
    main()