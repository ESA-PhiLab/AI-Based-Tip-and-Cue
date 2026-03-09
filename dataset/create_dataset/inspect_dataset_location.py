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

# Dictionary to store counts of ocean and whale images per location (subfolder)
location_ocean_count = defaultdict(int)
location_whale_count = defaultdict(int)

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
# Count Ocean and Whale Images per Location
# =========================
def count_images_in_subfolders(main_dir: Path):
    """Count images in subfolders (locations) for ocean and whale."""
    location_ocean_count = defaultdict(int)
    location_whale_count = defaultdict(int)

    # Load COCO annotations
    coco = load_json(ANNOTATIONS_PATH)
    anns = coco.get("annotations", [])

    for subfolder in os.listdir(main_dir):
        subfolder_path = os.path.join(main_dir, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it is a directory (location)
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                if os.path.isfile(file_path):
                    # Check if the file is an ocean image
                    ocean_match = ocean_pattern.search(filename)
                    if ocean_match:
                        location_ocean_count[subfolder] += 1

                    # Check if the file is a whale image
                    whale_match = whale_pattern.search(filename)
                    if whale_match:
                        location_whale_count[subfolder] += 1

    return location_ocean_count, location_whale_count

# =========================
# Print the Results
# =========================
def print_counts(location_ocean_count, location_whale_count):
    """Print the counts for ocean and whale images per location (subfolder)."""
    print("\nOcean Image Counts per Location:")
    for location, count in location_ocean_count.items():
        print(f"Location {location}: {count} ocean images")

    print("\nWhale Image Counts per Location:")
    for location, count in location_whale_count.items():
        print(f"Location {location}: {count} whale images")


# =========================
# Main Function
# =========================
def main():
    location_ocean_count, location_whale_count = count_images_in_subfolders(BASE_DIR)
    print_counts(location_ocean_count, location_whale_count)


if __name__ == "__main__":
    main()