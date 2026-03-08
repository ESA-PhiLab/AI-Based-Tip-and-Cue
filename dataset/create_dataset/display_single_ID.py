import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# =========================
# Config
# =========================


ID_INPUT = 1000       # ID of input image

# Get the current script's directory
main_path = Path(__file__).resolve().parents[2]  # Adjust according to your file structure
os.chdir(main_path)

# Dataset and annotations
DATASET_PATH = Path("dataset")
mode = "reflection_offnadir_glint_255"  # Example mode

BASE_DIR = DATASET_PATH / "create_dataset" / "0_merged" / mode
ANNOTATIONS_PATH = BASE_DIR / "final_annotations_repaired.json"  # Path to the annotation file

DISPLAY_ZOOM = 6  # VISUAL zoom only; annotations are still drawn in original patch pixels
WINDOW_TITLE = "Annotation Preview"


# =========================
# Helpers
# =========================
def load_json(path: str) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def anns_by_image(anns: list) -> dict:
    """anns_by_image(anns) -> {image_id: [anns...] }."""
    d = {}
    for a in anns:
        d.setdefault(a["image_id"], []).append(a)
    return d


def draw_overlay(img: Image.Image, anns: list, scale: float = 1.0) -> Image.Image:
    """draw_overlay(img,anns,scale) -> Image: draw polygons and bboxes."""
    draw = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg: continue
            pts = [(seg[i] * scale, seg[i + 1] * scale) for i in range(0, len(seg), 2)]
            if len(pts) >= 3: draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            x, y, w, h = x * scale, y * scale, w * scale, h * scale
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    return img


# =========================
# Display Image with Annotations using Matplotlib
# =========================
def show_image_with_annotations(image_path: Path, anns: list) -> None:
    """show_image_with_annotations(image_path, anns) -> None: Display image with annotations using matplotlib."""
    try:
        base = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[Error reading {image_path}]: {e}")
        base = Image.new("RGB", (640, 360), color=(20, 20, 20))
        ImageDraw.Draw(base).text((10, 10), f"Read error: {image_path}", fill=(255, 80, 80))

    # Draw annotations (segmentations and bounding boxes)
    base = draw_overlay(base, anns)

    # Convert image for matplotlib
    img = base.convert("RGB")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Correct image orientation for matplotlib

    # Plot image using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.title(WINDOW_TITLE)
    plt.show()


# =========================
# Main
# =========================
def show_data() -> None:
    """show_data() -> None: Load image and annotations, and run viewer."""
    # Load annotations
    try:
        coco = load_json(ANNOTATIONS_PATH)
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return

    anns = coco.get("annotations", [])

    # Assuming we want to display the first image in the "images" list
    if len(coco["images"]) == 0:
        print("No images found in the dataset.")
        return

    image = coco["images"][ID_INPUT]  # Get the first image
    image_path = BASE_DIR / image["file_name"]

    # Check if the image file exists
    if not image_path.is_file():
        print(f"Image file not found: {image_path}")
        return

    # Gather all annotations for the selected image (no dependency on IDs)
    annotations_for_image = []
    for ann in anns:
        if ann.get("image_id") == image["id"]:
            annotations_for_image.append(ann)

    # Display image with annotations using matplotlib
    show_image_with_annotations(image_path, annotations_for_image)


if __name__ == "__main__":
    show_data()