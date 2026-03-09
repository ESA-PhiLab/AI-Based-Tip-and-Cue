import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw

# =========================
# Config
# =========================

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

ALLOWED_modes = {
    "patch_raw_255",
    "patch_raw_rot_255",

    "texture_nadir_255",
    "texture_offnadir_255",

    "radiance_nadir_glint_255",
    "radiance_nadir_glint_npy",
    "radiance_nadir_no_glint_255",
    "radiance_nadir_no_glint_npy",

    "radiance_offnadir_glint_255",
    "radiance_offnadir_glint_npy",
    "radiance_offnadir_no_glint_255",
    "radiance_offnadir_no_glint_npy",

    "reflection_nadir_glint_255",
    "reflection_nadir_glint_npy",
    "reflection_nadir_no_glint_255",
    "reflection_nadir_no_glint_npy",

    "reflection_offnadir_glint_255",
    "reflection_offnadir_glint_npy",
    "reflection_offnadir_no_glint_255",
    "reflection_offnadir_no_glint_npy",
}

mode = "reflection_offnadir_glint_255"
mode = "texture_offnadir_255"

DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "create_dataset" / "0_merged" / mode
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "0_merged" / mode / "final_annotations_repaired.json"

# Output directory to save images
OUTPUT_DIR = Path("output_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

GRID_ROWS = 15
GRID_COLS = 10
IMAGE_SIZE = (64, 64)  # Resize images for display


# =========================
# Helpers
# =========================
def load_json(path: str | Path) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_images_from_folder(folder: Path, segmentation: bool = False) -> list:
    """Load images from the folder and its subdirectories, returning them as a list."""
    images = []
    coco = load_json(ANNOTATIONS_PATH)
    anns = coco.get("annotations", [])

    for root, x, files in os.walk(folder):
        path = Path(folder)

        # Get the last folder
        last_folder = path.name

        for filename in files:


            img_path = Path(root) / filename
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:

                if not segmentation:
                    try:
                        img = Image.open(img_path)
                        img = img.resize(IMAGE_SIZE)  # Resize for display

                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")

                else:
                    FILENAMEFULL = last_folder + "/" + filename
                    # Find the image in the dataset that matches the filename
                    image = None
                    for img in coco["images"]:
                        if img["file_name"] == FILENAMEFULL:
                            image = img

                    if not image:
                        print(f"Image with filename {FILENAMEFULL} not found in dataset.")

                    image_path = BASE_DIR / image["file_name"]

                    # Check if the image file exists
                    if not image_path.is_file():
                        print(f"Image file not found: {image_path}")

                    # Gather all annotations for the selected image (matching by filename)
                    annotations_for_image = []
                    for ann in anns:
                        if ann.get("image_id") == image["id"]:  # Matching by image_id (which is unique per image)
                            annotations_for_image.append(ann)

                    img = get_image_with_annotations(image_path, annotations_for_image)

                images.append(img)

            else:
                print(f"Skipping unsupported image format: {img_path}")
    print(f"Loaded {len(images)} images.")
    return images


# =========================
# Displaying in 10x20 grid
# =========================
def display_images_in_grid(images: list, plot_index: int, segmentation: bool, savename: str) -> None:
    """Display images in a grid using Matplotlib and save the plot to a file."""

    # Calculate the figure size dynamically based on grid size and image size
    fig_width = GRID_COLS * IMAGE_SIZE[0] / 100  # Adjust the figure width
    fig_height = GRID_ROWS * IMAGE_SIZE[1] / 100  # Adjust the figure height

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(fig_width, fig_height), gridspec_kw={'hspace': 0, 'wspace': 0})
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide axes if there are fewer than 200 images

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove extra whitespace around the image grid

    # Save the figure to the output directory with the segmentation tag
    segmentation_tag = "_segmentation" if segmentation else "_no_segmentation"
    plot_filename = OUTPUT_DIR / f"plot_{savename}_{plot_index}{segmentation_tag}.png"
    plt.savefig(plot_filename, bbox_inches='tight')  # Save with tight bounding box
    print(f"Saved plot {plot_index} to {plot_filename}")
    plt.close(fig)  # Close the figure to free up memory


# =========================
# Helpers for Annotations
# =========================
def get_image_with_annotations(image_path: Path, anns: list) -> None:
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
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Correct image orientation for matplotlib

    return img


def draw_overlay(img: Image.Image, anns: list, scale: float = 1.0) -> Image.Image:
    """draw_overlay(img, anns, scale) -> Image: draw polygons and bboxes."""
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
# Main
# =========================
def show_data(n: int = 1, segmentation: bool = False, save_name: str = None) -> None:
    """show_data() -> None: Load images and save them as grid images."""
    # Check if annotations file exists
    if not ANNOTATIONS_PATH.is_file():
        raise FileNotFoundError(f"Missing COCO json: {ANNOTATIONS_PATH}")

    # Load COCO annotations (just to confirm structure, not used for grid)
    coco = load_json(ANNOTATIONS_PATH)
    images = coco.get("images", [])

    # Get subdirectories of BASE_DIR (folders containing image data)
    subdirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]

    # Loop through the subdirectories and generate n plots
    for i in range(n):
        if i < len(subdirs):  # Make sure we don't exceed the available subdirectories
            subdir = subdirs[i]
            print(f"Loading images from folder: {subdir}")

            # Load images from the subdirectory
            loaded_images = load_images_from_folder(subdir, segmentation)

            # Display the images in a grid format and save them
            print(f"Saving plot {i + 1}")
            display_images_in_grid(loaded_images, i + 1, segmentation, savename=save_name)
        else:
            print(f"Not enough subdirectories to generate {n} plots. Only {len(subdirs)} available.")
            break


if __name__ == "__main__":
    # Set the number of plots to generate (e.g., 3 plots)
    show_data(n=9, segmentation=True, save_name="offnadir_no_glint")  # Set segmentation=True or False depending on the requirement