from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# =========================
# Config
# =========================

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

ALLOWED_MODES = {
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

mode = "texture_offnadir_255"

# Example 1: original dataset structure
DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "create_dataset" / "0_merged" / mode
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "0_merged" / mode / "final_annotations_repaired.json"

# Example 2: custom folder with image subfolders
DATASET_PATH = Path(r"C:/Users/nadine/Documents/Phi-Lab_MasterThesis/2_Full_Thesis/Report/V5_Report/figures/results/no_radiometric_geometric")
BASE_DIR = DATASET_PATH
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "0_merged" / mode / "final_annotations_repaired.json"
GRID_ROWS = 3
GRID_COLS = 7

# Example 3: whales from space visualization
# DATASET_PATH = Path(r"C:/Users/nadine/Documents/Phi-Lab_MasterThesis/2_Full_Thesis/Report/V5_Report/figures/dataset/whales_from_space_samples")
# BASE_DIR = DATASET_PATH
# ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "0_merged" / mode / "final_annotations_repaired.json"
# GRID_ROWS = 9
# GRID_COLS = 6

OUTPUT_DIR = Path("figures/output_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_SIZE = (64, 64)
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# =========================
# Helpers
# =========================
def load_json(path: str | Path) -> dict:
    """Read JSON file and return dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_plain_image(image_path: Path) -> Image.Image | None:
    """Load image, convert to RGB, resize, return PIL image."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img.resize(IMAGE_SIZE)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def prepare_annotation_data(annotation_path: Path) -> tuple[dict | None, dict[str, dict], dict[int, list[dict]]]:
    """Load COCO annotations and build lookup maps, else return empty maps."""
    if not annotation_path.is_file():
        print("No annotations found, generating plots without annotations.")
        return None, {}, {}

    try:
        coco = load_json(annotation_path)
    except Exception as e:
        print(f"Could not read annotations file {annotation_path}: {e}")
        print("No annotations found, generating plots without annotations.")
        return None, {}, {}

    images_by_file_name = {}
    anns_by_image_id = defaultdict(list)

    for img in coco.get("images", []):
        file_name = str(img.get("file_name", "")).replace("\\", "/")
        images_by_file_name[file_name] = img

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id is not None:
            anns_by_image_id[image_id].append(ann)

    return coco, images_by_file_name, dict(anns_by_image_id)


def draw_overlay(img: Image.Image, anns: list, scale_x: float = 1.0, scale_y: float = 1.0) -> Image.Image:
    """Draw segmentation polygons and boxes on image."""
    draw = ImageDraw.Draw(img)

    for ann in anns:
        for seg in ann.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i] * scale_x, seg[i + 1] * scale_y) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    return img


def get_image_with_annotations(image_path: Path, anns: list) -> Image.Image:
    """Load image, draw overlays, resize, return PIL image."""
    try:
        base = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        fallback = Image.new("RGB", IMAGE_SIZE, color=(20, 20, 20))
        ImageDraw.Draw(fallback).text((5, 5), "Read error", fill=(255, 80, 80))
        return fallback

    original_w, original_h = base.size
    resized = base.resize(IMAGE_SIZE)
    scale_x = IMAGE_SIZE[0] / original_w if original_w else 1.0
    scale_y = IMAGE_SIZE[1] / original_h if original_h else 1.0

    resized = draw_overlay(resized, anns, scale_x=scale_x, scale_y=scale_y)
    return resized


def build_relative_filename(image_path: Path, base_dir: Path) -> str:
    """Build COCO-style relative path using forward slashes."""
    try:
        return image_path.relative_to(base_dir).as_posix()
    except Exception:
        return image_path.name


def load_images_from_folder(folder: Path, segmentation: bool = False, images_by_file_name: dict | None = None, anns_by_image_id: dict | None = None) -> list[Image.Image]:
    """Load images from folder, optionally with annotation overlays."""
    images = []
    images_by_file_name = images_by_file_name or {}
    anns_by_image_id = anns_by_image_id or {}

    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            img_path = Path(root) / filename

            if img_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
                continue

            if not segmentation:
                img = load_plain_image(img_path)
                if img is not None:
                    images.append(img)
                continue

            relative_name = build_relative_filename(img_path, BASE_DIR)
            image_info = images_by_file_name.get(relative_name)

            if image_info is None:
                img = load_plain_image(img_path)
                if img is not None:
                    images.append(img)
                continue

            image_id = image_info.get("id")
            anns = anns_by_image_id.get(image_id, [])
            img = get_image_with_annotations(img_path, anns)
            images.append(img)

    print(f"Loaded {len(images)} images from {folder}.")
    return images


# =========================
# Displaying in grid
# =========================
def display_images_in_grid(images: list[Image.Image], plot_index: int, segmentation: bool, savename: str) -> None:
    """Display images in grid and save figure."""
    fig_width = GRID_COLS * IMAGE_SIZE[0] / 100
    fig_height = GRID_ROWS * IMAGE_SIZE[1] / 100

    fig, axes = plt.subplots(
        GRID_ROWS,
        GRID_COLS,
        figsize=(fig_width, fig_height),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    segmentation_tag = "_segmentation" if segmentation else "_no_segmentation"
    plot_filename = OUTPUT_DIR / f"plot_{savename}_{plot_index}{segmentation_tag}.png"
    plt.savefig(plot_filename, bbox_inches="tight", pad_inches=0)
    print(f"Saved plot {plot_index} to {plot_filename}")
    plt.close(fig)


def collect_plot_folders(base_dir: Path) -> list[Path]:
    """Return subfolders, or base_dir itself if it directly contains images."""
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if subdirs:
        return subdirs

    has_images = any(p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES for p in base_dir.iterdir())
    if has_images:
        return [base_dir]

    return []


# =========================
# Main
# =========================
def show_data(n: int = 1, segmentation: bool = False, save_name: str | None = None) -> None:
    """Load images and save them as grid figures."""
    save_name = save_name or "plot"

    coco, images_by_file_name, anns_by_image_id = prepare_annotation_data(ANNOTATIONS_PATH)
    effective_segmentation = segmentation and coco is not None

    folders = collect_plot_folders(BASE_DIR)
    if not folders:
        raise FileNotFoundError(f"No image folders or images found in BASE_DIR: {BASE_DIR}")

    for i in range(n):
        if i >= len(folders):
            print(f"Not enough folders to generate {n} plots. Only {len(folders)} available.")
            break

        folder = folders[i]
        print(f"Loading images from folder: {folder}")

        loaded_images = load_images_from_folder(
            folder=folder,
            segmentation=effective_segmentation,
            images_by_file_name=images_by_file_name,
            anns_by_image_id=anns_by_image_id,
        )

        if not loaded_images:
            print(f"No images found in {folder}, skipping plot {i + 1}.")
            continue

        print(f"Saving plot {i + 1}")
        display_images_in_grid(
            images=loaded_images,
            plot_index=i + 1,
            segmentation=effective_segmentation,
            savename=save_name,
        )


if __name__ == "__main__":
    show_data(n=9, segmentation=True, save_name="offnadir_no_glint")