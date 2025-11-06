import os
import json
import re
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import pandas as pd

# === CONFIG ===
BASE_DIR = Path(__file__).parent.resolve()
RAW_ROOT = BASE_DIR / "2022_whalesfromspace/data_cue_raw"
SPLITS_ROOT = BASE_DIR / "Whales from Satellite"
SPLIT_DIRS = ["train", "test", "valid"]

OUT_JSON = BASE_DIR / "new_annotations.json"

MATCHED_DIR = BASE_DIR / "whales_from_space_matched"
UNMATCHED_DIR = BASE_DIR / "whales_from_space_unmatched"

MATCHED_DIR.mkdir(exist_ok=True)
UNMATCHED_DIR.mkdir(exist_ok=True)

EXCEL_PATH = r"C:/Users/Nadine/ESA/Phi-lab Onboard AI - Onboard tip and cue - Onboard tip and cue/3_Software/1_Datasets/whales/2022_whalesfromspace/WhaleFromSpaceDB_Whales.csv"
# === Helper functions ===

def infer_rotation_angle(raw_img, candidate_img):
    best_angle = 0
    best_score = -1
    for angle in [0, 90, 180, 270]:
        rotated = candidate_img.rotate(angle, expand=True)
        rotated = rotated.resize(raw_img.size)
        score = ssim(np.array(raw_img), np.array(rotated), channel_axis=-1)
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle

def rotate_point(x, y, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle_rad) - ty * math.sin(angle_rad)
    ry = tx * math.sin(angle_rad) + ty * math.cos(angle_rad)
    return rx + cx, ry + cy

def rotate_bbox(bbox, cx, cy, angle_deg):
    x, y, bw, bh = bbox
    corners = [
        (x, y),
        (x + bw, y),
        (x + bw, y + bh),
        (x, y + bh)
    ]
    rotated_corners = [rotate_point(px, py, cx, cy, angle_deg) for px, py in corners]
    xs, ys = zip(*rotated_corners)
    nx, ny = min(xs), min(ys)
    nw, nh = max(xs) - nx, max(ys) - ny
    return [nx, ny, nw, nh]

def rotate_segmentation(segmentation, cx, cy, angle_deg):
    rotated_seg = []
    for polygon in segmentation:
        coords = []
        for i in range(0, len(polygon), 2):
            x, y = polygon[i], polygon[i+1]
            nx, ny = rotate_point(x, y, cx, cy, angle_deg)
            coords.extend([nx, ny])
        rotated_seg.append(coords)
    return rotated_seg

def plot_img_and_annots(ax, image, anns, title):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    for ann in anns:
        bbox = ann['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if 'segmentation' in ann and ann['segmentation']:
            for polygon in ann['segmentation']:
                poly_xy = np.array(polygon).reshape(-1, 2)
                ax.plot(poly_xy[:, 0], poly_xy[:, 1], color='lime', linewidth=1.5)

# === LOAD ALL COCO JSONS ===
df = pd.read_csv(EXCEL_PATH)

coco_data = {}
for split in SPLIT_DIRS:
    split_folder = SPLITS_ROOT / split
    coco_path = split_folder / "_annotations.coco.json"
    with open(coco_path) as f:
        coco = json.load(f)
    file_to_img = {img['file_name']: img for img in coco['images']}
    imgid_to_ann = {}
    for ann in coco['annotations']:
        imgid_to_ann.setdefault(ann['image_id'], []).append(ann)
    coco_data[split] = {
        "coco": coco,
        "folder": split_folder,
        "file_to_img": file_to_img,
        "imgid_to_ann": imgid_to_ann
    }
    print(f"Loaded COCO for {split}: {len(file_to_img)} images")

new_coco = {
    "images": [],
    "annotations": [],
    "categories": coco["categories"],
    "licenses": coco["licenses"],
}

new_image_id = 1
new_annotation_id = 1

total_raw = 0
matched_raw = 0

def matches_base_name(stem, base_name):
    return stem == base_name or stem.startswith(base_name + "_")




for subfolder in RAW_ROOT.iterdir():
    if not subfolder.is_dir():
        continue
    if subfolder.name.startswith("0_"):
        print(f"Skipping folder: {subfolder.name}")
        continue

    for raw_file in subfolder.rglob("*"):
        if not raw_file.is_file():
            continue
        if raw_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            print(f"Skipping non-image file: {raw_file.name}")
            continue

        total_raw += 1
        print(f"\n[{total_raw}] Processing raw image: {raw_file.relative_to(RAW_ROOT)}")

        raw_img = Image.open(raw_file).convert("RGB")
        raw_arr = np.array(raw_img)

        best_score = -1
        best_match = None
        best_split = None

        all_candidates = []
        base_name = raw_file.stem
        for split in SPLIT_DIRS:
            split_folder = coco_data[split]["folder"]
            candidates = list(split_folder.glob(f"{base_name}*"))
            candidates = [c for c in candidates if c.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            filtered_candidates = [c for c in candidates if matches_base_name(c.stem, base_name)]
            for cand in filtered_candidates:
                all_candidates.append((cand, split))

        target_image = Path(raw_file).stem
        filtered_rows = df[df['BoxID/ImageChip'] == target_image]

        # --- Extract image center and GCS ---
        certainty = filtered_rows.iloc[0]['Certainty2']
        print(f"  Whale certainty: {certainty}")

        if not all_candidates:
            print(f"  No candidates found in any split.")
            # Copy unmatched raw image
            target_path = UNMATCHED_DIR / raw_file.relative_to(RAW_ROOT)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_file, target_path)
            continue

        for candidate, split in all_candidates:
            cand_img = Image.open(candidate).convert("RGB")
            cand_img = cand_img.resize(raw_img.size)
            cand_arr = np.array(cand_img)
            score = ssim(raw_arr, cand_arr, channel_axis=-1)
            if score > best_score:
                best_score = score
                best_match = candidate
                best_split = split

        if best_match is None:
            print(f"  No match found in any split.")
            target_path = UNMATCHED_DIR / raw_file.relative_to(RAW_ROOT)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_file, target_path)
            continue

        matched_raw += 1
        print(f"  Best match: {best_match.name} (Split: {best_split}, SSIM: {best_score:.4f})")

        # Copy matched raw image
        matched_target_path = MATCHED_DIR / raw_file.relative_to(RAW_ROOT)
        matched_target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_file, matched_target_path)

        # Infer rotation angle to align best_img to raw_img
        best_img = Image.open(best_match).convert("RGB")
        angle = infer_rotation_angle(raw_img, best_img)
        print(f"  Inferred rotation angle: {angle} degrees")

        # Prepare images for plotting
        best_img_rotated = best_img.rotate(angle, expand=True).resize(raw_img.size)

        if best_score < 0.7 or "rotated" in best_match.name.lower():

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            # Raw image (no annotations)
            plot_img_and_annots(axs[0], raw_img, [], f"Raw Image")

            # Candidate original image with annotations
            split_data = coco_data[best_split]
            matched_img_entry = split_data["file_to_img"].get(best_match.name)
            candidate_anns = split_data["imgid_to_ann"].get(matched_img_entry['id'], [])
            plot_img_and_annots(axs[1], best_img, candidate_anns, f"Best Match (orig): {best_match.name}")

            # Candidate rotated back with rotated annotations
            inv_angle = (360 - angle) % 360
            cx, cy = raw_img.width / 2, raw_img.height / 2
            rotated_anns = []
            for ann in candidate_anns:
                new_ann = ann.copy()
                new_ann['bbox'] = rotate_bbox(new_ann['bbox'], cx, cy, inv_angle)
                if 'segmentation' in new_ann and new_ann['segmentation']:
                    new_ann['segmentation'] = rotate_segmentation(new_ann['segmentation'], cx, cy, inv_angle)
            
                new_ann['bbox'] = np.round(new_ann['bbox'], 3).tolist()
                new_ann['segmentation'] = np.round(new_ann['segmentation'], 3).tolist()

                rotated_anns.append(new_ann)


            plot_img_and_annots(axs[2], best_img_rotated, rotated_anns, f"Best Match (rotated {angle}°)")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(3)
            plt.close(fig)

        # Save annotations rotated back to raw orientation in new COCO
        cx, cy = raw_img.width / 2, raw_img.height / 2

        split_data = coco_data[best_split]
        matched_img_entry = split_data["file_to_img"].get(best_match.name)
        if not matched_img_entry:
            print(f"  Warning: no COCO image entry for {best_match.name}")
            pause(3)
            continue

        new_img = matched_img_entry.copy()
        new_img['file_name'] = raw_file.relative_to(RAW_ROOT).as_posix()
        new_img['id'] = new_image_id
        new_coco['images'].append(new_img)

        anns = split_data["imgid_to_ann"].get(matched_img_entry['id'], [])
        for ann in anns:
            new_ann = ann.copy()
            new_ann['image_id'] = new_image_id
            new_ann['id'] = new_annotation_id

            inv_angle = (360 - angle) % 360
            new_ann['bbox'] = rotate_bbox(new_ann['bbox'], cx, cy, inv_angle)
            if 'segmentation' in new_ann and new_ann['segmentation']:
                new_ann['segmentation'] = rotate_segmentation(new_ann['segmentation'], cx, cy, inv_angle)

            new_annotation_id += 1
            new_coco['annotations'].append(new_ann)

        new_image_id += 1

# === SAVE NEW COCO ===
with open(OUT_JSON, "w") as f:
    json.dump(new_coco, f)

print(f"\nDone. New combined COCO saved to: {OUT_JSON}")
print(f"Total raw images processed: {total_raw}")
print(f"Total matched with COCO:     {matched_raw}")
print(f"Total unmatched:             {total_raw - matched_raw}")
