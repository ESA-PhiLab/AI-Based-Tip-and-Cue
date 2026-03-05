import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

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

# mode can be one of ALLOWED_MODES or "all"
mode = "all"

DATASET_PATH = Path("dataset")
MERGED_ROOT = DATASET_PATH / "create_dataset" / "0_merged"

# Optional: restrict repair to one location (folder name in file_name)
LOCATION_FILTER: str | None = None

MAKE_CATEGORY_IDS_0_BASED = True
REPAIR_BBOX = True

COPY_TO_DEIM = True
DEIM_DST_ROOT = Path("onboard_ai") / "DEIMv2-main" / "data" / "0_merged"

INDEX_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

# Debug printing
DEBUG = False
DEBUG_MAX_PRINT = 50


# =========================
# Helpers
# =========================
def load_json(path: Path) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    """save_json(path,data) -> None: Write JSON utf-8 (pretty)."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_disk_files(base_dir: Path) -> tuple[set[str], dict[str, str]]:
    """scan_disk_files(base_dir) -> (set,dict): Return (exact_relpaths, lower_relpath->actual_relpath)."""
    exact: set[str] = set()
    lower_map: dict[str, str] = {}
    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in INDEX_SUFFIXES:
            continue
        rel = p.relative_to(base_dir).as_posix()
        exact.add(rel)
        lower_map.setdefault(rel.lower(), rel)
    return exact, lower_map


def get_location(file_name: str) -> str:
    """get_location(file_name) -> str: Location is first folder in file_name."""
    s = (file_name or "").strip()
    return s.split("/", 1)[0] if s and "/" in s else (s if s else "UNKNOWN")


def safe_pct(n: int, d: int) -> float:
    """safe_pct(n,d) -> float: Percentage n/d*100, safe for d=0."""
    return (100.0 * n / d) if d else 0.0


def sanitize_location(loc: str) -> str:
    """sanitize_location(loc) -> str: Make location safe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", loc.strip()) or "UNKNOWN"


def counts_per_location(coco: dict) -> tuple[dict[str, int], dict[str, int]]:
    """counts_per_location(coco) -> (dict,dict): Return (images_per_loc, anns_per_loc)."""
    images: list[dict] = coco.get("images", [])
    anns: list[dict] = coco.get("annotations", [])

    img_per: dict[str, int] = defaultdict(int)
    ann_per: dict[str, int] = defaultdict(int)

    id_to_loc: dict[int, str] = {}
    for im in images:
        loc = get_location(im.get("file_name", ""))
        img_per[loc] += 1
        if "id" in im:
            id_to_loc[im["id"]] = loc

    for a in anns:
        loc = id_to_loc.get(a.get("image_id"))
        if loc is not None:
            ann_per[loc] += 1

    return dict(img_per), dict(ann_per)


def print_overall_line(images_kept: int, images_total: int, anns_kept: int, anns_total: int) -> None:
    """print_overall_line(images_kept,images_total,anns_kept,anns_total) -> None: Print repaired/original with % first."""
    img_pct = safe_pct(images_kept, images_total)
    ann_pct = safe_pct(anns_kept, anns_total)
    print(f"Images: {img_pct:6.2f}%  {images_kept} / {images_total}    Annotations: {ann_pct:6.2f}%  {anns_kept} / {anns_total}")


def print_per_location_stats(orig_img: dict[str, int], orig_ann: dict[str, int], rep_img: dict[str, int], rep_ann: dict[str, int]) -> None:
    """print_per_location_stats(orig_img,orig_ann,rep_img,rep_ann) -> None: Print per-location repaired/original with % first."""
    locs = sorted(set(orig_img.keys()) | set(rep_img.keys()))
    print("\nPER LOCATION (repaired / original)")
    for loc in locs:
        oi = orig_img.get(loc, 0)
        oa = orig_ann.get(loc, 0)
        ri = rep_img.get(loc, 0)
        ra = rep_ann.get(loc, 0)
        ip = safe_pct(ri, oi)
        ap = safe_pct(ra, oa)
        print(f"{loc:<18} Images: {ip:6.2f}%  {ri:5d} / {oi:5d}    Annotations: {ap:6.2f}%  {ra:5d} / {oa:5d}")
    print("")


def remap_categories_0_based(coco: dict) -> dict:
    """remap_categories_0_based(coco) -> dict: Remap category ids to 0..K-1 and update annotations."""
    coco2 = dict(coco)
    cats: list[dict] = [dict(c) for c in coco.get("categories", [])]
    anns: list[dict] = [dict(a) for a in coco.get("annotations", [])]

    ids = set()
    for c in cats:
        if "id" in c:
            ids.add(c["id"])
    for a in anns:
        if "category_id" in a:
            ids.add(a["category_id"])

    if not ids:
        coco2["categories"] = cats
        coco2["annotations"] = anns
        return coco2

    old_ids = sorted(ids)
    mapping = {old: new for new, old in enumerate(old_ids)}

    for c in cats:
        if "id" in c:
            c["id"] = mapping[c["id"]]

    for a in anns:
        if "category_id" in a:
            a["category_id"] = mapping[a["category_id"]]

    coco2["categories"] = cats
    coco2["annotations"] = anns
    return coco2


def bbox_from_segmentation(seg: object) -> list[float] | None:
    """bbox_from_segmentation(seg) -> list[float]|None: Compute [x,y,w,h] from COCO polygon segmentation."""
    if not isinstance(seg, list) or not seg:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        if len(poly) % 2 != 0:
            continue
        for i in range(0, len(poly), 2):
            try:
                xs.append(float(poly[i]))
                ys.append(float(poly[i + 1]))
            except Exception:
                return None
    if not xs or not ys:
        return None
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    w = x1 - x0
    h = y1 - y0
    if w <= 0 or h <= 0:
        return None
    return [x0, y0, w, h]


def is_valid_bbox(b: object) -> bool:
    """is_valid_bbox(b) -> bool: True if bbox is [x,y,w,h] with w,h>0."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return False
    try:
        w = float(b[2])
        h = float(b[3])
    except Exception:
        return False
    return w > 0 and h > 0


def count_polygons(seg: object) -> int:
    """count_polygons(seg) -> int: Count polygon lists in COCO segmentation."""
    if not isinstance(seg, list):
        return 0
    return sum(1 for poly in seg if isinstance(poly, list) and len(poly) >= 6)


def repair_annotations_bboxes(coco: dict) -> tuple[dict, dict, list[dict], list[dict]]:
    """repair_annotations_bboxes(coco) -> (dict,stats,fixed_list,dropped_list): Fix missing bbox from seg or drop."""
    coco2 = dict(coco)
    anns: list[dict] = [dict(a) for a in coco.get("annotations", [])]

    fixed = 0
    dropped = 0
    kept: list[dict] = []
    fixed_list: list[dict] = []
    dropped_list: list[dict] = []

    for a in anns:
        if is_valid_bbox(a.get("bbox")):
            kept.append(a)
            continue

        b = bbox_from_segmentation(a.get("segmentation"))
        if b is not None:
            a["bbox"] = b
            fixed += 1
            kept.append(a)
            fixed_list.append(a)
            continue

        dropped += 1
        dropped_list.append(a)

    coco2["annotations"] = kept
    stats = {"bbox_fixed_from_segmentation": fixed, "bbox_dropped_no_bbox_no_seg": dropped}
    return coco2, stats, fixed_list, dropped_list


def copy_repaired_to_deim(src_path: Path, one_mode: str) -> None:
    """copy_repaired_to_deim(src_path,one_mode) -> None: Copy repaired json to onboard_ai/DEIMv2-main/data/0_merged/<mode>/."""
    if not COPY_TO_DEIM:
        return
    dst_dir = DEIM_DST_ROOT / one_mode
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    print(f"COPIED: {dst_path}")


# =========================
# Repair core
# =========================
def repair_one_coco(coco: dict, exact_disk: set[str], lower_disk: dict[str, str], location_filter: str | None) -> dict:
    """repair_one_coco(coco,exact_disk,lower_disk,location_filter) -> dict: Drop missing images + orphan anns; keep same layout."""
    images: list[dict] = coco.get("images", [])
    anns: list[dict] = coco.get("annotations", [])

    kept_images: list[dict] = []
    kept_image_ids: set[int] = set()

    for im in images:
        rel = (im.get("file_name") or "").strip()
        if not rel:
            continue

        loc = get_location(rel)
        if location_filter is not None and loc != location_filter:
            continue

        if rel in exact_disk:
            kept_images.append(im)
            kept_image_ids.add(im["id"])
            continue

        rel_lower = rel.lower()
        if rel_lower in lower_disk:
            im2 = dict(im)
            im2["file_name"] = lower_disk[rel_lower]
            kept_images.append(im2)
            kept_image_ids.add(im2["id"])
            continue

    kept_anns = [a for a in anns if a.get("image_id") in kept_image_ids]

    coco2 = dict(coco)
    coco2["images"] = kept_images
    coco2["annotations"] = kept_anns
    return coco2


def run_for_mode(one_mode: str) -> None:
    """run_for_mode(one_mode) -> None: Repair one mode and write final_annotations_repaired.json."""
    base_dir = MERGED_ROOT / one_mode
    ann_path = base_dir / "final_annotations_merged.json"

    out_path = (
        base_dir / "final_annotations_repaired.json"
        if LOCATION_FILTER is None
        else base_dir / f"final_annotations_repaired_{sanitize_location(LOCATION_FILTER)}.json"
    )

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Missing BASE_DIR: {base_dir}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing COCO json: {ann_path}")

    coco_orig = load_json(ann_path)
    exact_disk, lower_disk = scan_disk_files(base_dir)

    coco_rep = repair_one_coco(coco_orig, exact_disk, lower_disk, LOCATION_FILTER)

    bbox_stats = {"bbox_fixed_from_segmentation": 0, "bbox_dropped_no_bbox_no_seg": 0}
    fixed_list: list[dict] = []
    dropped_list: list[dict] = []
    if REPAIR_BBOX:
        coco_rep, bbox_stats, fixed_list, dropped_list = repair_annotations_bboxes(coco_rep)

    if MAKE_CATEGORY_IDS_0_BASED:
        coco_rep = remap_categories_0_based(coco_rep)

    save_json(out_path, coco_rep)
    print(f"WROTE:   {out_path}")
    if bbox_stats["bbox_fixed_from_segmentation"] or bbox_stats["bbox_dropped_no_bbox_no_seg"]:
        print(f"  bbox fixed from segmentation: {bbox_stats['bbox_fixed_from_segmentation']}, dropped: {bbox_stats['bbox_dropped_no_bbox_no_seg']}")

    copy_repaired_to_deim(out_path, one_mode)

    if LOCATION_FILTER is None:
        images_total = len(coco_orig.get("images", []))
        anns_total = len(coco_orig.get("annotations", []))
    else:
        images_total = sum(
            1 for im in coco_orig.get("images", [])
            if get_location(im.get("file_name", "")) == LOCATION_FILTER
        )
        img_ids_loc = {
            im["id"] for im in coco_orig.get("images", [])
            if get_location(im.get("file_name", "")) == LOCATION_FILTER
        }
        anns_total = sum(1 for a in coco_orig.get("annotations", []) if a.get("image_id") in img_ids_loc)

    print_overall_line(
        images_kept=len(coco_rep.get("images", [])),
        images_total=images_total,
        anns_kept=len(coco_rep.get("annotations", [])),
        anns_total=anns_total,
    )

    # DEBUG: print which images/annotations were repaired/dropped and multi-polygon anns
    if DEBUG:
        id_to_name = {im.get("id"): im.get("file_name") for im in coco_rep.get("images", [])}

        print("\nDEBUG: annotations with bbox computed from segmentation")
        for k, a in enumerate(fixed_list[:DEBUG_MAX_PRINT]):
            fn = id_to_name.get(a.get("image_id"), "UNKNOWN_IMAGE")
            npoly = count_polygons(a.get("segmentation"))
            print(f"  FIXED bbox | image_id={a.get('image_id')} ann_id={a.get('id')} polys={npoly} file={fn}")
        if len(fixed_list) > DEBUG_MAX_PRINT:
            print(f"  ... ({len(fixed_list) - DEBUG_MAX_PRINT} more)")

        print("\nDEBUG: annotations dropped (no bbox and no usable segmentation)")
        for k, a in enumerate(dropped_list[:DEBUG_MAX_PRINT]):
            fn = id_to_name.get(a.get("image_id"), "UNKNOWN_IMAGE")
            npoly = count_polygons(a.get("segmentation"))
            print(f"  DROPPED    | image_id={a.get('image_id')} ann_id={a.get('id')} polys={npoly} file={fn}")
        if len(dropped_list) > DEBUG_MAX_PRINT:
            print(f"  ... ({len(dropped_list) - DEBUG_MAX_PRINT} more)")

        # likely "two whales in one ann" cases
        multi_poly = []
        for a in coco_rep.get("annotations", []):
            if count_polygons(a.get("segmentation")) >= 2:
                multi_poly.append(a)

        print("\nDEBUG: annotations with >=2 polygons (bbox will cover all polygons in that annotation)")
        for k, a in enumerate(multi_poly[:DEBUG_MAX_PRINT]):
            fn = id_to_name.get(a.get("image_id"), "UNKNOWN_IMAGE")
            npoly = count_polygons(a.get("segmentation"))
            print(f"  MULTI-POLY | image_id={a.get('image_id')} ann_id={a.get('id')} polys={npoly} file={fn}")
        if len(multi_poly) > DEBUG_MAX_PRINT:
            print(f"  ... ({len(multi_poly) - DEBUG_MAX_PRINT} more)")

    # Per-location stats ONLY when mode != "all"
    if mode != "all":
        if LOCATION_FILTER is None:
            orig_img_per, orig_ann_per = counts_per_location(coco_orig)
            rep_img_per, rep_ann_per = counts_per_location(coco_rep)
            print_per_location_stats(orig_img_per, orig_ann_per, rep_img_per, rep_ann_per)
        else:
            orig_img_per, orig_ann_per = counts_per_location(coco_orig)
            rep_img_per, rep_ann_per = counts_per_location(coco_rep)
            loc = LOCATION_FILTER
            print_per_location_stats(
                {loc: orig_img_per.get(loc, 0)},
                {loc: orig_ann_per.get(loc, 0)},
                {loc: rep_img_per.get(loc, 0)},
                {loc: rep_ann_per.get(loc, 0)},
            )


# =========================
# Main
# =========================
def main() -> None:
    """main() -> None: Repair one mode or all; optionally print debug list of repaired images."""
    if mode != "all" and mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)} or 'all'")

    if COPY_TO_DEIM:
        DEIM_DST_ROOT.mkdir(parents=True, exist_ok=True)

    if mode == "all":
        for m in sorted(ALLOWED_MODES):
            print(f"\n===== MODE: {m} =====")
            run_for_mode(m)
        return

    print(f"\n===== MODE: {mode} =====")
    run_for_mode(mode)


if __name__ == "__main__":
    main()