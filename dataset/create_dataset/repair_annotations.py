import json
import os
import re
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

# mode can be:
#   - one of ALLOWED_MODES
#   - "all"  -> repair every mode in ALLOWED_MODES
mode = "all"

DATASET_PATH = Path("dataset")
MERGED_ROOT = DATASET_PATH / "create_dataset" / "0_merged"

# Optional: restrict repair to one location (folder name in file_name, e.g. "Maui2015")
#   - None -> keep all locations together (recommended; writes final_annotations_repaired.json)
#   - "Maui2015" -> write final_annotations_repaired_Maui2015.json for that location only
LOCATION_FILTER: str | None = None


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
    """scan_disk_files(base_dir) -> (set[str],dict[str,str]): Return (exact_paths, lower->actual map)."""
    exact = set()
    lower_map: dict[str, str] = {}
    for p in base_dir.rglob("*"):
        if not p.is_file():
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


def print_overall_stats(images_kept: int, images_total: int, anns_kept: int, anns_total: int) -> None:
    """print_overall_stats(images_kept,images_total,anns_kept,anns_total) -> None: Print repaired/original with % first."""
    img_pct = safe_pct(images_kept, images_total)
    ann_pct = safe_pct(anns_kept, anns_total)
    print("\n==============================")
    print("OVERALL (repaired / original)")
    print("==============================")
    print(f"Images:       {img_pct:6.2f}%   {images_kept} / {images_total}")
    print(f"Annotations:  {ann_pct:6.2f}%   {anns_kept} / {anns_total}")
    print("==============================\n")


def print_per_location_stats(orig_img: dict[str, int], orig_ann: dict[str, int], rep_img: dict[str, int], rep_ann: dict[str, int]) -> None:
    """print_per_location_stats(orig_img,orig_ann,rep_img,rep_ann) -> None: Print per-location repaired/original with % first."""
    locs = sorted(set(orig_img.keys()) | set(rep_img.keys()))
    print("\n==============================")
    print("PER LOCATION (repaired / original)")
    print("==============================")
    for loc in locs:
        oi = orig_img.get(loc, 0)
        oa = orig_ann.get(loc, 0)
        ri = rep_img.get(loc, 0)
        ra = rep_ann.get(loc, 0)
        ip = safe_pct(ri, oi)
        ap = safe_pct(ra, oa)
        print(f"{loc:<18} Images: {ip:6.2f}%  {ri:5d} / {oi:5d}    Annotations: {ap:6.2f}%  {ra:5d} / {oa:5d}")
    print("==============================\n")


# =========================
# Repair core
# =========================
def repair_one_coco(coco: dict, exact_disk: set[str], lower_disk: dict[str, str], location_filter: str | None) -> dict:
    """repair_one_coco(coco,exact_disk,lower_disk,location_filter) -> dict: Return repaired COCO (same layout; drops missing images and orphaned anns)."""
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


def run_for_mode(one_mode: str, print_locations: bool) -> None:
    """run_for_mode(one_mode,print_locations) -> None: Repair one mode and write final_annotations_repaired.json (combined layout)."""
    if one_mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)} or 'all'")

    base_dir = MERGED_ROOT / one_mode
    ann_path = base_dir / "final_annotations_merged.json"

    if not base_dir.is_dir():
        raise FileNotFoundError(f"Missing BASE_DIR: {base_dir}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing COCO json: {ann_path}")

    coco_orig = load_json(ann_path)
    exact_disk, lower_disk = scan_disk_files(base_dir)

    coco_rep = repair_one_coco(coco_orig, exact_disk, lower_disk, LOCATION_FILTER)

    if LOCATION_FILTER is None:
        out_path = base_dir / "final_annotations_repaired.json"
    else:
        out_path = base_dir / f"final_annotations_repaired_{sanitize_location(LOCATION_FILTER)}.json"

    save_json(out_path, coco_rep)
    print(f"WROTE: {out_path}")

    # Overall stats (for this run)
    print_overall_stats(
        images_kept=len(coco_rep.get("images", [])),
        images_total=len(coco_orig.get("images", [])) if LOCATION_FILTER is None else sum(1 for im in coco_orig.get("images", []) if get_location(im.get("file_name", "")) == LOCATION_FILTER),
        anns_kept=len(coco_rep.get("annotations", [])),
        anns_total=len(coco_orig.get("annotations", [])) if LOCATION_FILTER is None else sum(
            1
            for a in coco_orig.get("annotations", [])
            if any(im.get("id") == a.get("image_id") and get_location(im.get("file_name", "")) == LOCATION_FILTER for im in coco_orig.get("images", []))
        ),
    )

    # Per-location stats only when mode != "all" (i.e., print_locations=True)
    if print_locations:
        if LOCATION_FILTER is None:
            orig_img_per, orig_ann_per = counts_per_location(coco_orig)
            rep_img_per, rep_ann_per = counts_per_location(coco_rep)
            print_per_location_stats(orig_img_per, orig_ann_per, rep_img_per, rep_ann_per)
        else:
            # One-line per-location for the chosen location
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
    """main() -> None: mode can be one mode or 'all'. Writes combined repaired json with same layout as merged."""
    if mode != "all" and mode not in ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_MODES)} or 'all'")

    if mode == "all":
        for m in sorted(ALLOWED_MODES):
            print(f"\n===== MODE: {m} =====")
            run_for_mode(m, print_locations=False)
        return

    run_for_mode(mode, print_locations=True)


if __name__ == "__main__":
    main()