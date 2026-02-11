#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
from typing import Any


# ============================================================
# ===================== USER SETTINGS ========================
# ============================================================

OUT_FOLDER_NAME = "0_merged"

# If the same relative image path exists in both datasets:
# "rename" (recommended), "skip", or "error"
ON_CONFLICT = "rename"
WHALES_PREFIX = "whales"
OCEAN_PREFIX = "ocean"

# ============================================================
# ============================================================


def read_json(path: Path) -> dict[str, Any]:
    """read_json(path: Path) -> dict[str, Any]"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """write_json(path: Path, obj: dict[str, Any]) -> None"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def is_coco(obj: dict[str, Any]) -> bool:
    """is_coco(obj: dict[str, Any]) -> bool"""
    return isinstance(obj, dict) and "images" in obj and "annotations" in obj and "categories" in obj


def find_coco_ann(modality_dir: Path) -> Path:
    """find_coco_ann(modality_dir: Path) -> Path"""
    # Your case: final_annotations.json directly under modality folder
    preferred = [
        modality_dir / "final_annotations.json",
        modality_dir / "final_annotations.coco.json",
        modality_dir / "instances_all.json",
        modality_dir / "instances.json",
        modality_dir / "coco.json",
    ]
    for p in preferred:
        if p.exists():
            obj = read_json(p)
            if is_coco(obj):
                return p

    # Otherwise: try any json in modality root that looks like COCO
    for p in sorted(modality_dir.glob("*.json")):
        try:
            obj = read_json(p)
            if is_coco(obj):
                return p
        except Exception:
            pass

    raise FileNotFoundError(f"No COCO annotation json found in: {modality_dir}")


def list_modalities(root: Path) -> dict[str, tuple[Path, Path]]:
    """list_modalities(root: Path) -> dict[str, tuple[Path, Path]]"""
    mods: dict[str, tuple[Path, Path]] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            ann = find_coco_ann(p)
            mods[p.name] = (p, ann)
        except Exception:
            continue
    return mods


def merge_categories(c1: list[dict[str, Any]], c2: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    """merge_categories(c1: list[dict[str, Any]], c2: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int,int], dict[int,int]]"""
    merged: list[dict[str, Any]] = []
    name_to_newid: dict[str, int] = {}

    def add(cat: dict[str, Any]) -> int:
        name = str(cat["name"])
        if name in name_to_newid:
            return name_to_newid[name]
        new_id = len(merged) + 1
        merged.append({"id": new_id, "name": name, "supercategory": cat.get("supercategory", name)})
        name_to_newid[name] = new_id
        return new_id

    map1: dict[int, int] = {}
    map2: dict[int, int] = {}

    for cat in c1:
        map1[int(cat["id"])] = add(cat)
    for cat in c2:
        map2[int(cat["id"])] = add(cat)

    return merged, map1, map2


def copy_images(src_mod: Path, dst_mod: Path, prefix: str, ann_json_name: str) -> dict[str, str]:
    """copy_images(src_mod: Path, dst_mod: Path, prefix: str, ann_json_name: str) -> dict[str, str]"""
    rename_map: dict[str, str] = {}

    for root, _, files in os.walk(src_mod):
        root_p = Path(root)

        # skip modality root annotation json (and any jsons in root)
        if root_p == src_mod:
            files = [f for f in files if f != ann_json_name and not f.lower().endswith(".json")]

        rel_dir = root_p.relative_to(src_mod)
        dst_dir = dst_mod / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_file = root_p / fname
            if src_file.is_dir():
                continue
            # do not copy json files at all
            if src_file.suffix.lower() == ".json":
                continue

            dst_file = dst_dir / fname

            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                continue

            if ON_CONFLICT == "skip":
                continue
            if ON_CONFLICT == "error":
                raise RuntimeError(f"Collision: {dst_file}")

            new_name = f"{prefix}__{fname}"
            dst_file2 = dst_dir / new_name
            shutil.copy2(src_file, dst_file2)

            old_rel = str((rel_dir / fname)).replace("\\", "/")
            new_rel = str((rel_dir / new_name)).replace("\\", "/")
            rename_map[old_rel] = new_rel

    return rename_map


def remap_coco(
    coco: dict[str, Any],
    cat_map: dict[int, int],
    img_id_offset: int,
    ann_id_offset: int,
    rename_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """remap_coco(coco: dict[str, Any], cat_map: dict[int,int], img_id_offset: int, ann_id_offset: int, rename_map: dict[str,str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]"""
    new_images: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    img_id_map: dict[int, int] = {}

    for im in coco["images"]:
        old_img_id = int(im["id"])
        new_img_id = old_img_id + img_id_offset
        img_id_map[old_img_id] = new_img_id

        fn = str(im["file_name"]).replace("\\", "/")
        fn = rename_map.get(fn, fn)

        im2 = dict(im)
        im2["id"] = new_img_id
        im2["file_name"] = fn
        new_images.append(im2)

    for ann in coco["annotations"]:
        ann2 = dict(ann)
        ann2["id"] = int(ann["id"]) + ann_id_offset
        ann2["image_id"] = img_id_map[int(ann["image_id"])]
        ann2["category_id"] = cat_map[int(ann["category_id"])]
        new_annotations.append(ann2)

    return new_images, new_annotations


def merge_one_modality(
    modality_name: str,
    whales_mod_dir: Path,
    whales_ann: Path,
    ocean_mod_dir: Path,
    ocean_ann: Path,
    out_root: Path,
) -> None:
    """merge_one_modality(modality_name: str, whales_mod_dir: Path, whales_ann: Path, ocean_mod_dir: Path, ocean_ann: Path, out_root: Path) -> None"""
    print(f"\n=== Merging modality: {modality_name} ===")
    print(f"Whales: {whales_ann}")
    print(f"Ocean : {ocean_ann}")

    out_mod = out_root / modality_name
    out_mod.mkdir(parents=True, exist_ok=True)

    w = read_json(whales_ann)
    o = read_json(ocean_ann)
    if not is_coco(w):
        raise RuntimeError(f"Not COCO: {whales_ann}")
    if not is_coco(o):
        raise RuntimeError(f"Not COCO: {ocean_ann}")

    merged_categories, w_cat_map, o_cat_map = merge_categories(w["categories"], o["categories"])

    w_rename = copy_images(whales_mod_dir, out_mod, WHALES_PREFIX, whales_ann.name)
    o_rename = copy_images(ocean_mod_dir, out_mod, OCEAN_PREFIX, ocean_ann.name)

    w_imgs, w_anns = remap_coco(w, w_cat_map, 0, 0, w_rename)
    o_imgs, o_anns = remap_coco(o, o_cat_map, 1_000_000, 1_000_000, o_rename)

    merged = {"images": w_imgs + o_imgs, "annotations": w_anns + o_anns, "categories": merged_categories}

    out_ann = out_mod / "final_annotations_merged.json"
    write_json(out_ann, merged)

    print(f"Saved: {out_ann}")
    print(f"Images: {len(merged['images'])} | Annotations: {len(merged['annotations'])} | Categories: {len(merged['categories'])}")


def main() -> None:
    """main() -> None"""
    base_dir = Path(__file__).resolve().parent
    whales_root = base_dir / "0_whales"
    ocean_root = base_dir / "0_ocean"
    out_root = base_dir / OUT_FOLDER_NAME

    if not whales_root.exists():
        raise FileNotFoundError(f"Missing: {whales_root}")
    if not ocean_root.exists():
        raise FileNotFoundError(f"Missing: {ocean_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    w_mods = list_modalities(whales_root)
    o_mods = list_modalities(ocean_root)

    if not w_mods:
        raise RuntimeError(f"No modality folders with COCO json found under: {whales_root}")
    if not o_mods:
        raise RuntimeError(f"No modality folders with COCO json found under: {ocean_root}")

    common = sorted(set(w_mods.keys()) & set(o_mods.keys()))
    if not common:
        raise RuntimeError(
            "No common modality folders found.\n"
            f"Whales modalities: {sorted(w_mods.keys())}\n"
            f"Ocean  modalities: {sorted(o_mods.keys())}"
        )

    print("Common modalities:", common)
    for mod in common:
        w_dir, w_ann = w_mods[mod]
        o_dir, o_ann = o_mods[mod]
        merge_one_modality(mod, w_dir, w_ann, o_dir, o_ann, out_root)

    print("\nDONE")
    print("Merged dataset at:", out_root.resolve())


if __name__ == "__main__":
    main()
