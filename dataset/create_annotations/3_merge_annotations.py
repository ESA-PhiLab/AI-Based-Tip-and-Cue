import os, sys, re, json, pandas as pd

def load_json(path: str) -> dict: """Read JSON."""; return json.load(open(path, "r", encoding="utf-8"))
def dump_json(obj: dict, path: str) -> None: """Write JSON (indent=2)."""; json.dump(obj, open(path, "w", encoding="utf-8"), indent=2)

def upper_png(name: str) -> str: """Ensure .PNG extension."""; return os.path.splitext(name)[0] + ".PNG"
def normalize_basestem(s: str) -> str: """Drop _rotated suffixes."""; return re.sub(r'(_rotated-?|_rotated)?$', '', os.path.splitext(os.path.basename(s))[0], flags=re.IGNORECASE)
def norm_stem_from_img(img: dict) -> str: """Lowercase normalized stem."""; src = (img.get("extra") or {}).get("name") or img.get("file_name",""); return normalize_basestem(src).lower()

def load_csv_any(csv_path: str) -> pd.DataFrame:
    """Robust CSV load."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        return pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")

def resolve_column(df: pd.DataFrame, want: str) -> str:
    """Find column ignoring case/space."""
    wn = "".join(want.split()).lower()
    for c in df.columns:
        cn = "".join(str(c).split()).lower()
        if cn == wn: return c
    for c in df.columns:
        cn = "".join(str(c).split()).lower()
        if wn in cn or cn in wn: return c
    raise KeyError(f"Column '{want}' not found in {list(df.columns)}")

def excel_order_and_caps(df: pd.DataFrame) -> tuple[list[str], dict]:
    """Excel row order and stem mapping."""
    col = resolve_column(df, "BoxID/ImageChip")
    order, seen, stem2orig = [], set(), {}
    for v in df[col].dropna().astype(str):
        base = os.path.basename(v)
        s = normalize_basestem(base).lower()
        if s and s not in seen:
            seen.add(s); order.append(s); stem2orig[s] = base
    return order, stem2orig

def index_images_by_stem(imgs: list) -> dict:
    """stem->image."""
    out = {}
    for im in imgs or []:
        s = norm_stem_from_img(im)
        if s and s not in out: out[s] = im
    return out

def anns_by_image(anns: list) -> dict:
    """image_id->annotations list."""
    d = {}
    for a in anns or []: d.setdefault(a["image_id"], []).append(a)
    return d

def folder_from_added(img: dict) -> str:
    """Folder (location+year) from added file_name."""
    fn = img.get("file_name","")
    return os.path.normpath(fn).split(os.sep)[0] if fn else ""

def infer_folder_from_name(base_name: str) -> str:
    """Infer Auckland2006 from filename."""
    token = base_name.split("_",1)[0]
    m = re.search(r"(20\d{2})\d{4}", base_name)  # YYYYMMDD
    year = m.group(1) if m else ""
    token = "Pelagos" if token.lower().startswith("pelagos") else token
    return f"{token}{year}" if year else token

def canonicalize_image_for_output(src_img: dict, folder_hint: str) -> dict:
    """Return normalized image entry with folder/year naming."""
    base_from = (src_img.get("extra") or {}).get("name") or os.path.basename(src_img.get("file_name",""))
    norm_base = upper_png(normalize_basestem(base_from))
    folder = folder_hint or infer_folder_from_name(norm_base)
    out = dict(src_img)
    out["file_name"] = f"{folder}/{norm_base}"
    extra = dict(src_img.get("extra") or {})
    extra["name"] = norm_base
    out["extra"] = extra
    return out

def main(initial_path: str, added_path: str, excel_path: str, out_path: str) -> None:
    """Merge initial+added annotations ordered by Excel; keep initial layout/categories."""
    initial, added = load_json(initial_path), load_json(added_path)
    df = load_csv_any(excel_path)
    order, _ = excel_order_and_caps(df)

    init_imgs, add_imgs = index_images_by_stem(initial.get("images", [])), index_images_by_stem(added.get("images", []))
    init_anns = anns_by_image(initial.get("annotations", []))

    merged_imgs, merged_anns = [], []
    next_img_id, next_ann_id = 1, 1

    for s in order:
        src_img = init_imgs.get(s) or add_imgs.get(s)
        if not src_img: continue
        folder = folder_from_added(add_imgs.get(s)) if add_imgs.get(s) else ""
        out_img = canonicalize_image_for_output(src_img, folder)
        out_img["id"] = next_img_id
        merged_imgs.append(out_img)

        if s in init_imgs:
            anns = init_anns.get(init_imgs[s]["id"], [])
            for a in anns:
                na = dict(a)
                na["id"] = next_ann_id
                na["image_id"] = next_img_id
                merged_anns.append(na)
                next_ann_id += 1

        next_img_id += 1

    merged = {"images": merged_imgs, "annotations": merged_anns, "categories": initial.get("categories", [])}
    dump_json(merged, out_path)
    print(f"OK -> {out_path} | images={len(merged_imgs)} anns={len(merged_anns)} cats={len(merged['categories'])}")

if __name__ == "__main__":
    init = sys.argv[1] if len(sys.argv) > 1 else "initial_annotations.json"
    add  = sys.argv[2] if len(sys.argv) > 2 else "added_annotations.json"
    csv  = sys.argv[3] if len(sys.argv) > 3 else os.path.join("..","whales_from_space","WhaleFromSpaceDB_Whales.csv")
    out  = sys.argv[4] if len(sys.argv) > 4 else "merged_initial_plus_added_names.json"
    main(init, add, csv, out)
