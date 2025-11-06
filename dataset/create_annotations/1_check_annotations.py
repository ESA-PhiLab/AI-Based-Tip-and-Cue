import os
import re
import json
import pandas as pd


def stem(p: str) -> str:
    """Lowercase filename stem without path/ext."""
    return os.path.splitext(os.path.basename(str(p)))[0].lower()


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV (robust to BOM/delimiters)."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        return pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")


def resolve_column(df: pd.DataFrame, want: str) -> str:
    """Find CSV column 'want' (case/space insensitive)."""
    want_norm = "".join(want.split()).lower()
    for col in df.columns:
        col_norm = "".join(str(col).split()).lower()
        if col_norm == want_norm:
            return col
    for col in df.columns:
        col_norm = "".join(str(col).split()).lower()
        if want_norm in col_norm or col_norm in want_norm:
            return col
    raise KeyError(f"Column '{want}' not found. Columns: {list(df.columns)}")


def infer_folder(filename: str) -> str:
    """Infer top-level folder (e.g., Auckland2006) from filename like Auckland_SRW_QB2_PS_20060812_B0.PNG."""
    base = os.path.basename(filename)
    first = base.split("_", 1)[0]
    m = re.search(r"(20\d{2})\d{4}", base)  # yyyymmdd
    year = m.group(1) if m else ""
    if first.lower().startswith("pelagos"):
        first = "Pelagos"
    if not year:
        return first
    return f"{first}{year}"


def main(json_path: str, csv_path: str, csv_wanted_col: str = "boxID/ImageChip"):
    """Compare CSV images to JSON annotations and print missing as <folder>/<filename>.png (case preserved)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = load_csv(csv_path)
    col = resolve_column(df, csv_wanted_col)
    print(f"Using CSV column: '{col}'")
    print("Sample values:", df[col].dropna().astype(str).head(10).tolist())

    images = data.get("images", [])
    anns = data.get("annotations", [])

    imgid_by_stem = {}
    for img in images:
        orig_name = img.get("file_name", "") or img.get("extra", {}).get("name", "")
        s = stem(orig_name)
        if s:
            imgid_by_stem.setdefault(s, set()).add(img["id"])

    annotated_ids = {a["image_id"] for a in anns}
    annotated_stems = {
        s for s, ids in imgid_by_stem.items() if any(i in annotated_ids for i in ids)
    }

    csv_files = df[col].dropna().astype(str).tolist()
    csv_stem_map = {stem(v): v for v in csv_files}

    missing_stems = sorted(set(csv_stem_map.keys()) - annotated_stems)
    missing_files = [csv_stem_map[s] for s in missing_stems]

    print(f"CSV images total: {len(csv_stem_map)}")
    print(f"Annotated images (>=1 annotation): {len(annotated_stems)}")
    print(f"Images without annotation: {len(missing_files)}")

    if missing_files:
        print("\nMissing images (folder/filename.png):")
        for fname in missing_files:
            folder = infer_folder(fname)
            name = os.path.splitext(os.path.basename(fname))[0] + ".png"
            out = f"{folder}/{name}" if folder else name
            print(out)


if __name__ == "__main__":
    main(
        json_path="new_annotations.json",
        csv_path=os.path.join("..", "whales_from_space", "WhaleFromSpaceDB_Whales.csv"),
        csv_wanted_col="boxID/ImageChip",
    )
