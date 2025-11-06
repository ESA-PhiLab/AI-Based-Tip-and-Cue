import os
import re
import json
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


# ----------------------------
# Utility functions
# ----------------------------
def stem(p: str) -> str:
    """Lowercase stem (basename without extension)."""
    return os.path.splitext(os.path.basename(str(p)))[0].lower()


def upper_png(name: str) -> str:
    """Force .PNG extension (uppercase)."""
    root, _ = os.path.splitext(name)
    return f"{root}.PNG"


def tidy_area(val: float):
    """Return int if integral; else float rounded to 3 dp (strip trailing zeros)."""
    r = round(val, 3)
    return int(r) if abs(r - round(r)) < 1e-9 else float(f"{r:.3f}".rstrip("0").rstrip("."))


def load_csv(csv_path: str) -> pd.DataFrame:
    """Robust CSV load (handles BOM/unknown delimiter)."""
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    except Exception:
        return pd.read_csv(csv_path, encoding="latin-1", sep=None, engine="python")


def resolve_column(df: pd.DataFrame, want: str) -> str:
    """Find column matching 'want' (case/space-insensitive)."""
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
    """Infer folder like Auckland2006 from e.g. Auckland_SRW_QB2_PS_20060812_B0.PNG."""
    base = os.path.basename(filename)
    token = base.split("_", 1)[0]
    match = re.search(r"(20\d{2})\d{4}", base)
    year = match.group(1) if match else ""
    if token.lower().startswith("pelagos"):
        token = "Pelagos"
    return f"{token}{year}" if year else token


def load_json(path: str) -> dict:
    """Load JSON dict if file exists; else {}."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotated_stems_from(json_dict: dict) -> set:
    """Return set of lowercase stems that have at least one annotation in a COCO dict."""
    if not json_dict:
        return set()
    images = json_dict.get("images", [])
    anns = json_dict.get("annotations", [])
    annotated_ids = {a.get("image_id") for a in anns}
    stems = set()
    for img in images:
        if img.get("id") in annotated_ids:
            nm = img.get("file_name") or img.get("extra", {}).get("name", "")
            if nm:
                stems.add(stem(nm))
    return stems


def compute_missing_from_new(new_json_path: str, csv_path: str, csv_col: str = "boxID/ImageChip") -> list:
    """Row-by-row over Excel; check presence ONLY in new_annotations.json; return relative paths with .PNG."""
    df = load_csv(csv_path)
    col = resolve_column(df, csv_col)

    present = annotated_stems_from(load_json(new_json_path))

    missing_rel = []
    for v in df[col].dropna().astype(str):
        s = stem(v)
        if not s or s in present:
            continue
        base_name = upper_png(os.path.basename(v))
        folder = infer_folder(base_name)
        rel = f"{folder}/{base_name}"
        missing_rel.append(rel)

    seen = set()
    out = []
    for r in missing_rel:
        k = r.lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

def split_existing(paths: list, base_dir: str) -> tuple[list, list]:
    """Return (present_full_paths, missing_full_paths)."""
    present, missing = [], []
    for rel in paths:
        full = os.path.join(base_dir, rel.replace("/", os.sep))
        (present if os.path.isfile(full) else missing).append(full)
    return present, missing

# ----------------------------
# Annotator (opens given paths; writes added_annotations.json)
# ----------------------------
class Annotator:
    def __init__(self, root, image_paths: list, whale_csv_path: str, output_path: str):
        self.root = root
        self.root.title("Multi-Whale COCO Annotator")
        self.root.geometry("1200x800")

        self.image_list = list(image_paths)
        self.out_json = output_path
        self.annotations = []
        self.image_index = 0
        self.points, self.point_circles, self.temp_lines = [], [], []
        self.scale_factor = 1.0
        self.current_polygon_id = None
        self.draggable_points = []
        self.drawn_ids = set()

        self.whale_lookup = self._load_whale_info(whale_csv_path)

        self.frame = tk.Frame(self.root)
        self.frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.frame, bg="gray", cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y = tk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")

        self.control = tk.Frame(self.root)
        self.control.pack(fill="x")
        self.label = tk.Label(self.control, text="", font=("Arial", 12))
        self.label.pack(side="left", padx=10)
        tk.Button(self.control, text="Next Image", command=self.save_and_next).pack(side="right", padx=5)
        tk.Button(self.control, text="Clear Polygon", command=self.clear_polygon).pack(side="right", padx=5)
        tk.Button(self.control, text="Finish", command=self.export_and_exit).pack(side="right", padx=5)

        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)

        self._load_next_image()

    def _load_whale_info(self, csv_path: str) -> dict:
        """Map lowercase stem (no extension) -> (NumWhale, Certainty2)."""
        try:
            df = load_csv(csv_path)
            col = resolve_column(df, "BoxID/ImageChip")
            lookup_df = df[[col, "NumWhale", "Certainty2"]].copy()
            lookup = {}
            for _, row in lookup_df.dropna(subset=[col]).iterrows():
                key = stem(str(row[col]))
                lookup[key] = (row["NumWhale"], row["Certainty2"])
            print(f"Loaded whale info for {len(lookup)} entries.")
            return lookup
        except Exception as e:
            print(f"Failed to load whale info: {e}")
            return {}

    def _load_next_image(self) -> None:
        if self.image_index >= len(self.image_list):
            self.export_and_exit()
            return

        path = self.image_list[self.image_index]
        self.folder_name = os.path.basename(os.path.dirname(path))
        self.img = Image.open(path)
        max_dim = max(self.img.width, self.img.height)
        self.scale_factor = max(1, 768 // max_dim)

        disp = self.img.resize(
            (self.img.width * self.scale_factor, self.img.height * self.scale_factor),
            resample=Image.NEAREST,
        )
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        filename = os.path.basename(path)
        key = stem(filename)
        num_whales, certainty = self.whale_lookup.get(key, ("?", "?"))

        self.label.config(
            text=f"[{self.image_index+1}/{len(self.image_list)}] "
                 f"{self.folder_name}/{filename} — Expected whales: {num_whales} ({certainty})"
        )
        print(f"{self.folder_name}/{filename} → Expected whales: {num_whales}, Certainty: {certainty}")

        self.points.clear()
        self.point_circles.clear()
        self.temp_lines.clear()
        self.current_polygon_id = None
        self.draggable_points.clear()
        self.drawn_ids.clear()

    def left_click(self, event) -> None:
        x, y = int(event.x / self.scale_factor), int(event.y / self.scale_factor)
        self.points.append((x, y))
        px, py = x * self.scale_factor, y * self.scale_factor

        pid = self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4,
                                      fill="green", tags=("poly", "point"))
        self.point_circles.append(pid)
        self.drawn_ids.add(pid)

        if len(self.points) > 1:
            prev = self.points[-2]
            lid = self.canvas.create_line(prev[0] * self.scale_factor, prev[1] * self.scale_factor,
                                          px, py, fill="#90ee90", width=2,
                                          tags=("poly", "temp_line"))
            self.temp_lines.append(lid)
            self.drawn_ids.add(lid)

    def right_click(self, event) -> None:
        if len(self.points) < 3:
            messagebox.showwarning("Too Few Points", "You need at least 3 points for a polygon.")
            return

        screen_pts = [(x * self.scale_factor, y * self.scale_factor) for x, y in self.points]
        poly_id = self.canvas.create_polygon(screen_pts, outline="lime", fill="", width=2, tags=("poly", "polygon"))
        self.drawn_ids.add(poly_id)

        xs, ys = zip(*self.points)
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        bbox_id = self.canvas.create_rectangle(xmin * self.scale_factor, ymin * self.scale_factor,
                                               xmax * self.scale_factor, ymax * self.scale_factor,
                                               outline="red", width=2, tags=("poly", "bbox"))
        self.drawn_ids.add(bbox_id)

        self.annotations.append({
            "filename": os.path.basename(self.image_list[self.image_index]),
            "bbox": (xmin, ymin, xmax, ymax),
            "segmentation": self.points.copy()
        })

        self.draggable_points = list(self.points)
        for i, cid in enumerate(self.point_circles):
            self.canvas.tag_bind(cid, "<B1-Motion>", lambda e, idx=i: self.move_point(e, idx))

        self.current_polygon_id = (poly_id, bbox_id)
        self.points.clear()
        self.point_circles.clear()
        self.temp_lines.clear()

    def move_point(self, event, index) -> None:
        x_new, y_new = int(event.x / self.scale_factor), int(event.y / self.scale_factor)
        self.draggable_points[index] = (x_new, y_new)

        self.canvas.coords(
            self.canvas.find_withtag("point")[index],
            x_new * self.scale_factor - 4, y_new * self.scale_factor - 4,
            x_new * self.scale_factor + 4, y_new * self.scale_factor + 4,
        )

        screen_pts = [(x * self.scale_factor, y * self.scale_factor) for x, y in self.draggable_points]
        if self.current_polygon_id:
            poly_id, bbox_id = self.current_polygon_id
            self.canvas.coords(poly_id, *sum(screen_pts, ()))

            xs, ys = zip(*self.draggable_points)
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
            self.canvas.coords(bbox_id,
                               xmin * self.scale_factor, ymin * self.scale_factor,
                               xmax * self.scale_factor, ymax * self.scale_factor)

    def clear_polygon(self) -> None:
        self.canvas.delete("poly")
        if self.drawn_ids:
            for iid in list(self.drawn_ids):
                try:
                    self.canvas.delete(iid)
                except Exception:
                    pass
            self.drawn_ids.clear()

        cur_name = os.path.basename(self.image_list[self.image_index]) if self.image_list else None
        if cur_name:
            self.annotations = [a for a in self.annotations if a.get("filename") != cur_name]

        self.points.clear()
        self.point_circles.clear()
        self.temp_lines.clear()
        self.current_polygon_id = None
        self.draggable_points.clear()

    def save_and_next(self) -> None:
        self.image_index += 1
        self._load_next_image()

    def export_and_exit(self) -> None:
        images, annotations = [], []
        categories = [{"id": 1, "name": "whale", "supercategory": "animal"}]
        annotation_id = 1
        image_id_map = {}
        now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "+00:00"

        for image_id, img_path in enumerate(self.image_list, 1):
            img = Image.open(img_path)
            filename_disk = os.path.basename(img_path)
            folder = os.path.basename(os.path.dirname(img_path))
            filename_png = upper_png(filename_disk)
            image_id_map[filename_disk] = image_id
            images.append(OrderedDict(
                id=image_id, license=1,
                file_name=f"{folder}/{filename_png}",
                height=img.height, width=img.width,
                date_captured=now_iso, extra={"name": filename_png}
            ))

        for ann in self.annotations:
            filename = ann["filename"]
            image_id = image_id_map.get(filename)
            if image_id is None:
                continue
            x1, y1, x2, y2 = map(float, ann["bbox"])
            w, h = x2 - x1, y2 - y1
            segmentation_flat = [float(c) for pt in ann["segmentation"] for c in pt]
            annotations.append(OrderedDict(
                id=annotation_id, image_id=image_id, category_id=1,
                bbox=[x1, y1, w, h], area=tidy_area(w * h),
                segmentation=[segmentation_flat], iscrowd=0
            ))
            annotation_id += 1

        coco_dict = OrderedDict(images=images, annotations=annotations, categories=categories)
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2)
        messagebox.showinfo("Saved", f"Annotations saved to {self.out_json}")
        self.root.quit()

if __name__ == "__main__":
    base_dir = os.path.join("..", "whales_from_space")
    csv_path = os.path.join(base_dir, "WhaleFromSpaceDB_Whales.csv")
    new_json = "new_annotations.json"
    out_json = "added_annotations.json"

    # === Load data ===
    new_data = load_json(new_json)
    df = load_csv(csv_path)
    col = resolve_column(df, "boxID/ImageChip")
    excel_files = df[col].dropna().astype(str).tolist()

    # === Find which stems already annotated ===
    annotated = annotated_stems_from(new_data)

    # === Count how many Excel entries are already annotated ===
    total_excel = len(excel_files)
    excel_annotated = 0
    excel_missing = []
    for v in excel_files:
        s = stem(v)
        if s in annotated:
            excel_annotated += 1
        else:
            excel_missing.append(v)

    print(f"Excel total entries: {total_excel}")
    print(f"Excel entries already annotated: {excel_annotated}")
    print(f"Excel entries missing annotations: {total_excel - excel_annotated}")

    # === Prepare missing list (row-by-row, only those absent from JSON) ===
    missing_rel = []
    for v in excel_missing:
        base_name = upper_png(os.path.basename(v))
        folder = infer_folder(base_name)
        rel = f"{folder}/{base_name}"
        missing_rel.append(rel)

    # === Check which missing images exist on disk ===
    present, missing = split_existing(missing_rel, base_dir)

    # === Summary before opening GUI ===
    num_existing = len(new_data.get("annotations", []))
    num_missing = len(missing_rel)
    print(f"\nSummary:")
    print(f"  Existing annotations in {new_json}: {num_existing}")
    print(f"  Excel images already annotated: {excel_annotated}")
    print(f"  Excel images still missing annotations: {num_missing}")
    print(f"  Missing images found on disk: {len(present)}")
    print(f"  Missing images not on disk: {len(missing)}")

    if missing:
        for p in missing[:10]:
            print("    -", p)
        if len(missing) > 10:
            print("    ...")
    # Uncomment to stop if missing
    # if missing:
    #     raise FileNotFoundError(f"{len(missing)} expected image files are missing on disk.")

    # === Launch GUI or exit ===
    if not present:
        print("\nNo new images to annotate. Exiting.")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {"images": [], "annotations": [], "categories": [{"id": 1, "name": "whale", "supercategory": "animal"}]},
                f, indent=2)
    else:
        print(f"\nLaunching annotator for {len(present)} images ...")
        root = tk.Tk()
        app = Annotator(root, image_paths=present, whale_csv_path=csv_path, output_path=out_json)
        root.mainloop()