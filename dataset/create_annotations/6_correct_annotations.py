import json
import copy
import math
from pathlib import Path
from typing import List, Tuple, Optional
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk

# =========================
# Config
# =========================
ANNOTATIONS_PATH = "new_annotations.json"           # input COCO
CORRECTED_PATH   = "corrected_annotations.json"     # output COCO
BASE_DIR = Path("..") / "whales_from_space"         # ../whales_from_space/<LocationYear>/<ImageFile>
DISPLAY_MS = 500                                    # auto-advance delay (ms) when not paused
UPSCALE_FACTOR = 6                                  # enlarge small images by this factor
WINDOW_TITLE = "Annot Viewer/Editor | ←/→ prev/next | R/E ±90 | A add poly | U undo | C clear | S save | Q quit"


# =========================
# Helpers
# =========================
def load_json(path: str) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: dict, path: str) -> None:
    """dump_json(obj,path) -> None: Write JSON indent=2 utf-8."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def anns_by_image(anns: List[dict]) -> dict:
    """anns_by_image(anns) -> {image_id: [anns...] }."""
    d = {}
    for a in anns:
        d.setdefault(a["image_id"], []).append(a)
    return d


def upscale_image(img: Image.Image, factor: int) -> Image.Image:
    """upscale_image(img,factor) -> Image: NEAREST scaling."""
    w, h = img.size
    return img.resize((w * factor, h * factor), resample=Image.NEAREST)


def draw_overlay(img: Image.Image, anns: List[dict], scale: float = 1.0) -> Image.Image:
    """draw_overlay(img,anns,scale) -> Image: polygons (green) + bboxes (red)."""
    dr = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i] * scale, seg[i + 1] * scale) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                dr.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = [v * scale for v in a["bbox"]]
            dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    return img


def rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    """rotate_point(x,y,cx,cy,deg) -> (x',y'): rotate around center."""
    a = math.radians(angle_deg)
    tx, ty = x - cx, y - cy
    return tx * math.cos(a) - ty * math.sin(a) + cx, tx * math.sin(a) + ty * math.cos(a) + cy


def rotate_bbox(bbox: List[float], cx: float, cy: float, angle_deg: float) -> List[float]:
    """rotate_bbox([x,y,w,h],cx,cy,deg) -> [x',y',w',h']."""
    x, y, w, h = bbox
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    rs = [rotate_point(px, py, cx, cy, angle_deg) for (px, py) in corners]
    xs, ys = zip(*rs)
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def rotate_segmentation(segmentation: List[List[float]], cx: float, cy: float, angle_deg: float) -> List[List[float]]:
    """rotate_segmentation(polylist,cx,cy,deg) -> polylist rotated."""
    out = []
    for poly in segmentation or []:
        if not poly:
            out.append(poly)
            continue
        coords = []
        for i in range(0, len(poly), 2):
            nx, ny = rotate_point(poly[i], poly[i + 1], cx, cy, angle_deg)
            coords.extend([round(nx, 3), round(ny, 3)])
        out.append(coords)
    return out


def rotate_annotations_inplace(anns: List[dict], img_w: int, img_h: int, angle_deg: float) -> None:
    """rotate_annotations_inplace(anns,w,h,deg) -> None: rotate bbox+seg around center."""
    cx, cy = img_w / 2.0, img_h / 2.0
    for a in anns:
        if "bbox" in a and len(a["bbox"]) == 4:
            a["bbox"] = [round(v, 3) for v in rotate_bbox(a["bbox"], cx, cy, angle_deg)]
        if a.get("segmentation"):
            a["segmentation"] = rotate_segmentation(a["segmentation"], cx, cy, angle_deg)


def bbox_from_points(pts: List[Tuple[float, float]]) -> List[float]:
    """bbox_from_points(pts) -> [x,y,w,h]."""
    xs, ys = zip(*pts)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def flat_from_points(pts: List[Tuple[float, float]]) -> List[float]:
    """flat_from_points(pts) -> [x1,y1,...]."""
    out = []
    for x, y in pts:
        out.extend([float(x), float(y)])
    return out


# =========================
# Viewer/Editor with Reannotation
# =========================
class AnnotationViewer:
    def __init__(self, root: tk.Tk, coco: dict):
        """__init__(root,coco) -> None: setup state/UI and working copies."""
        self.root = root
        self.coco = coco
        self.images = coco.get("images", [])
        self.by_img_orig = anns_by_image(coco.get("annotations", []))
        self.by_img_work = {k: copy.deepcopy(v) for k, v in self.by_img_orig.items()}
        self.idx = 0
        self.after_id: Optional[str] = None
        self.paused = False
        self.tk_img = None

        # drawing state
        self.reannot_mode = False
        self.draw_pts: List[Tuple[int, int]] = []
        self.draw_point_ids: List[int] = []
        self.draw_line_ids: List[int] = []
        self.status_text = tk.StringVar(value="")

        # --- layout ---
        top = tk.Frame(root)
        top.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(top, bg="black", cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        ctrl = tk.Frame(root)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="← Prev", command=self.prev_image).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Next →", command=self.next_image).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Rotate −90°", command=lambda: self._rotate_current(-90)).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Rotate +90°", command=lambda: self._rotate_current(+90)).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Start Reannotate (Replace)", command=self.start_reannotate_replace).pack(side="left", padx=8, pady=4)
        tk.Button(ctrl, text="Add Polygon", command=self.toggle_add_polygon).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Undo Point", command=self.undo_point).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Clear Current", command=self.clear_current_annotations).pack(side="left", padx=3, pady=4)
        tk.Button(ctrl, text="Save", command=self.save_now).pack(side="left", padx=10, pady=4)
        tk.Button(ctrl, text="Quit", command=self.quit).pack(side="left", padx=3, pady=4)

        self.status = tk.Label(root, anchor="w", textvariable=self.status_text)
        self.status.pack(fill="x")

        # keys
        root.bind("<space>", self.toggle_pause)
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.prev_image)
        root.bind("<r>", lambda e: self._rotate_current(+90))
        root.bind("<e>", lambda e: self._rotate_current(-90))
        root.bind("<a>", self.toggle_add_polygon)
        root.bind("<u>", self.undo_point)
        root.bind("<c>", self._clear_current_and_redraw)
        root.bind("<s>", self.save_now)
        root.bind("<q>", lambda e: self.quit())

        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.title(WINDOW_TITLE)

        # mouse for drawing
        self.canvas.bind("<Button-1>", self._on_left_click)   # add point in add-polygon mode
        self.canvas.bind("<Button-3>", self._on_right_click)  # finish polygon

        self.schedule_show(100)

    # ----- scheduling -----
    def schedule_show(self, delay_ms: int) -> None:
        """schedule_show(delay_ms) -> None: schedule render."""
        self.cancel_pending()
        self.after_id = self.root.after(delay_ms, self.show_current)

    def cancel_pending(self) -> None:
        """cancel_pending() -> None: cancel after()."""
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    # ----- controls -----
    def toggle_pause(self, _=None) -> None:
        """toggle_pause() -> None: pause/resume auto-advance."""
        self.paused = not self.paused
        self.root.title(WINDOW_TITLE + (" [PAUSED]" if self.paused else ""))
        if not self.paused:
            self.schedule_show(DISPLAY_MS)

    def next_image(self, _=None) -> None:
        """next_image() -> None: go next."""
        self._abort_drawing()
        self.idx = min(self.idx + 1, len(self.images) - 1)
        self.schedule_show(0)

    def prev_image(self, _=None) -> None:
        """prev_image() -> None: go prev."""
        self._abort_drawing()
        self.idx = max(self.idx - 1, 0)
        self.schedule_show(0)

    def start_reannotate_replace(self) -> None:
        """start_reannotate_replace() -> None: clear current annotations and enter reannot mode."""
        iid = self._current_image_id()
        if iid is None:
            return
        self.by_img_work[iid] = []  # replace
        self.reannot_mode = True
        self.status_text.set("Reannotate mode: L-click to add points, R-click to close polygon.")

    def toggle_add_polygon(self, _=None) -> None:
        """toggle_add_polygon() -> None: toggle add-poly mode (keeps existing)."""
        self.reannot_mode = not self.reannot_mode
        if self.reannot_mode:
            self.status_text.set("Add polygon: L-click to add points, R-click to close.")
        else:
            self.status_text.set("")

    def undo_point(self, _=None) -> None:
        """undo_point() -> None: remove last point while drawing."""
        if not self.reannot_mode or not self.draw_pts:
            return
        self.draw_pts.pop()
        if self.draw_point_ids:
            pid = self.draw_point_ids.pop()
            try:
                self.canvas.delete(pid)
            except Exception:
                pass
        if self.draw_line_ids:
            lid = self.draw_line_ids.pop()
            try:
                self.canvas.delete(lid)
            except Exception:
                pass

    def clear_current_annotations(self, _=None) -> None:
        """clear_current_annotations() -> None: clear all anns for current image."""
        iid = self._current_image_id()
        if iid is None:
            return
        self.by_img_work[iid] = []
        self._abort_drawing()
        self.schedule_show(0)

    def _clear_current_and_redraw(self, _=None) -> None:
        """_clear_current_and_redraw() -> None: keybinding wrapper."""
        self.clear_current_annotations()

    def save_now(self, _=None) -> None:
        """save_now() -> None: write corrected_annotations.json."""
        out = list(self._flatten_work_annotations())
        coco_out = {
            "images": self.images,
            "annotations": out,
            "categories": self.coco.get("categories", []),
            "licenses": self.coco.get("licenses", []),
        }
        dump_json(coco_out, CORRECTED_PATH)
        self.status_text.set(f"Saved: {CORRECTED_PATH} ({len(out)} anns)")

    def quit(self) -> None:
        """quit() -> None: save then exit."""
        self.save_now()
        self.cancel_pending()
        self.root.destroy()

    # ----- drawing -----
    def _on_left_click(self, event) -> None:
        """_on_left_click(event) -> None: add point when in add/reannot mode."""
        if not self.reannot_mode:
            return
        x = int(event.x / UPSCALE_FACTOR)
        y = int(event.y / UPSCALE_FACTOR)
        self.draw_pts.append((x, y))
        px, py = x * UPSCALE_FACTOR, y * UPSCALE_FACTOR
        pid = self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="yellow", outline="", tags=("drawpt",))
        self.draw_point_ids.append(pid)
        if len(self.draw_pts) > 1:
            x0, y0 = self.draw_pts[-2]
            lid = self.canvas.create_line(
                x0 * UPSCALE_FACTOR, y0 * UPSCALE_FACTOR, px, py,
                fill="#ffd966", width=2, tags=("drawline",)
            )
            self.draw_line_ids.append(lid)

    def _on_right_click(self, _event) -> None:
        """_on_right_click(event) -> None: close polygon and add annotation."""
        if not self.reannot_mode or len(self.draw_pts) < 3:
            return
        iid = self._current_image_id()
        if iid is None:
            self._abort_drawing()
            return
        flat = flat_from_points(self.draw_pts)
        bbox = bbox_from_points(self.draw_pts)
        new_ann = {
            "id": self._next_ann_id(),
            "image_id": iid,
            "category_id": 1,
            "bbox": [round(v, 3) for v in bbox],
            "area": round(bbox[2] * bbox[3], 3),
            "segmentation": [[round(v, 3) for v in flat]],
            "iscrowd": 0,
        }
        self.by_img_work.setdefault(iid, []).append(new_ann)
        self._abort_drawing()
        self.schedule_show(0)

    def _abort_drawing(self) -> None:
        """_abort_drawing() -> None: clear temp drawing aids and stop add mode (keeps mode)."""
        for pid in self.draw_point_ids:
            try:
                self.canvas.delete(pid)
            except Exception:
                pass
        for lid in self.draw_line_ids:
            try:
                self.canvas.delete(lid)
            except Exception:
                pass
        self.draw_pts.clear()
        self.draw_point_ids.clear()
        self.draw_line_ids.clear()

    # ----- internals -----
    def _current_image_id(self) -> Optional[int]:
        """_current_image_id() -> int|None: id for current image."""
        if not (0 <= self.idx < len(self.images)):
            return None
        return self.images[self.idx].get("id")

    def _next_ann_id(self) -> int:
        """_next_ann_id() -> int: next annotation id (max+1 over working set)."""
        m = 0
        for lst in self.by_img_work.values():
            for a in lst:
                if isinstance(a.get("id"), int):
                    m = max(m, a["id"])
        return m + 1

    def _rotate_current(self, angle: float) -> None:
        """_rotate_current(angle) -> None: rotate anns for current image around center."""
        iid = self._current_image_id()
        if iid is None:
            return
        im = self.images[self.idx]
        w, h = im.get("width"), im.get("height")
        if not (w and h):
            p = BASE_DIR / (im.get("file_name") or "")
            try:
                iw, ih = Image.open(p).size
                w, h = iw, ih
            except Exception:
                self.status_text.set("Warning: missing width/height; cannot rotate.")
                return
        rotate_annotations_inplace(self.by_img_work.setdefault(iid, []), w, h, angle)
        self.schedule_show(0)

    def _flatten_work_annotations(self):
        """_flatten_work_annotations() -> iter: yield annotations from working dict."""
        for _, lst in self.by_img_work.items():
            for a in lst:
                yield a

    # ----- rendering -----
    def show_current(self) -> None:
        """show_current() -> None: render image + current annotations."""
        if not (0 <= self.idx < len(self.images)):
            self.quit()
            return

        im = self.images[self.idx]
        rel = im.get("file_name", "")
        path = BASE_DIR / rel

        if not path.is_file():
            img = Image.new("RGB", (640, 360), (20, 20, 20))
            ImageDraw.Draw(img).text((10, 10), f"Missing: {rel}", fill=(255, 80, 80))
        else:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (640, 360), (20, 20, 20))
                ImageDraw.Draw(img).text((10, 10), f"Read error: {rel}", fill=(255, 80, 80))
            img = upscale_image(img, UPSCALE_FACTOR)
            img = draw_overlay(img, self.by_img_work.get(im["id"], []), scale=UPSCALE_FACTOR)

        w, h = img.size
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(width=w, height=h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.create_text(
            10, 20, anchor="nw", fill="white",
            text=f"{rel}  ({w//UPSCALE_FACTOR}x{h//UPSCALE_FACTOR})  [{self.idx+1}/{len(self.images)}]",
            font=("Arial", 14, "bold"),
        )
        if self.reannot_mode:
            self.canvas.create_text(
                10, 44, anchor="nw", fill="#ffd966",
                text="ADDING POLYGON: L-click to add points, R-click to close",
                font=("Arial", 12, "bold"),
            )

        if not self.paused and self.idx < len(self.images) - 1:
            self.idx += 1
            self.schedule_show(DISPLAY_MS)


# =========================
# Main
# =========================
def main() -> None:
    """main() -> None: run viewer/editor with reannotation and save on exit."""
    coco = load_json(ANNOTATIONS_PATH)
    root = tk.Tk()
    AnnotationViewer(root, coco)
    root.mainloop()


if __name__ == "__main__":
    main()
