import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk

# =========================
# Config
# =========================

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

ALLOWED_modes = {
    "patch_raw_255",
    "patch_raw_rot_255",
    "texture_nadir_255",
    "radiance_nadir_255",
    "radiance_nadir_npy",
    "reflection_nadir_255",
    "reflection_nadir_npy",
    "texture_offnadir_255",
    "radiance_offnadir_255",
    "radiance_offnadir_npy",
    "reflection_offnadir_255",
    "reflection_offnadir_npy",
}

mode = "texture_offnadir_255"

DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "create_dataset" / mode
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / mode / "final_annotations.json"

DISPLAY_MS = 500
DISPLAY_ZOOM = 6  # VISUAL zoom only; annotations are still drawn in original patch pixels
WINDOW_TITLE = "Annotation Preview  |  Space: pause/resume  |  ←/→: prev/next  |  Q: quit"

# =========================
# Helpers
# =========================
def load_json(path: str | Path) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def anns_by_image(anns: list) -> dict:
    """anns_by_image(anns) -> dict: Map image_id -> list of anns."""
    d = {}
    for a in anns:
        d.setdefault(a["image_id"], []).append(a)
    return d


def draw_overlay(img: Image.Image, anns: list) -> Image.Image:
    """draw_overlay(img,anns) -> Image: Draw polygons/bboxes in ORIGINAL patch pixel coords."""
    draw = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
        if "bbox" in a and isinstance(a["bbox"], (list, tuple)) and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
    return img


def make_display_image(img: Image.Image, zoom: int) -> Image.Image:
    """make_display_image(img,zoom) -> Image: Upscale for display only (keeps crisp pixels)."""
    if zoom <= 1:
        return img
    w, h = img.size
    return img.resize((w * zoom, h * zoom), resample=Image.NEAREST)


# =========================
# Viewer
# =========================
class AnnotationViewer:
    def __init__(self, root: tk.Tk, images: list, anns_by_img: dict):
        """__init__(root,images,anns_by_img) -> None: Setup UI/state."""
        self.root = root
        self.images = images
        self.anns_by_img = anns_by_img
        self.idx = 0
        self.after_id = None
        self.paused = False
        self.tk_img = None

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        root.bind("<space>", self.toggle_pause)
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.prev_image)
        root.bind("<q>", lambda e: self.quit())
        root.bind("<Q>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.schedule_show(100)

    def schedule_show(self, delay_ms: int) -> None:
        """schedule_show(delay_ms) -> None: Schedule render."""
        self.cancel_pending()
        self.after_id = self.root.after(delay_ms, self.show_current)

    def cancel_pending(self) -> None:
        """cancel_pending() -> None: Cancel pending callback."""
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def toggle_pause(self, _evt=None) -> None:
        """toggle_pause() -> None: Pause/resume auto-advance."""
        self.paused = not self.paused
        self.root.title(WINDOW_TITLE + ("  [PAUSED]" if self.paused else ""))
        if not self.paused:
            self.schedule_show(DISPLAY_MS)

    def next_image(self, _evt=None) -> None:
        """next_image() -> None: Next image."""
        self.idx = min(self.idx + 1, len(self.images) - 1)
        self.schedule_show(0)

    def prev_image(self, _evt=None) -> None:
        """prev_image() -> None: Previous image."""
        self.idx = max(self.idx - 1, 0)
        self.schedule_show(0)

    def quit(self) -> None:
        """quit() -> None: Close window."""
        self.cancel_pending()
        self.root.destroy()

    def show_current(self) -> None:
        """show_current() -> None: Render current image+anns; upscale for DISPLAY only."""
        if not (0 <= self.idx < len(self.images)):
            self.quit()
            return

        im = self.images[self.idx]
        rel = im.get("file_name", "")
        path = BASE_DIR / rel

        if not path.is_file():
            print(f"[Missing] {rel}")
            base = Image.new("RGB", (640, 360), color=(20, 20, 20))
            ImageDraw.Draw(base).text((10, 10), f"Missing: {rel}", fill=(255, 80, 80))
        else:
            try:
                base = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[Error reading {rel}]: {e}")
                base = Image.new("RGB", (640, 360), color=(20, 20, 20))
                ImageDraw.Draw(base).text((10, 10), f"Read error: {rel}", fill=(255, 80, 80))

            base = draw_overlay(base, self.anns_by_img.get(im["id"], []))

        raw_w, raw_h = base.size
        disp = make_display_image(base, DISPLAY_ZOOM)
        disp_w, disp_h = disp.size

        self.tk_img = ImageTk.PhotoImage(disp)

        self.canvas.delete("all")
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.canvas.create_text(
            10, 20, anchor="nw", fill="white",
            text=f"{rel}  raw=({raw_w}x{raw_h})  display=({disp_w}x{disp_h})  zoom={DISPLAY_ZOOM}x  [{self.idx+1}/{len(self.images)}]",
            font=("Arial", 14, "bold"),
        )

        # Make the window big enough to actually show the zoomed image
        self.root.geometry(f"{disp_w}x{disp_h + 40}")

        if not self.paused and self.idx < len(self.images) - 1:
            self.idx += 1
            self.schedule_show(DISPLAY_MS)


# =========================
# Main
# =========================
def show_data() -> None:
    """show_data() -> None: Run viewer."""
    if mode not in ALLOWED_modes:
        raise ValueError(f"mode must be one of {sorted(ALLOWED_modes)}")
    if not ANNOTATIONS_PATH.is_file():
        raise FileNotFoundError(f"Missing COCO json: {ANNOTATIONS_PATH}")

    coco = load_json(ANNOTATIONS_PATH)
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    by_img = anns_by_image(anns)

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    AnnotationViewer(root, images, by_img)
    root.mainloop()


if __name__ == "__main__":
    show_data()
