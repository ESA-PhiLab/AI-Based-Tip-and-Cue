import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
import os

# =========================
# Config
# =========================

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = "dataset"
DATASET_PATH = Path("dataset")
mode = "reflection_offnadir_glint_255"

BASE_DIR = DATASET_PATH / "create_dataset" / "0_merged" / mode
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "0_merged" / mode / "final_annotations_repaired.json"


DISPLAY_MS = 500                                    # auto-advance delay (ms) when not paused
UPSCALE_FACTOR = 6                                  # enlarge small images by this factor
WINDOW_TITLE = "Annotation Preview  |  Space: pause/resume  |  ←/→: prev/next  |  Q: quit"


# =========================
# Helpers
# =========================
def load_json(path: str) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def anns_by_image(anns: list) -> dict:
    """anns_by_image(anns) -> {image_id: [anns...] }."""
    d = {}
    for a in anns: d.setdefault(a["image_id"], []).append(a)
    return d

def upscale_image(img: Image.Image, factor: int) -> Image.Image:
    """upscale_image(img,factor) -> Image: NEAREST upscaling for crisp polygons."""
    w, h = img.size
    return img.resize((w * factor, h * factor), resample=Image.NEAREST)

def draw_overlay(img: Image.Image, anns: list, scale: float = 1.0) -> Image.Image:
    """draw_overlay(img,anns,scale) -> Image: draw polygons and bboxes."""
    draw = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg: continue
            pts = [(seg[i] * scale, seg[i+1] * scale) for i in range(0, len(seg), 2)]
            if len(pts) >= 3: draw.line(pts + [pts[0]], fill=(0,255,0), width=2)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            x, y, w, h = x*scale, y*scale, w*scale, h*scale
            draw.rectangle([x, y, x+w, y+h], outline=(255,0,0), width=2)
    return img


# =========================
# Viewer
# =========================
class AnnotationViewer:
    def __init__(self, root: tk.Tk, images: list, anns_by_img: dict):
        """__init__(root,images,anns_by_img) -> None: setup UI/state."""
        self.root = root
        self.images = images
        self.anns_by_img = anns_by_img
        self.idx = 0
        self.after_id = None
        self.paused = False
        self.tk_img = None

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # key bindings
        root.bind("<space>", self.toggle_pause)
        root.bind("<Right>", self.next_image)
        root.bind("<Left>", self.prev_image)
        root.bind("<q>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.schedule_show(100)  # initial render

    # ----- scheduling helpers -----
    def schedule_show(self, delay_ms: int) -> None:
        """schedule_show(delay_ms) -> None: schedule self.show_current after delay."""
        self.cancel_pending()
        self.after_id = self.root.after(delay_ms, self.show_current)

    def cancel_pending(self) -> None:
        """cancel_pending() -> None: cancel any pending after() callback."""
        if self.after_id is not None:
            try: self.root.after_cancel(self.after_id)
            except Exception: pass
            self.after_id = None

    # ----- playback controls -----
    def toggle_pause(self, _evt=None) -> None:
        """toggle_pause() -> None: pause/resume auto-advance."""
        self.paused = not self.paused
        self.root.title(WINDOW_TITLE + ("  [PAUSED]" if self.paused else ""))
        if not self.paused:
            self.schedule_show(DISPLAY_MS)

    def next_image(self, _evt=None) -> None:
        """next_image() -> None: advance by one and show immediately."""
        self.idx = min(self.idx + 1, len(self.images) - 1)
        self.schedule_show(0)

    def prev_image(self, _evt=None) -> None:
        """prev_image() -> None: go back by one and show immediately."""
        self.idx = max(self.idx - 1, 0)
        self.schedule_show(0)

    def quit(self) -> None:
        """quit() -> None: close window."""
        self.cancel_pending()
        self.root.destroy()

    # ----- rendering -----
    def show_current(self) -> None:
        """show_current() -> None: render image+ann at current index; optionally auto-advance."""
        if not (0 <= self.idx < len(self.images)):
            self.quit()
            return

        im = self.images[self.idx]
        rel = im.get("file_name", "")
        path = BASE_DIR / rel

        if not path.is_file():
            print(f"[Missing] {rel}")
            # draw placeholder
            img = Image.new("RGB", (640, 360), color=(20, 20, 20))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Missing: {rel}", fill=(255, 80, 80))
        else:
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[Error reading {rel}]: {e}")
                img = Image.new("RGB", (640, 360), color=(20, 20, 20))
                ImageDraw.Draw(img).text((10, 10), f"Read error: {rel}", fill=(255, 80, 80))

            img = upscale_image(img, UPSCALE_FACTOR)
            img = draw_overlay(img, self.anns_by_img.get(im["id"], []), scale=UPSCALE_FACTOR)

        w, h = img.size
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(width=w, height=h)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.create_text(10, 20, anchor="nw", fill="white",
                                text=f"{rel}  ({w}x{h})  [{self.idx+1}/{len(self.images)}]",
                                font=("Arial", 14, "bold"))

        # auto-advance if not paused and not at the end
        if not self.paused and self.idx < len(self.images) - 1:
            self.idx += 1
            self.schedule_show(DISPLAY_MS)

# =========================
# Main
# =========================
def show_data() -> None:
    """main() -> None: run viewer."""
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
