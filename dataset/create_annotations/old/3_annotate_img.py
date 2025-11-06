import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from collections import OrderedDict
import pandas as pd

class Annotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Whale COCO Annotator with Metadata")
        self.root.geometry("1200x800")

        self.load_whale_info("2022_whalesfromspace/WhaleFromSpaceDB_Whales.csv")

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

        self.points = []
        self.point_circles = []
        self.temp_lines = []
        self.image_list = []
        self.image_index = 0
        self.annotations = []
        self.scale_factor = 1.0
        self.current_polygon_id = None
        self.draggable_points = []

        self.canvas.bind("<Button-1>", self.left_click)
        self.canvas.bind("<Button-3>", self.right_click)

        self.load_images()
        self.load_next_image()

    def load_whale_info(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            lookup_df = df[["BoxID/ImageChip", "NumWhale", "Certainty2"]].copy()
            self.whale_lookup = {
                f"{row['BoxID/ImageChip']}.PNG": (row["NumWhale"], row["Certainty2"])
                for _, row in lookup_df.iterrows()
            }
        except Exception as e:
            self.whale_lookup = {}
            print(f"Failed to load whale info: {e}")

    def load_images(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        self.folder_name = os.path.basename(folder)
        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_list.sort()

    def load_next_image(self):
        if self.image_index >= len(self.image_list):
            self.export_and_exit()
            return

        path = self.image_list[self.image_index]
        self.img = Image.open(path)
        max_dim = max(self.img.width, self.img.height)
        self.scale_factor = max(1, 768 // max_dim)

        display_img = self.img.resize((self.img.width * self.scale_factor,
                                       self.img.height * self.scale_factor),
                                      resample=Image.NEAREST)

        self.tk_img = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        filename = os.path.basename(path)
        num_whales, certainty = self.whale_lookup.get(filename, ("?", "?"))
        self.label.config(
            text=f"[{self.image_index+1}/{len(self.image_list)}] {filename} — Expected whales: {num_whales} ({certainty})"
        )

        self.points.clear()
        self.point_circles.clear()
        self.temp_lines.clear()
        self.current_polygon_id = None
        self.draggable_points.clear()

    def left_click(self, event):
        x, y = int(event.x / self.scale_factor), int(event.y / self.scale_factor)
        self.points.append((x, y))
        px, py = x * self.scale_factor, y * self.scale_factor

        point_id = self.canvas.create_oval(px-4, py-4, px+4, py+4, fill='green', tags='point')
        self.point_circles.append(point_id)

        if len(self.points) > 1:
            prev = self.points[-2]
            line_id = self.canvas.create_line(prev[0]*self.scale_factor, prev[1]*self.scale_factor,
                                              px, py, fill='#90ee90', width=2)
            self.temp_lines.append(line_id)

    def right_click(self, event):
        if len(self.points) < 3:
            messagebox.showwarning("Too Few Points", "You need at least 3 points for a polygon.")
            return

        screen_pts = [(x * self.scale_factor, y * self.scale_factor) for x, y in self.points]
        self.current_polygon_id = self.canvas.create_polygon(screen_pts, outline='lime', fill='', width=2)

        xs, ys = zip(*self.points)
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        self.canvas.create_rectangle(
            xmin * self.scale_factor, ymin * self.scale_factor,
            xmax * self.scale_factor, ymax * self.scale_factor,
            outline='red', width=2
        )

        self.annotations.append({
            "filename": os.path.basename(self.image_list[self.image_index]),
            "bbox": (xmin, ymin, xmax, ymax),
            "segmentation": self.points.copy()
        })

        self.draggable_points = list(self.points)
        for i, cid in enumerate(self.point_circles):
            self.canvas.tag_bind(cid, "<B1-Motion>", lambda e, idx=i: self.move_point(e, idx))

        self.points.clear()
        self.point_circles.clear()
        self.temp_lines.clear()

    def move_point(self, event, index):
        x_new, y_new = int(event.x / self.scale_factor), int(event.y / self.scale_factor)
        self.draggable_points[index] = (x_new, y_new)

        self.canvas.coords(self.canvas.find_withtag('point')[index],
                           x_new*self.scale_factor - 4, y_new*self.scale_factor - 4,
                           x_new*self.scale_factor + 4, y_new*self.scale_factor + 4)

        screen_pts = [(x * self.scale_factor, y * self.scale_factor) for x, y in self.draggable_points]
        self.canvas.coords(self.current_polygon_id, *sum(screen_pts, ()))

    def clear_polygon(self):
        self.points.clear()
        for cid in self.point_circles:
            self.canvas.delete(cid)
        for lid in self.temp_lines:
            self.canvas.delete(lid)
        if self.current_polygon_id:
            self.canvas.delete(self.current_polygon_id)
        self.point_circles.clear()
        self.temp_lines.clear()
        self.current_polygon_id = None

    def save_and_next(self):
        self.image_index += 1
        self.load_next_image()

    def export_and_exit(self):
        if not self.annotations:
            messagebox.showinfo("Done", "No annotations made.")
            self.root.quit()
            return

        images = []
        annotations = []
        categories = [{
            "id": 1,
            "name": "whale",
            "supercategory": "animal"
        }]
        annotation_id = 1
        image_id_map = {}
        round1 = lambda x: round(float(x), 1)

        for image_id, img_path in enumerate(self.image_list, 1):
            img = Image.open(img_path)
            filename = os.path.basename(img_path)
            file_name_with_folder = f"{self.folder_name}/{filename}"
            image_id_map[filename] = image_id

            image_obj = OrderedDict()
            image_obj["id"] = image_id
            image_obj["license"] = 1
            image_obj["file_name"] = file_name_with_folder
            image_obj["height"] = img.height
            image_obj["width"] = img.width
            image_obj["date_captured"] = datetime.utcnow().isoformat() + "+00:00"
            image_obj["extra"] = {"name": filename}
            images.append(image_obj)

        for ann in self.annotations:
            filename = ann["filename"]
            image_id = image_id_map.get(filename)
            if image_id is None:
                continue

            segmentation = [round1(coord) for pt in ann["segmentation"] for coord in pt]
            x, y, x2, y2 = ann["bbox"]
            width = x2 - x
            height = y2 - y

            ann_obj = OrderedDict()
            ann_obj["id"] = annotation_id
            ann_obj["image_id"] = image_id
            ann_obj["category_id"] = 1
            ann_obj["bbox"] = [round1(x), round1(y), round1(width), round1(height)]
            ann_obj["area"] = round1(width * height)
            ann_obj["segmentation"] = [segmentation]
            ann_obj["iscrowd"] = 0
            annotations.append(ann_obj)
            annotation_id += 1

        coco_dict = OrderedDict()
        coco_dict["images"] = images
        coco_dict["annotations"] = annotations
        coco_dict["categories"] = categories

        output_filename = f"{self.folder_name}.json"
        with open(output_filename, "w") as f:
            json.dump(coco_dict, f, indent=2, sort_keys=False)

        messagebox.showinfo("Saved", f"Annotations saved to {output_filename}")
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = Annotator(root)
    root.mainloop()
