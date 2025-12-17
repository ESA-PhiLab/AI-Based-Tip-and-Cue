import json
import os
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import mitsuba as mi
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from offnadir_imaging.rendering import generate_image
from offnadir_imaging.create_DEM.create_dummy_DEM import get_DEM
from offnadir_imaging.create_DEM.convert_DEM import convert_DEM
from offnadir_imaging.functions.get_satellite_data import get_satellite, get_spatial_res, get_band_data
from offnadir_imaging.functions.convert_reference_frames import get_ecef_from_lat_lon, compute_max_glint_satellite_ecef
from offnadir_imaging.functions.intermediate_functions import get_scene_characteristics, is_dark_from_sun_dir


# =========================
# Config
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = Path("dataset")
BASE_DIR = DATASET_PATH / "whales_from_space"
ANNOTATIONS_PATH = DATASET_PATH / "create_dataset" / "final_annotations.json"
CSV_PATH = BASE_DIR / "WhaleFromSpaceDB_Whales.csv"

IMG_FILE = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"
DATETIME_UTC = datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)

SAT_LAT, SAT_LON, SAT_ALT = 58.0, -5.0, 617000.0
TGT_LAT, TGT_LON, TGT_ALT = 53.0, 0.0, 0.0

BOOLS = {
    "plot_3d": False,
    "plot_result": False,
    "max_glint": False,
    "print_values": True,
    "crop_black_border": False,   # IGNORE CROPPING (as requested)
    "generate_radiation": True,
}

WAVE_PROPERTIES = {"wind_speed": 10.0, "num_waves": 50, "wave_min": 0.05, "wave_max": 0.5}
RENDER_RESOLUTION = 124
SAMPLE_COUNT = 512
DEM_SEED = 42

# ID-pass mapping (annotation translation) happens BEFORE generate_image
ID_RENDER_RES = RENDER_RESOLUTION
ID_SPP = 1
ID_POINT_RADIUS_PX = 4  # increase if points disappear at steep angles


# =========================
# COCO helpers
# =========================
def load_json(path: Path) -> dict:
    """load_json(path) -> dict: Read JSON utf-8."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def anns_by_image(anns: list) -> dict:
    """anns_by_image(anns) -> dict: Map image_id -> list[ann]."""
    d = {}
    for a in anns:
        d.setdefault(a["image_id"], []).append(a)
    return d


# =========================
# Overlay helpers
# =========================
def draw_overlay(img: Image.Image, anns: list) -> Image.Image:
    """draw_overlay(img,anns) -> Image: Draw polygons and bboxes."""
    draw = ImageDraw.Draw(img)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    return img

def draw_overlay_from_polys(img: Image.Image, polys_xy: list, boxes_xywh: list) -> Image.Image:
    """draw_overlay_from_polys(img,polys_xy,boxes_xywh) -> Image: Draw projected polys/boxes."""
    draw = ImageDraw.Draw(img)
    for pts in polys_xy:
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)
    for (x, y, w, h) in boxes_xywh:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    return img

def to_uint8_rgb(arr) -> np.ndarray:
    """to_uint8_rgb(arr) -> np.ndarray: Clamp/cast to uint8 RGB array."""
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


# =========================
# ID encoding helpers
# =========================
def id_to_rgb_u8(i: int) -> tuple[int, int, int]:
    """id_to_rgb_u8(i) -> (r,g,b): Encode 1..16,777,215 into RGB."""
    i = int(i) & 0xFFFFFF
    return (int((i >> 16) & 255), int((i >> 8) & 255), int(i & 255))

def rgb_u8_to_id(rgb: np.ndarray) -> np.ndarray:
    """rgb_u8_to_id(rgb) -> np.ndarray: Decode RGB uint8 image to int32 IDs."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    return (r << 16) | (g << 8) | b

def id_to_display_color(i: int) -> tuple[int, int, int]:
    """id_to_display_color(i) -> (r,g,b): Deterministic visible color for plotting only."""
    x = int(i) * 2654435761 & 0xFFFFFFFF
    r = (x >> 16) & 255
    g = (x >> 8) & 255
    b = x & 255
    # avoid black
    if r + g + b < 60:
        r = (r + 120) & 255
        g = (g + 60) & 255
    return int(r), int(g), int(b)


# =========================
# Build point-ID texture from COCO
# =========================
def build_point_id_texture(anns: list, tex_w: int, tex_h: int, radius_px: int) -> tuple[np.ndarray, dict, np.ndarray]:
    """build_point_id_texture(anns,tex_w,tex_h,radius_px) -> (tex_rgb01,point_meta,debug_vis_u8)."""
    id_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    vis_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    id_draw = ImageDraw.Draw(id_img)
    vis_draw = ImageDraw.Draw(vis_img)

    next_id = 1
    point_meta = {}  # pid -> {"ann_id":..., "kind":"poly|bbox", "idx":...}

    def stamp(x: float, y: float, ann_id: int, kind: str, idx: int) -> int:
        nonlocal next_id
        pid = next_id
        next_id += 1

        r, g, b = id_to_rgb_u8(pid)
        rc, gc, bc = id_to_display_color(pid)

        cx = int(round(x))
        cy = int(round(y))
        x0, y0 = cx - radius_px, cy - radius_px
        x1, y1 = cx + radius_px, cy + radius_px

        id_draw.ellipse([x0, y0, x1, y1], fill=(r, g, b), outline=None)
        vis_draw.ellipse([x0, y0, x1, y1], fill=(rc, gc, bc), outline=None)

        point_meta[pid] = {"ann_id": int(ann_id), "kind": kind, "idx": int(idx)}
        return pid

    for a in anns:
        aid = int(a.get("id", -1))

        for seg in a.get("segmentation", []):
            if not seg:
                continue
            vi = 0
            for i in range(0, len(seg), 2):
                stamp(seg[i], seg[i + 1], aid, "poly", vi)
                vi += 1

        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for ci, (cx, cy) in enumerate(corners):
                stamp(cx, cy, aid, "bbox", ci)

    id_u8 = np.asarray(id_img, dtype=np.uint8)
    vis_u8 = np.asarray(vis_img, dtype=np.uint8)
    id_rgb01 = id_u8.astype(np.float32) / 255.0
    return id_rgb01, point_meta, vis_u8


# =========================
# Mitsuba ID render (unshaded emission)
# =========================
def render_id_pass(dem_obj_path: str, id_texture_rgb01: np.ndarray, to_world_scene, to_world_sensor, fov_deg: float, res: int, spp: int) -> np.ndarray:
    """render_id_pass(dem_obj_path,id_texture_rgb01,to_world_scene,to_world_sensor,fov_deg,res,spp) -> np.ndarray."""
    mi.set_variant("llvm_ad_rgb")

    tex = {
        "type": "bitmap",
        "data": id_texture_rgb01,
        "raw": True,
        "wrap_mode": "clamp",
        "filter_type": "nearest",
    }

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 1},
        "earth_surface": {
            "type": "obj",
            "filename": dem_obj_path,
            "to_world": to_world_scene,
            "bsdf": {"type": "diffuse", "reflectance": 0.0},
            "emitter": {"type": "area", "radiance": tex},
        },
        "sensor": {
            "type": "perspective",
            "to_world": to_world_sensor,
            "fov": float(fov_deg),
            "far_clip": 1e8,
            "film": {
                "type": "hdrfilm",
                "width": int(res),
                "height": int(res),
                "rfilter": {"type": "box"},
                "sample_border": True,
                "compensate": False,
            },
            "sampler": {"type": "independent", "sample_count": int(max(1, spp))},
        },
    })

    img = np.array(mi.render(scene), copy=False)  # float RGB
    img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img_u8


# =========================
# Decode off-nadir ID image -> point positions -> rebuild polygons/boxes
# =========================
def decode_point_positions(id_img_u8: np.ndarray) -> dict:
    """decode_point_positions(id_img_u8) -> dict: pid -> (x,y) centroid."""
    ids = rgb_u8_to_id(id_img_u8)
    unique = np.unique(ids)
    pos = {}
    for pid in unique:
        if pid == 0:
            continue
        ys, xs = np.where(ids == pid)
        if xs.size == 0:
            continue
        pos[int(pid)] = (float(xs.mean()), float(ys.mean()))
    return pos

def rebuild_annotations_from_points(anns: list, point_meta: dict, point_pos: dict) -> tuple[list, list]:
    """rebuild_annotations_from_points(anns,point_meta,point_pos) -> (polys_xy,boxes_xywh)."""
    by_ann_poly = {}
    by_ann_bbox = {}
    for pid, meta in point_meta.items():
        aid = meta["ann_id"]
        if meta["kind"] == "poly":
            by_ann_poly.setdefault(aid, []).append((meta["idx"], pid))
        else:
            by_ann_bbox.setdefault(aid, []).append((meta["idx"], pid))

    polys_xy = []
    boxes_xywh = []

    for a in anns:
        aid = int(a.get("id", -1))

        if aid in by_ann_poly:
            items = sorted(by_ann_poly[aid], key=lambda t: t[0])
            pts = []
            for _, pid in items:
                if pid in point_pos:
                    pts.append(point_pos[pid])
            if len(pts) >= 3:
                polys_xy.append([(int(round(x)), int(round(y))) for (x, y) in pts])

        if aid in by_ann_bbox:
            items = sorted(by_ann_bbox[aid], key=lambda t: t[0])
            corners = []
            for _, pid in items:
                if pid in point_pos:
                    corners.append(point_pos[pid])
            if len(corners) == 4:
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                boxes_xywh.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))

    return polys_xy, boxes_xywh

def make_offnadir_points_debug_image(res: int, point_pos: dict) -> np.ndarray:
    """make_offnadir_points_debug_image(res,point_pos) -> np.ndarray: colored dots on black for decoded points."""
    img = Image.new("RGB", (res, res), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for pid, (x, y) in point_pos.items():
        r, g, b = id_to_display_color(pid)
        cx = int(round(x))
        cy = int(round(y))
        d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(r, g, b), outline=None)
    return np.asarray(img, dtype=np.uint8)


# =========================
# Main
# =========================
def main() -> None:
    """main() -> None: 1) Translate annotations via ID back-trace, 2) run generate_image, 3) plot progress."""
    coco = load_json(ANNOTATIONS_PATH)
    images = coco.get("images", [])
    anns_all = coco.get("annotations", [])
    by_img = anns_by_image(anns_all)

    im = next((x for x in images if x.get("file_name") == IMG_FILE), None)
    if im is None:
        raise FileNotFoundError(f"IMG_FILE not found in COCO images: {IMG_FILE}")

    img_path = BASE_DIR / IMG_FILE
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing image file on disk: {img_path}")

    anns = by_img.get(im["id"], [])

    # ---- load original image (no crop) ----
    orig_rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.uint8)
    tex_h, tex_w = orig_rgb.shape[:2]
    orig_overlay = draw_overlay(Image.fromarray(orig_rgb, mode="RGB").copy(), anns)

    print(f"[1/5] Loaded original image {tex_w}x{tex_h}, annotations={len(anns)}")

    # ---- build DEM mesh (needed for mapping) ----
    satellite = get_satellite(str(img_path), str(CSV_PATH))
    gsd = get_spatial_res(str(img_path), str(CSV_PATH))

    dem_folder = main_path / "offnadir_imaging" / "create_DEM"
    dem_tiff_path = str(dem_folder / "input_dem_WV.tiff")
    dem_obj_path = str(dem_folder / "dem_mesh_WV.obj")

    print("[2/5] Building synthetic DEM + mesh...")
    get_DEM(str(img_path), dem_tiff_path, gsd, WAVE_PROPERTIES, random_seed=DEM_SEED, waves=True, curvature=True, plot_DEM=False)
    convert_DEM(str(img_path), dem_tiff_path, dem_obj_path, gsd, scale_km=False, print_output=False, plot_DEM=False)

    # keep pipeline init consistent
    _ = get_band_data(satellite, str(main_path / "offnadir_imaging" / "spd_files"))

    # ---- compute camera geometry (same math as pipeline) ----
    satellite_ecef, target_ecef, sun_ecef = get_ecef_from_lat_lon(
        SAT_LAT, SAT_LON, SAT_ALT,
        TGT_LAT, TGT_LON, TGT_ALT,
        DATETIME_UTC,
    )
    if BOOLS["max_glint"]:
        satellite_ecef = compute_max_glint_satellite_ecef(target_ecef, sun_ecef, glint_distance_m=700 * 10**3)

    is_dark, *_ = is_dark_from_sun_dir(
        target_ecef=target_ecef,
        sun_ecef=sun_ecef,
        threshold_deg=-18.0,
        model="wgs84",
        dir_type="target_to_sun",
    )
    if is_dark:
        raise RuntimeError("Dark hours, no image possible")

    satellite_local, target_local, _sun_dir, fov_deg, _off_nadir_rad, azimuth_rad = get_scene_characteristics(
        satellite_ecef, target_ecef, sun_ecef, tex_h, tex_w, gsd
    )

    mi.set_variant("llvm_ad_rgb")

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-azimuth_rad),
    )
    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    to_world_sensor_off = mi.ScalarTransform4f().look_at(
        origin=satellite_local,
        target=target_local,
        up=[0, 0, 1],
    )

    print(f"[3/5] Camera ready. FOV={fov_deg:.6f} deg, azimuth={azimuth_rad * 180 / math.pi:.2f} deg")

    # ---- build ID texture and render ID pass (translation) ----
    print("[4/5] Creating ID texture from annotation points...")
    id_tex_rgb01, point_meta, id_vis_u8 = build_point_id_texture(
        anns=anns,
        tex_w=tex_w,
        tex_h=tex_h,
        radius_px=ID_POINT_RADIUS_PX,
    )

    total_points = len(point_meta)
    print(f"      Stamped {total_points} points (poly vertices + bbox corners). Rendering off-nadir ID pass...")

    id_off_u8 = render_id_pass(
        dem_obj_path=dem_obj_path,
        id_texture_rgb01=id_tex_rgb01,
        to_world_scene=to_world_scene,
        to_world_sensor=to_world_sensor_off,
        fov_deg=fov_deg,
        res=ID_RENDER_RES,
        spp=ID_SPP,
    )

    point_pos = decode_point_positions(id_off_u8)
    found_points = len(point_pos)
    print(f"      Decoded {found_points}/{total_points} points in off-nadir render ({100.0 * found_points / max(1,total_points):.1f}%).")

    polys_off, boxes_off = rebuild_annotations_from_points(anns, point_meta, point_pos)

    off_points_vis = make_offnadir_points_debug_image(ID_RENDER_RES, point_pos)

    # Progress plot: show mapping artifacts BEFORE generate_image
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1).imshow(orig_overlay); plt.axis("off"); plt.title("Original + annotation")
    fig.add_subplot(2, 3, 2).imshow(id_vis_u8); plt.axis("off"); plt.title("Original point IDs (debug colors)")
    fig.add_subplot(2, 3, 3).imshow(id_off_u8); plt.axis("off"); plt.title("Off-nadir ID pass (encoded)")
    fig.add_subplot(2, 3, 5).imshow(off_points_vis); plt.axis("off"); plt.title("Decoded off-nadir points (debug colors)")
    plt.tight_layout()
    plt.show()

    # ---- now run generate_image (after translation) ----
    print("[5/5] Running generate_image AFTER annotation translation...")
    sensor_characteristics = {"resolution": RENDER_RESOLUTION, "sample_count": SAMPLE_COUNT, "GSD": gsd}

    DN255_offnadir, DN255_sunglint, _rad_sunglint, DN255_combined = generate_image(
        str(img_path),
        satellite,
        SAT_LAT, SAT_LON, SAT_ALT,
        TGT_LAT, TGT_LON, TGT_ALT,
        DATETIME_UTC,
        sensor_characteristics,
        WAVE_PROPERTIES,
        BOOLS,
        DEM_SEED,
    )
    if DN255_offnadir is None:
        raise RuntimeError("Renderer returned None (dark hours).")

    offnadir_u8 = to_uint8_rgb(DN255_offnadir)
    sunglint_u8 = to_uint8_rgb(DN255_sunglint)
    combined_u8 = to_uint8_rgb(DN255_combined)

    off_overlay = draw_overlay_from_polys(Image.fromarray(offnadir_u8, mode="RGB").copy(), polys_off, boxes_off)
    comb_overlay = None
    if combined_u8 is not None:
        comb_overlay = draw_overlay_from_polys(Image.fromarray(combined_u8, mode="RGB").copy(), polys_off, boxes_off)

    fig2 = plt.figure(figsize=(18, 10))
    fig2.add_subplot(1, 4, 1).imshow(orig_overlay); plt.axis("off"); plt.title("original + annotation")
    fig2.add_subplot(1, 4, 2).imshow(off_overlay);  plt.axis("off"); plt.title("off-nadir + projected annotation")
    fig2.add_subplot(1, 4, 3).imshow(sunglint_u8 if sunglint_u8 is not None else np.zeros_like(offnadir_u8)); plt.axis("off"); plt.title("sun glint")
    fig2.add_subplot(1, 4, 4).imshow(comb_overlay if comb_overlay is not None else np.zeros_like(offnadir_u8)); plt.axis("off"); plt.title("combined + projected")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
