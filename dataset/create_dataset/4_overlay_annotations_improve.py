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
# IMG_FILE = "Ignacio2017/Ignacio_GW_WV3_PS_20170220_B58.PNG"
DATETIME_UTC = datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)

SAT_LAT, SAT_LON, SAT_ALT = 58.0, -5.0, 617000.0
TGT_LAT, TGT_LON, TGT_ALT = 53.0, 0.0, 0.0

BOOLS = {
    "plot_3d": False,
    "plot_result": False,
    "max_glint": False,
    "print_values": True,
    "crop_black_border": False,  # ignore cropping
    "generate_radiation": True,
    "generate_nadir": False
}

WAVE_PROPERTIES = {"wind_speed": 10.0, "num_waves": 50, "wave_min": 0.05, "wave_max": 0.5}
RENDER_RESOLUTION = 124
SAMPLE_COUNT = 512
DEM_SEED = 42

# Segmentation translation:
# Render mask at higher internal resolution then downsample back to RENDER_RESOLUTION
MASK_SUPERSAMPLE = 2          # 2x is usually enough
MASK_SPP = 1                  # keep 1, we do supersampling via resolution
MASK_FILTER = "bilinear"      # crucial for avoiding holes
MASK_CLOSE_RADIUS = 2         # try 2..4

# BBox translation:
ID_RENDER_RES = RENDER_RESOLUTION
ID_SPP = 1
ID_POINT_RADIUS_PX = 4        # in original image pixels (texture space)
ID_FILTER = "nearest"         # keep IDs sharp


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
# Drawing helpers
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
                draw.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
    return img

def draw_overlay_from_polys(img: Image.Image, polys_xy: list, boxes_xywh: list) -> Image.Image:
    """draw_overlay_from_polys(img,polys_xy,boxes_xywh) -> Image: Draw projected polys/boxes."""
    draw = ImageDraw.Draw(img)
    for pts in polys_xy:
        if len(pts) >= 3:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=1)
    for (x, y, w, h) in boxes_xywh:
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
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
# Mitsuba emission render
# =========================
def render_emission_texture(dem_obj_path: str, tex_rgb01: np.ndarray, to_world_scene, to_world_sensor, fov_deg: float, res: int, spp: int, filter_type: str) -> np.ndarray:
    """render_emission_texture(dem_obj_path,tex_rgb01,to_world_scene,to_world_sensor,fov_deg,res,spp,filter_type) -> np.ndarray."""
    mi.set_variant("cuda_ad_rgb")

    tex = {
        "type": "bitmap",
        "data": mi.TensorXf(tex_rgb01.astype(np.float32)),
        "raw": True,
        "wrap_mode": "clamp",
        "filter_type": str(filter_type),
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

    img = np.array(mi.render(scene), copy=False)
    img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img_u8


# =========================
# Segmentation: mask build/render/postprocess/contour
# =========================
def build_binary_mask_texture(anns: list, tex_w: int, tex_h: int) -> tuple[np.ndarray, np.ndarray]:
    """build_binary_mask_texture(anns,tex_w,tex_h) -> (mask_rgb01,mask_u8)."""
    m = Image.new("L", (tex_w, tex_h), 0)
    d = ImageDraw.Draw(m)
    for a in anns:
        for seg in a.get("segmentation", []):
            if not seg:
                continue
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            if len(pts) >= 3:
                d.polygon(pts, fill=255, outline=255)
    mask_u8 = np.asarray(m, dtype=np.uint8)
    mask_rgb01 = np.stack([mask_u8, mask_u8, mask_u8], axis=-1).astype(np.float32) / 255.0
    return mask_rgb01, mask_u8

def threshold_mask(off_mask_u8: np.ndarray, thr: int) -> np.ndarray:
    """threshold_mask(off_mask_u8,thr) -> np.ndarray: bool mask."""
    g = off_mask_u8[..., 0].astype(np.uint8)
    return g > np.uint8(thr)

def downsample_binary_mask(mask_bool: np.ndarray, out_res: int) -> np.ndarray:
    """downsample_binary_mask(mask_bool,out_res) -> np.ndarray: Downsample (H,W)->(out_res,out_res) by block mean."""
    h, w = mask_bool.shape
    if h == out_res and w == out_res:
        return mask_bool
    if h % out_res != 0 or w % out_res != 0:
        # fallback: nearest resize via PIL
        img = Image.fromarray((mask_bool.astype(np.uint8) * 255), mode="L")
        img = img.resize((out_res, out_res), resample=Image.NEAREST)
        return (np.asarray(img, dtype=np.uint8) > 0)
    sy = h // out_res
    sx = w // out_res
    m = mask_bool.reshape(out_res, sy, out_res, sx).mean(axis=(1, 3))
    return m > 0.5

def postprocess_binary_mask(binary_mask: np.ndarray, close_radius: int) -> np.ndarray:
    """postprocess_binary_mask(binary_mask,close_radius) -> np.ndarray: Close gaps and fill holes."""
    try:
        import cv2  # type: ignore
        m = (binary_mask.astype(np.uint8) * 255)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        h, w = m2.shape
        ff = m2.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        filled = cv2.bitwise_or(m2, holes)
        return filled > 127
    except Exception:
        pass

    try:
        from scipy import ndimage  # type: ignore
        m = binary_mask.astype(bool)
        structure = ndimage.generate_binary_structure(2, 1)
        m2 = ndimage.binary_closing(m, structure=structure, iterations=int(max(1, close_radius)))
        m3 = ndimage.binary_fill_holes(m2)
        return m3.astype(bool)
    except Exception:
        raise RuntimeError("Need either opencv-python (cv2) or scipy for mask hole filling/closing.")

def mask_to_polygons(binary_mask: np.ndarray, simplify_eps: float) -> list:
    """mask_to_polygons(binary_mask,simplify_eps) -> list: list of polygons [(x,y),...]."""
    try:
        import cv2  # type: ignore
        m = (binary_mask.astype(np.uint8) * 255)
        contours, _hier = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polys = []
        for c in contours:
            if c.shape[0] < 10:
                continue
            approx = cv2.approxPolyDP(c, epsilon=float(simplify_eps), closed=True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
            if len(pts) >= 3:
                polys.append(pts)
        return polys
    except Exception:
        pass

    try:
        from skimage import measure  # type: ignore
        contours = measure.find_contours(binary_mask.astype(np.float32), 0.5)
        polys = []
        for c in contours:
            if c.shape[0] < 20:
                continue
            pts = [(int(round(p[1])), int(round(p[0]))) for p in c]
            if len(pts) >= 3:
                polys.append(pts)
        return polys
    except Exception:
        raise RuntimeError("Need opencv-python (cv2) or scikit-image for contour extraction.")

def polygons_to_coco_segmentation(polys_xy: list) -> list:
    """polygons_to_coco_segmentation(polys_xy) -> list: COCO polygon list (flattened)."""
    seg = []
    for pts in polys_xy:
        flat = []
        for x, y in pts:
            flat.extend([float(x), float(y)])
        if len(flat) >= 6:
            seg.append(flat)
    return seg


# =========================
# BBox: ID pass (only for bbox corners)
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

def build_bbox_id_texture(anns: list, tex_w: int, tex_h: int, radius_px: int) -> tuple[np.ndarray, dict, np.ndarray]:
    """build_bbox_id_texture(anns,tex_w,tex_h,radius_px) -> (tex_rgb01,meta,debug_u8)."""
    id_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    dbg_img = Image.new("RGB", (tex_w, tex_h), (0, 0, 0))
    id_draw = ImageDraw.Draw(id_img)
    dbg_draw = ImageDraw.Draw(dbg_img)

    next_id = 1
    meta = {}  # pid -> (ann_id, corner_idx)

    def stamp(cx: float, cy: float, ann_id: int, corner_idx: int) -> int:
        nonlocal next_id
        pid = next_id
        next_id += 1
        r, g, b = id_to_rgb_u8(pid)
        rc, gc, bc = (pid * 53) % 255, (pid * 97) % 255, (pid * 193) % 255

        x = int(round(cx))
        y = int(round(cy))
        x0, y0 = x - radius_px, y - radius_px
        x1, y1 = x + radius_px, y + radius_px
        id_draw.ellipse([x0, y0, x1, y1], fill=(r, g, b), outline=None)
        dbg_draw.ellipse([x0, y0, x1, y1], fill=(rc, gc, bc), outline=None)

        meta[pid] = (int(ann_id), int(corner_idx))
        return pid

    for a in anns:
        aid = int(a.get("id", -1))
        if "bbox" in a and len(a["bbox"]) == 4:
            x, y, w, h = a["bbox"]
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for ci, (cx, cy) in enumerate(corners):
                stamp(cx, cy, aid, ci)

    id_rgb01 = np.asarray(id_img, dtype=np.uint8).astype(np.float32) / 255.0
    dbg_u8 = np.asarray(dbg_img, dtype=np.uint8)
    return id_rgb01, meta, dbg_u8

def decode_id_centroids(id_img_u8: np.ndarray) -> dict:
    """decode_id_centroids(id_img_u8) -> dict: pid -> (x,y) centroid."""
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

def rebuild_bboxes_from_id(anns: list, meta: dict, pos: dict) -> list:
    """rebuild_bboxes_from_id(anns,meta,pos) -> list: boxes_xywh in off-nadir."""
    by_ann = {}
    for pid, (aid, ci) in meta.items():
        if pid in pos:
            by_ann.setdefault(aid, {})[ci] = pos[pid]

    boxes = []
    for a in anns:
        aid = int(a.get("id", -1))
        if aid not in by_ann:
            continue
        corners = by_ann[aid]
        if len(corners) != 4:
            continue
        pts = [corners[i] for i in range(4)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
    return boxes


# =========================
# Main
# =========================
def main() -> None:
    """main() -> None: Translate annotations at off-nadir resolution, then run generate_image and overlay."""
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

    # ---- original image (no crop) ----
    orig_rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.uint8)
    tex_h, tex_w = orig_rgb.shape[:2]
    orig_overlay = draw_overlay(Image.fromarray(orig_rgb, mode="RGB").copy(), anns)

    print(f"[1/7] Loaded original {tex_w}x{tex_h}, anns={len(anns)}")

    # ---- DEM mesh ----
    satellite = get_satellite(str(img_path), str(CSV_PATH))
    gsd = get_spatial_res(str(img_path), str(CSV_PATH))

    dem_folder = main_path / "offnadir_imaging" / "create_DEM"
    dem_tiff_path = str(dem_folder / "input_dem_WV.tiff")
    dem_obj_path = str(dem_folder / "dem_mesh_WV.obj")

    print("[2/7] Building DEM mesh...")
    get_DEM(str(img_path), dem_tiff_path, gsd, WAVE_PROPERTIES, random_seed=DEM_SEED, waves=True, curvature=True, plot_DEM=False)
    convert_DEM(str(img_path), dem_tiff_path, dem_obj_path, gsd, scale_km=False, print_output=False, plot_DEM=False)
    _ = get_band_data(satellite, str(main_path / "offnadir_imaging" / "spd_files"))

    # ---- camera geometry ----
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

    mi.set_variant("cuda_ad_rgb")
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

    print(f"[3/7] Camera ready. FOV={fov_deg:.6f} deg, azimuth={azimuth_rad * 180 / math.pi:.2f} deg")

    # ==========================================================
    # (A) SEGMENTATION: mask render at SS-res -> downsample to RENDER_RESOLUTION
    # ==========================================================
    print("[4/7] Translating segmentation via mask render...")
    mask_tex_rgb01, mask_u8 = build_binary_mask_texture(anns, tex_w, tex_h)

    mask_res_hi = int(RENDER_RESOLUTION * MASK_SUPERSAMPLE)
    mask_off_u8_hi = render_emission_texture(
        dem_obj_path=dem_obj_path,
        tex_rgb01=mask_tex_rgb01,
        to_world_scene=to_world_scene,
        to_world_sensor=to_world_sensor_off,
        fov_deg=fov_deg,
        res=mask_res_hi,
        spp=MASK_SPP,
        filter_type=MASK_FILTER,
    )

    mask_off_bin_hi = threshold_mask(mask_off_u8_hi, thr=127)
    mask_off_bin_hi = postprocess_binary_mask(mask_off_bin_hi, close_radius=MASK_CLOSE_RADIUS)

    # downsample back to exactly the off-nadir render resolution
    mask_off_bin = downsample_binary_mask(mask_off_bin_hi, out_res=RENDER_RESOLUTION)

    polys_off = mask_to_polygons(mask_off_bin, simplify_eps=1.5)

    # ==========================================================
    # (B) BBOX: ID corners at off-nadir res
    # ==========================================================
    print("[5/7] Translating bbox via ID corner render...")
    bbox_id_tex_rgb01, bbox_meta, bbox_dbg_u8 = build_bbox_id_texture(anns, tex_w, tex_h, radius_px=ID_POINT_RADIUS_PX)

    bbox_id_off_u8 = render_emission_texture(
        dem_obj_path=dem_obj_path,
        tex_rgb01=bbox_id_tex_rgb01,
        to_world_scene=to_world_scene,
        to_world_sensor=to_world_sensor_off,
        fov_deg=fov_deg,
        res=ID_RENDER_RES,
        spp=ID_SPP,
        filter_type=ID_FILTER,
    )

    bbox_pos = decode_id_centroids(bbox_id_off_u8)
    boxes_off = rebuild_bboxes_from_id(anns, bbox_meta, bbox_pos)

    # Progress plots (all in correct final resolution for off-nadir annotations)
    contour_dbg = Image.new("RGB", (RENDER_RESOLUTION, RENDER_RESOLUTION), (0, 0, 0))
    contour_dbg = draw_overlay_from_polys(contour_dbg, polys_off, boxes_off)

    fig = plt.figure(figsize=(22, 12))
    fig.add_subplot(2, 3, 1).imshow(orig_overlay); plt.axis("off"); plt.title("Original + annotation")
    fig.add_subplot(2, 3, 2).imshow(mask_u8, cmap="gray"); plt.axis("off"); plt.title("Original binary mask")
    fig.add_subplot(2, 3, 3).imshow(mask_off_u8_hi); plt.axis("off"); plt.title(f"Off-nadir mask (hi-res {mask_res_hi}x{mask_res_hi})")
    fig.add_subplot(2, 3, 4).imshow(mask_off_bin.astype(np.uint8) * 255, cmap="gray"); plt.axis("off"); plt.title(f"Off-nadir mask (downsampled {RENDER_RESOLUTION}x{RENDER_RESOLUTION})")
    fig.add_subplot(2, 3, 5).imshow(np.asarray(contour_dbg)); plt.axis("off"); plt.title("Extracted contour + bbox (final res)")
    fig.add_subplot(2, 3, 6).imshow(bbox_id_off_u8); plt.axis("off"); plt.title("Off-nadir bbox ID pass")
    plt.tight_layout()
    plt.show()

    # (Optional) produce COCO segmentation lists at off-nadir resolution
    coco_seg_off = polygons_to_coco_segmentation(polys_off)
    print(f"[6/7] Off-nadir polygons extracted: {len(polys_off)} (COCO seg lists: {len(coco_seg_off)})")
    print(f"      Off-nadir boxes extracted: {len(boxes_off)}")

    # ==========================================================
    # (C) Now run generate_image and overlay
    # ==========================================================
    print("[7/7] Running generate_image after translation...")
    sensor_characteristics = {"resolution": RENDER_RESOLUTION, "sample_count": SAMPLE_COUNT, "GSD": gsd}

    DN255_texture, DN255_no_glint, DN255_glint, radiance_glint, rho_glint, rho_disp, black_mask_full, scale, offnadir_deg = generate_image(
        str(img_path), str(ANNOTATIONS_PATH),
        satellite,
        SAT_LAT, SAT_LON, SAT_ALT,
        TGT_LAT, TGT_LON, TGT_ALT,
        DATETIME_UTC,
        sensor_characteristics,
        WAVE_PROPERTIES,
        BOOLS,
        DEM_SEED,
    )

    if DN255_texture is None:
        raise RuntimeError("Renderer returned None (dark hours).")

    tex_u8 = to_uint8_rgb(DN255_texture)
    no_glint_u8 = to_uint8_rgb(DN255_no_glint) if DN255_no_glint is not None else np.zeros_like(tex_u8)
    glint_u8 = to_uint8_rgb(DN255_glint) if DN255_glint is not None else np.zeros_like(tex_u8)

    rho_u8 = (np.clip(rho_disp, 0.0, 1.0) * 255.0).astype(np.uint8) if rho_disp is not None else np.zeros_like(tex_u8)

    tex_overlay = draw_overlay_from_polys(Image.fromarray(tex_u8, mode="RGB").copy(), polys_off, boxes_off)
    glint_overlay = draw_overlay_from_polys(Image.fromarray(glint_u8, mode="RGB").copy(), polys_off, boxes_off) if DN255_glint is not None else None
    rho_overlay = draw_overlay_from_polys(Image.fromarray(rho_u8, mode="RGB").copy(), polys_off, boxes_off) if rho_disp is not None else None

    fig2 = plt.figure(figsize=(18, 10))
    fig2.add_subplot(1, 4, 1).imshow(orig_overlay);
    plt.axis("off");
    plt.title("original + annotation")
    fig2.add_subplot(1, 4, 2).imshow(tex_overlay);
    plt.axis("off");
    plt.title("off-nadir + translated (final res)")
    fig2.add_subplot(1, 4, 3).imshow(glint_overlay if glint_overlay is not None else glint_u8);
    plt.axis("off");
    plt.title("sun glint")
    fig2.add_subplot(1, 4, 4).imshow(rho_overlay if rho_overlay is not None else rho_u8);
    plt.axis("off");
    plt.title("TOA reflectance (scaled)")
    plt.tight_layout()
    plt.show()

    # Example: how you'd store translated annotations back to COCO for the off-nadir image
    # (You still need to create a new "image" entry for the off-nadir render and then attach these.)
    # Here we just print an example payload for the first annotation:
    if anns:
        example = {
            "segmentation": coco_seg_off[:1],  # replace with per-instance segmentation if you split masks by instance
            "bbox": list(boxes_off[0]) if boxes_off else None,
            "iscrowd": 0,
        }
        print("Example translated COCO fields (not written to file):")
        print(example)


if __name__ == "__main__":
    main()
