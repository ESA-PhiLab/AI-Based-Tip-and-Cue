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
from offnadir_imaging.functions.convert_reference_frames import get_ecef_from_lat_lon
from offnadir_imaging.functions.intermediate_functions import get_scene_characteristics, is_dark_from_sun_dir


# =========================
# Path handling (same style)
# =========================
main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

DATASET_PATH = Path("dataset")
CREATE_DATASET_DIR = DATASET_PATH / "create_dataset"
NADIR_DIR = CREATE_DATASET_DIR / "nadir"
CSV_PATH = DATASET_PATH / "whales_from_space" / "WhaleFromSpaceDB_Whales.csv"

DEFAULT_DATETIME_UTC = datetime(2025, 6, 11, 8, 0, 0, tzinfo=timezone.utc)

BOOLS_BASE = {
    "plot_3d": False,
    "plot_result": False,
    "max_glint": False,
    "print_values": True,
    "crop_black_border": False,  # ignore cropping
    "generate_radiation": True,
}

WAVE_PROPERTIES = {"wind_speed": 10.0, "num_waves": 50, "wave_min": 0.05, "wave_max": 0.5}
SAMPLE_COUNT = 512
DEM_SEED = 42

# Segmentation translation:
MASK_SUPERSAMPLE = 2
MASK_SPP = 1
MASK_FILTER = "bilinear"
MASK_CLOSE_RADIUS = 2

# BBox translation:
ID_SPP = 1
ID_POINT_RADIUS_PX = 4
ID_FILTER = "nearest"


# =========================
# Drawing helpers (baseline)
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
def render_emission_texture(dem_obj_path: str,
                            tex_rgb01: np.ndarray,
                            to_world_scene,
                            to_world_sensor,
                            fov_deg: float,
                            res: int,
                            spp: int,
                            filter_type: str) -> np.ndarray:
    """render_emission_texture(dem_obj_path,tex_rgb01,to_world_scene,to_world_sensor,fov_deg,res,spp,filter_type) -> np.ndarray."""
    tex = {
        "type": "bitmap",
        "data": tex_rgb01,
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
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =========================
# Segmentation: per-annotation mask -> render -> polygons
# =========================
def build_binary_mask_texture_single(ann: dict, tex_w: int, tex_h: int) -> np.ndarray:
    """build_binary_mask_texture_single(ann,tex_w,tex_h) -> np.ndarray: RGB(0..1) emission texture for one instance."""
    m = Image.new("L", (tex_w, tex_h), 0)
    d = ImageDraw.Draw(m)
    for seg in ann.get("segmentation", []):
        if not seg:
            continue
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        if len(pts) >= 3:
            d.polygon(pts, fill=255, outline=255)
    mask_u8 = np.asarray(m, dtype=np.uint8)
    return np.stack([mask_u8, mask_u8, mask_u8], axis=-1).astype(np.float32) / 255.0


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
# BBox: ID pass
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
        if "bbox" in a and isinstance(a["bbox"], (list, tuple)) and len(a["bbox"]) == 4:
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


def rebuild_bboxes_from_id_by_ann(meta: dict, pos: dict) -> dict:
    """rebuild_bboxes_from_id_by_ann(meta,pos) -> dict: ann_id -> (x,y,w,h)."""
    by_ann = {}
    for pid, (aid, ci) in meta.items():
        if pid in pos:
            by_ann.setdefault(aid, {})[ci] = pos[pid]

    out = {}
    for aid, corners in by_ann.items():
        if len(corners) != 4:
            continue
        pts = [corners[i] for i in range(4)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        out[int(aid)] = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return out


# =========================
# Patch resolver (bundle stores no paths)
# =========================
def _resolve_nadir_patch_path(patch_bundle: dict) -> Path:
    """_resolve_nadir_patch_path(patch_bundle) -> Path: dataset/create_dataset/nadir/<subdir>/<patch_name><ext>."""
    img_file = patch_bundle.get("img_file", None)
    patch_name = patch_bundle.get("patch_name", None)
    if not isinstance(img_file, str):
        raise ValueError("patch_bundle['img_file'] must be the original image file_name string")
    if not isinstance(patch_name, str) or not patch_name:
        raise ValueError("patch_bundle['patch_name'] must exist (set by save_patch('nadir', ...))")

    subdir = Path(img_file).parent
    ext = Path(img_file).suffix
    return NADIR_DIR / subdir / f"{patch_name}{ext}"


# =========================
# Public API
# =========================
def translate_offnadir(patch_bundle: dict,
                       render_resolution: int,
                       sat_lat: float, sat_lon: float, sat_alt: float,
                       tgt_lat: float, tgt_lon: float, tgt_alt: float,
                       dem_seed: int,
                       show_plot: bool = False,
                       datetime_utc: datetime | None = None) -> dict:

    """translate_offnadir(patch_bundle,render_resolution,sat_lat,sat_lon,sat_alt,tgt_lat,tgt_lon,tgt_alt,show_plot,datetime_utc) -> dict: Render offnadir + translate anns."""
    if not isinstance(patch_bundle, dict):
        raise TypeError("translate_offnadir expects patch_bundle as dict")

    img_path = _resolve_nadir_patch_path(patch_bundle)
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing nadir patch file: {img_path}")

    anns = patch_bundle.get("anns_patch", None)
    if not isinstance(anns, list):
        raise ValueError("patch_bundle['anns_patch'] missing. Call save_patch('nadir', patch_bundle) before translate_offnadir.")

    orig_rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    tex_h, tex_w = orig_rgb.shape[:2]
    orig_overlay = draw_overlay(Image.fromarray(orig_rgb, mode="RGB").copy(), anns)

    satellite = get_satellite(str(img_path), str(CSV_PATH))
    gsd = get_spatial_res(str(img_path), str(CSV_PATH))

    dem_folder = main_path / "offnadir_imaging" / "create_DEM"
    dem_tiff_path = str(dem_folder / "input_dem_WV.tiff")
    dem_obj_path = str(dem_folder / "dem_mesh_WV.obj")

    get_DEM(str(img_path), dem_tiff_path, gsd, WAVE_PROPERTIES, random_seed=int(dem_seed), waves=True, curvature=True, plot_DEM=False)
    convert_DEM(str(img_path), dem_tiff_path, dem_obj_path, gsd, scale_km=False, print_output=False, plot_DEM=False)
    _ = get_band_data(satellite, str(main_path / "offnadir_imaging" / "spd_files"))

    dt = datetime_utc if datetime_utc is not None else DEFAULT_DATETIME_UTC

    satellite_ecef, target_ecef, sun_ecef = get_ecef_from_lat_lon(
        float(sat_lat), float(sat_lon), float(sat_alt),
        float(tgt_lat), float(tgt_lon), float(tgt_alt),
        dt,
    )

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

    # ==========================================================
    # (A) SEGMENTATION: per-annotation mask render (NO global mask)
    # ==========================================================
    mask_res_hi = int(int(render_resolution) * MASK_SUPERSAMPLE)
    polys_off_by_ann: dict[int, list] = {}

    for ann in anns:
        ann_id = int(ann.get("id", -1))
        if ann_id < 0:
            continue

        mask_tex_rgb01 = build_binary_mask_texture_single(ann, tex_w, tex_h)

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
        mask_off_bin = downsample_binary_mask(mask_off_bin_hi, out_res=int(render_resolution))

        polys_off_by_ann[ann_id] = mask_to_polygons(mask_off_bin, simplify_eps=1.5)

    # ==========================================================
    # (B) BBOX: ID corners at off-nadir res
    # ==========================================================
    bbox_id_tex_rgb01, bbox_meta, bbox_dbg_u8 = build_bbox_id_texture(anns, tex_w, tex_h, radius_px=ID_POINT_RADIUS_PX)

    bbox_id_off_u8 = render_emission_texture(
        dem_obj_path=dem_obj_path,
        tex_rgb01=bbox_id_tex_rgb01,
        to_world_scene=to_world_scene,
        to_world_sensor=to_world_sensor_off,
        fov_deg=fov_deg,
        res=int(render_resolution),
        spp=ID_SPP,
        filter_type=ID_FILTER,
    )

    bbox_pos = decode_id_centroids(bbox_id_off_u8)
    boxes_off_by_ann = rebuild_bboxes_from_id_by_ann(bbox_meta, bbox_pos)
    boxes_off = list(boxes_off_by_ann.values())

    all_polys = [p for plist in polys_off_by_ann.values() for p in plist]
    contour_dbg = Image.new("RGB", (int(render_resolution), int(render_resolution)), (0, 0, 0))
    contour_dbg = draw_overlay_from_polys(contour_dbg, all_polys, boxes_off)

    if show_plot:
        fig = plt.figure(figsize=(22, 10))
        fig.add_subplot(2, 3, 1).imshow(orig_overlay); plt.axis("off"); plt.title("Original + annotation")
        fig.add_subplot(2, 3, 2).imshow(np.asarray(contour_dbg)); plt.axis("off"); plt.title("Per-instance contours + bboxes (final res)")
        fig.add_subplot(2, 3, 3).imshow(bbox_dbg_u8); plt.axis("off"); plt.title("BBox ID stamps (texture space)")
        fig.add_subplot(2, 3, 4).imshow(bbox_id_off_u8); plt.axis("off"); plt.title("Off-nadir bbox ID pass")
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # (C) Now run generate_image and overlay (debug only)
    # ==========================================================
    sensor_characteristics = {"resolution": int(render_resolution), "sample_count": int(SAMPLE_COUNT), "GSD": gsd}
    bools = dict(BOOLS_BASE)
    bools["plot_result"] = False

    DN255_offnadir, DN255_sunglint, _rad_sunglint, DN255_combined = generate_image(
        str(img_path),
        satellite,
        float(sat_lat), float(sat_lon), float(sat_alt),
        float(tgt_lat), float(tgt_lon), float(tgt_alt),
        dt,
        sensor_characteristics,
        WAVE_PROPERTIES,
        bools,
        DEM_SEED,
    )
    if DN255_offnadir is None:
        raise RuntimeError("Renderer returned None (dark hours).")

    offnadir_u8 = to_uint8_rgb(DN255_offnadir)
    sunglint_u8 = to_uint8_rgb(DN255_sunglint)
    combined_u8 = to_uint8_rgb(DN255_combined)

    off_overlay = draw_overlay_from_polys(Image.fromarray(offnadir_u8, mode="RGB").copy(), all_polys, boxes_off)
    comb_overlay = None
    if combined_u8 is not None:
        comb_overlay = draw_overlay_from_polys(Image.fromarray(combined_u8, mode="RGB").copy(), all_polys, boxes_off)

    if show_plot:
        fig2 = plt.figure(figsize=(18, 10))
        fig2.add_subplot(1, 4, 1).imshow(orig_overlay); plt.axis("off"); plt.title("original + annotation")
        fig2.add_subplot(1, 4, 2).imshow(off_overlay);  plt.axis("off"); plt.title("off-nadir + translated (per-instance)")
        fig2.add_subplot(1, 4, 3).imshow(sunglint_u8 if sunglint_u8 is not None else np.zeros_like(offnadir_u8)); plt.axis("off"); plt.title("sun glint")
        fig2.add_subplot(1, 4, 4).imshow(comb_overlay if comb_overlay is not None else np.zeros_like(offnadir_u8)); plt.axis("off"); plt.title("combined + translated")
        plt.tight_layout()
        plt.show()

    # ==========================================================
    # (D) Translate annotations, preserving keys exactly
    # ==========================================================
    translated_anns: list[dict] = []
    for ann in anns:
        aid = int(ann.get("id", -1))
        polys = polys_off_by_ann.get(aid, [])
        bbox = boxes_off_by_ann.get(aid, None)

        a2 = dict(ann)  # keep everything
        a2["segmentation"] = polygons_to_coco_segmentation(polys)
        if bbox is not None:
            a2["bbox"] = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        translated_anns.append(a2)

    out = dict(patch_bundle)
    out["patch"] = offnadir_u8                # image to save (raw, no overlay)
    out["anns_patch"] = translated_anns       # anns to save in json
    out["combined_u8"] = combined_u8          # used for sunglint (combined)
    out["render_resolution"] = int(render_resolution)
    return out


def add_sunglint(offnadir_bundle: dict, show_plot: bool = False) -> dict:
    """add_sunglint(offnadir_bundle,show_plot) -> dict: Save combined image (4th plot), keep anns."""
    combined_u8 = offnadir_bundle.get("combined_u8", None)
    if combined_u8 is None:
        raise ValueError("offnadir_bundle missing 'combined_u8'")

    combined_u8 = np.asarray(combined_u8)
    if combined_u8.dtype != np.uint8:
        combined_u8 = np.clip(combined_u8, 0, 255).astype(np.uint8)

    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(combined_u8)
        plt.axis("off")
        plt.title("combined (saved for sunglint)")
        plt.tight_layout()
        plt.show()

    out = dict(offnadir_bundle)
    out["patch"] = combined_u8  # what save_patch("sunglint", ...) writes
    return out
