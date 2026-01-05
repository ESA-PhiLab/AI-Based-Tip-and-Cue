import math
import os, sys
import numpy as np

import mitsuba as mi
import drjit as dr
from datetime import datetime, timezone
from matplotlib import pyplot as plt
from PIL import Image
from pathlib import Path
import gc

from .RTM import generate_sun_and_sky_spds
from .create_DEM.create_dummy_DEM import get_DEM
from .create_DEM.convert_DEM import convert_DEM

from .functions.plotfunctions import plot_earth_with_pyvista, plot_earth_slice_with_sun, plot_target_perspective, get_rgb
from .functions.get_satellite_data import get_band_data, get_satellite, get_spatial_res
from .functions.convert_reference_frames import get_lat_lon_alt_from_ecef, get_ecef_from_lat_lon, compute_max_glint_satellite_ecef
from .functions.intermediate_functions import rmse, normalize, get_scene_characteristics, is_dark_from_sun_dir, dbg_sun_elevation, masked_abs_radiance, masked_percentile, masked_channel_percentiles, masked_channel_means, masked_mean, plot_radiance_timeline
from .functions import image_utils as iu
from .functions.mask_functions import get_whale_mask_for_image, coco_segmentation_to_mask, load_coco_index, rgb_png_to_reflectance_proxy, compute_hit_mask_full

_COCO_CACHE = {}

def load_coco_index_cached(anns_path: str):
    """load_coco_index_cached(anns_path) -> tuple: Cache COCO index by path."""
    p = os.path.normpath(str(anns_path))
    if p not in _COCO_CACHE:
        _COCO_CACHE[p] = load_coco_index(p)
    return _COCO_CACHE[p]

def resolve_coco_file_name(img_path: str, by_file: dict) -> str:
    """resolve_coco_file_name(img_path,by_file) -> str: Find COCO file_name key matching img_path."""
    p = Path(str(img_path)).as_posix()

    # 1) Strong match: full suffix equals COCO key
    for k in by_file.keys():
        k_posix = Path(str(k)).as_posix()
        if p.endswith(k_posix):
            return str(k_posix)

    # 2) Weaker match: unique basename match
    base = Path(p).name
    matches = [Path(str(k)).as_posix() for k in by_file.keys() if Path(str(k)).name == base]
    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        "Image file_name not found in annotations for img_path="
        f"{p}. Tried suffix match and basename match (matches={len(matches)})."
    )



def render_band_radiance(input_img_lin, dem_path, spd_path, sun_spd, sky_spd, satellite_local, target_local, sun_direction, sensor_characteristics, alpha, specular_weight):
    """Render one band as band-integrated at-sensor radiance (diffuse texture + water specular)."""
    mi.set_variant('cuda_ad_spectral')

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-sensor_characteristics['azimuth_rad'])
    )
    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    sun_dir_target_to_sun = np.array(sun_direction, dtype=float)
    sun_dir_target_to_sun /= np.linalg.norm(sun_dir_target_to_sun) + 1e-12

    mi_light_dir = sun_dir_target_to_sun  # IMPORTANT: sun -> target

    # Diffuse albedo texture (treat your image as reflectance/albedo proxy)
    texture = {
        "type": "bitmap",
        "data": mi.TensorXf(input_img_lin.astype(np.float32)),
        "wrap_mode": "clamp",
        "filter_type": "nearest",
        "raw": True
    }

    diffuse_bsdf = {
        "type": "diffuse",
        "reflectance": texture
    }

    # Specular water glint (use beckmann to match Gaussian slopes better than GGX)
    specular_bsdf = {
        "type": "roughdielectric",
        "distribution": "beckmann",
        "alpha": float(alpha),
        "int_ior": 1.3330,
        "ext_ior": 1.000277
    }

    # Mix diffuse + specular
    bsdf = {
        "type": "blendbsdf",
        "weight": float(specular_weight),  # 0 -> diffuse only, 1 -> specular only
        "bsdf_0": diffuse_bsdf,
        "bsdf_1": specular_bsdf
    }


    to_world_sensor = mi.ScalarTransform4f().look_at(
        origin=satellite_local,
        target=target_local,
        up=[0, 0, 1]
    )

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},

        "earth_surface": {
            "type": "obj",
            "filename": dem_path,
            "to_world": to_world_scene,
            "bsdf": bsdf
        },

        "sun": {
            "type": "directional",
            "direction": mi.ScalarVector3f(mi_light_dir),
            "irradiance": {
                "type": "spectrum",
                "filename": sun_spd  # direct normal irradiance
            }
        },

        "sky": {
            "type": "constant",
            "radiance": {
                "type": "spectrum",
                "filename": sky_spd  # diffuse horizontal / pi
            }
        },

        "sensor": {
            "type": "perspective",
            "to_world": to_world_sensor,
            "fov": sensor_characteristics['fov_deg'],
            "far_clip": 1e8,
            "film": {
                "type": "specfilm",
                "width": sensor_characteristics['resolution'],
                "height": sensor_characteristics['resolution'],
                "spectral_band": {
                    "type": "spectrum",
                    "filename": spd_path
                }
            },
            "sampler": {
                "type": "independent",
                "sample_count": sensor_characteristics['sample_count']
            }
        }
    }

    scene = mi.load_dict(scene_dict)
    L = mi.render(scene)
    _ = mi.util.convert_to_bitmap(L)
    return L



def render_projected_texture(input_img, dem_path, satellite_local, target_local, sensor_characteristics):

    mi.set_variant('cuda_ad_rgb')

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-sensor_characteristics['azimuth_rad'])
    )

    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    texture = {
        "type": "bitmap",
        "data": mi.TensorXf(np.asarray(input_img, dtype=np.float32)),

        "filter_type": "nearest",
        "wrap_mode": "clamp",
        "raw": True
    }

    brdf = {
        "type": "diffuse",
        "reflectance": texture
    }

    to_world_sensor = mi.ScalarTransform4f().look_at(
        origin=satellite_local,
        target=target_local,
        up=[0, 0, 1]
    )

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},

        "earth_surface": {
            "type": "obj",
            "filename": dem_path,
            "bsdf": brdf,
            "to_world": to_world_scene,
        },

        "light": {"type": "constant"},

        "sensor": {
            "type": "perspective",
            "to_world": to_world_sensor,
            "fov": sensor_characteristics['fov_deg'],
            'far_clip': 1e8,
            "film": {
                "type": "hdrfilm",
                "width": sensor_characteristics['resolution'],
                "height": sensor_characteristics['resolution'],
                "rfilter": {"type": "box"},
                "sample_border": False,
                "compensate": True
            },

            "sampler": {
                "type": "independent",
                "sample_count": sensor_characteristics['sample_count']
                ,
            },
        },
    }

    # Load and render the scene
    scene = mi.load_dict(scene_dict)
    image_offnadir = mi.render(scene)

    _ = mi.util.convert_to_bitmap(image_offnadir)

    return image_offnadir

def render_rgb_with_optional_glint(img_refl, dem_path, band_data, sun_spd_path, sky_spd_path, satellite_local, target_local, sun_direction, sensor_characteristics, alpha, specular_weight, tonemap_percentile=99.5, return_linear=False):

    """Render physically consistent RGB under the same sun/camera, with glint controlled by specular_weight; returns (DN255_rgb, radiance_rgb_linear_or_None, scale)."""


    R_lin = img_refl[:, :, 0][:, :, None].astype(np.float32)
    G_lin = img_refl[:, :, 1][:, :, None].astype(np.float32)
    B_lin = img_refl[:, :, 2][:, :, None].astype(np.float32)

    # Render each band with the same geometry and sun; glint on/off is just specular_weight
    L_R = np.array(render_band_radiance(
    input_img_lin=R_lin,
    dem_path=dem_path,
    spd_path=band_data["red"]["spd"],
    sun_spd=sun_spd_path,
    sky_spd=sky_spd_path,
    satellite_local=satellite_local,
    target_local=target_local,
    sun_direction=sun_direction,
    sensor_characteristics=sensor_characteristics,
    alpha=float(alpha),
    specular_weight=float(specular_weight)
))[:, :, 0]

    L_G = np.array(render_band_radiance(
        input_img_lin=G_lin,
        dem_path=dem_path,
        spd_path=band_data['green']['spd'],
        sun_spd=sun_spd_path,
        sky_spd=sky_spd_path,
        satellite_local=satellite_local,
        target_local=target_local,
        sun_direction=sun_direction,
        sensor_characteristics=sensor_characteristics,
        alpha=float(alpha),
        specular_weight=float(specular_weight)
    ))[:, :, 0]

    L_B = np.array(render_band_radiance(
        input_img_lin=B_lin,
        dem_path=dem_path,
        spd_path=band_data['blue']['spd'],
        sun_spd=sun_spd_path,
        sky_spd=sky_spd_path,
        satellite_local=satellite_local,
        target_local=target_local,
        sun_direction=sun_direction,
        sensor_characteristics=sensor_characteristics,
        alpha=float(alpha),
        specular_weight=float(specular_weight)
    ))[:, :, 0]

    radiance_rgb = np.stack([L_R, L_G, L_B], axis=-1)  # HxWx3, linear proxy radiance

    # Tone map (robust)
    scale = float(np.percentile(radiance_rgb, float(tonemap_percentile)))
    rgb_lin = np.clip(radiance_rgb / (scale + 1e-12), 0.0, 1.0)
    DN255_rgb = iu.linear_to_DN255(rgb_lin)

    return (DN255_rgb, radiance_rgb if return_linear else None, scale)




def generate_image(img_path, anns_path, satellite, satellite_lat, satellite_lon, satellite_alt, target_lat, target_lon, target_alt, datetime_utc, sensor_characteristics, wave_properties, bools, dem_seed):

    # dr.set_flag(dr.JitFlag.Debug, True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Temp file paths
    dem_tiff_path = os.path.join(script_dir, 'create_DEM/input_dem_WV.tiff')
    dem_path = os.path.join(script_dir, "create_DEM/dem_mesh_WV.obj")

    # --- SPDs (relative to this module) ---
    spd_folder = os.path.join(script_dir, "spd_files")
    sun_spd = os.path.join(spd_folder, "solar_direct_normal.spd")
    sky_spd = os.path.join(spd_folder, "sky_diffuse_radiance.spd")

    band_data = get_band_data(satellite, spd_folder)
    if bools['print_values']:
        print(f"Retrieved band data for {satellite} satellite\n")

    GSD = sensor_characteristics['GSD']
    wind_speed = wave_properties['wind_speed']
    sigma2 = 0.003 + 0.00512 * wind_speed
    alpha = np.sqrt(sigma2)

    get_DEM(img_path, dem_tiff_path, GSD, wave_properties, random_seed=dem_seed, waves=True, curvature=True, plot_DEM=False)
    convert_DEM(img_path, dem_tiff_path, dem_path, GSD, scale_km=False, print_output=False, plot_DEM=False)

    if bools['print_values']:
        print(f"Saved synthetic DEM to {dem_path}\n")

    if bools['print_values']:
        print("Convert lat lon to ecef coordinates")

    if bools['generate_nadir']:
        satellite_lat = target_lat
        satellite_lon = target_lon

    satellite_ecef, target_ecef, sun_ecef = get_ecef_from_lat_lon(satellite_lat, satellite_lon, satellite_alt, target_lat, target_lon, target_alt, datetime_utc)

    if bools['max_glint']:
        satellite_ecef = compute_max_glint_satellite_ecef(target_ecef, sun_ecef, glint_distance_m=700 * 10**3)

    if bools['print_values']:
        print("Target ECEF:    ", np.round(target_ecef / 1000, 3), ' km')
        print("Satellite ECEF: ", np.round(satellite_ecef / 1000, 3), ' km')
        formatted = " ".join(f"{x:.2e}" for x in sun_ecef / 1000)
        print(f"Sun ECEF       : [{formatted}]  km\n")

    if bools['plot_3d']:
        plot_earth_with_pyvista(satellite_ecef, target_ecef, sun_ecef, R_earth=6378137.0)

    is_dark, elev_deg, thr = is_dark_from_sun_dir(
        target_ecef=target_ecef,
        sun_ecef=sun_ecef,
        threshold_deg=-18.0,  # Astronomical night
        model="wgs84",
        dir_type="target_to_sun"
    )

    if is_dark:
        print("Dark hours, no image possible")
        return None, None, None, None, None, None, None, None

    img_rgb = np.asarray(Image.open(img_path).convert('RGB'))

    img_np = np.array(img_rgb)
    img_height, img_width = img_np.shape[:2]

    img_lin = iu.DN255_to_linear(img_rgb)

    BY_FILE, ANNS_BY_IMAGE_ID, IMAGES_BY_ID = load_coco_index_cached(str(anns_path))
    img_file_name_for_coco = resolve_coco_file_name(img_path, BY_FILE)

    whale_mask = get_whale_mask_for_image(
        img_rgb_uint8=img_rgb,
        img_file_name=img_file_name_for_coco,
        by_file=BY_FILE,
        anns_by_image_id=ANNS_BY_IMAGE_ID,
        images_by_id=IMAGES_BY_ID,
        whale_category_id=0
    )

    # Anchor region = "everything except whale"
    anchor_mask = ~whale_mask

    # Optional: also remove very bright pixels from anchor to avoid boats/foam bias
    gray = img_rgb.mean(axis=2)
    anchor_mask &= (gray < 220)

    img_refl = rgb_png_to_reflectance_proxy(
        img_rgb_uint8=img_rgb,
        anchor_mask=anchor_mask,
        target_reflectance_rgb=wave_properties.get("target_reflectance_rgb", (0.04, 0.03, 0.02))
    )

    if bools['print_values']:
        print("Loaded input image with shape ", img_height, "h x", img_width, " w, and DN255 min ", np.min(img_rgb), 'max ', np.max(img_rgb))

    satellite_local, target_local, sun_direction, fov_deg, off_nadir_rad, azimuth_rad = get_scene_characteristics(
        satellite_ecef, target_ecef, sun_ecef, img_height, img_width, GSD)

    if bools['print_values']:
        print(f"\nFOV \t\t: {fov_deg:.5f} deg")
        print(f"Off Nadir \t: {off_nadir_rad * 180 / np.pi:.2f} deg")
        print(f"Azimuth \t: {azimuth_rad * 180 / np.pi:.2f} deg\n")

    sensor_characteristics['fov_deg'] = fov_deg
    sensor_characteristics['azimuth_rad'] = azimuth_rad

    if bools['generate_radiation']:

        sun_direction_away = -np.array(sun_direction)
        min_wvl, max_wvl = 250, 1000  # nm

        try:
            sun_spd, sky_spd = generate_sun_and_sky_spds(
                datetime_utc=datetime_utc,
                target_lat=target_lat,
                target_lon=target_lon,
                target_alt_m=target_alt,
                out_sun_spd=sun_spd,
                out_sky_spd=sky_spd,
                wavelength_range=(min_wvl, max_wvl),
                timezone_offset=0,
                material="Water",
                plot_spd=False
            )

            if bools['print_values']:
                print("UTC time:", datetime_utc.isoformat())
                print("Target lat/lon:", target_lat, target_lon)
                print("Sun elevation from geometry (deg):", elev_deg)


        except Exception as e:
            if bools.get("print_values", False):
                print(f"Failed to generate sun/sky SPD: {repr(e)}")
            return None, None, None, None, None, None, None, None

        if bools['print_values']:
            print(f"Saved solar SPD to {sun_spd}\n")

        if bools['print_values']:
            print(f"Generate off nadir image\n")



        off_nadir_image = render_projected_texture(img_lin, dem_path, satellite_local, target_local, sensor_characteristics)

        THIS_DIR = Path(__file__).resolve().parent  # offnadir_imaging/
        PROJECT_ROOT = THIS_DIR.parent  # AI-Based-Tip-and-Cue/

        black_folder = PROJECT_ROOT / "dataset" / "utils_images"
        black_rgb = np.asarray(Image.open(os.path.join(black_folder, 'black.png')).convert('RGB'))
        black_lin = iu.DN255_to_linear(black_rgb)

        black_offnadir = render_projected_texture(black_lin, dem_path, satellite_local, target_local, sensor_characteristics)
        DN255_black = iu.linear_to_DN255(black_offnadir)

        # off_nadir_image = np.flip(off_nadir_image, axis=0)

        DN255_texture = iu.linear_to_DN255(off_nadir_image)

        # Physically consistent baseline: same sun/camera, NO glint
        DN255_no_glint, radiance_no_glint, scale = render_rgb_with_optional_glint(
            img_refl=img_refl,
            dem_path=dem_path,
            band_data=band_data,
            sun_spd_path=sun_spd,
            sky_spd_path=sky_spd,
            satellite_local=satellite_local,
            target_local=target_local,
            sun_direction=sun_direction,
            sensor_characteristics=sensor_characteristics,
            alpha=float(alpha),
            specular_weight=0.0,
            tonemap_percentile=float(wave_properties.get('tonemap_percentile', 99.5)),
            return_linear=True
        )

        # Physically consistent glint render: same sun/camera, glint enabled
        DN255_glint, radiance_glint, _ = render_rgb_with_optional_glint(
            img_refl=img_refl,
            dem_path=dem_path,
            band_data=band_data,
            sun_spd_path=sun_spd,
            sky_spd_path=sky_spd,
            satellite_local=satellite_local,
            target_local=target_local,
            sun_direction=sun_direction,
            sensor_characteristics=sensor_characteristics,
            alpha=float(alpha),
            specular_weight=float(wave_properties.get('specular_weight', 0.2)),
            tonemap_percentile=float(wave_properties.get('tonemap_percentile', 99.5)),
            return_linear=True
        )

        # Re-apply the NO-GLINT tone scale to the glint radiance for fair comparison
        rgb_lin_glint = np.clip(radiance_glint / (scale + 1e-12), 0.0, 1.0)
        DN255_glint2 = iu.linear_to_DN255(rgb_lin_glint)

        black_mask_full, black_mask_raw, y0, y1, xL, xR, dbg = compute_hit_mask_full(
            dn255_blackproj=DN255_black,
            tol=50,  # black if each channel <= 100
            row_black_frac_keep=0.8,  # relative to the best (max) row in THIS image
            min_row_black_frac_abs=0.02,
            width_keep_frac=0.4,
            interval_mode="median"
        )

        radiance_glint[~black_mask_full] = 0.0
        DN255_texture[~black_mask_full] = 0
        DN255_no_glint[~black_mask_full] = 0
        DN255_glint2[~black_mask_full] = 0

        print("DN255_black min/max:", DN255_black.min(), DN255_black.max())
        print("fraction black (raw):", np.mean(np.linalg.norm(DN255_black.astype(np.float32), axis=2) <= 2.0))

        # --- TOA reflectance (compute once, independent of plotting) ---
        cos_theta_s = float(np.sin(np.deg2rad(elev_deg)))  # WV-3: cos(theta_s)=sin(sunEl)
        d_au = 1.0

        E_R = iu.band_weighted_irradiance_integral(sun_spd, band_data["red"]["spd"])
        E_G = iu.band_weighted_irradiance_integral(sun_spd, band_data["green"]["spd"])
        E_B = iu.band_weighted_irradiance_integral(sun_spd, band_data["blue"]["spd"])

        if min(E_R, E_G, E_B) <= 0.0:
            raise ValueError(f"Non-positive band irradiance integrals: E_R={E_R}, E_G={E_G}, E_B={E_B}")

        if cos_theta_s <= 0.0:
            rho_rgb = np.zeros_like(radiance_glint, dtype=np.float32)
        else:
            rho_R = iu.radiance_to_toa_reflectance(radiance_glint[..., 0], E_R, cos_theta_s, d_au)
            rho_G = iu.radiance_to_toa_reflectance(radiance_glint[..., 1], E_G, cos_theta_s, d_au)
            rho_B = iu.radiance_to_toa_reflectance(radiance_glint[..., 2], E_B, cos_theta_s, d_au)
            rho_glint = np.stack([rho_R, rho_G, rho_B], axis=-1).astype(np.float32)

        rho_glint[~black_mask_full] = 0.0

        print(f"Reflectance min: {np.min(rho_glint)}, max: {np.max(rho_glint)}")

        m = black_mask_full.astype(bool)
        p = float(np.percentile(rho_glint[m], 99.5)) if np.any(m) else 1.0
        rho_disp = np.clip(rho_glint / (p + 1e-12), 0.0, 1.0)

        if bools['plot_result'] == True:

            fig = plt.figure(figsize=(22, 8))
            fig.add_subplot(1, 5, 1).imshow(img_rgb);
            plt.axis('off'); plt.title('original PNG')
            fig.add_subplot(1, 5, 2).imshow(DN255_texture); plt.axis('off'); plt.title('reproject (constant light)')
            fig.add_subplot(1, 5, 3).imshow(DN255_no_glint); plt.axis('off'); plt.title('render (sun, no glint)')
            fig.add_subplot(1, 5, 4).imshow(DN255_glint2); plt.axis('off'); plt.title('render (sun + glint)')
            fig.add_subplot(1, 5, 5).imshow(rho_disp); plt.axis('off'); plt.title('TOA reflectance')
            plt.show()






    else:

        off_nadir_image = render_projected_texture(img_lin, dem_path, satellite_local, target_local, sensor_characteristics)
        DN255_texture = iu.linear_to_DN255(off_nadir_image)
        DN255_no_glint = None
        DN255_glint2 = None
        radiance_glint = None
        rho_glint = None
        rho_disp = None
        black_mask_full = None
        scale = None

        if bools['plot_result'] == True:
            fig = plt.figure(figsize=(18,10))
            fig.add_subplot(1, 3, 1).imshow(img_rgb);plt.axis('off');plt.title('original');
            fig.add_subplot(1, 3, 2).imshow(DN255_texture);plt.axis('off');plt.title('off-nadir');
            fig.add_subplot(1, 3, 3).imshow(np.abs(img_rgb - DN255_texture));plt.axis('off');plt.title('difference');
            plt.show()

    gc.collect()
    dr.sync_thread()
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    return DN255_texture, DN255_no_glint, DN255_glint2, radiance_glint, rho_glint, rho_disp, black_mask_full, scale


if __name__ == "__main__":

    from pathlib import Path
    from settings import *

    bools["plot_result"] = False

    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent

    images_folder = PROJECT_ROOT / "dataset" / "whales_from_space"
    img_file = "Pelagos2016/PelagosIm2_FW_WV3_PS_20160619_B2.PNG"
    img_path = str(images_folder / img_file)
    anns_folder = PROJECT_ROOT / "dataset" / "create_dataset"
    anns_path = str(anns_folder / "final_annotations.json")

    outdir = THIS_DIR / "images"
    outdir.mkdir(parents=True, exist_ok=True)

    hour_lst = np.arange(4, 22, 1)
    minute_lst = [0, 15, 30, 45]

    sat_lat, sat_lon, sat_alt = 58.0, 0.0, 617000.0
    tgt_lat, tgt_lon, tgt_alt = 53.0, 0.0, 0.0

    p95_rad_lst_R, p95_rad_lst_G, p95_rad_lst_B = [], [], []
    mean_rad_lst_R, mean_rad_lst_G, mean_rad_lst_B = [], [], []
    p95_abs_rad_lst, mean_abs_rad_lst = [], []
    datetime_lst = []

    for hour in hour_lst:
        for minute in minute_lst:

            bools["max_glint"] = True

            dt = datetime(2025, 6, 11, int(hour), int(minute), 0, tzinfo=timezone.utc)

            DN255_texture, DN255_no_glint, DN255_glint, radiance_glint, rho_glint, rho_disp, black_mask_full, scale = generate_image(
                img_path, anns_path, satellite,
                sat_lat, sat_lon, sat_alt,
                tgt_lat, tgt_lon, tgt_alt,
                dt, sensor_characteristics, wave_properties, bools, seed_dem
            )

            if radiance_glint is None or DN255_glint is None or black_mask_full is None:
                continue

            mask = black_mask_full.astype(bool)
            if np.count_nonzero(mask) == 0:
                continue

            abs_rad = np.sqrt(np.sum(np.square(radiance_glint.astype(np.float64)), axis=2))
            vals = abs_rad[mask]
            if vals.size == 0 or float(np.max(vals)) <= 1e-12:
                continue

            save_path_img = outdir / f"{hour}-{minute}h.png"
            Image.fromarray(np.clip(DN255_glint, 0, 255).astype(np.uint8)).save(save_path_img)

            p95_abs_rad_lst.append(masked_percentile(abs_rad, mask, 95.0))
            p95_R, p95_G, p95_B = masked_channel_percentiles(radiance_glint, mask, 95.0)
            p95_rad_lst_R.append(p95_R)
            p95_rad_lst_G.append(p95_G)
            p95_rad_lst_B.append(p95_B)

            mean_abs_rad_lst.append(masked_mean(abs_rad, mask))
            mR, mG, mB = masked_channel_means(radiance_glint, mask)
            mean_rad_lst_R.append(mR)
            mean_rad_lst_G.append(mG)
            mean_rad_lst_B.append(mB)

            datetime_lst.append(dt)

            print(f"Saved image for {hour}:{minute:02d}h -> {save_path_img}")
            print("Max glint (masked abs):", float(np.max(vals)))

    plot_radiance_timeline(
        datetime_lst=datetime_lst,
        series=[p95_rad_lst_R, p95_rad_lst_G, p95_rad_lst_B],
        labels=["Red (p95)", "Green (p95)", "Blue (p95)"],
        styles=["r", "g", "b"],
        ylabel=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
        title="95th percentile band radiance (masked)",
        save_path=outdir / "radiance_timeline_rgb_p95.png"
    )

    plot_radiance_timeline(
        datetime_lst=datetime_lst,
        series=[mean_rad_lst_R, mean_rad_lst_G, mean_rad_lst_B],
        labels=["Red (mean)", "Green (mean)", "Blue (mean)"],
        styles=["r--", "g--", "b--"],
        ylabel=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
        title="Mean band radiance (masked)",
        save_path=outdir / "radiance_timeline_rgb_mean.png"
    )

    plot_radiance_timeline(
        datetime_lst=datetime_lst,
        series=[p95_abs_rad_lst, mean_abs_rad_lst],
        labels=["‖L‖ p95", "‖L‖ mean"],
        styles=["k", "k--"],
        ylabel=r"‖Radiance‖ [W m$^{-2}$ sr$^{-1}$]",
        title="Absolute radiance over time (masked)",
        save_path=outdir / "radiance_timeline_abs_p95_mean.png"
    )
