import math
import os, sys
import numpy as np
import mitsuba as mi
import drjit as dr
from datetime import datetime, timezone
from matplotlib import pyplot as plt
from PIL import Image

from .RTM import generate_solar_spd_from_target
from .create_DEM.create_dummy_DEM import get_DEM
from .create_DEM.convert_DEM import convert_DEM

from .functions.plotfunctions import plot_earth_with_pyvista, plot_earth_slice_with_sun, plot_target_perspective, get_rgb
from .functions.get_satellite_data import get_band_data, get_satellite, get_spatial_res
from .functions.convert_reference_frames import get_lat_lon_alt_from_ecef, get_ecef_from_lat_lon, compute_max_glint_satellite_ecef
from .functions.intermediate_functions import rmse, normalize, get_scene_characteristics
from .functions import image_utils as iu

def get_radiance_sunglint(input_img, dem_path, spd_path, solar_spd, satellite_local, target_local, sun_direction, sensor_characteristics, alpha):
    mi.set_variant('llvm_ad_spectral')

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-sensor_characteristics['azimuth_rad'])
    )

    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    # Load scene with textured mesh

    texture = {
        "type": "bitmap",
        "data": input_img,
        "wrap_mode": "clamp",  # or "clamp"
        "filter_type": "nearest",
        "raw": True  # Treat image as raw (no gamma)
    }

    brdf = {
        "type": "roughdielectric",
        "distribution": "ggx",  # or 'ggx'
        "alpha": alpha,
        "int_ior": 1.3330,  # Water
        "ext_ior": 1.000277,  # Air
    }

    to_world_sensor = mi.ScalarTransform4f().look_at(
        origin=satellite_local,
        target=target_local,
        up=[0, 0, 1]
    )

    #    brdf = {
    #        "type": "roughplastic",
    #        "distribution": "beckmann",  # or ggx
    #        "alpha": alpha,
    #        "int_ior": 1.33,
    #        "ext_ior": 1.0,
    #        "diffuse_reflectance": texture
    #    }

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "path"},

        "earth_surface": {
            "type": "obj",
            "filename": dem_path,
            "to_world": to_world_scene,
            #  "to_world": mi.ScalarTransform4f().translate([0, 0, 0]),
            "bsdf": brdf
        },

        "sun": {
            "type": "directional",
            "direction":  mi.ScalarVector3f(sun_direction),
            "irradiance": {
                    'type': 'spectrum',
                    'filename': solar_spd
                 },
        },

        "sensor": {
            "type": "perspective",
            "to_world": to_world_sensor,
            "fov": sensor_characteristics['fov_deg'],
            'far_clip': 1e8,
            "film": {
                "type": "specfilm",
                "width": sensor_characteristics['resolution'],
                "height": sensor_characteristics['resolution'],
                'spectral_band': {
                    'type': 'spectrum',
                    'filename': spd_path
                },
                # "rfilter": {"type": "box"}
            },

            "sampler": {
                "type": "independent",
                "sample_count": sensor_characteristics['sample_count']
            },
        },

    }

    # Load and render the scene
    scene = mi.load_dict(scene_dict)
    radiance_sunglint = mi.render(scene)

    return radiance_sunglint


def get_image_offnadir(input_img, dem_path, satellite_local, target_local, sensor_characteristics):

    mi.set_variant('llvm_ad_rgb')

    scene_rotation = mi.ScalarTransform4f().rotate(
        axis=mi.ScalarVector3f(0, 0, 1),
        angle=math.degrees(-sensor_characteristics['azimuth_rad'])
    )

    scene_mirror = mi.ScalarTransform4f().scale([-1, 1, 1])
    to_world_scene = scene_rotation @ scene_mirror

    texture = {
        "type": "bitmap",
        "data": input_img,
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
                "sample_border": True,
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

    return image_offnadir

def generate_image(img_path, satellite, satellite_lat, satellite_lon, satellite_alt, target_lat, target_lon, target_alt, datetime_utc, sensor_characteristics, wave_properties, bools, dem_seed):

    dr.set_flag(dr.JitFlag.Debug, True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Temp file paths
    dem_tiff_path = os.path.join(script_dir, 'create_DEM/input_dem_WV.tiff')
    dem_path = os.path.join(script_dir, "create_DEM/dem_mesh_WV.obj")
    spd_folder = os.path.join(script_dir, 'spd_files')
    solar_spd = os.path.join(spd_folder, 'solar_spectral_irradiance.spd')

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

    img_rgb = np.asarray(Image.open(img_path).convert('RGB'))
    if bools['crop_black_border']:
        img_rgb = iu.crop_black_border_image(img_rgb, threshold=1)

    img_np = np.array(img_rgb)
    img_height, img_width = img_np.shape[:2]

    img_lin = iu.DN255_to_linear(img_rgb)
    if bools['print_values']:
        print("Loaded input image with DN255 min ", np.min(img_rgb), 'max ', np.max(img_rgb))

    satellite_local, target_local, sun_direction, fov_deg, off_nadir_rad, azimuth_rad = get_scene_characteristics(
        satellite_ecef, target_ecef, sun_ecef, img_height, img_width, GSD)

    if bools['print_values']:
        print(f"\nFOV \t\t: {fov_deg:.5f} deg")
        print(f"Off Nadir \t: {off_nadir_rad * 180 / np.pi:.2f} deg")
        print(f"Azimuth \t: {azimuth_rad * 180 / np.pi:.2f} deg\n")

    sensor_characteristics['fov_deg'] = fov_deg
    sensor_characteristics['azimuth_rad'] = azimuth_rad

    if bools['print_values']:
        print(f"Generate off nadir image\n")
    off_nadir_image = get_image_offnadir(img_lin, dem_path, satellite_local, target_local, sensor_characteristics)
    # off_nadir_image = np.flip(off_nadir_image, axis=0)

    # Split bands
    R_img = img_rgb[:, :, 0];    G_img = img_rgb[:, :, 1];    B_img =img_rgb[:, :, 2]
    band_data['red']['input_img'] = R_img[:, :, np.newaxis]; band_data['green']['input_img'] = G_img[:, :, np.newaxis]; band_data['blue']['input_img'] = B_img[:, :, np.newaxis]

    sun_direction_away = -np.array(sun_direction)
    min_wvl, max_wvl = 250, 1000  # nm

    generate_solar_spd_from_target(
        datetime_utc=datetime_utc,
        target_lat=target_lat,
        target_lon=target_lon,
        target_alt=target_alt,
        sun_direction=sun_direction_away,
        output_path=solar_spd,
        wavelength_range=[min_wvl, max_wvl],
        timezone_offset=0,
        material="Water"
    )
    if bools['print_values']:
        print(f"Saved solar SPD to {solar_spd}\n")

    abs_cal_factor = band_data['abs_cal_factor']
    gains_arr = []
    offset_arr = []
    eff_bw_arr = []

    if bools['print_values']:
        print("Generate radiance image for channels: (W m-2 µm-1 sr-1)")
    for band_name in ['red', 'green', 'blue']:

        input_img = band_data[band_name]["input_img"]
        spd_path = band_data[band_name]["spd"]
        gain = band_data[band_name]["gain"]
        offset = band_data[band_name]["offset"]
        eff_bw = band_data[band_name]["eff_bw"]

        # Obtain radiance per band
        img_lin = iu.DN255_to_linear(input_img)
        radiance_sunglint_n = get_radiance_sunglint(img_lin, dem_path, spd_path, solar_spd, satellite_local, target_local, sun_direction, sensor_characteristics, alpha)
        radiance_sunglint_n = radiance_sunglint_n * 1000  # from [W nm-1 m-2 sr-1] to [W µm-1 m-2 sr-1]

        # Save and convert to Digital Number
        band_data[band_name]['radiance_sunglint'] = radiance_sunglint_n
        if bools['print_values']:
            print(f"{band_name} min: {np.min(radiance_sunglint_n):.1f}, max: {np.max(radiance_sunglint_n):.1f} ")

        # Append array values for future computations
        gains_arr.append(gain);        offset_arr.append(offset);        eff_bw_arr.append(eff_bw)

    # Convert to numpy
    gains_arr = np.array(gains_arr);    offset_arr = np.array(offset_arr);    eff_bw_arr = np.array(eff_bw_arr)

    # Convert image formats
    radiance_sunglint = iu.stack_rgb_img('radiance_sunglint', band_data)
    DN2047_sunglint = iu.radiance_to_DN2047(radiance_sunglint, gains_arr, offset_arr, eff_bw_arr, abs_cal_factor)
    DN255_sunglint = iu.DN2047_to_DN255(DN2047_sunglint)

    DN255_offnadir = iu.linear_to_DN255(off_nadir_image)
    DN2047_offnadir = iu.linear_to_DN2047(off_nadir_image)
    radiance_offnadir = iu.DN2047_to_radiance(DN2047_offnadir, gains_arr, offset_arr, eff_bw_arr, abs_cal_factor)

    radiance_combined = radiance_offnadir + radiance_sunglint
    DN2047_combined = iu.radiance_to_DN2047(radiance_combined, gains_arr, offset_arr, eff_bw_arr, abs_cal_factor)
    DN255_combined = iu.DN2047_to_DN255(DN2047_combined)

    if bools['plot_result'] == True:

        fig = plt.figure(figsize=(18,10))
        fig.add_subplot(1, 4, 1).imshow(img_rgb);plt.axis('off');plt.title('original');
        fig.add_subplot(1, 4, 2).imshow(DN255_offnadir);plt.axis('off');plt.title('off-nadir');
        fig.add_subplot(1, 4, 3).imshow(DN255_sunglint);plt.axis('off');plt.title('sun glint');
        fig.add_subplot(1, 4, 4).imshow(DN255_combined);plt.axis('off');plt.title('off-nadir + sun glint');
        plt.show()

    return DN255_offnadir, DN255_sunglint, radiance_sunglint, DN255_combined


if __name__ == "__main__":

    images_folder = "../dataset/whales_from_space/"
    img_file = 'Pelagos2016/PelagosIm2_FW_WV3_PS_20160619_B11.PNG'

    csv_path = os.path.join(images_folder, "WhaleFromSpaceDB_Whales.csv")
    img_path = os.path.join(images_folder, img_file)

    print_values = True
    plot_3d = False
    plot_result = True
    max_glint = False
    crop_black_border = True

    satellite = get_satellite(img_path, csv_path)

    hour_lst = np.arange(6,22, 1)
    minute_lst = [0, 15, 30, 45]

    satellite_lat, satellite_lon, satellite_alt = 58, -5, 617000.0  # lat, lon, m
    target_lat, target_lon, target_alt = 53, 0, 0.0  # lat, lon, meters

    GSD = get_spatial_res(img_path, csv_path)
    resolution = 124    # pixels of render
    sample_count = 512  # 8192 min, 2048 * 2**7 max

    dem_seed = 42

    wind_speed = 10.0  # m/s
    num_waves = 50
    wave_min = 0.05  # m
    wave_max = 0.5  # m

    wave_properties = {}
    wave_properties['wind_speed'] = wind_speed
    wave_properties['num_waves'] = num_waves
    wave_properties['wave_min'] = wave_min
    wave_properties['wave_max'] = wave_max

    bools = {}
    bools['plot_3d'] = plot_3d
    bools['plot_result'] = plot_result
    bools['max_glint'] = max_glint
    bools['print_values'] = print_values
    bools['crop_black_border'] = crop_black_border

    sensor_characteristics = {}
    sensor_characteristics['resolution'] = resolution
    sensor_characteristics['sample_count'] = sample_count
    sensor_characteristics['GSD'] = GSD

    max_rad_lst_R = []
    max_rad_lst_G = []
    max_rad_lst_B = []

    max_rad_lst = []
    datetime_lst = []
    for hour in hour_lst:
        for minute in minute_lst:

            save_name =  'images/' + str(hour) + '-' + str(minute) + 'h.png'
            datetime_utc = datetime(2025, 6, 11, hour, minute, 0, tzinfo=timezone.utc)
            DN255_rgb_offnadir, DN255_rgb_sunglint, radiance_sunglint, DN255_combined = generate_image(img_path, satellite, satellite_lat, satellite_lon, satellite_alt, target_lat, target_lon, target_alt, datetime_utc, sensor_characteristics, wave_properties, bools, dem_seed)

            image_uint8 = np.clip(DN255_combined, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_uint8)
            img.save(save_name)

            max_rad = np.max(radiance_sunglint)
            max_rad_lst.append(max_rad)
            datetime_lst.append(datetime_utc)

            max_rad_R = np.max(radiance_sunglint[:, :, 0])
            max_rad_G = np.max(radiance_sunglint[:, :, 1])
            max_rad_B = np.max(radiance_sunglint[:, :, 2])

            max_rad_lst_R.append(max_rad_R)
            max_rad_lst_G.append(max_rad_G)
            max_rad_lst_B.append(max_rad_B)

            print("Saved image under ", save_name + '\n')
            print('Max glint:', np.max(radiance_sunglint))

    plt.plot(datetime_lst, max_rad_lst_R, 'r')
    plt.plot(datetime_lst, max_rad_lst_G, 'g')
    plt.plot(datetime_lst, max_rad_lst_B, 'b')
    plt.grid(True)
    plt.show()




















