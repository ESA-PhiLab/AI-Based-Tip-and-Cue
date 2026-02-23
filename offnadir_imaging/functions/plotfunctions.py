import numpy as np
import pyvista
from pyvista import examples

from paseos.custom_paseos.utils.point_transformation import Point_ECEF2Geodetic

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import shutil
import math

from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.dates as mdates


from .convert_reference_frames import  get_ecef_from_lat_lon

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def plot_earth_with_pyvista(satellite, feature, sun, R_earth):
    light = pyvista.Light()

    P_sun = Point_ECEF2Geodetic(sun[0], sun[1], sun[2])
    light.set_direction_angle(-P_sun[0] + 180,P_sun[1])

    earth = examples.planets.load_earth(radius=R_earth)
    earth_texture = examples.load_globe_texture()

    satellite_copy = satellite.copy()
    feature_copy = feature.copy()
    sun_copy = sun.copy()

    pl = pyvista.Plotter(shape=(1, 1), lighting='none')
    cubemap = examples.download_cubemap_space_4k()
    _ = pl.add_actor(cubemap.to_skybox())
    pl.set_environment_texture(cubemap, True)
    pl.add_light(light)

    pl.subplot(0, 0)
    pl.add_text("3D View", font_size = 12)
    pl.add_mesh(earth, texture=earth_texture, smooth_shading=True)
    pl.link_views()

    satellite_copy[0] = -satellite_copy[0]
    satellite_copy[1] = -satellite_copy[1]
    feature_copy[0] = -feature_copy[0]
    feature_copy[1] = -feature_copy[1]
    sun_copy[0] = -sun_copy[0]
    sun_copy[1] = -sun_copy[1]

    # Plot satellite (represented as a point)
    pl.add_points(satellite_copy, color="teal", point_size=16, render_points_as_spheres=True, label = 'Satellite')
    pl.add_points(feature_copy, color="violet", point_size=16, render_points_as_spheres=True, label = 'Feature')

    vector = sun_copy - feature_copy
    distance = np.linalg.norm(vector)
    direction = vector / distance
    short_vector = direction * (distance / 20000)

    pl.add_lines(np.array([satellite_copy, feature_copy]), color='black', width=3)
    pl.add_lines(np.array([feature_copy, feature_copy+short_vector]), color='peachpuff', width=3)
    pl.add_lines(np.array([[0,0,0], sun_copy/10000]), color='peachpuff', width=3)

    # pl.add_lines(np.array([np.array([0, 0, 0]), np.array([sun_copy[0], 0, 0] )/ 10000]), color='green', width=3)   # x axis

    # Set view options
    # pl.set_background('black')
    pl.show_axes()
    pl.view_isometric()
    pl.add_legend(bcolor='w', face='circle', size = (0.12, 0.12))

    pl.show(cpos="xy")

def plot_earth_slice_with_sun(satellite, feature, sun_direction, R_earth, ax2d):
    satellite_vector = np.array(satellite)
    feature_vector = np.array(feature)
    sun_vector = np.array(sun_direction)

    # Calculate the normal to the plane formed by the Earth center, satellite, and feature
    normal_vector = np.cross(satellite_vector, feature_vector)
    normal_vector /= np.linalg.norm(normal_vector)

    # Plot Earth as a circle in 2D projection on the plane
    earth_circle = plt.Circle((0, 0), R_earth, color='lightblue', alpha=0.5, label='Earth')
    ax2d.add_patch(earth_circle)

    # Compute rotation to align plane normal with Z-axis
    def rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        ux, uy, uz = axis
        return np.array([
            [cos_a + ux**2 * (1 - cos_a),      ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
            [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a),      uy*uz*(1 - cos_a) - ux*sin_a],
            [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
        ])

    rotation_axis = np.cross(normal_vector, [0, 0, 1])
    if np.linalg.norm(rotation_axis) < 1e-8:
        rotation_mat = np.eye(3)  # Already aligned
    else:
        rotation_angle = np.arccos(np.clip(np.dot(normal_vector, [0, 0, 1]), -1.0, 1.0))
        rotation_mat = rotation_matrix(rotation_axis, rotation_angle)

    # Rotate vectors
    sat_rot = rotation_mat @ satellite_vector
    feat_rot = rotation_mat @ feature_vector
    sun_rot = rotation_mat @ sun_vector

    x_s, y_s, _ = sat_rot
    x_f, y_f, _ = feat_rot
    x_sun, y_sun, _ = sun_rot

    # Plot line of sight and points
    ax2d.plot([x_s, x_f], [y_s, y_f], color='black', label='Line of Sight')
    ax2d.scatter(x_s, y_s, color='teal', s=100, label='Satellite')
    ax2d.scatter(x_f, y_f, color='violet', s=100, label='Feature')

    # Normalize and scale sun direction
    sun_dir_2d = np.array([x_sun, y_sun])
    sun_dir_2d = sun_dir_2d / np.linalg.norm(sun_dir_2d) * R_earth * 1.2

    # From Sun to Earth (arrow pointing toward Earth center)
    arrow_start = sun_dir_2d * R_earth * 1.2
    arrow_vec = -sun_dir_2d * R_earth * 1.2

    ax2d.arrow(
        arrow_start[0], arrow_start[1],
        arrow_vec[0], arrow_vec[1],
        head_width=R_earth * 0.07,
        color='orange',
        label='Sun Direction',
        length_includes_head=True
    )

    # Axes and plot settings
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Projection of Earth Slice with Sun Direction')
    ax2d.set_aspect('equal', 'box')

    distance = np.sqrt(x_s**2 + y_s**2)
    margin = 1.5 * (distance - R_earth)
    ax2d.set_xlim(-R_earth - margin, R_earth + margin)
    ax2d.set_ylim(-R_earth - margin, R_earth + margin)
    ax2d.legend()



def plot_target_perspective(satellite, feature, sun_direction, ax2d):
    satellite_vector = np.array(satellite)
    feature_vector = np.array(feature)
    sun_vector = np.array(sun_direction)

    print('sun', sun_vector , 'sat', satellite, 'feature', feature)

    if not (feature_vector==np.zeros_like(feature_vector)).all():
        # Calculate the normal to the plane formed by the Earth center, satellite, and feature
        normal_vector = np.cross(satellite_vector, feature_vector)
        normal_vector /= np.linalg.norm(normal_vector)


        # Compute rotation to align plane normal with Z-axis
        def rotation_matrix(axis, angle):
            axis = axis / np.linalg.norm(axis)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            ux, uy, uz = axis
            return np.array([
                [cos_a + ux**2 * (1 - cos_a),      ux*uy*(1 - cos_a) - uz*sin_a, ux*uz*(1 - cos_a) + uy*sin_a],
                [uy*ux*(1 - cos_a) + uz*sin_a, cos_a + uy**2 * (1 - cos_a),      uy*uz*(1 - cos_a) - ux*sin_a],
                [uz*ux*(1 - cos_a) - uy*sin_a, uz*uy*(1 - cos_a) + ux*sin_a, cos_a + uz**2 * (1 - cos_a)]
            ])

        rotation_axis = np.cross(normal_vector, [0, 0, 1])
        if np.linalg.norm(rotation_axis) < 1e-8:
            rotation_mat = np.eye(3)  # Already aligned
        else:
            rotation_angle = np.arccos(np.clip(np.dot(normal_vector, [0, 0, 1]), -1.0, 1.0))
            rotation_mat = rotation_matrix(rotation_axis, rotation_angle)

        # Rotate vectors
        sat_rot = rotation_mat @ satellite_vector
        feat_rot = rotation_mat @ feature_vector
        sun_rot = rotation_mat @ sun_vector

        x_s, y_s, _ = sat_rot
        x_f, y_f, _ = feat_rot
        x_sun, y_sun, _ = sun_rot

        x_s_target = x_s - x_f
        y_s_target = y_s - y_f
        x_f_target = x_f - x_f
        y_f_target = y_f - y_f
        x_sun_target = x_sun - x_f
        y_sun_target = y_sun - y_f

    print('sun', [x_sun_target, y_sun_target], 'sat', [x_s_target, y_s_target], 'feature', [x_f_target, y_f_target])
    # Normalize and scale sun direction
    sun_dir_2d_target = np.array([x_sun_target, y_sun_target])
    sun_dir_2d_target = sun_dir_2d_target / np.linalg.norm(sun_dir_2d_target)

    # Plot line of sight and points
    ax2d.plot([x_s_target, x_f_target], [y_s_target, y_f_target], color='black', label='Line of Sight')
    ax2d.scatter(x_s_target, y_s_target, color='teal', s=100, label='Satellite')
    ax2d.scatter(x_f_target, y_f_target, color='violet', s=100, label='Feature')

    scaling = np.linalg.norm(x_s_target)
    print(scaling)
    # From Sun to Earth (arrow pointing toward Earth center)
    arrow_start = sun_dir_2d_target  * scaling * 1.2
    arrow_vec = -sun_dir_2d_target  * scaling * 1.2

    ax2d.arrow(
        arrow_start[0], arrow_start[1],
        arrow_vec[0], arrow_vec[1],
        color='orange',
        label='Sun Direction',
        length_includes_head=True
    )

    # Axes and plot settings
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('2D Projection of Earth Slice with Sun Direction')
    ax2d.set_aspect('equal', 'box')
    ax2d.legend()

    distance = np.sqrt(x_s_target**2 + x_s_target**2)
    margin = 1.5 * (distance - scaling)
    ax2d.set_xlim(-scaling  - margin, scaling + margin)
    ax2d.set_ylim(-scaling - margin, scaling + margin)
    ax2d.legend()

def get_rgb(bmp):
    return np.power(np.clip(np.array(bmp), 0.0, 1.0), 1 / 2.2)



def save_off_nadir_plots(img_rgb, texture_disp, radiance_disp_final, rho_disp_no_glint, rho_disp_final, output_dir="test", tag=""):
    """Save individual and combined off-nadir comparison plots. Returns list of saved file paths."""

    os.makedirs(output_dir, exist_ok=True)

    if tag != "":
        tag = f"_{tag}"

    saved_files = []

    images = [
        ("01_original", img_rgb, "original PNG"),
        ("02_reproject_constant_light", texture_disp, "reproject (constant light)"),
        ("03_radiance_glint", radiance_disp_final, "Radiance (glint)"),
        ("04_reflectance_no_glint", rho_disp_no_glint, "Reflectance (no glint)"),
        ("05_reflectance_glint", rho_disp_final, "Reflectance (glint)")
    ]

    # --------- Individual plots ---------
    for name, img, title in images:
        path = os.path.join(output_dir, f"{name}{tag}.png")
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_files.append(path)

    # --------- Combined plot ---------
    combined_path = os.path.join(output_dir, f"00_full_comparison{tag}.png")

    fig = plt.figure(figsize=(22, 8))
    for i, (_, img, title) in enumerate(images):
        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(combined_path)

    return saved_files

def plot_radiance_timeline_show_save(datetime_lst, series, labels, styles, ylabel, title, save_path, show=True):
    """Plot timeline series, format x-axis as HH:MM, save to save_path, optionally show. Returns Path(save_path)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(datetime_lst) == 0:
        return save_path

    x = np.array(datetime_lst, dtype=object)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    for y, lab, sty in zip(series, labels, styles):
        ax.plot(x, y, sty, label=lab)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Force HH:MM tick labels (no date)
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return save_path

def solar_zenith_deg(target_ecef: np.ndarray, sun_ecef: np.ndarray) -> float:
    """solar_zenith_deg(target_ecef, sun_ecef) -> float: Solar zenith angle at target [deg]."""
    r_t = np.asarray(target_ecef, dtype=float)
    r_s = np.asarray(sun_ecef, dtype=float)

    n = r_t / np.linalg.norm(r_t)                  # local (spherical) surface normal
    s = (r_s - r_t)
    s = s / np.linalg.norm(s)                      # direction from target to Sun

    cos_sza = float(np.clip(np.dot(n, s), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_sza)))


def cos_solar_zenith(target_ecef: np.ndarray, sun_ecef: np.ndarray) -> float:
    """cos_solar_zenith(target_ecef, sun_ecef) -> float: cos(theta_s) at target [-]."""
    r_t = np.asarray(target_ecef, dtype=float)
    r_s = np.asarray(sun_ecef, dtype=float)

    n = r_t / np.linalg.norm(r_t)
    s = (r_s - r_t)
    s = s / np.linalg.norm(s)

    return float(np.dot(n, s))





def run_glint_timeline(img_path, anns_path, satellite, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, sensor_characteristics, wave_properties, bools, seed_dem, outdir, hours, minutes, generate_image_fn, masked_percentile_fn, masked_channel_percentiles_fn, masked_mean_fn, masked_channel_means_fn, image_prefix="glint", save_images=True, show=True, date_ymd=(2025, 6, 11)):
    """Run generate_image over a time grid, save per-time images, and save+display radiance timelines. Returns results dict."""

    import numpy as np
    outdir = Path(outdir)

    # Remove existing folder completely
    if outdir.exists():
        shutil.rmtree(outdir)

    # Recreate clean directory
    outdir.mkdir(parents=True, exist_ok=True)

    p95_rad_lst_R, p95_rad_lst_G, p95_rad_lst_B = [], [], []
    mean_rad_lst_R, mean_rad_lst_G, mean_rad_lst_B = [], [], []
    p95_abs_rad_lst, mean_abs_rad_lst = [], []
    datetime_lst = []
    saved_images_rad = []
    saved_images_refl = []

    Y, M, D = int(date_ymd[0]), int(date_ymd[1]), int(date_ymd[2])
    solar_sza_deg_lst = []
    cos_sza_lst = []

    for hour in hours:
        for minute in minutes:
            dt = datetime(Y, M, D, int(hour), int(minute), 0, tzinfo=timezone.utc)


            (texture_disp,
             radiance_no_glint, radiance_disp_no_glint, rho_no_glint, rho_disp_no_glint,
             radiance_final, radiance_disp_final, rho_final, rho_disp_final,
             black_mask_full, scale, offnadir_deg) = generate_image_fn(
                img_path, anns_path, satellite,
                sat_lat, sat_lon, sat_alt,
                tgt_lat, tgt_lon, tgt_alt,
                dt, sensor_characteristics, wave_properties, bools, seed_dem
            )

            if radiance_final is None or radiance_disp_final is None or black_mask_full is None:
                continue

            mask = black_mask_full.astype(bool)
            if np.count_nonzero(mask) == 0:
                continue

            abs_rad = np.sqrt(np.sum(np.square(radiance_final.astype(np.float64)), axis=2))
            vals = abs_rad[mask]
            if vals.size == 0 or float(np.max(vals)) <= 1e-12:
                continue

            if save_images:

                save_path_img_refl = outdir / "reflection" / f"{image_prefix}_{hour:02d}-{minute:02d}h.png"
                save_path_img_refl.parent.mkdir(parents=True, exist_ok=True)
                img_refl = (rho_disp_final * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img_refl).save(save_path_img_refl)
                saved_images_refl.append(save_path_img_refl)

                save_path_img_rad = outdir / "radiance" / f"{image_prefix}_{hour:02d}-{minute:02d}h.png"
                save_path_img_rad.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.clip(radiance_disp_final, 0, 255).astype(np.uint8)).save(save_path_img_rad)
                saved_images_rad.append(save_path_img_rad)

            p95_abs_rad_lst.append(masked_percentile_fn(abs_rad, mask, 95.0))
            p95_R, p95_G, p95_B = masked_channel_percentiles_fn(radiance_final, mask, 95.0)
            p95_rad_lst_R.append(p95_R)
            p95_rad_lst_G.append(p95_G)
            p95_rad_lst_B.append(p95_B)

            mean_abs_rad_lst.append(masked_mean_fn(abs_rad, mask))
            mR, mG, mB = masked_channel_means_fn(radiance_final, mask)
            mean_rad_lst_R.append(mR)
            mean_rad_lst_G.append(mG)
            mean_rad_lst_B.append(mB)

            _, target_ecef, sun_ecef = get_ecef_from_lat_lon(
                sat_lat, sat_lon, sat_alt,
                tgt_lat, tgt_lon, tgt_alt,
                dt,
                generate_nadir=bools.get("generate_nadir", False),
            )

            sza = solar_zenith_deg(target_ecef, sun_ecef)
            cos_sza = cos_solar_zenith(target_ecef, sun_ecef)

            solar_sza_deg_lst.append(sza)
            cos_sza_lst.append(cos_sza)
            datetime_lst.append(dt)

    # Save + (optionally) display timelines
    plot_radiance_timeline_show_save(
        datetime_lst=datetime_lst,
        series=[p95_rad_lst_R, p95_rad_lst_G, p95_rad_lst_B],
        labels=["Red (p95)", "Green (p95)", "Blue (p95)"],
        styles=["r", "g", "b"],
        ylabel=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
        title="95th percentile band radiance (masked)",
        save_path=outdir / "radiance_timeline_rgb_p95.png",
        show=show
    )

    plot_radiance_timeline_show_save(
        datetime_lst=datetime_lst,
        series=[mean_rad_lst_R, mean_rad_lst_G, mean_rad_lst_B],
        labels=["Red (mean)", "Green (mean)", "Blue (mean)"],
        styles=["r", "g", "b"],
        ylabel=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
        title="Mean band radiance (masked)",
        save_path=outdir / "radiance_timeline_rgb_mean.png",
        show=show
    )

    plot_radiance_timeline_show_save(
        datetime_lst=datetime_lst,
        series=[p95_abs_rad_lst, mean_abs_rad_lst],
        labels=["‖L‖ p95", "‖L‖ mean"],
        styles=["k", "k--"],
        ylabel=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
        title="Absolute radiance over time (masked)",
        save_path=outdir / "radiance_timeline_abs_p95_mean.png",
        show=show
    )

    # Solar azimuth timeline (deg)
    # Solar zenith timeline (deg) + cos(theta_s)
    if len(datetime_lst) > 0 and len(cos_sza_lst) == len(datetime_lst):
        plot_radiance_timeline_show_save(
            datetime_lst=datetime_lst,
            series=[solar_sza_deg_lst],
            labels=[r"Solar zenith $\theta_s$"],
            styles=["k"],
            ylabel=r"Solar zenith [deg]",
            title="Solar zenith over time",
            save_path=outdir / "solar_zenith_timeline_deg.png",
            show=show
        )

        plot_radiance_timeline_show_save(
            datetime_lst=datetime_lst,
            series=[cos_sza_lst],
            labels=[r"$\cos(\theta_s)$"],
            styles=["k"],
            ylabel=r"$\cos(\theta_s)$ [-]",
            title="Cosine solar zenith over time",
            save_path=outdir / "cos_solar_zenith_timeline.png",
            show=show
        )


    timeline_strip_path = None
    if save_images and len(saved_images_rad) > 0:
        timeline_strip_path_rad = make_horizontal_timeline_image(
            image_paths=saved_images_rad,
            datetimes=datetime_lst,
            save_path=outdir / "radiance" / f"{image_prefix}_timeline_strip_rad.png",
            target_height=256,
            pad=6,
            bg_rgb=(0, 0, 0),
            label_every=4,
            time_fmt="%H:%M",
            label_margin_px=80,
            font_size=48  # <-- bigger
        )

        timeline_strip_path_refl = make_horizontal_timeline_image(
            image_paths=saved_images_refl,
            datetimes=datetime_lst,
            save_path=outdir / "reflection"/ f"{image_prefix}_timeline_strip_refl.png",
            target_height=256,
            pad=6,
            bg_rgb=(0, 0, 0),
            label_every=4,
            time_fmt="%H:%M",
            label_margin_px=80,
            font_size=48  # <-- bigger
        )

        if show:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                strip = np.asarray(Image.open(timeline_strip_path_rad))
                plt.figure(figsize=(16, 4))
                plt.imshow(strip)
                plt.axis("off")
                plt.title("Horizontal timeline strip")
                plt.show()
                plt.close()

                strip = np.asarray(Image.open(timeline_strip_path_refl))
                plt.figure(figsize=(16, 4))
                plt.imshow(strip)
                plt.axis("off")
                plt.title("Horizontal timeline strip")
                plt.show()
                plt.close()
            except Exception:
                pass

    return {
        "datetime_lst": datetime_lst,
        "p95_rgb": (p95_rad_lst_R, p95_rad_lst_G, p95_rad_lst_B),
        "mean_rgb": (mean_rad_lst_R, mean_rad_lst_G, mean_rad_lst_B),
        "p95_abs": p95_abs_rad_lst,
        "mean_abs": mean_abs_rad_lst,
        "saved_images_rad": saved_images_rad,
        "saved_images_refl": saved_images_rad,
        "timeline_strip_path": timeline_strip_path,
        "outdir": outdir,
    }



def make_horizontal_timeline_image(image_paths, save_path, datetimes=None, target_height=256, pad=6, bg_rgb=(0, 0, 0), label_every=4, time_fmt="%H:%M", label_margin_px=60, font_size=36):
    """Concatenate images left-to-right, add large time labels below with fixed step, save to save_path. Returns Path(save_path)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in image_paths if p is not None]
    if len(paths) == 0:
        return save_path

    if datetimes is not None:
        datetimes = list(datetimes)
        if len(datetimes) != len(paths):
            datetimes = None

    imgs = []
    widths = []

    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue

        if target_height and target_height > 0:
            w, h = im.size
            if h > 0:
                new_w = int(round(w * (float(target_height) / float(h))))
                im = im.resize((max(new_w, 1), int(target_height)), resample=Image.BILINEAR)

        imgs.append(im)
        widths.append(im.size[0])

    if len(imgs) == 0:
        return save_path

    total_w = sum(widths) + pad * (len(imgs) - 1)
    img_h = max(im.size[1] for im in imgs)

    extra_h = int(label_margin_px) if datetimes is not None and label_every and label_every > 0 else 0
    canvas_h = img_h + extra_h

    canvas = Image.new("RGB", (total_w, canvas_h), color=tuple(int(x) for x in bg_rgb))
    draw = ImageDraw.Draw(canvas)

    # Try to load a scalable font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    x = 0
    for i, im in enumerate(imgs):
        canvas.paste(im, (x, 0))

        if datetimes is not None and label_every and label_every > 0 and (i % int(label_every) == 0):
            t = datetimes[i].strftime(time_fmt)

            bbox = draw.textbbox((0, 0), t, font=font)
            txt_w = bbox[2] - bbox[0]
            txt_h = bbox[3] - bbox[1]

            cx = x + im.size[0] // 2
            tx = int(cx - txt_w // 2)
            ty = int(img_h + max((extra_h - txt_h) // 2, 4))

            draw.text((tx, ty), t, fill=(255, 255, 255), font=font)

        x += im.size[0] + pad

    canvas.save(save_path)
    return save_path
