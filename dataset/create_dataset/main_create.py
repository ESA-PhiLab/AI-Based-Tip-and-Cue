import json
import os
import subprocess
import sys
import time
from pathlib import Path
import shutil
from typing import Iterator, List, Sequence
import numpy as np


from read_and_write_data import (
    cleanup_previous_outputs,
    cleanup_meta_only,
    open_overview_book,
    append_run_rows,
    pick_random_pose, count_images_in_subfolders
)

main_path = Path(__file__).resolve().parents[2]
os.chdir(main_path)

SCRIPT_DIR = Path(__file__).resolve().parent

def iter_images_round_robin(dataset_root: Path,
                            allowed_ext: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> Iterator[Path]:
    """iter_images_round_robin(dataset_root,allowed_ext) -> Iterator[Path]: Yield images round-robin over subfolders."""
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    folders = sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    per_folder: List[List[Path]] = []
    allowed = set(e.lower() for e in allowed_ext)
    for folder in folders:
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in allowed]
        files = sorted(files, key=lambda p: p.name.lower())
        per_folder.append(files)

    total_images = sum(len(x) for x in per_folder)
    if total_images == 0:
        return

    k = 0
    yielded = 0
    while yielded < total_images:
        progressed = False
        for files in per_folder:
            if k < len(files):
                yield files[k]
                yielded += 1
                progressed = True
        if not progressed:
            break
        k += 1


def load_image(n: int,
               dataset_root: Path,
               allowed_ext: tuple = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")) -> str:
    """load_image(n,dataset_root,allowed_ext) -> str: Cyclic round-robin image loader."""
    if n < 0:
        raise ValueError("n must be >= 0")

    images = list(iter_images_round_robin(dataset_root, allowed_ext=allowed_ext))
    if not images:
        raise FileNotFoundError(f"No images found in {dataset_root}")

    idx = n % len(images)
    return images[idx].relative_to(dataset_root).as_posix()


def _run_worker(stage: str,
                i: int,
                img_file: str,
                crop_patch_seed_i: int,
                dem_seed_i: int,
                show_plot: bool,
                rotation_angle_deg: float,
                mirror_bool: int,
                sat_lat: float, sat_lon: float, sat_alt: float,
                tgt_lat: float, tgt_lon: float, tgt_alt: float,
                datetime_utc: str,
                patch_parameters: dict,
                sensor_characteristics: dict,
                wave_properties: dict,
                meta_out: Path,
                patch_name: str = "") -> None:
    """_run_worker(stage,...) -> None: Spawn worker_run.py with a stage, passing configs as JSON."""
    shutil.rmtree(Path.home() / "AppData/Local/Temp/drjit", ignore_errors=True)

    env = os.environ.copy()
    env["GENERATED_ROOT_REL"] = os.environ.get("GENERATED_ROOT_REL", "").strip()
    if not env["GENERATED_ROOT_REL"]:
        raise RuntimeError("GENERATED_ROOT_REL missing in worker env.")

    base = Path(r"C:\drjit_temp") / f"worker_{stage}_{i}"
    tmp = base / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)

    meta_out_abs = meta_out.resolve()
    meta_out_abs.parent.mkdir(parents=True, exist_ok=True)

    sensor_characteristics = dict(sensor_characteristics)

    sensor_json = json.dumps(sensor_characteristics)
    wave_json = json.dumps(wave_properties)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "worker_run.py"),
        "--stage", stage,
        "--img_file", img_file,
        "--patch_seed", str(crop_patch_seed_i),
        "--dem_seed", str(dem_seed_i),
        "--show_plot", "1" if show_plot else "0",

        "--sat_lat", str(sat_lat),
        "--sat_lon", str(sat_lon),
        "--sat_alt", str(sat_alt),

        "--tgt_lat", str(tgt_lat),
        "--tgt_lon", str(tgt_lon),
        "--tgt_alt", str(tgt_alt),

        "--datetime_utc", str(datetime_utc),

        "--mode_single", str(patch_parameters["mode_single"]),
        "--mode_multiple_allow_partial", "1" if bool(patch_parameters["mode_multiple_allow_partial"]) else "0",
        "--window_size", str(int(patch_parameters["window_size"])),
        "--rotation_angle_deg", str(float(rotation_angle_deg)),
        "--mirror_bool", "1" if mirror_bool else "0",

        "--nowhale_max_fraction", str(float(patch_parameters["nowhale_max_fraction"])),
        "--whale_min_fraction", str(float(patch_parameters["whale_min_fraction"])),
        "--half_fraction_low", str(float(patch_parameters["half_fraction_range"][0])),
        "--half_fraction_high", str(float(patch_parameters["half_fraction_range"][1])),
        "--mask_alpha", str(int(patch_parameters["mask_alpha"])),

        "--sensor_json", sensor_json,
        "--wave_json", wave_json,

        "--meta_out", str(meta_out_abs),
    ]

    if patch_name:
        cmd += ["--patch_name", patch_name]

    try:
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr, env=env)
    finally:
        shutil.rmtree(base, ignore_errors=True)




def run_one(i: int,
            img_file: str,
            crop_patch_seed_i: int,
            dem_seed_i: int,
            show_plot: bool,
            rotation_angle_deg: float,
            mirror_bool: int,
            sat_lat: float, sat_lon: float, sat_alt: float,
            tgt_lat: float, tgt_lon: float, tgt_alt: float,
            datetime_utc: str,
            patch_parameters: dict,
            sensor_characteristics: dict,
            wave_properties: dict,
            meta_out: Path) -> None:
    """run_one(...) -> None: Run nadir worker then offnadir worker."""
    if meta_out.exists():
        meta_out.unlink()

    _run_worker(
        stage="nadir",
        i=i,
        img_file=img_file,
        crop_patch_seed_i=crop_patch_seed_i,
        dem_seed_i=dem_seed_i,
        show_plot=show_plot,
        rotation_angle_deg=rotation_angle_deg,
        mirror_bool=mirror_bool,
        sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
        datetime_utc=datetime_utc,
        patch_parameters=patch_parameters,
        sensor_characteristics=sensor_characteristics,
        wave_properties=wave_properties,
        meta_out=meta_out,
    )

    if not meta_out.exists():
        raise FileNotFoundError(f"Nadir worker did not write meta_out: {meta_out}")

    meta = json.loads(meta_out.read_text(encoding="utf-8"))
    patch_name = meta.get("patch_name", "")
    if not isinstance(patch_name, str) or not patch_name:
        raise RuntimeError("meta_out missing patch_name after nadir stage")

    _run_worker(
        stage="offnadir",
        i=i,
        img_file=img_file,
        crop_patch_seed_i=crop_patch_seed_i,
        dem_seed_i=dem_seed_i,
        show_plot=show_plot,
        rotation_angle_deg=rotation_angle_deg,
        mirror_bool=mirror_bool,
        sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
        datetime_utc=datetime_utc,
        patch_parameters=patch_parameters,
        sensor_characteristics=sensor_characteristics,
        wave_properties=wave_properties,
        meta_out=meta_out,
        patch_name=patch_name,
    )




def run_dataset(n_runs: int,
                sensor_characteristics: dict,
                wave_properties: dict,
                pick_img_seed: int,
                img_rot_seed: int,
                crop_patch_seed: int,
                dem_seed: int,
                pick_pose_seed: int,
                wind_speed_seed: int,
                wind_speed_range: tuple[float, float],
                show_plot: bool,
                patch_parameters: dict,
                poses_xlsx: Path,
                overview_xlsx: Path,
                script_dir: Path,
                balanced_offnadir: bool = False,
                offnadir_angles: np.ndarray = np.arange(5, 55 + 1, 5)) -> None:


    """run_dataset(...) -> None: Run workers, read meta, append to dataset_overview.xlsx; optional balanced off-nadir sampling."""
    if not poses_xlsx.is_file():
        raise FileNotFoundError(f"Missing poses file: {poses_xlsx}")

    wb, ws_settings, ws_patch, ws_off, ws_ann_nadir, ws_ann_off, ws_radrefl = open_overview_book(
        overview_xlsx, patch_parameters, sensor_characteristics
    )

    # write wind settings once
    ws_settings.append(["wind_speed_seed", wind_speed_seed])
    ws_settings.append(["wind_speed_range_low", float(wind_speed_range[0])])
    ws_settings.append(["wind_speed_range_high", float(wind_speed_range[1])])

    meta_dir = (script_dir / "_meta").resolve()
    meta_dir.mkdir(parents=True, exist_ok=True)

    rng_rot = np.random.default_rng(img_rot_seed)
    rng_mirr = np.random.default_rng(img_rot_seed + 1)
    rng_wind = np.random.default_rng(wind_speed_seed)
    wlo, whi = float(wind_speed_range[0]), float(wind_speed_range[1])
    if not (np.isfinite(wlo) and np.isfinite(whi) and wlo <= whi):
        raise ValueError("wind_speed_range must be (low, high) with low <= high")

    if balanced_offnadir:

        run_idx = 0

        total_runs = int(n_runs) * len(offnadir_angles)

        for img_i in range(int(n_runs)):
            pick_img_seed_i = pick_img_seed + img_i

            whales_root = Path("dataset") / "whales_from_space"
            img_file = load_image(pick_img_seed_i, whales_root)

            for ang in offnadir_angles:
                print(f"\n ====================== Start new process {run_idx+1} / {total_runs} ====================== \n")
                print(f"Image: {img_file} | Offnadir request: {ang} deg")

                rotation_angle_deg = float(rng_rot.choice([0, 90, 180, -90]))
                mirror_bool = bool(rng_mirr.integers(0, 2))

                crop_patch_seed_i = crop_patch_seed + run_idx
                dem_seed_i = dem_seed + run_idx
                pick_pose_seed_i = pick_pose_seed + run_idx

                result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
                    poses_xlsx, pick_pose_seed=pick_pose_seed_i, offnadir_angle=float(ang), selection_method='exact'
                )


                print(f"Pose: sat=({sat_lat},{sat_lon},{sat_alt}) tgt=({tgt_lat},{tgt_lon},{tgt_alt}) dt={datetime_utc}")

                wind_speed = float(rng_wind.uniform(wlo, whi))
                wave_properties_i = dict(wave_properties)
                wave_properties_i["wind_speed"] = wind_speed

                max_attempts = 5
                success = False

                for attempt in range(max_attempts):
                    crop_patch_seed_i_try = crop_patch_seed + run_idx + 100000 * attempt
                    dem_seed_i_try = dem_seed + run_idx + 100000 * attempt
                    pick_pose_seed_i_try = pick_pose_seed + run_idx + 100000 * attempt

                    result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
                        poses_xlsx,
                        pick_pose_seed=pick_pose_seed_i_try,
                        offnadir_angle=float(ang),
                        selection_method="exact",
                    )

                    meta_out = meta_dir / f"run_{run_idx:05d}.json"
                    if meta_out.exists():
                        meta_out.unlink()

                    print(f"[attempt {attempt + 1}/{max_attempts}] Pose: sat=({sat_lat},{sat_lon},{sat_alt}) tgt=({tgt_lat},{tgt_lon},{tgt_alt}) dt={datetime_utc}")

                    try:
                        run_one(
                            i=run_idx,
                            img_file=img_file,
                            crop_patch_seed_i=crop_patch_seed_i_try,
                            dem_seed_i=dem_seed_i_try,
                            show_plot=show_plot,
                            rotation_angle_deg=rotation_angle_deg,
                            mirror_bool=mirror_bool,
                            sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                            tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                            datetime_utc=datetime_utc,
                            patch_parameters=patch_parameters,
                            sensor_characteristics=sensor_characteristics,
                            wave_properties=wave_properties_i,
                            meta_out=meta_out,
                        )

                        meta = json.loads(meta_out.read_text(encoding="utf-8"))

                        append_run_rows(
                            ws_patch=ws_patch,
                            ws_off=ws_off,
                            ws_ann_nadir=ws_ann_nadir,
                            ws_ann_off=ws_ann_off,
                            ws_radrefl=ws_radrefl,
                            i=run_idx,
                            img_file=img_file,
                            result_name=result_name,
                            detection_id=detection_id,
                            pick_img_seed_i=pick_img_seed_i,
                            crop_patch_seed_i=crop_patch_seed_i_try,
                            dem_seed_i=dem_seed_i_try,
                            pick_pose_seed_i=pick_pose_seed_i_try,
                            sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                            tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                            datetime_utc=datetime_utc,
                            wind_speed=wind_speed,
                            meta=meta,
                        )

                        wb.save(overview_xlsx)
                        time.sleep(0.1)

                        success = True
                        break

                    except Exception as e:
                        print(f"[attempt {attempt + 1}/{max_attempts}] Failed: {e}")

                        # ---------- ROLLBACK NADIR + OFFNADIR OUTPUTS ----------
                        try:
                            if meta_out.exists():
                                meta_tmp = json.loads(meta_out.read_text(encoding="utf-8"))
                                patch_name = meta_tmp.get("patch_name", "")
                                if patch_name:
                                    from save_patch import rollback_patch_outputs
                                    rollback_patch_outputs(img_file=img_file, patch_name=patch_name)
                        except Exception as _:
                            pass

                if not success:
                    print(f"Giving up after {max_attempts} attempts: {img_file} @ {ang}deg (excluded)")

                run_idx += 1

        # count actually generated images from disk (robust against failures)
        out_dir = script_dir / "texture_offnadir_255"
        n_generated = 0
        if out_dir.exists():
            for root, _, files in os.walk(out_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                        n_generated += 1

        print(f"\n Dataset generation completed, with {n_generated} images")

        ws_settings.append(["n_images_generated", int(n_generated)])

        wb.save(overview_xlsx)
        wb.close()
        return

    total_runs = int(n_runs)

    # Option 1: one run per iteration, images advance by i, no offnadir filtering
    for i in range(int(n_runs)):
        pick_img_seed_i = pick_img_seed + i
        crop_patch_seed_i = crop_patch_seed + i
        dem_seed_i = dem_seed + i
        pick_pose_seed_i = pick_pose_seed + i

        rotation_angle_deg = float(rng_rot.choice([0, 90, 180, -90]))
        mirror_bool = bool(rng_mirr.integers(0, 2))

        result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
            poses_xlsx, pick_pose_seed=pick_pose_seed_i
        )

        whales_root = Path("dataset") / "whales_from_space"
        img_file = load_image(pick_img_seed_i, whales_root)

        meta_out = meta_dir / f"run_{i:04d}.json"

        wind_speed = float(rng_wind.uniform(wlo, whi))
        wave_properties_i = dict(wave_properties)
        wave_properties_i["wind_speed"] = wind_speed

        print(f"\n ====================== Start new process {i+1} / {total_runs} ====================== \n")
        print(f"Pose: sat=({sat_lat},{sat_lon},{sat_alt}) tgt=({tgt_lat},{tgt_lon},{tgt_alt}) dt={datetime_utc}")
        print(f"Image: {img_file}")
        print(f"Wind speed: {wind_speed:.3f} m/s")

        max_attempts = 5
        success = False

        for attempt in range(max_attempts):
            crop_patch_seed_i_try = crop_patch_seed + i + 100000 * attempt
            dem_seed_i_try = dem_seed + i + 100000 * attempt
            pick_pose_seed_i_try = pick_pose_seed + i + 100000 * attempt

            try:
                result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
                    poses_xlsx, pick_pose_seed=pick_pose_seed_i_try
                )

                meta_out = meta_dir / f"run_{i:04d}.json"
                if meta_out.exists():
                    meta_out.unlink()

                run_one(
                    i=i,
                    img_file=img_file,
                    crop_patch_seed_i=crop_patch_seed_i_try,
                    dem_seed_i=dem_seed_i_try,
                    show_plot=show_plot,
                    rotation_angle_deg=rotation_angle_deg,
                    mirror_bool=mirror_bool,
                    sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                    tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                    datetime_utc=datetime_utc,
                    patch_parameters=patch_parameters,
                    sensor_characteristics=sensor_characteristics,
                    wave_properties=wave_properties_i,
                    meta_out=meta_out,
                )

                meta = json.loads(meta_out.read_text(encoding="utf-8"))

                append_run_rows(
                    ws_patch=ws_patch,
                    ws_off=ws_off,
                    ws_ann_nadir=ws_ann_nadir,
                    ws_ann_off=ws_ann_off,
                    ws_radrefl=ws_radrefl,
                    i=i,
                    img_file=img_file,
                    result_name=result_name,
                    detection_id=detection_id,
                    pick_img_seed_i=pick_img_seed_i,
                    crop_patch_seed_i=crop_patch_seed_i_try,
                    dem_seed_i=dem_seed_i_try,
                    pick_pose_seed_i=pick_pose_seed_i_try,
                    sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                    tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                    datetime_utc=datetime_utc,
                    wind_speed=wind_speed,
                    meta=meta,
                )

                wb.save(overview_xlsx)
                time.sleep(0.1)

                success = True
                break

            except Exception as e:
                print(f"[attempt {attempt + 1}/{max_attempts}] Failed: {e}")

                # rollback any partial outputs if patch_name exists
                try:
                    if meta_out.exists():
                        meta_tmp = json.loads(meta_out.read_text(encoding="utf-8"))
                        patch_name = meta_tmp.get("patch_name", "")
                        if patch_name:
                            from save_patch import rollback_patch_outputs
                            rollback_patch_outputs(img_file=img_file, patch_name=patch_name)
                except Exception:
                    pass

        if not success:
            print(f"Giving up after {max_attempts} attempts: {img_file} (excluded)")
            continue

    # count actually generated images from disk (robust against failures)
    out_dir = script_dir / "texture_offnadir_255"

    n_generated = 0
    if out_dir.exists():
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                    n_generated += 1

    print(f"\n Dataset generation completed, with {n_generated} images")

    ws_settings.append(["n_images_generated", int(n_generated)])

    wb.save(overview_xlsx)
    wb.close()
    return

def main() -> None:
    """main() -> None: Configure and run dataset generation."""

    GENERATED_ROOT_REL = "0_whales"

    render_resolution = 64  # 64 * 2

    wave_properties = {"num_waves": 50, "wave_min": 0.05, "wave_max": 0.5}
    sensor_characteristics = {"resolution": int(render_resolution), "sample_count": 512, "specular_weight": 0.2, "refl_mode": "proxy", "refl_scale": None, "refl_offset": None}

    n_images = 5                         # count_images_in_subfolders(Path("dataset") / "whales_from_space")
    balanced_offnadir = False            # For full dataset: True. True = per-image angles 5..60, False = random offnadir, loop by one,.

    offnadir_angles = np.arange(5, 60 +1, 5)

    pick_img_seed_0 = 1
    crop_patch_seed = 42
    dem_seed = 1
    pick_pose_seed = 17
    img_rot_seed = 10

    wind_speed_seed = 123
    wind_speed_range = (4.0, 12.0)      # normal distribution

    show_plot = False

    # mode_single options:
    #   "full"      -> only full whales
    #   "half"      -> only half whales
    #   "ocean"     -> only ocean (no whales)
    #   "full_half" -> full OR half whales
    #   "all"       -> anything

    # mode_multiple_allow_partial:
    #   True  -> if multiple whales, allow other partial whales in the patch
    #   False -> forbid any whale in (nowhale_max_fraction, whale_min_fraction)

    patch_parameters = {
        "mode_single": "full",
        "mode_multiple_allow_partial": False,
        "window_size": 64,
        "nowhale_max_fraction": 0.10,
        "whale_min_fraction": 0.99,
        "half_fraction_range": (0.20, 0.8),
        "mask_alpha": 80,
    }

    if patch_parameters["mode_single"] == "ocean":
        pick_img_seed_0 += 1
        crop_patch_seed += 1
        dem_seed += 1
        pick_pose_seed += 1
        img_rot_seed += 1
        wind_speed_seed += 1

    os.environ["GENERATED_ROOT_REL"] = GENERATED_ROOT_REL  # ensure children inherit
    GENERATED_ROOT = Path("dataset") / "create_dataset" / GENERATED_ROOT_REL

    if GENERATED_ROOT.exists():
        shutil.rmtree(GENERATED_ROOT)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)

    poses_xlsx = SCRIPT_DIR / "combined_results.xlsx"
    overview_xlsx = GENERATED_ROOT / "dataset_overview.xlsx"

    run_dataset(
        n_runs=n_images,
        sensor_characteristics=sensor_characteristics,
        wave_properties=wave_properties,
        pick_img_seed=pick_img_seed_0,
        crop_patch_seed=crop_patch_seed,
        dem_seed=dem_seed,
        img_rot_seed=img_rot_seed,
        pick_pose_seed=pick_pose_seed,
        wind_speed_seed=wind_speed_seed,
        wind_speed_range=wind_speed_range,
        show_plot=show_plot,
        patch_parameters=patch_parameters,
        poses_xlsx=poses_xlsx,
        overview_xlsx=overview_xlsx,
        script_dir=GENERATED_ROOT,
        balanced_offnadir=balanced_offnadir,
        offnadir_angles=offnadir_angles,
    )

    cleanup_meta_only(GENERATED_ROOT)


if __name__ == "__main__":
    main()
