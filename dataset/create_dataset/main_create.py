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
    pick_random_pose,
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
                render_resolution: int,
                sat_lat: float, sat_lon: float, sat_alt: float,
                tgt_lat: float, tgt_lon: float, tgt_alt: float,
                datetime_utc: str,
                patch_parameters: dict,
                meta_out: Path,
                patch_name: str = "") -> None:
    """_run_worker(stage,...) -> None: Spawn worker_run.py with a stage."""
    shutil.rmtree(Path.home() / "AppData/Local/Temp/drjit", ignore_errors=True)

    env = os.environ.copy()
    base = Path(r"C:\drjit_temp") / f"worker_{stage}_{i}"
    tmp = base / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)

    meta_out_abs = meta_out.resolve()
    meta_out_abs.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT_DIR / "worker_run.py"),
        "--stage", stage,
        "--img_file", img_file,
        "--patch_seed", str(crop_patch_seed_i),
        "--dem_seed", str(dem_seed_i),
        "--show_plot", "1" if show_plot else "0",
        "--render_resolution", str(render_resolution),

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
        "--nowhale_max_fraction", str(float(patch_parameters["nowhale_max_fraction"])),
        "--whale_min_fraction", str(float(patch_parameters["whale_min_fraction"])),
        "--half_fraction_low", str(float(patch_parameters["half_fraction_range"][0])),
        "--half_fraction_high", str(float(patch_parameters["half_fraction_range"][1])),
        "--mask_alpha", str(int(patch_parameters["mask_alpha"])),

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
            render_resolution: int,
            sat_lat: float, sat_lon: float, sat_alt: float,
            tgt_lat: float, tgt_lon: float, tgt_alt: float,
            datetime_utc: str,
            patch_parameters: dict,
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
        render_resolution=render_resolution,
        sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
        datetime_utc=datetime_utc,
        patch_parameters=patch_parameters,
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
        render_resolution=render_resolution,
        sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
        tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
        datetime_utc=datetime_utc,
        patch_parameters=patch_parameters,
        meta_out=meta_out,
        patch_name=patch_name,
    )


def run_dataset(n_runs: int,
                render_resolution: int,
                pick_img_seed: int,
                crop_patch_seed: int,
                dem_seed: int,
                pick_pose_seed: int,
                show_plot: bool,
                patch_parameters: dict,
                poses_xlsx: Path,
                overview_xlsx: Path,
                script_dir: Path,
                balanced_offnadir: bool = False) -> None:
    """run_dataset(...) -> None: Run workers, read meta, append to dataset_overview.xlsx; optional balanced off-nadir sampling."""
    if not poses_xlsx.is_file():
        raise FileNotFoundError(f"Missing poses file: {poses_xlsx}")

    wb, _, ws_patch, ws_off, ws_ann = open_overview_book(
        overview_xlsx, patch_parameters, render_resolution
    )

    meta_dir = (script_dir / "_meta").resolve()
    meta_dir.mkdir(parents=True, exist_ok=True)

    if balanced_offnadir:

        offnadir_angles = np.arange(5, 65, 5)  # 5,10,...,60

        run_idx = 0
        for img_i in range(int(n_runs)):
            pick_img_seed_i = pick_img_seed + img_i

            whales_root = Path("dataset") / "whales_from_space"
            img_file = load_image(pick_img_seed_i, whales_root)

            for ang in offnadir_angles:
                crop_patch_seed_i = crop_patch_seed + run_idx
                dem_seed_i = dem_seed + run_idx
                pick_pose_seed_i = pick_pose_seed + run_idx

                result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
                    poses_xlsx, pick_pose_seed=pick_pose_seed_i, offnadir_angle=float(ang)
                )

                meta_out = meta_dir / f"run_{run_idx:05d}.json"

                print(f"\n ====================== Start new process {run_idx} ====================== \n")
                print(f"Pose: sat=({sat_lat},{sat_lon},{sat_alt}) tgt=({tgt_lat},{tgt_lon},{tgt_alt}) dt={datetime_utc}")
                print(f"Image: {img_file} | Offnadir request: {ang} deg")

                run_one(
                    i=run_idx,
                    img_file=img_file,
                    crop_patch_seed_i=crop_patch_seed_i,
                    dem_seed_i=dem_seed_i,
                    show_plot=show_plot,
                    render_resolution=render_resolution,
                    sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                    tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                    datetime_utc=datetime_utc,
                    patch_parameters=patch_parameters,
                    meta_out=meta_out,
                )

                meta = json.loads(meta_out.read_text(encoding="utf-8"))

                append_run_rows(
                    ws_patch=ws_patch,
                    ws_off=ws_off,
                    ws_ann=ws_ann,
                    i=run_idx,
                    img_file=img_file,
                    result_name=result_name,
                    detection_id=detection_id,
                    pick_img_seed_i=pick_img_seed_i,
                    crop_patch_seed_i=crop_patch_seed_i,
                    dem_seed_i=dem_seed_i,
                    pick_pose_seed_i=pick_pose_seed_i,
                    sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
                    tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
                    datetime_utc=datetime_utc,
                    meta=meta,
                )

                wb.save(overview_xlsx)
                time.sleep(0.1)

                run_idx += 1

        wb.save(overview_xlsx)
        wb.close()
        return

    # Option 1 (old behavior): one run per iteration, images advance by i, no offnadir filtering
    for i in range(int(n_runs)):
        pick_img_seed_i = pick_img_seed + i
        crop_patch_seed_i = crop_patch_seed + i
        dem_seed_i = dem_seed + i
        pick_pose_seed_i = pick_pose_seed + i

        result_name, detection_id, sat_lat, sat_lon, sat_alt, tgt_lat, tgt_lon, tgt_alt, datetime_utc = pick_random_pose(
            poses_xlsx, pick_pose_seed=pick_pose_seed_i
        )

        whales_root = Path("dataset") / "whales_from_space"
        img_file = load_image(pick_img_seed_i, whales_root)

        meta_out = meta_dir / f"run_{i:04d}.json"

        print(f"\n ====================== Start new process {i} ====================== \n")
        print(f"Pose: sat=({sat_lat},{sat_lon},{sat_alt}) tgt=({tgt_lat},{tgt_lon},{tgt_alt}) dt={datetime_utc}")
        print(f"Image: {img_file}")

        run_one(
            i=i,
            img_file=img_file,
            crop_patch_seed_i=crop_patch_seed_i,
            dem_seed_i=dem_seed_i,
            show_plot=show_plot,
            render_resolution=render_resolution,
            sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
            tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
            datetime_utc=datetime_utc,
            patch_parameters=patch_parameters,
            meta_out=meta_out,
        )

        meta = json.loads(meta_out.read_text(encoding="utf-8"))

        append_run_rows(
            ws_patch=ws_patch,
            ws_off=ws_off,
            ws_ann=ws_ann,
            i=i,
            img_file=img_file,
            result_name=result_name,
            detection_id=detection_id,
            pick_img_seed_i=pick_img_seed_i,
            crop_patch_seed_i=crop_patch_seed_i,
            dem_seed_i=dem_seed_i,
            pick_pose_seed_i=pick_pose_seed_i,
            sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
            tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
            datetime_utc=datetime_utc,
            meta=meta,
        )

        wb.save(overview_xlsx)
        time.sleep(0.1)

    wb.save(overview_xlsx)
    wb.close()



def main() -> None:
    """main() -> None: Configure and run dataset generation."""
    base = Path("dataset") / "create_dataset"
    cleanup_previous_outputs(base)

    render_resolution = 64 # 64 * 2

    pick_img_seed_0 = 12
    crop_patch_seed = 42
    dem_seed = 1
    pick_pose_seed = 17

    show_plot = False
    n_images = 5
    balanced_offnadir = False  # False = random offnadir, loop by one, True = per-image angles 5..60

    # mode_single options:
    #   "full"      -> only full whales
    #   "half"      -> only half whales
    #   "ocean"     -> only ocean (no whales)
    #   "full_half" -> full OR half whales
    #   "all"       -> anything
    #
    # mode_multiple_allow_partial:
    #   True  -> if multiple whales, allow other partial whales in the patch
    #   False -> forbid any whale in (nowhale_max_fraction, whale_min_fraction)

    patch_parameters = {
        "mode_single": "full",
        "mode_multiple_allow_partial": False,
        "window_size": 64,
        "nowhale_max_fraction": 0.10,
        "whale_min_fraction": 0.99,
        "half_fraction_range": (0.20, 0.80),
        "mask_alpha": 80,
    }

    poses_xlsx = SCRIPT_DIR / "combined_results.xlsx"
    overview_xlsx = SCRIPT_DIR / "dataset_overview.xlsx"

    run_dataset(
        n_runs=n_images,
        render_resolution=render_resolution,
        pick_img_seed=pick_img_seed_0,
        crop_patch_seed=crop_patch_seed,
        dem_seed=dem_seed,
        pick_pose_seed=pick_pose_seed,
        show_plot=show_plot,
        patch_parameters=patch_parameters,
        poses_xlsx=poses_xlsx,
        overview_xlsx=overview_xlsx,
        script_dir=SCRIPT_DIR,
        balanced_offnadir=balanced_offnadir,
    )

    cleanup_meta_only(base)


if __name__ == "__main__":
    main()
