# main_create.py
import subprocess
import sys
import time

from pathlib import Path
import shutil

def run_one(i: int,
            img_file: str,
            seed: int,
            dem_seed: int,
            show_plot: bool,
            render_resolution: int,
            sat_lat: float, sat_lon: float, sat_alt: float,
            tgt_lat: float, tgt_lon: float, tgt_alt: float,
            datetime_utc: str) -> None:
    """run_one(...) -> None: Spawn one worker with isolated TEMP (drjit cache isolation)."""
    import os
    import shutil
    from pathlib import Path
    import faulthandler, sys
    faulthandler.enable(all_threads=True)

    shutil.rmtree(Path.home() / "AppData/Local/Temp/drjit", ignore_errors=True)

    env = os.environ.copy()

    base = Path(r"C:\drjit_temp") / f"worker_{i}"
    tmp = base / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # drjit uses Windows temp -> put each worker in its own temp dir
    env["TEMP"] = str(tmp)
    env["TMP"] = str(tmp)

    # optional: ensure no leftovers if worker reruns with same i
    shutil.rmtree(tmp / "drjit", ignore_errors=True)

    cmd = [
        sys.executable, "worker_run.py",
        "--img_file", img_file,
        "--seed", str(seed),
        "--dem_seed", str(dem_seed),
        "--show_plot", "1" if show_plot else "0",
        "--render_resolution", str(render_resolution),
        "--sat_lat", str(sat_lat),
        "--sat_lon", str(sat_lon),
        "--sat_alt", str(sat_alt),
        "--tgt_lat", str(tgt_lat),
        "--tgt_lon", str(tgt_lon),
        "--tgt_alt", str(tgt_alt),
        "--datetime_utc", datetime_utc,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr, env=env)
    finally:
        # deletes drjit cache for that run (and everything else in that worker temp)
        shutil.rmtree(base, ignore_errors=True)



def main() -> None:
    """main() -> None: Launch isolated runs."""
    img_file = "Pelagos2016/PelagosIm4_FW_WV3_PS_20160619_B2.PNG"
    render_resolution = 124

    sat_lat, sat_lon, sat_alt = 58.0, -5.0, 617000.0

    targets = [
        (53.00, 0.00, 0.0),
        (53.02, 0.01, 0.0),
        (52.98, -0.01, 0.0),
        (53.01, 0.02, 0.0),
        (53.03, -0.02, 0.0),
    ]

    base_seed = 42
    base_dem_seed = 9000
    show_plot = False
    datetime_utc = "2025-06-11T08:00:00Z"

    for i, (tgt_lat, tgt_lon, tgt_alt) in enumerate(targets):
        print(f"\n ====================== Start new process {i} ====================== \n")
        run_one(
            i=i,
            img_file=img_file,
            seed=base_seed + i,
            dem_seed=base_dem_seed + i,
            show_plot=show_plot,
            render_resolution=render_resolution,
            sat_lat=sat_lat, sat_lon=sat_lon, sat_alt=sat_alt,
            tgt_lat=tgt_lat, tgt_lon=tgt_lon, tgt_alt=tgt_alt,
            datetime_utc=datetime_utc,
        )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
