import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# --- Make repo root the working dir ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)

from settings import satellite, sensor_characteristics, wave_properties, bools, seed_dem
from offnadir_imaging.rendering import generate_image


def load_rgb_uint8(path: str) -> np.ndarray:
    """load_rgb_uint8(path) -> np.ndarray: Load an image as uint8 RGB [H,W,3]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """to_uint8_rgb(arr) -> np.ndarray: Convert array to uint8 RGB [H,W,3] with clipping."""
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.repeat(x[:, :, None], 3, axis=2)
    if x.shape[-1] > 3:
        x = x[:, :, :3]
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def diff_abs_linear(a_uint8: np.ndarray, b_uint8: np.ndarray) -> np.ndarray:
    """diff_abs_linear(a_uint8,b_uint8) -> np.ndarray: |a-b|."""
    return np.clip(np.abs(a_uint8.astype(np.int16) - b_uint8.astype(np.int16)), 0, 255).astype(np.uint8)


def diff_abs_linear10(a_uint8: np.ndarray, b_uint8: np.ndarray) -> np.ndarray:
    """diff_abs_linear10(a_uint8,b_uint8) -> np.ndarray: |a-b| * 10."""
    d = np.abs(a_uint8.astype(np.int16) - b_uint8.astype(np.int16))
    return np.clip(d * 10, 0, 255).astype(np.uint8)


def diff_abs_log(a_uint8: np.ndarray, b_uint8: np.ndarray) -> np.ndarray:
    """diff_abs_log(a_uint8,b_uint8) -> np.ndarray: log1p(|a-b|) scaled to 0–255."""
    abs_d = np.abs(a_uint8.astype(np.int16) - b_uint8.astype(np.int16)).astype(np.float32)
    log_d = np.log1p(abs_d)
    m = float(np.max(log_d))
    if m > 0:
        log_d /= (m + 1e-12)
    return np.clip(log_d * 255.0, 0, 255).astype(np.uint8)


def print_stats(name: str, img_uint8: np.ndarray) -> None:
    """print_stats(name,img_uint8) -> None: Print min, max, mean."""
    x = img_uint8.astype(np.float32)
    print(f"{name:>25s}  min={x.min():.2f}  max={x.max():.2f}  mean={x.mean():.4f}")


def save_plain_png(img_uint8: np.ndarray, out_path: Path) -> None:
    """save_plain_png(img_uint8,out_path) -> None: Save image without axes/titles."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_uint8)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def display_5(orig, rend, diff_lin, diff_lin10, diff_log):
    fig = plt.figure(figsize=(26, 6))

    titles = [
        "Original",
        "Rendered",
        "|Diff| linear",
        "|Diff| linear * 10",
        "|Diff| log1p scaled",
    ]

    images = [orig, rend, diff_lin, diff_lin10, diff_log]

    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def print_stats_rgb(name: str, img_uint8: np.ndarray) -> None:
    """print_stats_rgb(name,img_uint8) -> None: Print min/max/mean overall and per RGB channel."""
    x = to_uint8_rgb(img_uint8).astype(np.float32)

    overall_min = float(x.min())
    overall_max = float(x.max())
    overall_mean = float(x.mean())

    ch_names = ["R", "G", "B"]
    ch_min = [float(x[:, :, i].min()) for i in range(3)]
    ch_max = [float(x[:, :, i].max()) for i in range(3)]
    ch_mean = [float(x[:, :, i].mean()) for i in range(3)]

    print(f"{name:>25s}  overall: min={overall_min:.2f}  max={overall_max:.2f}  mean={overall_mean:.4f}")
    print(f"{'':>25s}  per-band: R min={ch_min[0]:.2f} max={ch_max[0]:.2f} mean={ch_mean[0]:.4f} | "
          f"G min={ch_min[1]:.2f} max={ch_max[1]:.2f} mean={ch_mean[1]:.4f} | "
          f"B min={ch_min[2]:.2f} max={ch_max[2]:.2f} mean={ch_mean[2]:.4f}")



def main() -> None:
    images_folder = PROJECT_ROOT / "dataset" / "whales_from_space"
    img_file = "checkerboard_rgb.png"
    img_path = str(images_folder / img_file)

    anns_path = str(PROJECT_ROOT / "create_dataset" / "final_annotations.json")

    sat_lat, sat_lon, sat_alt = 0.0, 0.0, 617.0
    tgt_lat, tgt_lon, tgt_alt = 0.0, 0.0000000000000000000001, 0.0
    dt = datetime(2025, 6, 11, 12, 0, 0, tzinfo=timezone.utc)

    bools_local = dict(bools)
    bools_local["generate_radiation"] = False
    bools_local["plot_result"] = False
    bools_local["use_annotations"] = False
    bools_local["generate_nadir"] = False

    wave_properties['wave_min'] = 0.0
    wave_properties['wave_max'] = 0.0

    original = load_rgb_uint8(img_path)

    # --- Set render resolution equal to input image ---
    H_in, W_in = original.shape[:2]

    if H_in != W_in:
        print(f"Warning: input is not square ({H_in}x{W_in}). Using min dimension.")
    resolution_equal = min(H_in, W_in)

    print(resolution_equal)

    sensor_characteristics_local = dict(sensor_characteristics)
    sensor_characteristics_local["resolution"] = resolution_equal

    sensor_characteristics_local['sample_count'] = 512 * 4

    wave_properties_local = dict(wave_properties)
    wave_properties_local["wave_min"] = 0.0
    wave_properties_local["wave_max"] = 0.0

    original = load_rgb_uint8(img_path)

    texture_disp, *_ = generate_image(
        img_path,
        anns_path,
        satellite,
        sat_lat, sat_lon, sat_alt,
        tgt_lat, tgt_lon, tgt_alt,
        dt,
        sensor_characteristics_local,
        wave_properties_local,
        bools_local,
        seed_dem,
    )

    if texture_disp is None:
        raise RuntimeError("generate_image returned None texture_disp.")

    rendered = to_uint8_rgb(texture_disp)

    H = min(original.shape[0], rendered.shape[0])
    W = min(original.shape[1], rendered.shape[1])
    original = original[:H, :W]
    rendered = rendered[:H, :W]

    diff_lin = diff_abs_linear(original, rendered)
    diff_lin10 = diff_abs_linear10(original, rendered)
    diff_log = diff_abs_log(original, rendered)

    # ---- Print statistics ----
    print_stats_rgb("diff_abs_linear", diff_lin)
    print_stats_rgb("diff_abs_linear_x10", diff_lin10)
    print_stats_rgb("diff_abs_log1p_scaled", diff_log)

    # ---- Display ----
    display_5(original, rendered, diff_lin, diff_lin10, diff_log)

    # ---- Save ----
    out_dir = PROJECT_ROOT / "img_diff"
    stem = Path(img_path).stem
    save_plain_png(original, out_dir / f"{stem}_01_original.png")
    save_plain_png(rendered, out_dir / f"{stem}_02_rendered.png")
    save_plain_png(diff_lin, out_dir / f"{stem}_03_diff_abs_linear.png")
    save_plain_png(diff_lin10, out_dir / f"{stem}_04_diff_abs_linear_x10.png")
    save_plain_png(diff_log, out_dir / f"{stem}_05_diff_abs_log1p.png")

    print("Saved 5 images to:", out_dir)


if __name__ == "__main__":
    main()
