import os
import time
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
    """diff_abs_linear(a_uint8,b_uint8) -> np.ndarray: Absolute difference |a-b| as uint8 RGB."""
    return np.clip(np.abs(a_uint8.astype(np.int16) - b_uint8.astype(np.int16)), 0, 255).astype(np.uint8)


def compute_rgb_ratio(img_uint8: np.ndarray) -> dict[str, float]:
    """compute_rgb_ratio(img_uint8) -> dict[str,float]: Return per-channel energy ratios over R,G,B."""
    x = to_uint8_rgb(img_uint8).astype(np.float64)
    sums = x.reshape(-1, 3).sum(axis=0)
    total = float(sums.sum())
    eps = 1e-12
    return {
        "R": float(sums[0] / (total + eps)),
        "G": float(sums[1] / (total + eps)),
        "B": float(sums[2] / (total + eps)),
        "sum_R": float(sums[0]),
        "sum_G": float(sums[1]),
        "sum_B": float(sums[2]),
        "sum_total": total,
    }


def save_plot_no_title(
    x: list[int],
    y_series: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    out_path: Path,
    series_colors: dict[str, str] | None = None,
) -> None:
    """save_plot_no_title(x,y_series,xlabel,ylabel,out_path,series_colors) -> None: Plot, log2 x-axis, save w/o title."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    for label, y in y_series.items():
        c = None if series_colors is None else series_colors.get(label, None)
        ax.plot(x, y, marker="o", label=label, color=c)

    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.legend()

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.show()
    plt.close(fig)


def main() -> None:
    # ==============================
    # EDIT THESE
    # ==============================
    images_folder = PROJECT_ROOT / "dataset" / "whales_from_space"
    img_file = "checkerboard_rgb.png"
    img_path = str(images_folder / img_file)

    anns_path = str(PROJECT_ROOT / "create_dataset" / "final_annotations.json")
    sat_lat, sat_lon, sat_alt = 0.0, 0.0, 617.0
    tgt_lat, tgt_lon, tgt_alt = 0.0, 1e-22, 0.0
    dt = datetime(2025, 6, 11, 12, 0, 0, tzinfo=timezone.utc)
    # ==============================

    out_dir = PROJECT_ROOT / "img_diff"
    out_dir.mkdir(parents=True, exist_ok=True)

    original = load_rgb_uint8(img_path)

    # ---- Compute input RGB ratio (for later discussion) ----
    rgb_ratio = compute_rgb_ratio(original)
    print("Input image RGB ratio (sum over all pixels, normalized):")
    print(f"  R={rgb_ratio['R']:.6f}  G={rgb_ratio['G']:.6f}  B={rgb_ratio['B']:.6f}")
    print("Input image channel sums (DN units, summed over all pixels):")
    print(f"  sum_R={rgb_ratio['sum_R']:.0f}  sum_G={rgb_ratio['sum_G']:.0f}  sum_B={rgb_ratio['sum_B']:.0f}  total={rgb_ratio['sum_total']:.0f}")

    # --- Set render resolution equal to input image (square constraint in renderer) ---
    H_in, W_in = original.shape[:2]
    if H_in != W_in:
        print(f"Warning: input is not square ({H_in}x{W_in}). Using min dimension.")
    resolution_equal = int(min(H_in, W_in))
    print("Render resolution:", resolution_equal)

    bools_local = dict(bools)
    bools_local["generate_radiation"] = False
    bools_local["plot_result"] = False
    bools_local["use_annotations"] = False
    bools_local["generate_nadir"] = False

    wave_properties_local = dict(wave_properties)
    wave_properties_local["wave_min"] = 0.0
    wave_properties_local["wave_max"] = 0.0

    sensor_characteristics_local = dict(sensor_characteristics)
    sensor_characteristics_local["resolution"] = resolution_equal

    # Loop sample counts: 128 -> 512 * 2**9 (262144), powers of two
    start_sc = 128
    end_sc = 512 * (2 ** 9)

    sample_counts: list[int] = []
    sc = start_sc
    while sc <= end_sc:
        sample_counts.append(int(sc))
        sc *= 2

    mean_overall: list[float] = []
    max_overall: list[float] = []
    mean_R: list[float] = []
    mean_G: list[float] = []
    mean_B: list[float] = []
    max_R: list[float] = []
    max_G: list[float] = []
    max_B: list[float] = []
    render_time_s: list[float] = []

    time.sleep(1)

    for sc in sample_counts:
        print(f"\nRunning sample_count={sc}")
        sensor_characteristics_local["sample_count"] = int(sc)

        t0 = time.perf_counter()
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
        dt_s = time.perf_counter() - t0
        render_time_s.append(float(dt_s))

        if texture_disp is None:
            raise RuntimeError(f"generate_image returned None texture_disp at sample_count={sc}")

        rendered = to_uint8_rgb(texture_disp)

        H = min(original.shape[0], rendered.shape[0])
        W = min(original.shape[1], rendered.shape[1])
        orig_c = original[:H, :W]
        rend_c = rendered[:H, :W]

        diff = diff_abs_linear(orig_c, rend_c).astype(np.float32)

        mean_overall.append(float(diff.mean()))
        max_overall.append(float(diff.max()))

        mean_R.append(float(diff[:, :, 0].mean()))
        mean_G.append(float(diff[:, :, 1].mean()))
        mean_B.append(float(diff[:, :, 2].mean()))

        max_R.append(float(diff[:, :, 0].max()))
        max_G.append(float(diff[:, :, 1].max()))
        max_B.append(float(diff[:, :, 2].max()))

        print(
            f"  time={dt_s:.3f}s | mean={mean_overall[-1]:.6f} max={max_overall[-1]:.2f} | "
            f"mean(R,G,B)=({mean_R[-1]:.6f},{mean_G[-1]:.6f},{mean_B[-1]:.6f}) | "
            f"max(R,G,B)=({max_R[-1]:.2f},{max_G[-1]:.2f},{max_B[-1]:.2f})"
        )

    # ---- plots (saved w/o title, also displayed) ----
    save_plot_no_title(
        x=sample_counts,
        y_series={"overall": mean_overall},
        xlabel="Sample Count (log2 scale)",
        ylabel="Mean |diff| (DN)",
        out_path=out_dir / "mean_overall.png",
    )

    save_plot_no_title(
        x=sample_counts,
        y_series={"overall": max_overall},
        xlabel="Sample Count (log2 scale)",
        ylabel="Max |diff| (DN)",
        out_path=out_dir / "max_overall.png",
    )

    rgb_colors = {"R": "red", "G": "green", "B": "blue"}

    save_plot_no_title(
        x=sample_counts,
        y_series={"R": mean_R, "G": mean_G, "B": mean_B},
        xlabel="Sample Count (log2 scale)",
        ylabel="Mean |diff| (DN)",
        out_path=out_dir / "mean_per_band.png",
        series_colors=rgb_colors,
    )

    save_plot_no_title(
        x=sample_counts,
        y_series={"R": max_R, "G": max_G, "B": max_B},
        xlabel="Sample Count (log2 scale)",
        ylabel="Max |diff| (DN)",
        out_path=out_dir / "max_per_band.png",
        series_colors=rgb_colors,
    )

    save_plot_no_title(
        x=sample_counts,
        y_series={"render_time_s": render_time_s},
        xlabel="Sample Count (log2 scale)",
        ylabel="Render time (s)",
        out_path=out_dir / "render_time_seconds.png",
    )

    print("\nRender times (s) per sample_count:")
    for sc, tsec in zip(sample_counts, render_time_s):
        print(f"  {sc:>7d}: {tsec:.3f}s")

    print("\nSaved plots to:", out_dir)


if __name__ == "__main__":
    main()
