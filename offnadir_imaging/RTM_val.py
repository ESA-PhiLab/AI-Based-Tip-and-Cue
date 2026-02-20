#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

from RTM import generate_sun_and_sky_spds


@dataclass(frozen=True)
class BandSpec:
    name: str
    lo_nm: float
    hi_nm: float


def read_spd(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """read_spd(path) -> (wl_nm,val): Read 2-col SPD file (nm,value) as sorted float arrays."""
    data = np.loadtxt(Path(path), dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid SPD format: {path}")
    wl = data[:, 0].astype(float)
    val = data[:, 1].astype(float)
    order = np.argsort(wl)
    wl = wl[order]
    val = val[order]
    if wl.size < 2:
        raise ValueError(f"SPD has too few samples: {path}")
    return wl, val


def trapz_with_edge_interpolation(wl: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """trapz_with_edge_interpolation(wl,y,lo,hi) -> float: Integrate y(wl) over [lo,hi] with linear edge interpolation."""
    if hi <= lo:
        return 0.0

    wl_min = float(wl[0])
    wl_max = float(wl[-1])
    a = max(lo, wl_min)
    b = min(hi, wl_max)
    if b <= a:
        return 0.0

    mask = (wl > a) & (wl < b)
    wl_in = wl[mask]
    y_in = y[mask]

    y_a = float(np.interp(a, wl, y))
    y_b = float(np.interp(b, wl, y))

    wl_seg = np.concatenate(([a], wl_in, [b]))
    y_seg = np.concatenate(([y_a], y_in, [y_b]))

    return float(np.trapezoid(y_seg, wl_seg))


def compute_band_metrics(wl_nm: np.ndarray, e_w_m2_nm: np.ndarray, bands: Iterable[BandSpec]) -> tuple[list[dict], float]:
    """compute_band_metrics(wl,e,bands) -> (rows,total): Integrate spectrum per band and compute fractions."""
    rows = []
    for b in bands:
        irr = trapz_with_edge_interpolation(wl_nm, e_w_m2_nm, b.lo_nm, b.hi_nm)
        rows.append({"band": b.name, "lo_nm": b.lo_nm, "hi_nm": b.hi_nm, "irradiance_w_m2": irr})

    total = sum(r["irradiance_w_m2"] for r in rows)
    for r in rows:
        r["fraction_percent"] = 0.0 if total <= 0 else 100.0 * r["irradiance_w_m2"] / total
    return rows, total


def format_table(rows: list[dict], total: float) -> str:
    """format_table(rows,total) -> str: Build a fixed-width table string."""
    header = (
        f"{'Band':<10} {'Interval (nm)':<14} {'Irradiance (W/m^2)':>20} {'Fraction (%)':>14}\n"
        + "-" * 62
    )
    lines = [header]
    for r in rows:
        interval = f"{r['lo_nm']:.0f}-{r['hi_nm']:.0f}"
        lines.append(
            f"{r['band']:<10} {interval:<14} {r['irradiance_w_m2']:>20.3f} {r['fraction_percent']:>14.3f}"
        )
    lines.append("-" * 62)
    lines.append(f"{'TOTAL':<10} {'':<14} {total:>20.3f} {100.0:>14.3f}")
    return "\n".join(lines)


def plot_uv_vis_nir(wl: np.ndarray,
                    e: np.ndarray,
                    rows: list[dict],
                    out_path: str | Path | None,
                    show: bool) -> None:
    """plot_uv_vis_nir(wl,e,rows,out_path,show) -> None: Plot spectrum with filled UV/Visible/NIR regions."""
    bands = {r["band"]: (r["lo_nm"], r["hi_nm"], r["fraction_percent"]) for r in rows}

    plt.figure(figsize=(9, 4.8))
    plt.plot(wl, e)

    def fill_band(name: str, label_prefix: str) -> None:
        lo, hi, frac = bands[name]
        m = (wl >= lo) & (wl <= hi)
        plt.fill_between(wl[m], e[m], 0.0, alpha=0.5, label=f"~{frac:.0f}% {label_prefix} ({lo:.0f}-{hi:.0f} nm)")

    fill_band("UV", "UV")
    fill_band("Visible", "Visible")
    fill_band("NIR", "NIR")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Spectral Irradiance (W m$^{-2}$ nm$^{-1}$)")
    plt.xlim(min(wl), max(wl))
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main() -> int:
    """main() -> int: Generate SMARTS SPD, compute UV/Visible/NIR fractions, print metrics, optionally plot."""
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="spd_files_verify", help="Output folder for generated SPD files.")
    p.add_argument("--min-wvl", type=float, default=250.0, help="SMARTS min wavelength (nm).")
    p.add_argument("--max-wvl", type=float, default=2500.0, help="SMARTS max wavelength (nm).")
    p.add_argument("--lat", type=float, default=0.0)
    p.add_argument("--lon", type=float, default=0.0)
    p.add_argument("--alt-m", type=float, default=0.0)
    p.add_argument("--timezone-offset", type=int, default=0)
    p.add_argument("--material", type=str, default="Water")
    p.add_argument("--dt-utc", type=str, default="2025-06-11T16:00:00Z", help="UTC datetime ISO, e.g. 2025-06-11T16:00:00Z")
    p.add_argument("--plot", action="store_true", help="Create a UV/Visible/NIR filled plot like the reference figure.")
    p.add_argument("--plot-out", type=str, default="", help="Optional plot output path (png). If empty, uses out-dir.")
    p.add_argument("--show", action="store_true", help="Show the plot window (if --plot).")
    p.add_argument("--tol-frac-abs", type=float, default=6.0, help="Abs tolerance on fraction (%) vs plot reference.")
    args = p.parse_args()

    # Reference plot numbers (approximate, used only if you want a PASS/FAIL)
    expected_plot = {
        "UV": 5.0,
        "Visible": 43.0,
        "NIR": 52.0,
    }

    # Bands matching the plot (UV=300–400, Visible=400–700, NIR=700–2500)
    # NOTE: If SMARTS range doesn't cover the full band, the integral is clipped to available wavelengths.
    bands = [
        BandSpec("UV", 300.0, 400.0),
        BandSpec("Visible", 400.0, 700.0),
        BandSpec("NIR", 700.0, 2500.0),
    ]

    dt_str = args.dt_utc.strip()
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1]
    dt_utc = datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sun_spd = out_dir / "solar_direct_normal.spd"
    sky_spd = out_dir / "sky_diffuse_radiance.spd"

    out_sun, out_sky = generate_sun_and_sky_spds(
        datetime_utc=dt_utc,
        target_lat=args.lat,
        target_lon=args.lon,
        target_alt_m=args.alt_m,
        out_sun_spd=sun_spd,
        out_sky_spd=sky_spd,
        wavelength_range=(args.min_wvl, args.max_wvl),
        timezone_offset=args.timezone_offset,
        material=args.material,
        plot_spd=False,
    )

    wl, e_dir = read_spd(out_sun)

    # Compute only over the portion of the spectrum that covers the plot’s domain if possible
    rows, total_irr = compute_band_metrics(wl, e_dir, bands)

    print("\n=== SMARTS Direct Normal Irradiance Metrics (UV/Visible/NIR bands) ===")
    print(f"SPD: {out_sun}")
    print(f"Sky SPD: {out_sky}")
    print(f"SMARTS wavelength range used: {args.min_wvl:.1f}–{args.max_wvl:.1f} nm")
    print(format_table(rows, total_irr))

    print("\n=== Comparison vs Reference Plot Fractions (approx) ===")
    all_ok = True
    for r in rows:
        name = r["band"]
        frac = r["fraction_percent"]
        exp = expected_plot[name]
        abs_err = abs(frac - exp)
        ok = abs_err <= args.tol_frac_abs
        all_ok = all_ok and ok
        print(f"{name:<7}: computed={frac:6.2f}%, expected~{exp:5.1f}%, abs_err={abs_err:5.2f}% -> {'PASS' if ok else 'FAIL'}")

    print("\nOVERALL:", "PASS" if all_ok else "FAIL")

    if args.plot:
        plot_path = None
        if args.plot_out.strip():
            plot_path = args.plot_out.strip()
        else:
            plot_path = str(out_dir / "solar_uv_visible_nir.png")
        plot_uv_vis_nir(wl, e_dir, rows, plot_path, args.show)
        print(f"\nSaved plot: {plot_path}")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
