import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def extract_means_by_resolution(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Parse report-style Excel and return MEAN rows per RESOLUTION with AP50, AP50:95, and latency metrics."""
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    col0 = df_raw.iloc[:, 0].astype(str).fillna("")
    res_pat = re.compile(r"^\s*RESOLUTION\s+(\d+)\s*$", re.IGNORECASE)

    resolutions: List[int] = []
    ap50_vals: List[float] = []
    ap5095_vals: List[float] = []
    lat_vals: List[float] = []

    i = 0
    nrows = len(df_raw)

    while i < nrows:
        m = res_pat.match(col0.iat[i])
        if not m:
            i += 1
            continue

        res = int(m.group(1))

        header_row = None
        j = i + 1
        while j < nrows:
            v = str(df_raw.iat[j, 0]).strip().lower()
            if v == "fold":
                header_row = j
                break
            if res_pat.match(str(df_raw.iat[j, 0])):
                break
            j += 1

        if header_row is None:
            i += 1
            continue

        header_cells = df_raw.iloc[header_row, :].tolist()
        name_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(header_cells):
            s = str(name).strip()
            if s and s.lower() != "nan":
                name_to_idx[s] = idx

        missing = [c for c in (ap50_column, ap5095_column, latency_column) if c not in name_to_idx]
        if missing:
            raise KeyError(f"Missing columns in RESOLUTION {res} header: {missing}")

        ap50_idx = name_to_idx[ap50_column]
        ap5095_idx = name_to_idx[ap5095_column]
        lat_idx = name_to_idx[latency_column]

        mean_row = None
        k = header_row + 1
        while k < nrows:
            first = str(df_raw.iat[k, 0]).strip().upper()
            if first == "MEAN":
                mean_row = k
                break
            if res_pat.match(str(df_raw.iat[k, 0])):
                break
            k += 1

        if mean_row is None:
            raise ValueError(f"Could not find MEAN row for RESOLUTION {res}")

        ap50_val = float(df_raw.iat[mean_row, ap50_idx])
        ap5095_val = float(df_raw.iat[mean_row, ap5095_idx])
        lat_val = float(df_raw.iat[mean_row, lat_idx])

        resolutions.append(res)
        ap50_vals.append(ap50_val)
        ap5095_vals.append(ap5095_val)
        lat_vals.append(lat_val)

        i = k + 1

    out = pd.DataFrame(
        {
            "resolution": resolutions,
            "ap50": ap50_vals,
            "ap5095": ap5095_vals,
            "latency_ms": lat_vals,
        }
    ).sort_values("resolution")

    if out.empty:
        raise ValueError("No RESOLUTION blocks were parsed. Check the sheet name and formatting.")
    return out
def plot_map_vs_latency(
    excel_path: str,
    ap50_column: str,
    ap5095_column: str,
    latency_column: str,
    sheet_name: str | int | None = 0,
) -> None:
    """Create two separate connected-line plots: mAP@0.50 vs latency and mAP@0.50:0.95 vs latency."""

    data = extract_means_by_resolution(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    )

    # Sort by latency for smooth connected lines
    data = data.sort_values("latency_ms")

    resolutions = data["resolution"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    latency = data["latency_ms"].tolist()

    # ---- Plot 1: mAP@0.50 vs Latency ----
    plt.figure(figsize=(6, 5))
    plt.plot(latency, ap50, marker="o", color="tab:blue")
    for lat, ap, res in zip(latency, ap50, resolutions):
        plt.annotate(str(res), (lat, ap), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Latency [ms]")
    plt.ylabel("mAP@0.50")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out1 = Path(excel_path).parent / "map50_vs_latency.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved: {out1}")

    # ---- Plot 2: mAP@0.50:0.95 vs Latency ----
    plt.figure(figsize=(6, 5))
    plt.plot(latency, ap5095, marker="^", color="tab:green")
    for lat, ap, res in zip(latency, ap5095, resolutions):
        plt.annotate(str(res), (lat, ap), textcoords="offset points", xytext=(5, 5))

    plt.xlabel("Latency [ms]")
    plt.ylabel("mAP@0.50:0.95")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out2 = Path(excel_path).parent / "map5095_vs_latency.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved: {out2}")


def plot_resolution_ap_latency(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0, title: str = "AP and Latency") -> None:
    """Plot resolution vs AP50 and AP50:95 (left axis) and latency [ms] (right axis) from MEAN rows per RESOLUTION block."""
    data = extract_means_by_resolution(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    )

    x = data["resolution"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    lat = data["latency_ms"].tolist()

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("AP", color="tab:blue")
    ax1.plot(x, ap50, marker="o", color="tab:blue", label="mAP@0.50")
    ax1.plot(x, ap5095, marker="^", color="tab:green", label="mAP@0.50:0.95")
    ax1.set_xticks(x)
    ax1.set_ylim(bottom=0)  # Start AP axis at 0
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency [ms]", color="tab:red")
    ax2.plot(x, lat, marker="s", color="tab:red", label="Latency [ms]")
    ax2.set_ylim(bottom=0)  # Start latency axis at 0
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title)
    plt.tight_layout()

    out_path = Path(excel_path).parent / "resolution_map50_map5095_latency_ms_mean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    excel_file = r"Run 1 - Resolution.xlsx"

    ap50_metric = "AP_precision_iou_0.50_area_all_maxdets_100"
    ap5095_metric = "AP_precision_iou_0.50:0.95_area_all_maxdets_100"
    latency_metric = "latency_ms_per_image"

    plot_resolution_ap_latency(
        excel_path=excel_file,
        ap50_column=ap50_metric,
        ap5095_column=ap5095_metric,
        latency_column=latency_metric,
        sheet_name=0,
        title="mAP@0.50, mAP@0.50:0.95 and Latency",
    )

    plot_map_vs_latency(
        excel_path=excel_file,
        ap50_column=ap50_metric,
        ap5095_column=ap5095_metric,
        latency_column=latency_metric,
        sheet_name=0,
    )
