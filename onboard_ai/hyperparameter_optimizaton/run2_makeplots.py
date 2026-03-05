import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

MetricRow = Literal["MEAN"]


def extract_means_by_model_section(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Parse Excel sections and extract (title, AP50, AP50:0.95, latency) from VALIDATION MEAN rows; robust to header differences."""
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    def norm(s: object) -> str:
        return str(s).strip()

    def is_title_candidate(text: str) -> bool:
        low = text.strip().lower()
        if low == "" or low == "nan":
            return False
        if low in {"validation", "test", "mean", "final"}:
            return False
        if low.startswith("fold"):
            return False
        if low.startswith("stg"):
            return False
        return re.search(r"[a-zA-Z]", text) is not None

    def find_first_in_col0(start: int, end: int, target: str) -> Optional[int]:
        tgt = target.strip().lower()
        for i in range(start, min(end, len(df_raw))):
            if norm(df_raw.iat[i, 0]).lower() == tgt:
                return i
        return None

    def find_row_label_in_col0(start: int, end: int, label: str) -> Optional[int]:
        tgt = label.strip().upper()
        for i in range(start, min(end, len(df_raw))):
            if norm(df_raw.iat[i, 0]).upper() == tgt:
                return i
        return None

    def build_name_to_idx(header_row: int) -> Dict[str, int]:
        header_cells = df_raw.iloc[header_row, :].tolist()
        name_to_idx: Dict[str, int] = {}
        for idx, name in enumerate(header_cells):
            s = norm(name)
            if s and s.lower() != "nan":
                name_to_idx[s] = idx
        return name_to_idx

    def row_contains_required_columns(r: int) -> bool:
        vals = [norm(v) for v in df_raw.iloc[r, :].tolist()]
        s = set(v for v in vals if v and v.lower() != "nan")
        return (ap50_column in s) and (ap5095_column in s) and (latency_column in s)

    def header_rows_up_to(row_inclusive: int) -> List[int]:
        rows: List[int] = []
        for r in range(0, min(row_inclusive + 1, len(df_raw))):
            if norm(df_raw.iat[r, 0]).lower() == "fold" and row_contains_required_columns(r):
                rows.append(r)
        return rows

    def try_extract_with_header(mean_row: int, header_row: int) -> Optional[Tuple[float, float, float]]:
        name_to_idx = build_name_to_idx(header_row)
        if ap50_column not in name_to_idx or ap5095_column not in name_to_idx or latency_column not in name_to_idx:
            return None

        ap50_idx = name_to_idx[ap50_column]
        ap5095_idx = name_to_idx[ap5095_column]
        lat_idx = name_to_idx[latency_column]

        ap50_val = df_raw.iat[mean_row, ap50_idx] if ap50_idx < df_raw.shape[1] else None
        ap5095_val = df_raw.iat[mean_row, ap5095_idx] if ap5095_idx < df_raw.shape[1] else None
        lat_val = df_raw.iat[mean_row, lat_idx] if lat_idx < df_raw.shape[1] else None

        if pd.isna(ap50_val) or pd.isna(ap5095_val) or pd.isna(lat_val):
            return None

        return float(ap50_val), float(ap5095_val), float(lat_val)

    # Collect section titles
    titles: List[Tuple[str, int]] = []
    for i in range(len(df_raw)):
        t = norm(df_raw.iat[i, 0])
        if is_title_candidate(t):
            titles.append((t, i))

    if not titles:
        raise ValueError("No section titles detected in column 0.")

    out_rows: List[Dict[str, object]] = []

    for idx, (title, start_row) in enumerate(titles):
        end_row = titles[idx + 1][1] if idx + 1 < len(titles) else len(df_raw)

        # Find MEAN row within VALIDATION block (but still within section bounds)
        val_row = find_first_in_col0(start_row, end_row, "validation")
        search_start = (val_row + 1) if val_row is not None else start_row
        mean_row = find_row_label_in_col0(search_start, end_row, "MEAN")
        if mean_row is None:
            continue

        # Prefer local header in this section; fallback to previous valid headers above MEAN
        local_header = find_first_in_col0(start_row, end_row, "fold")
        candidate_headers: List[int] = []
        if local_header is not None and row_contains_required_columns(local_header):
            candidate_headers.append(local_header)

        for hr in reversed(header_rows_up_to(mean_row)):
            if hr not in candidate_headers:
                candidate_headers.append(hr)

        extracted: Optional[Tuple[float, float, float]] = None
        used_header: Optional[int] = None
        for hr in candidate_headers:
            res = try_extract_with_header(mean_row=mean_row, header_row=hr)
            if res is not None:
                extracted = res
                used_header = hr
                break

        if extracted is None:
            continue

        ap50_val, ap5095_val, lat_val = extracted
        out_rows.append(
            {
                "title": title,
                "ap50": ap50_val,
                "ap5095": ap5095_val,
                "latency_ms": lat_val,
                "header_row_used": used_header,
                "mean_row": mean_row,
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise ValueError("No sections produced MEAN rows. Check that each section contains a VALIDATION->MEAN row.")
    return out


def plot_map_vs_latency_titles_mean(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0) -> None:
    """Connected-line plots: mAP@0.50 vs latency and mAP@0.50:0.95 vs latency, labeled by section titles (MEAN only)."""
    data = extract_means_by_model_section(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    ).sort_values("latency_ms")

    titles = data["title"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    latency = data["latency_ms"].tolist()

    plt.figure(figsize=(6, 5))
    plt.plot(latency, ap50, marker="o")
    for lat, ap, t in zip(latency, ap50, titles):
        plt.annotate(str(t), (lat, ap), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Latency [ms]")
    plt.ylabel("mAP@0.50")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out1 = Path(excel_path).parent / "2_map50_vs_latency_titles_mean.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved: {out1}")

    plt.figure(figsize=(6, 5))
    plt.plot(latency, ap5095, marker="^")
    for lat, ap, t in zip(latency, ap5095, titles):
        plt.annotate(str(t), (lat, ap), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Latency [ms]")
    plt.ylabel("mAP@0.50:0.95")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out2 = Path(excel_path).parent / "2_map5095_vs_latency_titles_mean.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved: {out2}")


def plot_titles_ap_latency_mean(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0, title: str = "AP and Latency (MEAN)") -> None:
    """Plot title index vs AP50/AP50:0.95 (left axis) and latency [ms] (right axis), MEAN only."""
    data = extract_means_by_model_section(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    )

    x = list(range(len(data)))
    labels = data["title"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    lat = data["latency_ms"].tolist()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Model section title")
    ax1.set_ylabel("AP")
    ax1.plot(x, ap50, marker="o", label="mAP@0.50")
    ax1.plot(x, ap5095, marker="^", label="mAP@0.50:0.95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle="--", linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency [ms]")
    ax2.plot(x, lat, marker="s", label="Latency [ms]")
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(title)
    plt.tight_layout()
    out_path = Path(excel_path).parent / "2_titles_map50_map5095_latency_ms_mean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_bar_titles_ap_latency_mean(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0, title: str = "AP and Latency (MEAN)") -> None:
    """Grouped bar plot per section: mAP@0.50, mAP@0.50:0.95 (left axis) and latency [ms] (right axis), MEAN only."""
    data = extract_means_by_model_section(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    )

    titles = data["title"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    latency = data["latency_ms"].tolist()

    x = list(range(len(titles)))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ---- AP bars (left axis) ----
    ax1.set_ylabel("AP")
    bars1 = ax1.bar([i - width for i in x], ap50, width=width, label="mAP@0.50", color="tab:blue")
    bars2 = ax1.bar(x, ap5095, width=width, label="mAP@0.50:0.95", color="tab:green")
    ax1.set_xticks(x)
    ax1.set_xticklabels(titles, rotation=20, ha="right")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.5)

    # ---- Latency bars (right axis) ----
    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency [ms]")
    bars3 = ax2.bar([i + width for i in x], latency, width=width, label="Latency [ms]", color="tab:red", alpha=0.8)
    ax2.set_ylim(bottom=0)

    # ---- Combined legend ----
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")

    plt.title(title)
    plt.tight_layout()

    out_path = Path(excel_path).parent / "2_bar_titles_map_latency_mean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_separate_bar_plots_mean(excel_path: str, ap50_column: str, ap5095_column: str, latency_column: str, sheet_name: str | int | None = 0) -> None:
    """Create three separate bar plots (MEAN only): mAP@0.50, mAP@0.50:0.95, and latency."""

    data = extract_means_by_model_section(
        excel_path=excel_path,
        ap50_column=ap50_column,
        ap5095_column=ap5095_column,
        latency_column=latency_column,
        sheet_name=sheet_name,
    )

    titles = data["title"].tolist()
    ap50 = data["ap50"].tolist()
    ap5095 = data["ap5095"].tolist()
    latency = data["latency_ms"].tolist()

    x = list(range(len(titles)))

    # ---- Plot 1: mAP@0.50 ----
    plt.figure(figsize=(8, 5))
    plt.bar(x, ap50, color="tab:blue")
    plt.xticks(x, titles, rotation=20, ha="right")
    plt.ylabel("mAP@0.50")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out1 = Path(excel_path).parent / "2_bar_map50_mean.png"
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"Saved: {out1}")

    # ---- Plot 2: mAP@0.50:0.95 ----
    plt.figure(figsize=(8, 5))
    plt.bar(x, ap5095, color="tab:green")
    plt.xticks(x, titles, rotation=20, ha="right")
    plt.ylabel("mAP@0.50:0.95")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out2 = Path(excel_path).parent / "2_bar_map5095_mean.png"
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved: {out2}")

    # ---- Plot 3: Latency ----
    plt.figure(figsize=(8, 5))
    plt.bar(x, latency, color="tab:red")
    plt.xticks(x, titles, rotation=20, ha="right")
    plt.ylabel("Latency [ms]")
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    out3 = Path(excel_path).parent / "2_bar_latency_mean.png"
    plt.savefig(out3, dpi=300)
    plt.close()
    print(f"Saved: {out3}")


if __name__ == "__main__":
    excel_file = r"Run 2 - Augmentation.xlsx"

    ap50_metric = "AP_precision_iou_0.50_area_all_maxdets_100"
    ap5095_metric = "AP_precision_iou_0.50:0.95_area_all_maxdets_100"
    latency_metric = "latency_ms_per_image"

    plot_titles_ap_latency_mean(
        excel_path=excel_file,
        ap50_column=ap50_metric,
        ap5095_column=ap5095_metric,
        latency_column=latency_metric,
        sheet_name=0,
        title="mAP@0.50, mAP@0.50:0.95 and Latency (VALIDATION MEAN per section)",
    )

    plot_map_vs_latency_titles_mean(
        excel_path=excel_file,
        ap50_column=ap50_metric,
        ap5095_column=ap5095_metric,
        latency_column=latency_metric,
        sheet_name=0,
    )

    plot_separate_bar_plots_mean(
        excel_path=excel_file,
        ap50_column=ap50_metric,
        ap5095_column=ap5095_metric,
        latency_column=latency_metric,
        sheet_name=0,
    )