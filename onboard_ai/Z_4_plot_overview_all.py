from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_FONT_SIZE_TITLE = 16
PLOT_FONT_SIZE_AXIS = 16
PLOT_FONT_SIZE_TICKS = 14
PLOT_FONT_SIZE_LEGEND = 14
PLOT_FONT_SIZE_LEGEND_LARGE = 16

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "lines.linewidth": 1.4,
    "lines.antialiased": True,
    "axes.titlesize": PLOT_FONT_SIZE_TITLE,
    "axes.labelsize": PLOT_FONT_SIZE_AXIS,
    "xtick.labelsize": PLOT_FONT_SIZE_TICKS,
    "ytick.labelsize": PLOT_FONT_SIZE_TICKS,
    "legend.fontsize": PLOT_FONT_SIZE_LEGEND,
    "legend.frameon": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "figure.dpi": 120,
    "axes.labelpad": 12,
})


CASE_COLUMN = "case_name"

METRICS = [
    "C_succ_overall",
    "E_e2e",
    "L_mean_success_s",
    "V_mean_success_s",
    "Q_mean_success",
    "theta_mean_success_deg",
    "overall_f1_detection",
    "coco_ap50",
    "coco_ap50_95",
]


def _resolve_final_results_root(script_dir: Path, final_results_folder_name: str) -> Path:
    """Resolve FINAL_RESULTS root using the same folder logic as the overview script."""
    master_dir = script_dir.parent
    return master_dir / "0_results" / final_results_folder_name


def _resolve_sheet_name(xlsx_path: Path) -> str:
    """Use grouped_mean if available, otherwise mean."""
    xl = pd.ExcelFile(xlsx_path)
    if "grouped_mean" in xl.sheet_names:
        return "grouped_mean"
    if "mean" in xl.sheet_names:
        return "mean"
    raise ValueError(f"No suitable sheet found in {xlsx_path}. Available sheets: {xl.sheet_names}")


def _sanitize_filename(name: str) -> str:
    """Make a metric name safe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _load_overview_table(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load overview sheet and convert required columns to numeric."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    required_columns = [CASE_COLUMN, "offnadir_angle_deg", "time_delay_min"] + METRICS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {xlsx_path}: {missing}")

    output = df.copy()
    output[CASE_COLUMN] = output[CASE_COLUMN].astype(str)
    output["offnadir_angle_deg"] = pd.to_numeric(output["offnadir_angle_deg"], errors="coerce")
    output["time_delay_min"] = pd.to_numeric(output["time_delay_min"], errors="coerce")

    for metric in METRICS:
        output[metric] = pd.to_numeric(output[metric], errors="coerce")

    return output


def _filter_tc1x1(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only TC1x1 configurations."""
    return df[df[CASE_COLUMN].str.match(r"^TC_1x1sat_\d+deg_\d+min$", na=False)].copy()


def _save_subset_csv(df: pd.DataFrame, output_path: Path, sort_column: str) -> None:
    """Save one filtered sweep table to CSV."""
    preferred_columns = [
        CASE_COLUMN,
        "N_sats_total",
        "offnadir_angle_deg",
        "time_delay_min",
    ] + METRICS
    existing_columns = [column for column in preferred_columns if column in df.columns]
    df.sort_values(sort_column)[existing_columns].to_csv(output_path, index=False)


def _make_metric_plot(df: pd.DataFrame, x_column: str, metric: str, xlabel: str, title: str, output_path: Path) -> None:
    """Plot one metric against one sweep variable and save the figure."""
    plot_df = df[[CASE_COLUMN, x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column)

    if plot_df.empty:
        print(f"[SKIP] No valid data for plot: {output_path.name}")
        return

    ylabel = "Off-nadir angle [deg]" if metric == "theta_mean_success_deg" else metric

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(plot_df[x_column], plot_df[metric], marker="o", label=metric)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Create TC1x1 time-delay and off-nadir sweep plots from overview_random.xlsx."""
    script_dir = Path(__file__).resolve().parent

    final_results_folder_name = "EXPERIMENTS/reflection_offnadir_glint_255"
    overview_filename = "overview_random.xlsx"
    output_folder_name = "final_plots"

    final_results_root = _resolve_final_results_root(script_dir, final_results_folder_name)
    overview_path = final_results_root / overview_filename

    if not final_results_root.exists():
        raise FileNotFoundError(f"FINAL_RESULTS root does not exist: {final_results_root}")

    if not overview_path.exists():
        raise FileNotFoundError(f"Overview file does not exist: {overview_path}")

    sheet_name = _resolve_sheet_name(overview_path)
    df = _load_overview_table(overview_path, sheet_name)
    df_tc1x1 = _filter_tc1x1(df)

    if df_tc1x1.empty:
        raise ValueError("No TC1x1 rows found in the overview table.")

    time_delay_df = df_tc1x1[df_tc1x1["offnadir_angle_deg"] == 40].copy()
    offnadir_df = df_tc1x1[df_tc1x1["time_delay_min"] == 5].copy()

    if time_delay_df.empty:
        raise ValueError("No TC1x1 rows found for time-delay sweep with offnadir_angle_deg == 40.")
    if offnadir_df.empty:
        raise ValueError("No TC1x1 rows found for off-nadir sweep with time_delay_min == 5.")

    output_root = final_results_root / output_folder_name
    time_delay_output_dir = output_root / "time_delay_sweep"
    offnadir_output_dir = output_root / "offnadir_sweep"

    time_delay_output_dir.mkdir(parents=True, exist_ok=True)
    offnadir_output_dir.mkdir(parents=True, exist_ok=True)

    _save_subset_csv(
        df=time_delay_df,
        output_path=time_delay_output_dir / "time_delay_sweep_data_TC1x1.csv",
        sort_column="time_delay_min",
    )
    _save_subset_csv(
        df=offnadir_df,
        output_path=offnadir_output_dir / "offnadir_sweep_data_TC1x1.csv",
        sort_column="offnadir_angle_deg",
    )

    for metric in METRICS:
        _make_metric_plot(
            df=time_delay_df,
            x_column="time_delay_min",
            metric=metric,
            xlabel="Latency [min]",
            title=f"{metric} vs latency for TC1x1 at 40° off-nadir",
            output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg.png",
        )

        _make_metric_plot(
            df=offnadir_df,
            x_column="offnadir_angle_deg",
            metric=metric,
            xlabel="Off-nadir angle [deg]",
            title=f"{metric} vs off-nadir angle for TC1x1 at 5 min latency",
            output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min.png",
        )

    print(f"Overview file: {overview_path}")
    print(f"Sheet used: {sheet_name}")
    print(f"Total TC1x1 rows: {len(df_tc1x1)}")
    print(f"Time-delay sweep rows: {len(time_delay_df)}")
    print(f"Off-nadir sweep rows: {len(offnadir_df)}")
    print(f"Plots written to: {output_root}")


if __name__ == "__main__":
    main()