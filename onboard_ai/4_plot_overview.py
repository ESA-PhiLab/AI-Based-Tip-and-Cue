from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_FONT_SIZE_TITLE = 16
PLOT_FONT_SIZE_AXIS = 16
PLOT_FONT_SIZE_TICKS = 14
PLOT_FONT_SIZE_LEGEND = 14

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
    """Resolve FINAL_RESULTS root from script directory and result folder name."""
    master_dir = script_dir.parent
    return master_dir / "0_results" / final_results_folder_name


def _sanitize_filename(name: str) -> str:
    """Make a metric name safe for filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _load_grouped_mean_table(xlsx_path: Path) -> pd.DataFrame:
    """Load grouped_mean sheet and convert required columns to numeric."""
    df = pd.read_excel(xlsx_path, sheet_name="grouped_mean")

    required_columns = [CASE_COLUMN, "offnadir_angle_deg", "time_delay_min"] + METRICS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in grouped_mean of {xlsx_path}: {missing}")

    output = df.copy()
    output[CASE_COLUMN] = output[CASE_COLUMN].astype(str)
    output["offnadir_angle_deg"] = pd.to_numeric(output["offnadir_angle_deg"], errors="coerce")
    output["time_delay_min"] = pd.to_numeric(output["time_delay_min"], errors="coerce")

    for metric in METRICS:
        output[metric] = pd.to_numeric(output[metric], errors="coerce")

    return output


def _load_all_cases_table(xlsx_path: Path) -> pd.DataFrame:
    """Load all_cases sheet, extract seed and grouped case name, and convert required columns."""
    df = pd.read_excel(xlsx_path, sheet_name="all_cases")

    required_columns = [CASE_COLUMN, "offnadir_angle_deg", "time_delay_min"] + METRICS
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in all_cases of {xlsx_path}: {missing}")

    output = df.copy()
    output[CASE_COLUMN] = output[CASE_COLUMN].astype(str)
    output["offnadir_angle_deg"] = pd.to_numeric(output["offnadir_angle_deg"], errors="coerce")
    output["time_delay_min"] = pd.to_numeric(output["time_delay_min"], errors="coerce")

    for metric in METRICS:
        output[metric] = pd.to_numeric(output[metric], errors="coerce")

    output["seed"] = output[CASE_COLUMN].str.extract(r"_(\d+)sd$", expand=False)
    output["case_name_grouped"] = output[CASE_COLUMN].str.replace(r"_(\d+)sd$", "", regex=True)

    return output


def _filter_tc1x1_grouped_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only grouped TC1x1 configurations."""
    return df[df[CASE_COLUMN].str.match(r"^TC_1x1sat_\d+deg_\d+min$", na=False)].copy()


def _filter_tc1x1_all_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only seeded TC1x1 configurations from all_cases."""
    return df[df[CASE_COLUMN].str.match(r"^TC_1x1sat_\d+deg_\d+min_\d+sd$", na=False)].copy()


def _save_subset_csv(df: pd.DataFrame, output_path: Path, sort_columns: list[str]) -> None:
    """Save one filtered sweep table to CSV."""
    preferred_columns = [
        CASE_COLUMN,
        "case_name_grouped",
        "seed",
        "N_sats_total",
        "offnadir_angle_deg",
        "time_delay_min",
    ] + METRICS

    existing_columns = [column for column in preferred_columns if column in df.columns]
    df.sort_values(sort_columns)[existing_columns].to_csv(output_path, index=False)


def _make_mean_only_plot(df_mean: pd.DataFrame, x_column: str, metric: str, xlabel: str, title: str, output_path: Path) -> None:
    """Plot only the grouped mean line for one metric."""
    plot_df = df_mean[[CASE_COLUMN, x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column)

    if plot_df.empty:
        print(f"[SKIP] No valid data for mean-only plot: {output_path.name}")
        return

    ylabel = "Off-nadir angle [deg]" if metric == "theta_mean_success_deg" else metric

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(plot_df[x_column], plot_df[metric], marker="o", label="mean")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _make_mean_and_runs_plot(df_mean: pd.DataFrame, df_runs: pd.DataFrame, x_column: str, metric: str, xlabel: str, title: str, output_path: Path) -> None:
    """Plot grouped mean plus one aggregated line per seed from all_cases."""
    mean_plot_df = df_mean[[CASE_COLUMN, x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column)

    if mean_plot_df.empty:
        print(f"[SKIP] No valid data for mean+runs plot: {output_path.name}")
        return

    runs_plot_df = df_runs.copy()
    runs_plot_df = runs_plot_df[
        runs_plot_df["case_name_grouped"].str.match(r"^TC_1x1sat_\d+deg_\d+min$", na=False)
    ]
    runs_plot_df = runs_plot_df[["case_name_grouped", "seed", x_column, metric]].dropna(subset=[x_column, metric])

    duplicate_counts = (
        runs_plot_df.groupby(["seed", x_column], dropna=False)
        .size()
        .reset_index(name="count")
    )
    problematic = duplicate_counts[duplicate_counts["count"] > 1]

    if not problematic.empty:
        print(f"[WARN] Duplicate rows found for {output_path.name}. Aggregating by mean over seed + {x_column}.")
        print(problematic.sort_values(["seed", x_column]).to_string(index=False))

    runs_plot_df = (
        runs_plot_df.groupby(["seed", x_column], as_index=False)[metric]
        .mean()
        .sort_values(["seed", x_column])
    )

    ylabel = "Off-nadir angle [deg]" if metric == "theta_mean_success_deg" else metric

    plt.figure(figsize=(7.2, 4.8))

    for seed, seed_df in runs_plot_df.groupby("seed", dropna=True):
        seed_df = seed_df.sort_values(x_column)
        plt.plot(seed_df[x_column], seed_df[metric], marker="o", alpha=0.7, label=f"run {seed}")

    plt.plot(mean_plot_df[x_column], mean_plot_df[metric], marker="o", linewidth=2.4, label="mean")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Create TC1x1 time-delay and off-nadir sweep plots with mean-only and mean+run versions."""
    script_dir = Path(__file__).resolve().parent

    final_results_list = [
        "reflection_offnadir_glint_255",
        "reflection_nadir_glint_255",
        "texture_offnadir_255",
        "texture_nadir_255",
    ]
    location_plot_names = ["random", "Auckland2006", "Pelagos2016"]

    for final_results in final_results_list:
        print(f"\n=============== Start processing {final_results} ===============")
        final_results_folder_name = "EXPERIMENTS/" + final_results

        for location_plot in location_plot_names:
            overview_filename = f"overview_{location_plot}.xlsx"
            output_folder_name = f"final_plots_{location_plot}"

            final_results_root = _resolve_final_results_root(script_dir, final_results_folder_name)
            overview_path = final_results_root / overview_filename

            if not final_results_root.exists():
                raise FileNotFoundError(f"FINAL_RESULTS root does not exist: {final_results_root}")

            if not overview_path.exists():
                raise FileNotFoundError(f"Overview file does not exist: {overview_path}")

            df_mean = _load_grouped_mean_table(overview_path)
            df_runs = _load_all_cases_table(overview_path)

            df_mean_tc1x1 = _filter_tc1x1_grouped_mean(df_mean)
            df_runs_tc1x1 = _filter_tc1x1_all_cases(df_runs)

            if df_mean_tc1x1.empty:
                raise ValueError("No grouped TC1x1 rows found in grouped_mean.")
            if df_runs_tc1x1.empty:
                raise ValueError("No seeded TC1x1 rows found in all_cases.")

            time_delay_mean_df = df_mean_tc1x1[df_mean_tc1x1["offnadir_angle_deg"] == 40].copy()
            offnadir_mean_df = df_mean_tc1x1[df_mean_tc1x1["time_delay_min"] == 5].copy()

            time_delay_runs_df = df_runs_tc1x1[df_runs_tc1x1["offnadir_angle_deg"] == 40].copy()
            offnadir_runs_df = df_runs_tc1x1[df_runs_tc1x1["time_delay_min"] == 5].copy()

            if time_delay_mean_df.empty:
                raise ValueError("No grouped TC1x1 rows found for time-delay sweep with offnadir_angle_deg == 40.")
            if offnadir_mean_df.empty:
                raise ValueError("No grouped TC1x1 rows found for off-nadir sweep with time_delay_min == 5.")
            if time_delay_runs_df.empty:
                raise ValueError("No seeded TC1x1 rows found for time-delay sweep with offnadir_angle_deg == 40.")
            if offnadir_runs_df.empty:
                raise ValueError("No seeded TC1x1 rows found for off-nadir sweep with time_delay_min == 5.")

            output_root = final_results_root / output_folder_name
            time_delay_output_dir = output_root / "time_delay_sweep"
            offnadir_output_dir = output_root / "offnadir_sweep"

            time_delay_output_dir.mkdir(parents=True, exist_ok=True)
            offnadir_output_dir.mkdir(parents=True, exist_ok=True)

            _save_subset_csv(
                df=time_delay_mean_df,
                output_path=time_delay_output_dir / "time_delay_sweep_data_TC1x1_grouped_mean.csv",
                sort_columns=["time_delay_min"],
            )
            _save_subset_csv(
                df=offnadir_mean_df,
                output_path=offnadir_output_dir / "offnadir_sweep_data_TC1x1_grouped_mean.csv",
                sort_columns=["offnadir_angle_deg"],
            )
            _save_subset_csv(
                df=time_delay_runs_df,
                output_path=time_delay_output_dir / "time_delay_sweep_data_TC1x1_all_runs.csv",
                sort_columns=["seed", "time_delay_min"],
            )
            _save_subset_csv(
                df=offnadir_runs_df,
                output_path=offnadir_output_dir / "offnadir_sweep_data_TC1x1_all_runs.csv",
                sort_columns=["seed", "offnadir_angle_deg"],
            )

            for metric in METRICS:
                _make_mean_only_plot(
                    df_mean=time_delay_mean_df,
                    x_column="time_delay_min",
                    metric=metric,
                    xlabel="Latency [min]",
                    title=f"{metric} vs latency for TC1x1 at 40° off-nadir",
                    output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean.png",
                )

                _make_mean_only_plot(
                    df_mean=offnadir_mean_df,
                    x_column="offnadir_angle_deg",
                    metric=metric,
                    xlabel="Off-nadir angle [deg]",
                    title=f"{metric} vs off-nadir angle for TC1x1 at 5 min latency",
                    output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean.png",
                )

                _make_mean_and_runs_plot(
                    df_mean=time_delay_mean_df,
                    df_runs=time_delay_runs_df,
                    x_column="time_delay_min",
                    metric=metric,
                    xlabel="Latency [min]",
                    title=f"{metric} vs latency for TC1x1 at 40° off-nadir",
                    output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean_and_runs.png",
                )

                _make_mean_and_runs_plot(
                    df_mean=offnadir_mean_df,
                    df_runs=offnadir_runs_df,
                    x_column="offnadir_angle_deg",
                    metric=metric,
                    xlabel="Off-nadir angle [deg]",
                    title=f"{metric} vs off-nadir angle for TC1x1 at 5 min latency",
                    output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean_and_runs.png",
                )

            print(f"Overview file: {overview_path}")
            print("Sheets used: grouped_mean for mean, all_cases for individual seeded runs")
            print(f"Total grouped TC1x1 rows: {len(df_mean_tc1x1)}")
            print(f"Total seeded TC1x1 rows: {len(df_runs_tc1x1)}")
            print(f"Time-delay grouped rows: {len(time_delay_mean_df)}")
            print(f"Time-delay seeded rows: {len(time_delay_runs_df)}")
            print(f"Off-nadir grouped rows: {len(offnadir_mean_df)}")
            print(f"Off-nadir seeded rows: {len(offnadir_runs_df)}")
            print(f"Plots written to: {output_root}")


if __name__ == "__main__":
    main()