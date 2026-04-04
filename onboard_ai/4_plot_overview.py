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
    "C_mission_overall",
    "C_cue_task_received_overall",
    "C_cue_task_handled_overall",
    "C_succ_sat",
    "C_succ_overall",
    "C_succ_task_overall",
    "N_mission_task",
    "N_mission",
    "N_cue_task_received",
    "N_cue_task_handled",
    "N_succ",
    "N_succ_task",
    "L_mean_success_s",
    "V_mean_success_s",
    "offnadir_observed_mean_deg",
    "IoU_mean_success",
    "Q_mean_success",
    "coco_ap50",
    "coco_ap50_95",
    "detector_precision",
    "detector_recall",
    "detector_f1",
]

FINAL_RESULTS_LIST = [
    "reflection_offnadir_glint_255",
    "reflection_nadir_glint_255",
    "texture_offnadir_255",
    "texture_nadir_255",
]

LOCATION_PLOT_NAMES = ["random", "Auckland2006", "Pelagos2016"]
LOCATION_COMBINED_PLOT_NAMES = ["Auckland2006", "Pelagos2016"]

EXPERIMENT_LABELS = {
    "reflection_offnadir_glint_255": "Geometric and Radiometric Effects",
    "reflection_nadir_glint_255": "Only Radiometric Effects",
    "texture_offnadir_255": "Only Geometric Effects",
    "texture_nadir_255": "Raw Input Patches",
}

LOCATION_TITLE_LABELS = {
    "random": "random",
    "Auckland2006": "Auckland 2006",
    "Pelagos2016": "Pelagos 2016",
}

COMPARISON_GROUPS = {
    "reflection_offnadir_vs_reflection_nadir": [
        "reflection_offnadir_glint_255",
        "reflection_nadir_glint_255",
    ],
    "reflection_offnadir_vs_texture_offnadir": [
        "reflection_offnadir_glint_255",
        "texture_offnadir_255",
    ],
    "reflection_offnadir_vs_texture_nadir": [
        "reflection_offnadir_glint_255",
        "texture_nadir_255",
    ],
    "all_four": [
        "reflection_offnadir_glint_255",
        "reflection_nadir_glint_255",
        "texture_offnadir_255",
        "texture_nadir_255",
    ],
}

COLOR_MAP = {
    "reflection_offnadir_glint_255": "tab:blue",
    "reflection_nadir_glint_255": "tab:green",
    "texture_offnadir_255": "tab:orange",
    "texture_nadir_255": "tab:red",
}

LOCATION_COLOR_MAP = {
    "Auckland2006": "tab:blue",
    "Pelagos2016": "tab:orange",
}

METRIC_PLOT_META = {
    "C_mission_overall": {
        "symbol": r"$C_{\mathrm{mission}}$",
        "description": "Mission whale detections",
        "unit": "",
        "ylim": None,
    },
    "C_cue_task_received_overall": {
        "symbol": r"$C_{\mathrm{cue,received}}$",
        "description": "Cue tasks received",
        "unit": "",
        "ylim": None,
    },
    "C_cue_task_handled_overall": {
        "symbol": r"$C_{\mathrm{cue,handled}}$",
        "description": "Cue tasks handled",
        "unit": "",
        "ylim": (0, 80),
    },
    "C_succ_sat": {
        "symbol": r"$C_{\mathrm{succ}}$",
        "description": "Successful whale detections per satellite",
        "unit": "",
        "ylim": (0, 30),
    },
    "C_succ_overall": {
        "symbol": r"$C_{\mathrm{succ,mission}}$",
        "description": "Successful whale detections mission total",
        "unit": "",
        "ylim": None,
    },
    "C_succ_task_overall": {
        "symbol": r"$C_{\mathrm{succ,task}}$",
        "description": "Successful task confirmations mission total",
        "unit": "",
        "ylim": None,
    },
    "N_mission_task": {
        "symbol": r"$N_{\mathrm{mission,task}}$",
        "description": "Mission task events",
        "unit": "",
        "ylim": None,
    },
    "N_mission": {
        "symbol": r"$N_{\mathrm{mission}}$",
        "description": "Mission whale detections",
        "unit": "",
        "ylim": None,
    },
    "N_cue_task_received": {
        "symbol": r"$N_{\mathrm{cue,received}}$",
        "description": "Cue tasks received",
        "unit": "",
        "ylim": None,
    },
    "N_cue_task_handled": {
        "symbol": r"$N_{\mathrm{cue,handled}}$",
        "description": "Cue tasks handled",
        "unit": "",
        "ylim": (50, 80),
    },
    "N_succ": {
        "symbol": r"$N_{\mathrm{succ}}$",
        "description": "Successful whale detections",
        "unit": "",
        "ylim": None,
    },
    "N_succ_task": {
        "symbol": r"$N_{\mathrm{succ,task}}$",
        "description": "Successful task confirmations",
        "unit": "",
        "ylim": None,
    },
    "L_mean_success_s": {
        "symbol": r"$L$",
        "description": "Mean latency",
        "unit": "[s]",
        "ylim": (0, 650),
    },
    "V_mean_success_s": {
        "symbol": r"$V$",
        "description": "Mean viewing time",
        "unit": "[s]",
        "ylim": (0, 400),
    },
    "offnadir_observed_mean_deg": {
        "symbol": r"$\overline{\theta}_{\mathrm{obs}}$",
        "description": "Observed off-nadir angle",
        "unit": "[deg]",
        "ylim": (0, 60),
    },
    "IoU_mean_success": {
        "symbol": r"$\overline{\mathrm{IoU}}$",
        "description": "Mean IoU",
        "unit": "",
        "ylim": (0, 1),
    },
    "Q_mean_success": {
        "symbol": r"$Q$",
        "description": "Mean confidence",
        "unit": "",
        "ylim": (0, 1),
    },
    "coco_ap50": {
        "symbol": r"$\mathrm{AP}_{50}$",
        "description": "",
        "unit": "",
        "ylim": (0, 1),
    },
    "coco_ap50_95": {
        "symbol": r"$\mathrm{AP}_{50:95}$",
        "description": "",
        "unit": "",
        "ylim": (0, 1),
    },
    "detector_precision": {
        "symbol": r"$P$",
        "description": "Precision",
        "unit": "",
        "ylim": (0, 1),
    },
    "detector_recall": {
        "symbol": r"$R$",
        "description": "Recall",
        "unit": "",
        "ylim": (0, 1),
    },
    "detector_f1": {
        "symbol": r"$F_1$",
        "description": "score",
        "unit": "",
        "ylim": (0, 1),
    },
}


def _resolve_final_results_root(script_dir: Path, final_results_folder_name: str) -> Path:
    """Resolve results root path."""
    master_dir = script_dir.parent
    return master_dir / "0_results" / final_results_folder_name


def _sanitize_filename(name: str) -> str:
    """Make filename-safe string."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _get_experiment_label(final_results: str) -> str:
    """Return short human-readable experiment label."""
    return EXPERIMENT_LABELS.get(final_results, final_results)


def _get_location_label(location_plot: str) -> str:
    """Return short human-readable location label."""
    return LOCATION_TITLE_LABELS.get(location_plot, location_plot)


def _get_time_delay_scenario_title() -> str:
    """Return compact time-delay scenario title."""
    return "TC1x1, 40° planned off-nadir"


def _get_offnadir_scenario_title() -> str:
    """Return compact off-nadir scenario title."""
    return "TC1x1, 5 min latency"


def _build_single_experiment_title(final_results: str, location_plot: str, scenario_title: str) -> str:
    """Build compact title for one experiment and one location."""
    return f"{_get_experiment_label(final_results)}\n{_get_location_label(location_plot)} · {scenario_title}"


def _build_combined_location_title(final_results: str, scenario_title: str) -> str:
    """Build compact title for one experiment with Auckland and Pelagos combined."""
    return f"{_get_experiment_label(final_results)}\nAuckland 2006 + Pelagos 2016 · {scenario_title}"


def _build_comparison_title(location_plot: str, scenario_title: str) -> str:
    """Build compact title for cross-experiment comparison plots."""
    return f"{_get_location_label(location_plot)}\n{scenario_title}"


def _load_grouped_mean_table(xlsx_path: Path) -> pd.DataFrame:
    """Load grouped_mean sheet and coerce numeric columns."""
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
    """Load all_cases sheet and extract grouped case name plus seed."""
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
    """Keep grouped TC1x1 configurations only."""
    return df[df[CASE_COLUMN].str.match(r"^TC_1x1sat_\d+deg_\d+min$", na=False)].copy()


def _filter_tc1x1_all_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Keep seeded TC1x1 configurations only."""
    return df[df[CASE_COLUMN].str.match(r"^TC_1x1sat_\d+deg_\d+min_\d+sd$", na=False)].copy()


def _save_subset_csv(df: pd.DataFrame, output_path: Path, sort_columns: list[str]) -> None:
    """Save filtered subset to CSV."""
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


def _get_metric_plot_config(metric: str, y_unit_mode: str = "native") -> dict[str, object]:
    """Return ylabel, y-limits, and scale factor for one metric."""
    if metric not in METRIC_PLOT_META:
        raise KeyError(f"Missing plot metadata for metric: {metric}")

    meta = METRIC_PLOT_META[metric]
    scale = 1.0
    unit = meta["unit"]
    ylim = meta["ylim"]

    if y_unit_mode == "minutes":
        if metric not in {"L_mean_success_s", "V_mean_success_s"}:
            raise ValueError(f"Minute conversion is only supported for time metrics, got: {metric}")
        scale = 1.0 / 60.0
        unit = "[min]"
        if ylim is not None:
            ylim = (ylim[0] / 60.0, ylim[1] / 60.0)
    elif y_unit_mode != "native":
        raise ValueError(f"Unsupported y_unit_mode: {y_unit_mode}")

    label_parts = [meta["symbol"]]
    if meta["description"]:
        label_parts.append(meta["description"])
    if unit:
        label_parts.append(unit)

    ylabel = "\n".join(label_parts)
    return {
        "ylabel": ylabel,
        "ylim": ylim,
        "scale": scale,
    }


def _get_metric_plot_variants(metric: str) -> list[dict[str, str]]:
    """Return plot variants for one metric."""
    variants = [{"y_unit_mode": "native", "file_suffix": ""}]

    if metric in {"L_mean_success_s", "V_mean_success_s"}:
        variants.append({"y_unit_mode": "minutes", "file_suffix": "_ymin"})

    return variants


def _load_tc1x1_sweep_tables(overview_path: Path) -> dict[str, pd.DataFrame]:
    """Load grouped/all-case TC1x1 sweep tables from one overview file."""
    df_mean = _load_grouped_mean_table(overview_path)
    df_runs = _load_all_cases_table(overview_path)

    df_mean_tc1x1 = _filter_tc1x1_grouped_mean(df_mean)
    df_runs_tc1x1 = _filter_tc1x1_all_cases(df_runs)

    if df_mean_tc1x1.empty:
        raise ValueError(f"No grouped TC1x1 rows found in grouped_mean for {overview_path}")
    if df_runs_tc1x1.empty:
        raise ValueError(f"No seeded TC1x1 rows found in all_cases for {overview_path}")

    time_delay_mean_df = df_mean_tc1x1[df_mean_tc1x1["offnadir_angle_deg"] == 40].copy()
    offnadir_mean_df = df_mean_tc1x1[df_mean_tc1x1["time_delay_min"] == 5].copy()

    time_delay_runs_df = df_runs_tc1x1[df_runs_tc1x1["offnadir_angle_deg"] == 40].copy()
    offnadir_runs_df = df_runs_tc1x1[df_runs_tc1x1["time_delay_min"] == 5].copy()

    if time_delay_mean_df.empty:
        raise ValueError(f"No grouped TC1x1 rows found for time-delay sweep with offnadir_angle_deg == 40 in {overview_path}")
    if offnadir_mean_df.empty:
        raise ValueError(f"No grouped TC1x1 rows found for off-nadir sweep with time_delay_min == 5 in {overview_path}")
    if time_delay_runs_df.empty:
        raise ValueError(f"No seeded TC1x1 rows found for time-delay sweep with offnadir_angle_deg == 40 in {overview_path}")
    if offnadir_runs_df.empty:
        raise ValueError(f"No seeded TC1x1 rows found for off-nadir sweep with time_delay_min == 5 in {overview_path}")

    return {
        "df_mean_tc1x1": df_mean_tc1x1,
        "df_runs_tc1x1": df_runs_tc1x1,
        "time_delay_mean_df": time_delay_mean_df,
        "offnadir_mean_df": offnadir_mean_df,
        "time_delay_runs_df": time_delay_runs_df,
        "offnadir_runs_df": offnadir_runs_df,
    }


def _make_mean_only_plot(df_mean: pd.DataFrame, x_column: str, metric: str, xlabel: str, title: str, output_path: Path, y_unit_mode: str = "native") -> None:
    """Plot grouped mean only."""
    plot_config = _get_metric_plot_config(metric, y_unit_mode=y_unit_mode)
    plot_df = df_mean[[CASE_COLUMN, x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column).copy()

    if plot_df.empty:
        print(f"[SKIP] No valid data for mean-only plot: {output_path.name}")
        return

    plot_df[metric] = plot_df[metric] * float(plot_config["scale"])

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(plot_df[x_column], plot_df[metric], marker="o", label="mean")
    plt.xlabel(xlabel)
    plt.ylabel(str(plot_config["ylabel"]))
    plt.title(title)

    ylim = plot_config["ylim"]
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _make_mean_and_runs_plot(df_mean: pd.DataFrame, df_runs: pd.DataFrame, x_column: str, metric: str, xlabel: str, title: str, output_path: Path, y_unit_mode: str = "native") -> None:
    """Plot grouped mean and per-seed runs."""
    plot_config = _get_metric_plot_config(metric, y_unit_mode=y_unit_mode)
    mean_plot_df = df_mean[[CASE_COLUMN, x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column).copy()

    if mean_plot_df.empty:
        print(f"[SKIP] No valid data for mean+runs plot: {output_path.name}")
        return

    mean_plot_df[metric] = mean_plot_df[metric] * float(plot_config["scale"])

    runs_plot_df = df_runs.copy()
    runs_plot_df = runs_plot_df[
        runs_plot_df["case_name_grouped"].str.match(r"^TC_1x1sat_\d+deg_\d+min$", na=False)
    ]
    runs_plot_df = runs_plot_df[["case_name_grouped", "seed", x_column, metric]].dropna(subset=[x_column, metric])

    duplicate_counts = runs_plot_df.groupby(["seed", x_column], dropna=False).size().reset_index(name="count")
    problematic = duplicate_counts[duplicate_counts["count"] > 1]

    if not problematic.empty:
        print(f"[WARN] Duplicate rows found for {output_path.name}. Aggregating by mean over seed + {x_column}.")
        print(problematic.sort_values(["seed", x_column]).to_string(index=False))

    runs_plot_df = runs_plot_df.groupby(["seed", x_column], as_index=False)[metric].mean().sort_values(["seed", x_column]).copy()
    runs_plot_df[metric] = runs_plot_df[metric] * float(plot_config["scale"])

    plt.figure(figsize=(7.2, 4.8))

    for seed, seed_df in runs_plot_df.groupby("seed", dropna=True):
        seed_df = seed_df.sort_values(x_column)
        plt.plot(seed_df[x_column], seed_df[metric], marker="o", alpha=0.7, label=f"run {seed}")

    plt.plot(mean_plot_df[x_column], mean_plot_df[metric], marker="o", linewidth=2.4, label="mean")

    plt.xlabel(xlabel)
    plt.ylabel(str(plot_config["ylabel"]))
    plt.title(title)

    ylim = plot_config["ylim"]
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _make_combined_location_mean_only_plot(data_by_location: dict[str, dict[str, pd.DataFrame]], location_order: list[str], df_key: str, x_column: str, metric: str, xlabel: str, title: str, output_path: Path, y_unit_mode: str = "native") -> None:
    """Plot mean curves of multiple locations in one figure."""
    plot_config = _get_metric_plot_config(metric, y_unit_mode=y_unit_mode)
    plt.figure(figsize=(7.2, 4.8))
    plotted_any = False

    for location_name in location_order:
        if location_name not in data_by_location:
            continue

        df_mean = data_by_location[location_name][df_key]
        plot_df = df_mean[[x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column).copy()

        if plot_df.empty:
            continue

        plot_df[metric] = plot_df[metric] * float(plot_config["scale"])

        plt.plot(
            plot_df[x_column],
            plot_df[metric],
            marker="o",
            linewidth=2.4,
            label=location_name,
            color=LOCATION_COLOR_MAP.get(location_name, None),
        )
        plotted_any = True

    if not plotted_any:
        print(f"[SKIP] No valid data for combined-location mean-only plot: {output_path.name}")
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(str(plot_config["ylabel"]))
    plt.title(title)

    ylim = plot_config["ylim"]
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _make_combined_location_mean_and_runs_plot(data_by_location: dict[str, dict[str, pd.DataFrame]], location_order: list[str], mean_df_key: str, runs_df_key: str, x_column: str, metric: str, xlabel: str, title: str, output_path: Path, y_unit_mode: str = "native") -> None:
    """Plot mean and runs of multiple locations in one figure."""
    plot_config = _get_metric_plot_config(metric, y_unit_mode=y_unit_mode)
    plt.figure(figsize=(7.2, 4.8))
    plotted_any = False

    for location_name in location_order:
        if location_name not in data_by_location:
            continue

        color = LOCATION_COLOR_MAP.get(location_name, None)

        mean_plot_df = data_by_location[location_name][mean_df_key][[x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column).copy()
        runs_plot_df = data_by_location[location_name][runs_df_key][["case_name_grouped", "seed", x_column, metric]].dropna(subset=[x_column, metric]).copy()

        if mean_plot_df.empty:
            continue

        mean_plot_df[metric] = mean_plot_df[metric] * float(plot_config["scale"])

        duplicate_counts = runs_plot_df.groupby(["seed", x_column], dropna=False).size().reset_index(name="count")
        problematic = duplicate_counts[duplicate_counts["count"] > 1]

        if not problematic.empty:
            print(f"[WARN] Duplicate rows found for {output_path.name} / {location_name}. Aggregating by mean over seed + {x_column}.")
            print(problematic.sort_values(["seed", x_column]).to_string(index=False))

        runs_plot_df = runs_plot_df.groupby(["seed", x_column], as_index=False)[metric].mean().sort_values(["seed", x_column]).copy()
        runs_plot_df[metric] = runs_plot_df[metric] * float(plot_config["scale"])

        first_run = True
        for _, seed_df in runs_plot_df.groupby("seed", dropna=True):
            seed_df = seed_df.sort_values(x_column)
            plt.plot(
                seed_df[x_column],
                seed_df[metric],
                marker="o",
                alpha=0.30,
                color=color,
                label=f"{location_name} runs" if first_run else None,
            )
            first_run = False

        plt.plot(
            mean_plot_df[x_column],
            mean_plot_df[metric],
            marker="o",
            linewidth=2.8,
            color=color,
            label=f"{location_name} mean",
        )
        plotted_any = True

    if not plotted_any:
        print(f"[SKIP] No valid data for combined-location mean+runs plot: {output_path.name}")
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(str(plot_config["ylabel"]))
    plt.title(title)

    ylim = plot_config["ylim"]
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _load_tc1x1_mean_sweeps_for_location(script_dir: Path, final_results: str, location_plot: str) -> dict[str, pd.DataFrame]:
    """Load grouped mean TC1x1 sweeps for one experiment and location."""
    final_results_folder_name = "EXPERIMENTS/" + final_results
    final_results_root = _resolve_final_results_root(script_dir, final_results_folder_name)
    overview_path = final_results_root / f"overview_{location_plot}.xlsx"

    if not final_results_root.exists():
        raise FileNotFoundError(f"FINAL_RESULTS root does not exist: {final_results_root}")

    if not overview_path.exists():
        raise FileNotFoundError(f"Overview file does not exist: {overview_path}")

    sweep_tables = _load_tc1x1_sweep_tables(overview_path)

    return {
        "time_delay_mean_df": sweep_tables["time_delay_mean_df"],
        "offnadir_mean_df": sweep_tables["offnadir_mean_df"],
    }


def _make_combined_comparison_plot(data_by_experiment: dict[str, pd.DataFrame], experiment_order: list[str], x_column: str, metric: str, xlabel: str, title: str, output_path: Path, y_unit_mode: str = "native") -> None:
    """Plot grouped mean comparison across experiments."""
    plot_config = _get_metric_plot_config(metric, y_unit_mode=y_unit_mode)

    plt.figure(figsize=(7.2, 4.8))
    plotted_any = False

    for experiment_name in experiment_order:
        if experiment_name not in data_by_experiment:
            continue

        df = data_by_experiment[experiment_name]
        plot_df = df[[x_column, metric]].dropna(subset=[x_column, metric]).sort_values(x_column).copy()

        if plot_df.empty:
            continue

        plot_df[metric] = plot_df[metric] * float(plot_config["scale"])

        color = COLOR_MAP.get(experiment_name, "black")
        plt.plot(
            plot_df[x_column],
            plot_df[metric],
            marker="o",
            label=EXPERIMENT_LABELS.get(experiment_name, experiment_name),
            color=color,
        )
        plotted_any = True

    if not plotted_any:
        print(f"[SKIP] No valid data for combined plot: {output_path.name}")
        plt.close()
        return

    plt.xlabel(xlabel)
    plt.ylabel(str(plot_config["ylabel"]))
    plt.title(title)

    ylim = plot_config["ylim"]
    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(fontsize=PLOT_FONT_SIZE_LEGEND)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _write_combined_location_plots(script_dir: Path, final_results: str) -> None:
    """Create combined Auckland2006 + Pelagos2016 plots per experiment."""
    final_results_folder_name = "EXPERIMENTS/" + final_results
    final_results_root = _resolve_final_results_root(script_dir, final_results_folder_name)
    output_root = final_results_root / "final_plots_combined"
    time_delay_output_dir = output_root / "time_delay_sweep"
    offnadir_output_dir = output_root / "offnadir_sweep"

    time_delay_output_dir.mkdir(parents=True, exist_ok=True)
    offnadir_output_dir.mkdir(parents=True, exist_ok=True)

    data_by_location: dict[str, dict[str, pd.DataFrame]] = {}

    for location_plot in LOCATION_COMBINED_PLOT_NAMES:
        overview_path = final_results_root / f"overview_{location_plot}.xlsx"

        if not overview_path.exists():
            raise FileNotFoundError(f"Overview file does not exist: {overview_path}")

        data_by_location[location_plot] = _load_tc1x1_sweep_tables(overview_path)

        _save_subset_csv(
            df=data_by_location[location_plot]["time_delay_mean_df"],
            output_path=time_delay_output_dir / f"{location_plot}_time_delay_sweep_data_TC1x1_grouped_mean.csv",
            sort_columns=["time_delay_min"],
        )
        _save_subset_csv(
            df=data_by_location[location_plot]["time_delay_runs_df"],
            output_path=time_delay_output_dir / f"{location_plot}_time_delay_sweep_data_TC1x1_all_runs.csv",
            sort_columns=["seed", "time_delay_min"],
        )
        _save_subset_csv(
            df=data_by_location[location_plot]["offnadir_mean_df"],
            output_path=offnadir_output_dir / f"{location_plot}_offnadir_sweep_data_TC1x1_grouped_mean.csv",
            sort_columns=["offnadir_angle_deg"],
        )
        _save_subset_csv(
            df=data_by_location[location_plot]["offnadir_runs_df"],
            output_path=offnadir_output_dir / f"{location_plot}_offnadir_sweep_data_TC1x1_all_runs.csv",
            sort_columns=["seed", "offnadir_angle_deg"],
        )

    time_delay_title = _build_combined_location_title(final_results, _get_time_delay_scenario_title())
    offnadir_title = _build_combined_location_title(final_results, _get_offnadir_scenario_title())

    for metric in METRICS:
        for variant in _get_metric_plot_variants(metric):
            _make_combined_location_mean_only_plot(
                data_by_location=data_by_location,
                location_order=LOCATION_COMBINED_PLOT_NAMES,
                df_key="time_delay_mean_df",
                x_column="time_delay_min",
                metric=metric,
                xlabel="Tip-Cue time delay [min]",
                title=time_delay_title,
                output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean_Auckland2006_Pelagos2016{variant['file_suffix']}.png",
                y_unit_mode=variant["y_unit_mode"],
            )

            _make_combined_location_mean_and_runs_plot(
                data_by_location=data_by_location,
                location_order=LOCATION_COMBINED_PLOT_NAMES,
                mean_df_key="time_delay_mean_df",
                runs_df_key="time_delay_runs_df",
                x_column="time_delay_min",
                metric=metric,
                xlabel="Tip-Cue time delay [min]",
                title=time_delay_title,
                output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean_and_runs_Auckland2006_Pelagos2016{variant['file_suffix']}.png",
                y_unit_mode=variant["y_unit_mode"],
            )

        for variant in _get_metric_plot_variants(metric):
            _make_combined_location_mean_only_plot(
                data_by_location=data_by_location,
                location_order=LOCATION_COMBINED_PLOT_NAMES,
                df_key="offnadir_mean_df",
                x_column="offnadir_angle_deg",
                metric=metric,
                xlabel="Off-nadir angle [deg]",
                title=offnadir_title,
                output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean{variant['file_suffix']}.png",
                y_unit_mode=variant["y_unit_mode"],
            )

            _make_combined_location_mean_and_runs_plot(
                data_by_location=data_by_location,
                location_order=LOCATION_COMBINED_PLOT_NAMES,
                mean_df_key="offnadir_mean_df",
                runs_df_key="offnadir_runs_df",
                x_column="offnadir_angle_deg",
                metric=metric,
                xlabel="Off-nadir angle [deg]",
                title=offnadir_title,
                output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean_and_runs{variant['file_suffix']}.png",
                y_unit_mode=variant["y_unit_mode"],
            )

    print(f"[OK] Combined Auckland2006 + Pelagos2016 plots written to: {output_root}")


def _write_combined_comparison_plots(script_dir: Path, location_plot: str) -> None:
    """Create combined comparison plots for one location."""
    base_results_root = _resolve_final_results_root(script_dir, "EXPERIMENTS")
    combined_root = base_results_root / f"plots_combined_{location_plot}"
    combined_root.mkdir(parents=True, exist_ok=True)

    mean_sweep_tables: dict[str, dict[str, pd.DataFrame]] = {}

    for final_results in FINAL_RESULTS_LIST:
        mean_sweep_tables[final_results] = _load_tc1x1_mean_sweeps_for_location(
            script_dir=script_dir,
            final_results=final_results,
            location_plot=location_plot,
        )

    time_delay_title = _build_comparison_title(location_plot, _get_time_delay_scenario_title())
    offnadir_title = _build_comparison_title(location_plot, _get_offnadir_scenario_title())

    for group_name, experiment_names in COMPARISON_GROUPS.items():
        group_root = combined_root / group_name
        time_delay_output_dir = group_root / "time_delay_sweep"
        offnadir_output_dir = group_root / "offnadir_sweep"

        time_delay_output_dir.mkdir(parents=True, exist_ok=True)
        offnadir_output_dir.mkdir(parents=True, exist_ok=True)

        time_delay_data_by_experiment = {
            experiment_name: mean_sweep_tables[experiment_name]["time_delay_mean_df"]
            for experiment_name in experiment_names
        }
        offnadir_data_by_experiment = {
            experiment_name: mean_sweep_tables[experiment_name]["offnadir_mean_df"]
            for experiment_name in experiment_names
        }

        for metric in METRICS:
            for variant in _get_metric_plot_variants(metric):
                _make_combined_comparison_plot(
                    data_by_experiment=time_delay_data_by_experiment,
                    experiment_order=experiment_names,
                    x_column="time_delay_min",
                    metric=metric,
                    xlabel="Tip-Cue time delay [min]",
                    title=time_delay_title,
                    output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_comparison{variant['file_suffix']}.png",
                    y_unit_mode=variant["y_unit_mode"],
                )

            for variant in _get_metric_plot_variants(metric):
                _make_combined_comparison_plot(
                    data_by_experiment=offnadir_data_by_experiment,
                    experiment_order=experiment_names,
                    x_column="offnadir_angle_deg",
                    metric=metric,
                    xlabel="Off-nadir angle [deg]",
                    title=offnadir_title,
                    output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_comparison{variant['file_suffix']}.png",
                    y_unit_mode=variant["y_unit_mode"],
                )

        for experiment_name in experiment_names:
            time_delay_csv_path = time_delay_output_dir / f"{experiment_name}_time_delay_grouped_mean.csv"
            offnadir_csv_path = offnadir_output_dir / f"{experiment_name}_offnadir_grouped_mean.csv"

            _save_subset_csv(
                df=time_delay_data_by_experiment[experiment_name],
                output_path=time_delay_csv_path,
                sort_columns=["time_delay_min"],
            )
            _save_subset_csv(
                df=offnadir_data_by_experiment[experiment_name],
                output_path=offnadir_csv_path,
                sort_columns=["offnadir_angle_deg"],
            )

        print(f"[OK] Combined plots written for location '{location_plot}' in: {group_root}")


def main() -> None:
    """Create per-experiment, per-location, combined-location, and combined-comparison plots."""
    script_dir = Path(__file__).resolve().parent

    for final_results in FINAL_RESULTS_LIST:
        print(f"\n=============== Start processing {final_results} ===============")
        final_results_folder_name = "EXPERIMENTS/" + final_results

        for location_plot in LOCATION_PLOT_NAMES:
            overview_filename = f"overview_{location_plot}.xlsx"
            output_folder_name = f"final_plots_{location_plot}"

            final_results_root = _resolve_final_results_root(script_dir, final_results_folder_name)
            overview_path = final_results_root / overview_filename

            if not final_results_root.exists():
                raise FileNotFoundError(f"FINAL_RESULTS root does not exist: {final_results_root}")

            if not overview_path.exists():
                raise FileNotFoundError(f"Overview file does not exist: {overview_path}")

            sweep_tables = _load_tc1x1_sweep_tables(overview_path)

            df_mean_tc1x1 = sweep_tables["df_mean_tc1x1"]
            df_runs_tc1x1 = sweep_tables["df_runs_tc1x1"]
            time_delay_mean_df = sweep_tables["time_delay_mean_df"]
            offnadir_mean_df = sweep_tables["offnadir_mean_df"]
            time_delay_runs_df = sweep_tables["time_delay_runs_df"]
            offnadir_runs_df = sweep_tables["offnadir_runs_df"]

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

            time_delay_title = _build_single_experiment_title(final_results, location_plot, _get_time_delay_scenario_title())
            offnadir_title = _build_single_experiment_title(final_results, location_plot, _get_offnadir_scenario_title())

            for metric in METRICS:
                for variant in _get_metric_plot_variants(metric):
                    _make_mean_only_plot(
                        df_mean=time_delay_mean_df,
                        x_column="time_delay_min",
                        metric=metric,
                        xlabel="Tip-Cue time delay [min]",
                        title=time_delay_title,
                        output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean{variant['file_suffix']}.png",
                        y_unit_mode=variant["y_unit_mode"],
                    )

                    _make_mean_and_runs_plot(
                        df_mean=time_delay_mean_df,
                        df_runs=time_delay_runs_df,
                        x_column="time_delay_min",
                        metric=metric,
                        xlabel="Tip-Cue time delay [min]",
                        title=time_delay_title,
                        output_path=time_delay_output_dir / f"{_sanitize_filename(metric)}_vs_time_delay_TC1x1_40deg_mean_and_runs{variant['file_suffix']}.png",
                        y_unit_mode=variant["y_unit_mode"],
                    )

                for variant in _get_metric_plot_variants(metric):
                    _make_mean_only_plot(
                        df_mean=offnadir_mean_df,
                        x_column="offnadir_angle_deg",
                        metric=metric,
                        xlabel="Off-nadir angle [deg]",
                        title=offnadir_title,
                        output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean{variant['file_suffix']}.png",
                        y_unit_mode=variant["y_unit_mode"],
                    )

                    _make_mean_and_runs_plot(
                        df_mean=offnadir_mean_df,
                        df_runs=offnadir_runs_df,
                        x_column="offnadir_angle_deg",
                        metric=metric,
                        xlabel="Off-nadir angle [deg]",
                        title=offnadir_title,
                        output_path=offnadir_output_dir / f"{_sanitize_filename(metric)}_vs_offnadir_TC1x1_5min_mean_and_runs{variant['file_suffix']}.png",
                        y_unit_mode=variant["y_unit_mode"],
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

        _write_combined_location_plots(script_dir=script_dir, final_results=final_results)

    print("\n=============== Creating combined comparison plots ===============")
    for location_plot in LOCATION_PLOT_NAMES:
        _write_combined_comparison_plots(script_dir=script_dir, location_plot=location_plot)

    print("\nFinished all per-experiment and combined comparison plots.")


if __name__ == "__main__":
    main()