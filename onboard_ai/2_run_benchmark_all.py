from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def _normalize_name(text: str) -> str:
    """Normalize sheet/column names for robust matching."""
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _find_case_directories(master_results_root: Path) -> list[Path]:
    """Return all case directories inside the master results folder."""
    return sorted(path for path in master_results_root.iterdir() if path.is_dir())


def _resolve_variant_jobs(case_dir: Path, mode: str, distinct_locations: list[str]) -> list[dict[str, str | Path]]:
    """Resolve benchmark variants per case based on mode."""
    if mode not in {"random", "distinct", "all"}:
        raise ValueError("mode must be one of: 'random', 'distinct', 'all'")

    jobs: list[dict[str, str | Path]] = []
    case_name = case_dir.name

    if mode in {"random", "all"}:
        jobs.append(
            {
                "label": "default",
                "suffix": "",
                "benchmark_filename": f"benchmark_{case_name}.xlsx",
                "detection_dir": case_dir / "onboard_detection",
            }
        )

    if mode in {"distinct", "all"}:
        for location in distinct_locations:
            jobs.append(
                {
                    "label": location,
                    "suffix": f"_{location}",
                    "benchmark_filename": f"benchmark_{location}_{case_name}.xlsx",
                    "detection_dir": case_dir / f"onboard_detection_{location}",
                }
            )

    return jobs


def _find_mission_results_workbook(case_dir: Path) -> Path:
    """Find the main simulation results workbook inside one case directory."""
    candidates = sorted(
        path for path in case_dir.glob("results_*.xlsx")
        if path.is_file() and not path.name.lower().startswith("benchmark")
    )

    if not candidates:
        raise FileNotFoundError(f"No mission results workbook found in {case_dir}")

    return candidates[0]


def _find_required_workbook(path: Path) -> Path:
    """Validate workbook existence."""
    if not path.exists():
        raise FileNotFoundError(f"Required workbook does not exist: {path}")
    return path


def _read_sheet_with_aliases(xlsx_path: Path, aliases: list[str]) -> pd.DataFrame:
    """Read a sheet by trying multiple normalized alias names."""
    excel_file = pd.ExcelFile(xlsx_path)
    normalized_to_original = {_normalize_name(name): name for name in excel_file.sheet_names}

    for alias in aliases:
        key = _normalize_name(alias)
        if key in normalized_to_original:
            return pd.read_excel(xlsx_path, sheet_name=normalized_to_original[key])

    raise KeyError(f"None of the sheet aliases {aliases} were found in {xlsx_path}")


def _extract_overview_value(overview_df: pd.DataFrame, metric_name: str) -> float | int | str | None:
    """Extract one metric value from the Overview sheet."""
    if "Metric" not in overview_df.columns or "Value" not in overview_df.columns:
        return None

    metric_series = overview_df["Metric"].astype(str).fillna("")
    match = overview_df.loc[metric_series == metric_name, "Value"]
    if match.empty:
        return None
    return match.iloc[0]


def _extract_metric_from_long_table(df: pd.DataFrame, metric_name: str) -> float | int | str | None:
    """Extract a metric from a long two-column metric/value table."""
    if "metric" not in df.columns or "value" not in df.columns:
        return None

    metric_series = df["metric"].astype(str).fillna("")
    match = df.loc[metric_series == metric_name, "value"]
    if match.empty:
        return None
    return match.iloc[0]


def _to_float(value: object) -> float | None:
    """Convert to float when possible."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_total_satellites_from_case_name(case_name: str) -> int | None:
    """Extract total satellites from case name like TC_4x2sat_... or C_8x2sat_... ."""
    case_name = str(case_name)

    tc_match = re.match(r"^TC_(\d+)x(\d+)sat_", case_name)
    if tc_match:
        n_orbits = int(tc_match.group(1))
        n_sats_per_orbit = int(tc_match.group(2))
        return n_orbits * n_sats_per_orbit

    c_match = re.match(r"^C_(\d+)x(\d+)sat(?:_|$)", case_name)
    if c_match:
        n_orbits = int(c_match.group(1))
        n_sats_per_orbit = int(c_match.group(2))
        return n_orbits * n_sats_per_orbit

    return None


def _to_bool_positive_label(value: object) -> bool:
    """Interpret positive whale labels robustly."""
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"whale", "positive", "1", "true", "yes"}


def _parse_datetime_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Parse datetime values using fixed formats and numeric Excel timestamps only."""
    if column_name not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index)

    series = df[column_name]

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric_series.notna()
    if numeric_mask.any():
        parsed.loc[numeric_mask] = pd.to_datetime(
            numeric_series[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    text_series = series.astype(str).str.strip()
    known_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in known_formats:
        remaining_mask = parsed.isna()
        if not remaining_mask.any():
            break
        parsed_part = pd.to_datetime(text_series[remaining_mask], format=fmt, errors="coerce")
        parsed.loc[remaining_mask] = parsed_part

    return parsed


def _aggregate_detection_per_image(gt_sample_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one candidate per image using the highest-confidence prediction."""
    required_columns = {"image", "best_prediction_score", "best_prediction_iou", "sample_status"}
    missing = required_columns - set(gt_sample_df.columns)
    if missing:
        raise KeyError(f"Missing columns in gt_sample_summary: {sorted(missing)}")

    work_df = gt_sample_df.copy()
    work_df["best_prediction_score_num"] = pd.to_numeric(work_df["best_prediction_score"], errors="coerce")
    work_df["best_prediction_iou_num"] = pd.to_numeric(work_df["best_prediction_iou"], errors="coerce")

    aggregated_rows: list[dict[str, object]] = []

    for image_name, group in work_df.groupby("image", dropna=False):
        sortable = group.copy()
        sortable["score_sort"] = sortable["best_prediction_score_num"].fillna(-1.0)
        sortable["iou_sort"] = sortable["best_prediction_iou_num"].fillna(-1.0)

        best_row = sortable.sort_values(["score_sort", "iou_sort"], ascending=[False, False]).iloc[0]

        aggregated_rows.append(
            {
                "image": image_name,
                "candidate_score": _to_float(best_row["best_prediction_score_num"]),
                "candidate_iou": _to_float(best_row["best_prediction_iou_num"]),
                "any_matched_tp": bool((group["sample_status"].astype(str) == "matched_tp").any()),
                "num_gt_rows": int(len(group)),
                "num_fn_rows": int(pd.to_numeric(group["is_fn"], errors="coerce").fillna(0).sum()) if "is_fn" in group.columns else None,
            }
        )

    return pd.DataFrame(aggregated_rows)


def _build_event_table(
    combined_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    detection_image_df: pd.DataFrame,
    tau_max_seconds: float | None,
    iou_threshold: float,
) -> pd.DataFrame:
    """Build per-event benchmark table by joining mission and detection outputs."""
    combined = combined_df.copy()
    dataset = dataset_df.copy()
    detection = detection_image_df.copy()

    if "detection_id" not in combined.columns:
        raise KeyError("Combined sheet must contain 'detection_id'")
    if "detection_id" not in dataset.columns or "saved_image" not in dataset.columns:
        raise KeyError("dataset_generaton sheet must contain 'detection_id' and 'saved_image'")

    combined["true_label_positive"] = combined["true_label"].apply(_to_bool_positive_label) if "true_label" in combined.columns else False
    combined["cue_confirmation_time_dt"] = _parse_datetime_column(combined, "cue_confirmation_time")
    combined["cue_observation_time_dt"] = _parse_datetime_column(combined, "cue_observation_time")
    combined["tip_observation_time_dt"] = _parse_datetime_column(combined, "tip_observation_time")

    dataset_link_df = (
        dataset[["detection_id", "saved_image"]]
        .dropna(subset=["detection_id"])
        .drop_duplicates(subset=["detection_id"], keep="first")
        .copy()
    )

    merged = combined.merge(
        dataset_link_df,
        on="detection_id",
        how="left",
        validate="one_to_one",
    )

    merged = merged.merge(
        detection,
        left_on="saved_image",
        right_on="image",
        how="left",
        validate="many_to_one",
    )

    merged["latency_for_benchmark_s"] = pd.to_numeric(merged.get("latency_confirmation"), errors="coerce")
    if "latency_confirmation" not in merged.columns:
        merged["latency_for_benchmark_s"] = pd.to_numeric(merged.get("latency_observation"), errors="coerce")

    merged["viewing_time_s"] = pd.to_numeric(merged.get("viewing_time"), errors="coerce")
    merged["offnadir_deg_num"] = pd.to_numeric(merged.get("offnadir_deg"), errors="coerce")
    merged["candidate_score_num"] = pd.to_numeric(merged.get("candidate_score"), errors="coerce")
    merged["candidate_iou_num"] = pd.to_numeric(merged.get("candidate_iou"), errors="coerce")

    merged["has_detection_candidate"] = merged["candidate_score_num"].notna()
    merged["passes_iou"] = merged["candidate_iou_num"].fillna(0.0) >= float(iou_threshold)

    if tau_max_seconds is None:
        merged["passes_latency"] = True
    else:
        merged["passes_latency"] = merged["latency_for_benchmark_s"].notna() & (merged["latency_for_benchmark_s"] <= float(tau_max_seconds))

    merged["successful_detection"] = (
        merged["true_label_positive"]
        & merged["has_detection_candidate"]
        & merged["passes_iou"]
        & merged["passes_latency"]
    )

    merged["successful_detection_without_latency"] = (
        merged["true_label_positive"]
        & merged["has_detection_candidate"]
        & merged["passes_iou"]
    )

    preferred_columns = [
        "detection_id",
        "target_id",
        "saved_image",
        "true_label",
        "true_label_positive",
        "tip_actor",
        "cue_actor",
        "tip_observation_time",
        "cue_observation_time",
        "cue_confirmation_time",
        "latency_observation",
        "latency_confirmation",
        "latency_for_benchmark_s",
        "viewing_time",
        "viewing_time_s",
        "offnadir_deg",
        "offnadir_deg_num",
        "gsd_m",
        "candidate_score",
        "candidate_score_num",
        "candidate_iou",
        "candidate_iou_num",
        "any_matched_tp",
        "has_detection_candidate",
        "passes_iou",
        "passes_latency",
        "successful_detection_without_latency",
        "successful_detection",
    ]

    existing_columns = [column for column in preferred_columns if column in merged.columns]
    remaining_columns = [column for column in merged.columns if column not in existing_columns]

    return merged[existing_columns + remaining_columns].copy()


def _collapse_event_df_to_unique_events(event_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per detection_id, keeping the strongest valid candidate per event."""
    if "detection_id" not in event_df.columns:
        return event_df.copy()

    work_df = event_df.copy()

    work_df["candidate_score_num"] = pd.to_numeric(work_df.get("candidate_score_num"), errors="coerce")
    work_df["candidate_iou_num"] = pd.to_numeric(work_df.get("candidate_iou_num"), errors="coerce")
    work_df["successful_detection"] = work_df.get("successful_detection", False).fillna(False).astype(bool)
    work_df["successful_detection_without_latency"] = (
        work_df.get("successful_detection_without_latency", False).fillna(False).astype(bool)
    )

    work_df["score_sort"] = work_df["candidate_score_num"].fillna(-1.0)
    work_df["iou_sort"] = work_df["candidate_iou_num"].fillna(-1.0)
    work_df["succ_sort"] = work_df["successful_detection"].astype(int)
    work_df["succ_no_latency_sort"] = work_df["successful_detection_without_latency"].astype(int)

    collapsed = (
        work_df.sort_values(
            ["detection_id", "succ_sort", "succ_no_latency_sort", "score_sort", "iou_sort"],
            ascending=[True, False, False, False, False],
        )
        .drop_duplicates(subset=["detection_id"], keep="first")
        .drop(columns=["score_sort", "iou_sort", "succ_sort", "succ_no_latency_sort"], errors="ignore")
        .reset_index(drop=True)
    )

    return collapsed


def _build_benchmark_overview(
    case_name: str,
    variant_label: str,
    mission_workbook: Path,
    detection_results_workbook: Path,
    detection_per_sample_workbook: Path,
    overview_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    overall_detection_stats_df: pd.DataFrame,
    event_df: pd.DataFrame,
    tau_max_seconds: float | None,
    iou_threshold: float,
) -> pd.DataFrame:
    """Build tidy benchmark overview rows."""
    n_sat_from_name = _extract_total_satellites_from_case_name(case_name)
    n_sat_from_overview = _to_float(_extract_overview_value(overview_df, "Number of satellites"))
    n_sat = float(n_sat_from_name) if n_sat_from_name is not None else n_sat_from_overview
    t_sim_hours = _to_float(_extract_overview_value(overview_df, "Simulation time (h)"))
    n_gt = _to_float(_extract_overview_value(overview_df, "Total targets"))

    event_unique_df = _collapse_event_df_to_unique_events(event_df)

    n_obs = int(event_unique_df["detection_id"].nunique()) if "detection_id" in event_unique_df.columns else int(len(event_unique_df))
    n_positive_obs = (
        int(event_unique_df["true_label_positive"].fillna(False).sum())
        if "true_label_positive" in event_unique_df.columns
        else None
    )

    n_succ = int(event_unique_df["successful_detection"].fillna(False).sum())
    n_succ_no_latency = int(event_unique_df["successful_detection_without_latency"].fillna(False).sum())

    c_succ = None
    if n_sat is not None and t_sim_hours is not None and n_sat > 0:
        t_sim_days = t_sim_hours / 24.0
        if t_sim_days > 0:
            c_succ = n_succ / (n_sat * t_sim_days)

    e_e2e = n_succ / n_obs if n_obs > 0 else None
    l_mean = _to_float(event_unique_df.loc[event_unique_df["successful_detection"], "latency_for_benchmark_s"].mean())
    v_mean = _to_float(event_unique_df.loc[event_unique_df["successful_detection"], "viewing_time_s"].mean())
    q_mean = _to_float(event_unique_df.loc[event_unique_df["successful_detection"], "candidate_score_num"].mean())
    theta_mean_success = _to_float(event_unique_df.loc[event_unique_df["successful_detection"], "offnadir_deg_num"].mean())

    coco_ap50 = None
    coco_ap50_95 = None
    detection_threshold = None
    if not run_summary_df.empty:
        coco_ap50 = _to_float(run_summary_df.iloc[0].get("coco_ap50"))
        coco_ap50_95 = _to_float(run_summary_df.iloc[0].get("coco_ap50_95"))
        detection_threshold = _to_float(run_summary_df.iloc[0].get("individual_score_threshold"))

    overall_f1_detection = _to_float(_extract_metric_from_long_table(overall_detection_stats_df, "overall_f1"))

    rows = [
        {"metric": "case_name", "value": case_name, "comment": None},
        {"metric": "variant_label", "value": variant_label, "comment": "default / location-specific"},
        {"metric": "mission_results_workbook", "value": str(mission_workbook), "comment": None},
        {"metric": "detection_results_workbook", "value": str(detection_results_workbook), "comment": None},
        {"metric": "detection_per_sample_workbook", "value": str(detection_per_sample_workbook), "comment": None},
        {"metric": "tau_max_seconds", "value": tau_max_seconds, "comment": "None means no latency hard-filter"},
        {"metric": "iou_threshold", "value": iou_threshold, "comment": None},
        {"metric": "detector_score_threshold", "value": detection_threshold, "comment": None},
        {"metric": "N_sat", "value": n_sat, "comment": "Number of satellites"},
        {"metric": "T_sim_hours", "value": t_sim_hours, "comment": "Simulation time in hours"},
        {"metric": "N_gt", "value": n_gt, "comment": "Total targets from mission overview"},
        {"metric": "N_obs", "value": n_obs, "comment": "Unique end-to-end observation opportunities from Combined"},
        {"metric": "N_positive_obs", "value": n_positive_obs, "comment": "Positive whale-labelled opportunities in Combined"},
        {"metric": "N_succ_detection_only", "value": n_succ_no_latency, "comment": "Positive events with valid matched detection, ignoring latency hard-filter"},
        {"metric": "N_succ", "value": n_succ, "comment": "Positive events with valid matched detection and latency filter"},
        {"metric": "C_succ", "value": c_succ, "comment": "Successful detections per satellite per simulated day"},
        {"metric": "E_e2e", "value": e_e2e, "comment": "Successful detections divided by observation opportunities"},
        {"metric": "L_mean_success_s", "value": l_mean, "comment": "Mean latency over successful detections"},
        {"metric": "V_mean_success_s", "value": v_mean, "comment": "Mean viewing time over successful detections"},
        {"metric": "Q_mean_success", "value": q_mean, "comment": "Mean matched detection confidence over successful detections"},
        {"metric": "theta_mean_success_deg", "value": theta_mean_success, "comment": "Geometry-only fallback / diagnostic"},
        {"metric": "overall_f1_detection", "value": overall_f1_detection, "comment": "Detection workbook overall F1 at deployed threshold"},
        {"metric": "coco_ap50", "value": coco_ap50, "comment": "Detection workbook summary"},
        {"metric": "coco_ap50_95", "value": coco_ap50_95, "comment": "Detection workbook summary"},
    ]

    return pd.DataFrame(rows)


def _normalize_excel_value(value: object) -> object:
    """Convert values to Excel-safe entries."""
    if isinstance(value, (list, dict, tuple, set)):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _write_dataframe_sheet(workbook: Workbook, sheet_name: str, dataframe: pd.DataFrame) -> None:
    """Write one dataframe to one worksheet."""
    worksheet = workbook.create_sheet(title=sheet_name[:31])

    if dataframe.empty:
        worksheet.append(["no_data"])
        return

    headers = list(dataframe.columns)
    worksheet.append(headers)

    for _, row in dataframe.iterrows():
        worksheet.append([_normalize_excel_value(row.get(header)) for header in headers])

    for column_index, header in enumerate(headers, start=1):
        max_length = len(str(header))
        for row_index in range(2, worksheet.max_row + 1):
            cell_value = worksheet.cell(row=row_index, column=column_index).value
            cell_length = len(str(cell_value)) if cell_value is not None else 0
            if cell_length > max_length:
                max_length = cell_length
        worksheet.column_dimensions[get_column_letter(column_index)].width = min(max_length + 2, 40)


def _write_benchmark_workbook(
    benchmark_path: Path,
    benchmark_overview_df: pd.DataFrame,
    event_df: pd.DataFrame,
    event_unique_df: pd.DataFrame,
    mission_overview_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    detection_run_summary_df: pd.DataFrame,
    overall_detection_stats_df: pd.DataFrame,
    detection_image_summary_df: pd.DataFrame | None,
    gt_sample_df: pd.DataFrame,
    overwrite_results: bool,
) -> None:
    """Write benchmark workbook with overview and source sheets."""
    if benchmark_path.exists() and not overwrite_results:
        print(f"[SKIP] Benchmark already exists and overwrite_results=False: {benchmark_path}")
        return

    workbook = Workbook()
    workbook.remove(workbook.active)

    _write_dataframe_sheet(workbook, "benchmark_overview", benchmark_overview_df)
    _write_dataframe_sheet(workbook, "event_details", event_df)
    _write_dataframe_sheet(workbook, "event_details_unique", event_unique_df)
    _write_dataframe_sheet(workbook, "mission_overview_src", mission_overview_df)
    _write_dataframe_sheet(workbook, "mission_combined_src", combined_df)
    _write_dataframe_sheet(workbook, "dataset_generation_src", dataset_df)
    _write_dataframe_sheet(workbook, "detection_run_summary", detection_run_summary_df)
    _write_dataframe_sheet(workbook, "overall_detection_stats", overall_detection_stats_df)

    if detection_image_summary_df is not None:
        _write_dataframe_sheet(workbook, "image_summary_src", detection_image_summary_df)

    _write_dataframe_sheet(workbook, "gt_sample_summary_src", gt_sample_df)

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(benchmark_path)


def process_case_variant(
    case_dir: Path,
    variant_label: str,
    detection_dir: Path,
    benchmark_filename: str,
    overwrite_results: bool,
    tau_max_seconds: float | None,
    iou_threshold: float,
) -> None:
    """Process one case and one variant into one benchmark workbook."""
    mission_workbook = _find_mission_results_workbook(case_dir)
    detection_results_workbook = _find_required_workbook(detection_dir / "onboard_detection_results.xlsx")
    detection_per_sample_workbook = _find_required_workbook(detection_dir / "onboard_detection_per_sample.xlsx")

    overview_df = _read_sheet_with_aliases(mission_workbook, ["Overview"])
    combined_df = _read_sheet_with_aliases(mission_workbook, ["Combined"])
    dataset_df = _read_sheet_with_aliases(mission_workbook, ["dataset_generaton", "dataset_generation"])

    detection_run_summary_df = _read_sheet_with_aliases(detection_results_workbook, ["run_summary"])
    overall_detection_stats_df = _read_sheet_with_aliases(detection_results_workbook, ["overall_detection_stats"])
    gt_sample_df = _read_sheet_with_aliases(detection_per_sample_workbook, ["gt_sample_summary"])

    detection_image_summary_df: pd.DataFrame | None
    try:
        detection_image_summary_df = _read_sheet_with_aliases(detection_results_workbook, ["image_summary"])
    except Exception:
        detection_image_summary_df = None

    detection_image_df = _aggregate_detection_per_image(gt_sample_df)

    event_df = _build_event_table(
        combined_df=combined_df,
        dataset_df=dataset_df,
        detection_image_df=detection_image_df,
        tau_max_seconds=tau_max_seconds,
        iou_threshold=iou_threshold,
    )

    event_unique_df = _collapse_event_df_to_unique_events(event_df)

    benchmark_overview_df = _build_benchmark_overview(
        case_name=case_dir.name,
        variant_label=variant_label,
        mission_workbook=mission_workbook,
        detection_results_workbook=detection_results_workbook,
        detection_per_sample_workbook=detection_per_sample_workbook,
        overview_df=overview_df,
        run_summary_df=detection_run_summary_df,
        overall_detection_stats_df=overall_detection_stats_df,
        event_df=event_df,
        tau_max_seconds=tau_max_seconds,
        iou_threshold=iou_threshold,
    )

    benchmark_path = case_dir / benchmark_filename

    _write_benchmark_workbook(
        benchmark_path=benchmark_path,
        benchmark_overview_df=benchmark_overview_df,
        event_df=event_df,
        event_unique_df=event_unique_df,
        mission_overview_df=overview_df,
        combined_df=combined_df,
        dataset_df=dataset_df,
        detection_run_summary_df=detection_run_summary_df,
        overall_detection_stats_df=overall_detection_stats_df,
        detection_image_summary_df=detection_image_summary_df,
        gt_sample_df=gt_sample_df,
        overwrite_results=overwrite_results,
    )

    print(f"[OK] Wrote benchmark: {benchmark_path}")


def main() -> None:
    """Process benchmark workbooks across all cases using selected mode."""
    script_dir = Path(__file__).resolve().parent
    master_dir = script_dir.parent

    master_results_list = ["reflection_offnadir_glint_255", "reflection_nadir_glint_255", "texture_offnadir_255", "texture_nadir_255"]
    mode = "all"

    distinct_locations = ["Auckland2006", "Pelagos2016"]
    overwrite_results = True

    tau_max_seconds: float | None = None
    iou_threshold = 0.5

    for master_results in master_results_list:

        print(f"\n=============== Start processing {master_results} ===============")

        master_results =  "EXPERIMENTS/" + master_results

        master_results_root = master_dir / "0_results" / master_results

        if not master_results_root.exists():
            raise FileNotFoundError(f"Master results root does not exist: {master_results_root}")

        case_dirs = _find_case_directories(master_results_root)
        if not case_dirs:
            raise FileNotFoundError(f"No case directories found under {master_results_root}")

        all_jobs: list[dict[str, str | Path]] = []
        for case_dir in case_dirs:
            for job in _resolve_variant_jobs(case_dir=case_dir, mode=mode, distinct_locations=distinct_locations):
                all_jobs.append(
                    {
                        "case_dir": case_dir,
                        "variant_label": str(job["label"]),
                        "detection_dir": Path(job["detection_dir"]),
                        "benchmark_filename": str(job["benchmark_filename"]),
                    }
                )

        if not all_jobs:
            raise ValueError("No jobs were resolved. Check mode and distinct_locations.")

        print(f"Found {len(case_dirs)} case folders.")
        print(f"Resolved {len(all_jobs)} benchmark jobs.")
        print(f"Mode: {mode}")
        print(f"tau_max_seconds: {tau_max_seconds}")
        print(f"iou_threshold: {iou_threshold}")
        print()

        processed = 0
        skipped = 0
        failed = 0

        for index, job in enumerate(all_jobs, start=1):
            case_dir = Path(job["case_dir"])
            variant_label = str(job["variant_label"])
            detection_dir = Path(job["detection_dir"])
            benchmark_filename = str(job["benchmark_filename"])

            print(f"\n[{index}/{len(all_jobs)}] Case: {case_dir.name} | Variant: {variant_label}")

            try:
                process_case_variant(
                    case_dir=case_dir,
                    variant_label=variant_label,
                    detection_dir=detection_dir,
                    benchmark_filename=benchmark_filename,
                    overwrite_results=overwrite_results,
                    tau_max_seconds=tau_max_seconds,
                    iou_threshold=iou_threshold,
                )
                processed += 1
            except FileNotFoundError as exc:
                skipped += 1
                print(f"[SKIP] {case_dir.name} | {variant_label}: {exc}")
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {case_dir.name} | {variant_label}: {exc}")

        print("\n" + "=" * 100)
        print("Finished processing benchmark workbooks.")
        print(f"Processed successfully: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print("=" * 100)


if __name__ == "__main__":
    main()