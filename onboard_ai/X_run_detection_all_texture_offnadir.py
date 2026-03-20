from __future__ import annotations

import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from run_detection_one import (
    _compute_overall_detection_stats,
    _safe_mean,
    _safe_std,
    _save_workbook,
    compute_coco_ap_metrics,
    evaluate_predictions,
    extract_gt_boxes,
    filter_predictions,
    get_image_record,
    gt_details_to_rows,
    gt_sample_summary_rows,
    image_summary_row,
    load_annotations,
    load_detector,
    prediction_details_to_rows,
    run_loaded_detection_raw,
    save_detection_image,
    show_detections_for_image,
)

RANDOM_SEED = 42


def _clear_prediction_images_dir(prediction_images_dir: Path) -> None:
    """Delete existing rendered prediction images in one folder."""
    if not prediction_images_dir.exists():
        return

    for image_file in prediction_images_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            image_file.unlink()


def _prepare_output_dir(output_dir: Path, overwrite_results: bool) -> Path:
    """Create fixed or timestamped output directory."""
    if overwrite_results:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
    timestamped_output_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_output_dir


def _find_master_case_directories(master_results_root: Path) -> list[Path]:
    """Return all scenario/case directories inside the master results folder."""
    case_dirs = [path for path in master_results_root.iterdir() if path.is_dir()]
    return sorted(case_dirs)


def _resolve_image_jobs(case_root: Path, mode: str, distinct_locations: list[str]) -> list[dict[str, Path | str]]:
    """Resolve which image folders to process inside one case folder."""
    jobs: list[dict[str, Path | str]] = []

    if mode not in {"random", "distinct", "all"}:
        raise ValueError("mode must be one of: 'random', 'distinct', 'all'")

    default_images_dir = case_root / "satellite_images"

    if mode in {"random", "all"}:
        if default_images_dir.exists():
            jobs.append(
                {
                    "label": "default",
                    "image_folder_path": default_images_dir,
                    "output_dir_name": "onboard_detection",
                }
            )
        else:
            print(f"[WARN] Missing default image folder: {default_images_dir}")

    if mode in {"distinct", "all"}:
        for location in distinct_locations:
            image_dir = case_root / f"satellite_images_{location}"
            if image_dir.exists():
                jobs.append(
                    {
                        "label": location,
                        "image_folder_path": image_dir,
                        "output_dir_name": f"onboard_detection_{location}",
                    }
                )
            else:
                print(f"[WARN] Missing distinct image folder: {image_dir}")

    return jobs


def _extract_offnadir_angle_deg(case_name: str) -> int | None:
    """Extract off-nadir angle only for TC case names; otherwise return None."""
    case_name = str(case_name).strip()
    match = re.match(r"^TC_(\d+)x(\d+)sat_(\d+)deg_(\d+)min(?:_\d+sd)?$", case_name)
    if not match:
        return None
    return int(match.group(3))


def _infer_location_from_image_name(image_name: str) -> str | None:
    """Infer source location from image filename."""
    image_name_lower = str(image_name).lower()
    if "auckland2006" in image_name_lower:
        return "Auckland2006"
    if "pelagos2016" in image_name_lower:
        return "Pelagos2016"
    return None


def _infer_location_from_folder_label(folder_label: str) -> str | None:
    """Infer source location from folder label."""
    label_lower = str(folder_label).lower()
    if "auckland2006" in label_lower:
        return "Auckland2006"
    if "pelagos2016" in label_lower:
        return "Pelagos2016"
    return None


def _load_location_detection_factors(overview_path: Path) -> dict[str, dict[int, float]]:
    """Load success fraction per location and off-nadir angle."""
    whales_df = pd.read_excel(overview_path, sheet_name="whales_F")
    original_ids_df = pd.read_excel(overview_path, sheet_name="original_ids")

    whales_df = whales_df.copy()
    original_ids_df = original_ids_df.copy()

    whales_df.columns = [str(col).strip() for col in whales_df.columns]
    original_ids_df.columns = [str(col).strip() for col in original_ids_df.columns]

    whales_location_col = whales_df.columns[0]
    whales_df = whales_df.rename(columns={whales_location_col: "location"})
    whales_df["location"] = whales_df["location"].astype(str).str.strip()

    original_location_col = original_ids_df.columns[0]
    original_ids_df = original_ids_df.rename(columns={original_location_col: "location"})
    original_ids_df["location"] = original_ids_df["location"].astype(str).str.strip()

    whales_df = whales_df[whales_df["location"].str.lower() != "column_total"].copy()
    original_ids_df = original_ids_df[original_ids_df["location"].str.lower() != "column_total"].copy()

    unique_ids_col = None
    for col in original_ids_df.columns:
        if col != "location":
            unique_ids_col = col
            break

    if unique_ids_col is None:
        raise KeyError(f"No unique-id count column found in sheet 'original_ids' of {overview_path}")

    original_ids_map = {
        str(row["location"]): float(row[unique_ids_col])
        for _, row in original_ids_df.iterrows()
        if pd.notna(row[unique_ids_col])
    }

    factors: dict[str, dict[int, float]] = {}

    for _, row in whales_df.iterrows():
        location = str(row["location"])

        if location not in original_ids_map:
            continue

        total_possible = float(original_ids_map[location])
        if total_possible <= 0:
            continue

        factors[location] = {}

        for column in whales_df.columns:
            if column == "location":
                continue

            match = re.search(r"(\d+)", str(column))
            if not match:
                continue

            angle_deg = int(match.group(1))
            successful_count = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(successful_count):
                continue

            factors[location][angle_deg] = float(successful_count) / total_possible

    return factors


def _resolve_success_fraction_for_image(image_path: Path, case_name: str, folder_label: str, location_factors: dict[str, dict[int, float]]) -> tuple[float | None, str | None, int | None]:
    """Resolve success fraction f for one image."""
    location = _infer_location_from_image_name(image_path.name)
    if location is None:
        location = _infer_location_from_folder_label(folder_label)

    offnadir_angle_deg = _extract_offnadir_angle_deg(case_name)

    if location is None:
        return None, location, offnadir_angle_deg

    if offnadir_angle_deg is None:
        return None, location, None

    if location not in location_factors:
        return None, location, offnadir_angle_deg

    if offnadir_angle_deg not in location_factors[location]:
        return None, location, offnadir_angle_deg

    return location_factors[location][offnadir_angle_deg], location, offnadir_angle_deg


def _bbox_to_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    """Convert common bbox formats to xyxy."""
    if box is None:
        return None

    if isinstance(box, dict):
        if all(key in box for key in ["x1", "y1", "x2", "y2"]):
            return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])

        if "bbox" in box:
            bbox = box["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, a, b = bbox
                x1 = float(x1)
                y1 = float(y1)
                a = float(a)
                b = float(b)

                if a >= x1 and b >= y1:
                    return x1, y1, a, b
                return x1, y1, x1 + a, y1 + b

        if all(key in box for key in ["xmin", "ymin", "xmax", "ymax"]):
            return float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])

    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, a, b = box
        x1 = float(x1)
        y1 = float(y1)
        a = float(a)
        b = float(b)

        if a >= x1 and b >= y1:
            return x1, y1, a, b
        return x1, y1, x1 + a, y1 + b

    return None


def _bbox_iou_xyxy(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Compute IoU for two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def _extract_gt_objects_for_image(annotations_data: dict[str, Any], image_id: int) -> list[dict[str, Any]]:
    """Extract GT objects with stable ids and boxes for one image."""
    gt_objects: list[dict[str, Any]] = []

    for ann in annotations_data.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        if int(ann.get("image_id", -1)) != int(image_id):
            continue

        bbox_xyxy = _bbox_to_xyxy(ann)
        if bbox_xyxy is None:
            bbox_xyxy = _bbox_to_xyxy(ann.get("bbox"))

        if bbox_xyxy is None:
            continue

        gt_objects.append(
            {
                "ann_id": str(ann.get("id", len(gt_objects))),
                "bbox_xyxy": bbox_xyxy,
            }
        )

    return gt_objects


def _drop_predictions_for_dropped_gt(raw_predictions: list[dict[str, Any]], gt_objects: list[dict[str, Any]], gt_keep_map: dict[str, bool], match_iou_threshold: float) -> list[dict[str, Any]]:
    """Remove only predictions matched to GT whales that were dropped."""
    filtered_predictions: list[dict[str, Any]] = []

    for pred in raw_predictions:
        pred_bbox = _bbox_to_xyxy(pred)
        if pred_bbox is None:
            filtered_predictions.append(pred)
            continue

        best_gt_id = None
        best_iou = -1.0

        for gt_obj in gt_objects:
            iou = _bbox_iou_xyxy(pred_bbox, gt_obj["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_obj["ann_id"]

        if best_gt_id is None or best_iou < match_iou_threshold:
            filtered_predictions.append(pred)
            continue

        if gt_keep_map.get(best_gt_id, True):
            filtered_predictions.append(pred)

    return filtered_predictions


def _apply_stochastic_positive_sample_dropout(raw_predictions: list[dict[str, Any]], gt_objects: list[dict[str, Any]], image_path: Path, case_name: str, folder_label: str, location_factors: dict[str, dict[int, float]] | None, apply_offnadir_success_dropout: bool, match_iou_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply stochastic dropout per GT whale object, not per image."""
    debug_info = {
        "dropout_enabled": apply_offnadir_success_dropout,
        "dropout_applied": False,
        "dropout_fraction": None,
        "dropout_location": None,
        "dropout_offnadir_angle_deg": None,
        "gt_total": len(gt_objects),
        "gt_kept": None,
        "gt_dropped": None,
    }

    if not apply_offnadir_success_dropout:
        return raw_predictions, debug_info

    if not gt_objects:
        return raw_predictions, debug_info

    if location_factors is None:
        return raw_predictions, debug_info

    success_fraction, location, offnadir_angle_deg = _resolve_success_fraction_for_image(
        image_path=image_path,
        case_name=case_name,
        folder_label=folder_label,
        location_factors=location_factors,
    )

    debug_info["dropout_location"] = location
    debug_info["dropout_offnadir_angle_deg"] = offnadir_angle_deg
    debug_info["dropout_fraction"] = success_fraction

    if success_fraction is None:
        return raw_predictions, debug_info

    success_fraction = max(0.0, min(1.0, float(success_fraction)))

    gt_keep_map: dict[str, bool] = {}
    kept_count = 0
    dropped_count = 0

    for gt_obj in gt_objects:
        whale_key = gt_obj["ann_id"]
        deterministic_key = f"{case_name}|{folder_label}|{image_path.stem}|{whale_key}|{RANDOM_SEED}"
        local_rng = random.Random(deterministic_key)
        keep = local_rng.random() < success_fraction
        gt_keep_map[whale_key] = keep

        if keep:
            kept_count += 1
        else:
            dropped_count += 1

    filtered_predictions = _drop_predictions_for_dropped_gt(
        raw_predictions=raw_predictions,
        gt_objects=gt_objects,
        gt_keep_map=gt_keep_map,
        match_iou_threshold=match_iou_threshold,
    )

    debug_info["dropout_applied"] = True
    debug_info["gt_kept"] = kept_count
    debug_info["gt_dropped"] = dropped_count

    return filtered_predictions, debug_info


def process_model_for_image_folder(
    model_name: str,
    best_stg_path: Path,
    config_path: Path,
    deimv2_repo_root: Path,
    image_folder_path: Path,
    output_root_dir: Path,
    anns_path: Path,
    device: str,
    render_scale: int,
    line_width: int,
    max_images: int | None,
    individual_score_threshold: float,
    individual_iou_threshold: float,
    ap_score_threshold: float,
    max_detections_individual: int | None,
    max_detections_ap: int | None,
    model_label_to_category_id: dict[int, int] | None,
    show_detections: bool,
    save_prediction_images: bool,
    reset_prediction_images: bool,
    overwrite_results: bool,
    location_factors: dict[str, dict[int, float]] | None,
    case_name: str,
    folder_label: str,
    apply_offnadir_success_dropout: bool,
) -> None:
    """Run detection for one fixed model on one image folder."""
    if not best_stg_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_stg_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config YAML: {config_path}")

    if not anns_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {anns_path}")

    if not anns_path.is_file():
        raise FileNotFoundError(f"Annotations path is not a file: {anns_path}")

    run_output_root = _prepare_output_dir(output_root_dir, overwrite_results=overwrite_results)
    prediction_images_dir = run_output_root / "prediction_images"
    results_xlsx_path = run_output_root / "onboard_detection_results.xlsx"
    per_sample_xlsx_path = run_output_root / "onboard_detection_per_sample.xlsx"

    prediction_images_dir.mkdir(parents=True, exist_ok=True)

    if reset_prediction_images:
        _clear_prediction_images_dir(prediction_images_dir)

    print(f"\n{'=' * 100}")
    print(f"Processing model: {model_name}")
    print(f"Image folder: {image_folder_path}")
    print(f"Checkpoint: {best_stg_path}")
    print(f"Config: {config_path}")
    print(f"Annotations: {anns_path}")
    print(f"Output dir: {run_output_root}")
    print(f"Apply off-nadir success dropout: {apply_offnadir_success_dropout}")
    print(f"{'=' * 100}")

    detector = load_detector(
        pth=best_stg_path,
        config=config_path,
        device=device,
        repo_root=deimv2_repo_root,
        use_cache=True,
    )

    annotations_data = load_annotations(
        annotations=anns_path,
        use_cache=True,
    )

    annotation_category_ids = sorted(
        {
            int(ann["category_id"])
            for ann in annotations_data.get("annotations", [])
            if isinstance(ann, dict) and "category_id" in ann
        }
    )

    if not annotation_category_ids:
        raise ValueError("No annotation category ids were found in annotations_data['annotations'].")

    if len(annotation_category_ids) != 1:
        raise ValueError(
            f"Expected a single-class dataset, but found multiple annotation category ids: {annotation_category_ids}"
        )

    gt_category_id = int(annotation_category_ids[0])

    raw_categories = annotations_data.get("categories", [])
    raw_category_ids = sorted(
        int(cat["id"])
        for cat in raw_categories
        if isinstance(cat, dict) and "id" in cat
    )

    if raw_category_ids != [gt_category_id]:
        if len(raw_categories) == 1 and isinstance(raw_categories[0], dict):
            fixed_category = dict(raw_categories[0])
            fixed_category["id"] = gt_category_id
            annotations_data["categories"] = [fixed_category]
        else:
            annotations_data["categories"] = [
                {
                    "id": gt_category_id,
                    "name": "whale",
                    "supercategory": "whale",
                }
            ]

    if model_label_to_category_id is None:
        model_label_to_category_id = {0: gt_category_id}

    image_paths = sorted(image_folder_path.glob("*.png"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    image_summary_rows: list[dict[str, Any]] = []
    individual_prediction_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    gt_sample_rows: list[dict[str, Any]] = []
    ap_predictions_iou50_rows: list[dict[str, Any]] = []
    debug_model_output_rows: list[dict[str, Any]] = []

    ap_predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    positive_sample_best_ious: list[float] = []
    positive_sample_best_confidences: list[float] = []
    all_sample_best_ious: list[float] = []
    all_sample_best_confidences: list[float] = []

    print(f"GT category ids in annotations: {annotation_category_ids}")
    print(
        "Categories ids after normalization: "
        f"{[int(cat['id']) for cat in annotations_data.get('categories', []) if isinstance(cat, dict) and 'id' in cat]}"
    )
    print(f"MODEL_LABEL_TO_CATEGORY_ID: {model_label_to_category_id}")

    next_print = 100
    for k, image_path in enumerate(image_paths):
        if k == next_print:
            print(f"Progress: {k}/{len(image_paths)}", end="\r")
            next_print += 100

        image_record = get_image_record(annotations_data, image_path)
        image_id = int(image_record["id"])
        gt_boxes = extract_gt_boxes(
            annotations=annotations_data,
            image_path=image_path,
        )
        gt_objects = _extract_gt_objects_for_image(
            annotations_data=annotations_data,
            image_id=image_id,
        )

        raw_predictions_before_dropout = run_loaded_detection_raw(
            detector=detector,
            image=image_path,
        )

        raw_predictions, dropout_debug = _apply_stochastic_positive_sample_dropout(
            raw_predictions=raw_predictions_before_dropout,
            gt_objects=gt_objects,
            image_path=image_path,
            case_name=case_name,
            folder_label=folder_label,
            location_factors=location_factors,
            apply_offnadir_success_dropout=apply_offnadir_success_dropout,
            match_iou_threshold=float(individual_iou_threshold),
        )

        debug_model_output_rows.append(
            {
                "image": image_path.name,
                "image_id": image_id,
                "num_model_outputs_before_threshold": len(raw_predictions_before_dropout),
                "num_model_outputs_after_dropout": len(raw_predictions),
                "dropout_enabled": dropout_debug["dropout_enabled"],
                "dropout_applied": dropout_debug["dropout_applied"],
                "dropout_fraction": dropout_debug["dropout_fraction"],
                "dropout_location": dropout_debug["dropout_location"],
                "dropout_offnadir_angle_deg": dropout_debug["dropout_offnadir_angle_deg"],
                "gt_total": dropout_debug["gt_total"],
                "gt_kept": dropout_debug["gt_kept"],
                "gt_dropped": dropout_debug["gt_dropped"],
            }
        )

        individual_predictions = filter_predictions(
            predictions=raw_predictions,
            score_threshold=individual_score_threshold,
            max_detections=max_detections_individual,
        )

        individual_scores = evaluate_predictions(
            predictions=individual_predictions,
            gt_boxes=gt_boxes,
            iou_threshold=individual_iou_threshold,
        )
        individual_scores["image"] = str(image_path)
        individual_scores["checkpoint"] = str(best_stg_path)
        individual_scores["config"] = str(config_path)
        individual_scores["device"] = device
        individual_scores["score_threshold"] = float(individual_score_threshold)
        individual_scores["iou_threshold"] = float(individual_iou_threshold)

        ap_predictions = raw_predictions
        ap_predictions_by_image[image_path.name] = ap_predictions

        ap_iou50_scores = evaluate_predictions(
            predictions=ap_predictions,
            gt_boxes=gt_boxes,
            iou_threshold=0.50,
        )

        image_summary_rows.append(
            image_summary_row(
                image_name=image_path.name,
                image_id=image_id,
                evaluation_scores=individual_scores,
                score_threshold=individual_score_threshold,
                iou_threshold=individual_iou_threshold,
            )
        )
        individual_prediction_rows.extend(
            prediction_details_to_rows(
                image_name=image_path.name,
                image_id=image_id,
                evaluation_scores=individual_scores,
            )
        )
        gt_rows.extend(
            gt_details_to_rows(
                image_name=image_path.name,
                image_id=image_id,
                evaluation_scores=individual_scores,
            )
        )
        gt_sample_rows.extend(
            gt_sample_summary_rows(
                image_name=image_path.name,
                image_id=image_id,
                evaluation_scores=individual_scores,
                score_threshold=individual_score_threshold,
                iou_threshold=individual_iou_threshold,
            )
        )
        ap_predictions_iou50_rows.extend(
            prediction_details_to_rows(
                image_name=image_path.name,
                image_id=image_id,
                evaluation_scores=ap_iou50_scores,
            )
        )

        if gt_boxes:
            gt_best_ious = [float(item["best_prediction_iou"]) for item in individual_scores["gt_details"]]
            gt_best_confidences = [
                float(item["best_prediction_score"]) if item["best_prediction_score"] is not None else 0.0
                for item in individual_scores["gt_details"]
            ]
            best_iou_for_sample = max(gt_best_ious) if gt_best_ious else 0.0
            best_conf_for_sample = max(gt_best_confidences) if gt_best_confidences else 0.0
            positive_sample_best_ious.append(best_iou_for_sample)
            positive_sample_best_confidences.append(best_conf_for_sample)
            all_sample_best_ious.append(best_iou_for_sample)
            all_sample_best_confidences.append(best_conf_for_sample)
        else:
            all_sample_best_ious.append(0.0)
            all_sample_best_confidences.append(0.0)

        if save_prediction_images:
            prediction_image_path = prediction_images_dir / image_path.name
            save_detection_image(
                image=image_path,
                gt_boxes=gt_boxes,
                prediction_details=individual_scores["prediction_details"],
                output_path=prediction_image_path,
                render_scale=render_scale,
                line_width=line_width,
            )

        if show_detections:
            title = (
                f"{image_path.name} | "
                f"TP={individual_scores['tp']} FP={individual_scores['fp']} FN={individual_scores['fn']} "
                f"| F1={individual_scores['f1']:.3f}"
            )
            show_detections_for_image(
                image=image_path,
                gt_boxes=gt_boxes,
                prediction_details=individual_scores["prediction_details"],
                title=title,
                render_scale=render_scale,
                line_width=line_width,
            )

    total_thresholded_predictions_individual = int(sum(int(row["num_predictions"]) for row in image_summary_rows))
    total_model_outputs_before_threshold = int(sum(int(row["num_model_outputs_before_threshold"]) for row in debug_model_output_rows))
    total_model_outputs_after_dropout = int(sum(int(row["num_model_outputs_after_dropout"]) for row in debug_model_output_rows))

    coco_metrics = compute_coco_ap_metrics(
        annotations_data=annotations_data,
        image_paths=image_paths,
        predictions_by_image=ap_predictions_by_image,
        label_to_category_id=model_label_to_category_id,
    )

    ap_metrics_rows = coco_metrics["per_iou_rows"]
    ap50 = coco_metrics["ap50"]
    ap50_95 = coco_metrics["ap50_95"]
    total_gt_for_ap = coco_metrics["total_gt"]
    total_ap_predictions = coco_metrics["total_predictions"]
    gt_category_ids_for_ap = coco_metrics["gt_category_ids"]
    prediction_category_ids_for_ap = coco_metrics["prediction_category_ids"]

    overall_detection_stats_rows = _compute_overall_detection_stats(
        image_summary_rows=image_summary_rows,
        individual_prediction_rows=individual_prediction_rows,
        gt_rows=gt_rows,
        positive_sample_best_ious=positive_sample_best_ious,
        positive_sample_best_confidences=positive_sample_best_confidences,
        all_sample_best_ious=all_sample_best_ious,
        all_sample_best_confidences=all_sample_best_confidences,
    )

    run_summary_rows = [
        {
            "model_name": model_name,
            "checkpoint": str(best_stg_path),
            "config": str(config_path),
            "image_folder": str(image_folder_path),
            "case_name": case_name,
            "folder_label": folder_label,
            "dropout_enabled": apply_offnadir_success_dropout,
            "num_images_processed": len(image_paths),
            "positive_sample_count": len(positive_sample_best_ious),
            "all_sample_count": len(all_sample_best_ious),
            "individual_score_threshold": float(individual_score_threshold),
            "individual_iou_threshold": float(individual_iou_threshold),
            "ap_score_threshold": float(ap_score_threshold),
            "annotation_category_ids": annotation_category_ids,
            "ap_gt_category_ids": gt_category_ids_for_ap,
            "ap_prediction_category_ids": prediction_category_ids_for_ap,
            "model_label_to_category_id": model_label_to_category_id,
            "coco_ap50": ap50,
            "coco_ap50_95": ap50_95,
            "total_gt": total_gt_for_ap,
            "reported_predictions_individual_threshold": total_thresholded_predictions_individual,
            "predictions_used_for_coco_ap": total_ap_predictions,
            "debug_model_outputs_before_threshold_total": total_model_outputs_before_threshold,
            "debug_model_outputs_after_dropout_total": total_model_outputs_after_dropout,
            "avg_best_iou_positive_samples": _safe_mean(positive_sample_best_ious),
            "std_best_iou_positive_samples": _safe_std(positive_sample_best_ious),
            "avg_best_confidence_positive_samples": _safe_mean(positive_sample_best_confidences),
            "std_best_confidence_positive_samples": _safe_std(positive_sample_best_confidences),
            "avg_best_iou_all_samples_negative_zero": _safe_mean(all_sample_best_ious),
            "std_best_iou_all_samples_negative_zero": _safe_std(all_sample_best_ious),
            "avg_best_confidence_all_samples_negative_zero": _safe_mean(all_sample_best_confidences),
            "std_best_confidence_all_samples_negative_zero": _safe_std(all_sample_best_confidences),
            "prediction_images_dir": str(prediction_images_dir),
            "results_workbook": str(results_xlsx_path),
            "per_sample_workbook": str(per_sample_xlsx_path),
        }
    ]

    _save_workbook(
        results_xlsx_path,
        {
            "run_summary": run_summary_rows,
            "overall_detection_stats": overall_detection_stats_rows,
            "ap_metrics_by_iou": ap_metrics_rows,
            "image_summary": image_summary_rows,
            "individual_predictions": individual_prediction_rows,
            "gt_log": gt_rows,
            "AP_Predictions_IoU50": ap_predictions_iou50_rows,
            "debug_model_outputs": debug_model_output_rows,
        },
    )

    _save_workbook(
        per_sample_xlsx_path,
        {
            "run_summary": run_summary_rows,
            "gt_sample_summary": gt_sample_rows,
        },
    )

    print("\nOfficial COCO AP results:")
    print(f"AP50: {ap50:.6f}")
    print(f"AP50:95: {ap50_95:.6f}")
    print(f"Total GT: {total_gt_for_ap}")
    print(f"GT category ids used for AP: {gt_category_ids_for_ap}")
    print(f"Prediction category ids used for AP: {prediction_category_ids_for_ap}")
    print(f"Reported predictions (thresholded): {total_thresholded_predictions_individual}")
    print(f"Predictions used for COCO AP: {total_ap_predictions}")
    print(f"Debug model outputs before threshold: {total_model_outputs_before_threshold}")
    print(f"Debug model outputs after dropout: {total_model_outputs_after_dropout}")

    print("\nOutputs written to:")
    print(results_xlsx_path)
    print(per_sample_xlsx_path)
    print(prediction_images_dir)


def main() -> None:
    """Loop over all case folders and selected image folders for one fixed model."""
    script_dir = Path(__file__).resolve().parent
    master_dir = script_dir.parent
    deimv2_repo_root = script_dir / "DEIMv2-main"

    model_name = "texture_offnadir_255"
    master_results = f"EXPERIMENTS/{model_name}"

    mode = "all"
    distinct_locations = ["Auckland2006", "Pelagos2016"]
    overwrite_results = True

    device = "cpu"
    render_scale = 10
    line_width = 15

    max_images: int | None = None
    individual_score_threshold = 0.3
    individual_iou_threshold = 0.5
    ap_score_threshold = 0.0

    max_detections_individual = 100
    max_detections_ap = None

    model_path = master_dir / "onboard_ai" / "final_models"
    best_stg_path = model_path / model_name / "final_location_holdout" / "best_stg2.pth"
    config_path = model_path / model_name / "final_location_holdout" / "config" / "base_config_with_train_norm.yml"
    master_results_root = master_dir / "0_results" / master_results

    model_label_to_category_id: dict[int, int] | None = None

    show_detections = False
    save_prediction_images = True
    reset_prediction_images = True

    apply_offnadir_success_dropout = True

    location_overview_path = script_dir / "DEIMv2-main" / "data" / "0_merged" / "reflection_offnadir_glint_255" / "location_detection_overview.xlsx"

    if not deimv2_repo_root.exists():
        raise FileNotFoundError(f"DEIMv2 repo root does not exist: {deimv2_repo_root}")

    if not master_results_root.exists():
        raise FileNotFoundError(f"Master results root does not exist: {master_results_root}")

    if not best_stg_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {best_stg_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")

    if apply_offnadir_success_dropout:
        if not location_overview_path.exists():
            raise FileNotFoundError(f"Location detection overview file does not exist: {location_overview_path}")
        location_factors = _load_location_detection_factors(location_overview_path)
    else:
        location_factors = None

    case_dirs = _find_master_case_directories(master_results_root)
    if not case_dirs:
        raise FileNotFoundError(f"No case folders found under {master_results_root}")

    all_jobs: list[dict[str, Path | str]] = []
    for case_dir in case_dirs:
        case_jobs = _resolve_image_jobs(
            case_root=case_dir,
            mode=mode,
            distinct_locations=distinct_locations,
        )
        for job in case_jobs:
            all_jobs.append(
                {
                    "case_dir": case_dir,
                    "case_name": case_dir.name,
                    "label": job["label"],
                    "image_folder_path": job["image_folder_path"],
                    "output_dir_name": job["output_dir_name"],
                }
            )

    if not all_jobs:
        raise ValueError("No image jobs were resolved across all case folders.")

    print(f"Using fixed model: {model_name}")
    print(f"Checkpoint: {best_stg_path}")
    print(f"Config: {config_path}")
    print(f"Location overview path: {location_overview_path}")
    print(f"Apply off-nadir success dropout: {apply_offnadir_success_dropout}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Found {len(case_dirs)} case folders inside {master_results_root}.")
    print(f"Resolved {len(all_jobs)} image folder jobs.")
    print()

    for job in all_jobs:
        print(f"Case: {job['case_name']} | Image job: {job['image_folder_path']} -> {job['output_dir_name']}")
    print()

    processed = 0
    skipped = 0
    failed = 0

    total_jobs = len(all_jobs)
    current_job = 0

    for job in all_jobs:
        current_job += 1
        case_dir = Path(job["case_dir"])
        image_folder_path = Path(job["image_folder_path"])
        output_root_dir = case_dir / str(job["output_dir_name"])
        anns_path = image_folder_path / "annotations_postprocessed.json"
        folder_label = str(job["label"])

        if not anns_path.exists():
            raise FileNotFoundError(f"Annotations file does not exist: {anns_path}")

        print(f"\n\n[{current_job}/{total_jobs}] Starting model: {model_name}")
        print(f"Case: {case_dir.name}")
        print(f"Dataset: {image_folder_path.name}")
        print(f"Folder label: {folder_label}")

        try:
            process_model_for_image_folder(
                model_name=model_name,
                best_stg_path=best_stg_path,
                config_path=config_path,
                deimv2_repo_root=deimv2_repo_root,
                image_folder_path=image_folder_path,
                output_root_dir=output_root_dir,
                anns_path=anns_path,
                device=device,
                render_scale=render_scale,
                line_width=line_width,
                max_images=max_images,
                individual_score_threshold=individual_score_threshold,
                individual_iou_threshold=individual_iou_threshold,
                ap_score_threshold=ap_score_threshold,
                max_detections_individual=max_detections_individual,
                max_detections_ap=max_detections_ap,
                model_label_to_category_id=model_label_to_category_id,
                show_detections=show_detections,
                save_prediction_images=save_prediction_images,
                reset_prediction_images=reset_prediction_images,
                overwrite_results=overwrite_results,
                location_factors=location_factors,
                case_name=case_dir.name,
                folder_label=folder_label,
                apply_offnadir_success_dropout=apply_offnadir_success_dropout,
            )
            processed += 1
        except FileNotFoundError as exc:
            skipped += 1
            print(f"[SKIP] {case_dir.name} | {image_folder_path.name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {case_dir.name} | {image_folder_path.name}: {exc}")

    print("\n" + "=" * 100)
    print("Finished processing fixed model across all case folders and selected image folders.")
    print(f"Processed successfully: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print("=" * 100)


if __name__ == "__main__":
    main()