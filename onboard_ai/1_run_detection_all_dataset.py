from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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

    for image_path in image_paths:
        image_record = get_image_record(annotations_data, image_path)
        image_id = int(image_record["id"])
        gt_boxes = extract_gt_boxes(
            annotations=annotations_data,
            image_path=image_path,
        )

        raw_predictions = run_loaded_detection_raw(
            detector=detector,
            image=image_path,
        )

        debug_model_output_rows.append(
            {
                "image": image_path.name,
                "image_id": image_id,
                "num_model_outputs_before_threshold": len(raw_predictions),
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

        print(f"\nImage: {image_path.name}")
        print(f"Model outputs before threshold: {len(raw_predictions)}")
        print(f"Reported detections after threshold: {len(individual_predictions)}")
        print(
            f"TP={individual_scores['tp']} "
            f"FP={individual_scores['fp']} "
            f"FN={individual_scores['fn']} "
            f"Precision={individual_scores['precision']:.4f} "
            f"Recall={individual_scores['recall']:.4f} "
            f"F1={individual_scores['f1']:.4f} "
            f"mean_tp_iou={individual_scores['mean_tp_iou']:.4f} "
            f"best_tp_confidence={individual_scores['best_tp_confidence']:.4f}"
        )

    total_thresholded_predictions_individual = int(sum(int(row["num_predictions"]) for row in image_summary_rows))
    total_model_outputs_before_threshold = int(sum(int(row["num_model_outputs_before_threshold"]) for row in debug_model_output_rows))

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

    print("\nOutputs written to:")
    print(results_xlsx_path)
    print(per_sample_xlsx_path)
    print(prediction_images_dir)


def main() -> None:
    """Loop over all case folders and selected image folders for one fixed model."""
    script_dir = Path(__file__).resolve().parent
    master_dir = script_dir.parent
    deimv2_repo_root = script_dir / "DEIMV2-main"


    model_name = "04_reflection_offnadir_glint_255"
    master_results = "FINAL_RESULTS"


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

    if not deimv2_repo_root.exists():
        raise FileNotFoundError(f"DEIMV2 repo root does not exist: {deimv2_repo_root}")

    if not master_results_root.exists():
        raise FileNotFoundError(f"Master results root does not exist: {master_results_root}")

    if not best_stg_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {best_stg_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")



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

        if not anns_path.exists():
            raise FileNotFoundError(f"Annotations file does not exist: {anns_path}")

        print(f"\n\n[{current_job}/{total_jobs}] Starting model: {model_name}")
        print(f"Case: {case_dir.name}")
        print(f"Dataset: {image_folder_path.name}")

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