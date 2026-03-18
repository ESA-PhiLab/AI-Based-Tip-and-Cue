from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw


try:
    _NEAREST_RESAMPLE = Image.Resampling.NEAREST
except AttributeError:
    _NEAREST_RESAMPLE = Image.NEAREST


@dataclass
class LoadedDetector:
    """Reusable loaded DEIMv2 detector."""
    cfg: Any
    predictor: nn.Module
    device: torch.device
    transforms: T.Compose
    pth_path: Path
    config_path: Path
    repo_root_path: Path


_MODEL_CACHE: dict[tuple[str, str, str, str], LoadedDetector] = {}
_ANNOTATIONS_CACHE: dict[str, Any] = {}


def _find_repo_root(config_path: Path) -> Path:
    """Infer DEIMv2 repo root from config path."""
    for candidate in [config_path.parent, *config_path.parents]:
        if (candidate / "engine").exists() and (candidate / "train.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer the DEIMv2 repo root from config path: {config_path}\n"
        f"Expected a parent folder containing both 'engine' and 'train.py'."
    )


def _ensure_repo_on_sys_path(repo_root: Path) -> None:
    """Add DEIMv2 repo root to sys.path once."""
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _prepare_runtime_config(config_path: Path, repo_root: Path | None) -> Path:
    """Rewrite absolute __include__ paths to the local DEIMv2 repo."""
    if repo_root is None:
        return config_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    includes = cfg_dict.get("__include__")
    if not isinstance(includes, list):
        return config_path

    repo_root = repo_root.resolve()
    rewritten_includes = []
    changed = False

    for include in includes:
        include_str = str(include).replace("\\", "/")
        is_absolute_unix = include_str.startswith("/")
        is_absolute_windows = len(include_str) >= 3 and include_str[1] == ":" and include_str[2] == "/"

        if "/configs/" in include_str and (is_absolute_unix or is_absolute_windows):
            suffix = include_str.split("/configs/", 1)[1]
            new_include = (repo_root / "configs" / Path(suffix)).resolve().as_posix()
            rewritten_includes.append(new_include)
            changed = True
        else:
            rewritten_includes.append(include)

    if not changed:
        return config_path

    cfg_dict["__include__"] = rewritten_includes
    runtime_config_path = config_path.parent / f"{config_path.stem}__runtime.yml"

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    return runtime_config_path


def _get_device(device: str | None) -> torch.device:
    """Return requested torch device with CUDA fallback to CPU."""
    if device is not None:
        requested = str(device).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("Requested CUDA, but CUDA is not available in this PyTorch build. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_eval_size(yaml_cfg: dict[str, Any]) -> tuple[int, int]:
    """Return eval size as (height, width)."""
    eval_size = yaml_cfg.get("eval_spatial_size", (640, 640))
    if isinstance(eval_size, int):
        return int(eval_size), int(eval_size)
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])
    raise ValueError(f"Unsupported eval_spatial_size in config: {eval_size}")


def _build_image_transforms(yaml_cfg: dict[str, Any]) -> T.Compose:
    """Build eval transforms from val_dataloader config, matching solver.val() as closely as possible."""
    val_cfg = yaml_cfg.get("val_dataloader", {})
    dataset_cfg = val_cfg.get("dataset", {})
    transforms_cfg = dataset_cfg.get("transforms", {})
    ops_cfg = transforms_cfg.get("ops", [])

    ops: list[Any] = []

    if not isinstance(ops_cfg, list):
        size = _get_eval_size(yaml_cfg)
        return T.Compose([T.Resize(size), T.ToTensor()])

    for op_cfg in ops_cfg:
        if not isinstance(op_cfg, dict):
            continue

        op_type = str(op_cfg.get("type", "")).strip()

        if op_type == "Resize":
            size = op_cfg.get("size", _get_eval_size(yaml_cfg))
            if isinstance(size, int):
                size = (int(size), int(size))
            elif isinstance(size, (list, tuple)) and len(size) == 2:
                size = (int(size[0]), int(size[1]))
            else:
                raise ValueError(f"Unsupported Resize size in config: {size}")
            ops.append(T.Resize(size))

        elif op_type == "ConvertPILImage":
            ops.append(T.ToTensor())

        elif op_type == "Normalize":
            mean = op_cfg.get("mean")
            std = op_cfg.get("std")
            if mean is None or std is None:
                raise ValueError(f"Normalize op requires mean/std, got: {op_cfg}")
            ops.append(T.Normalize(mean=[float(v) for v in mean], std=[float(v) for v in std]))

        elif op_type in {"SanitizeBoundingBoxes", "ConvertBoxes"}:
            continue

    if not any(isinstance(op, T.ToTensor) for op in ops):
        ops.insert(1 if ops and isinstance(ops[0], T.Resize) else 0, T.ToTensor())

    if not ops:
        size = _get_eval_size(yaml_cfg)
        ops = [T.Resize(size), T.ToTensor()]

    return T.Compose(ops)


def _load_deimv2_predictor(config_path: Path, weights_path: Path, device: torch.device, use_cache: bool, repo_root: Path | None) -> LoadedDetector:
    """Load YAMLConfig and deployed DEIMv2 predictor once."""
    resolved_repo_root = repo_root.resolve() if repo_root is not None else _find_repo_root(config_path)
    cache_key = (str(config_path.resolve()), str(weights_path.resolve()), str(device), str(resolved_repo_root))

    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    _ensure_repo_on_sys_path(resolved_repo_root)

    from engine.core import YAMLConfig

    runtime_config_path = _prepare_runtime_config(config_path=config_path, repo_root=resolved_repo_root)
    cfg = YAMLConfig(str(runtime_config_path), resume=str(weights_path))

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if "ema" in checkpoint and isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
        state_dict = checkpoint["ema"]["module"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError("Could not find weights in checkpoint. Expected checkpoint['ema']['module'] or checkpoint['model'].")

    load_result = cfg.model.load_state_dict(state_dict, strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint/config mismatch.\n"
            f"Missing keys: {missing_keys[:20]}\n"
            f"Unexpected keys: {unexpected_keys[:20]}"
        )

    class Predictor(nn.Module):
        """Wrap deployed model and postprocessor."""

        def __init__(self, cfg_obj: Any) -> None:
            super().__init__()
            self.model = cfg_obj.model.deploy()
            self.postprocessor = cfg_obj.postprocessor.deploy()

        def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor) -> Any:
            """Run model and postprocessor."""
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    predictor = Predictor(cfg).to(device)
    predictor.eval()

    detector = LoadedDetector(
        cfg=cfg,
        predictor=predictor,
        device=device,
        transforms=_build_image_transforms(cfg.yaml_cfg),
        pth_path=weights_path.resolve(),
        config_path=config_path.resolve(),
        repo_root_path=resolved_repo_root.resolve(),
    )

    if use_cache:
        _MODEL_CACHE[cache_key] = detector

    return detector


def _prepare_image_tensor(image_path: Path, detector: LoadedDetector) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Load one image and prepare model input tensor."""
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_tensor = detector.transforms(image_pil).unsqueeze(0).to(detector.device)
    orig_size = torch.tensor([[orig_w, orig_h]], dtype=torch.int64, device=detector.device)
    return image_pil, image_tensor, orig_size


def _strip_batch_dim(array_like: Any) -> np.ndarray:
    """Convert tensor-like output to numpy and drop batch dim when batch size is 1."""
    if isinstance(array_like, torch.Tensor):
        array = array_like.detach().cpu().numpy()
    else:
        array = np.asarray(array_like)

    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _extract_predictor_outputs(outputs: Any) -> tuple[Any, Any, Any]:
    """Return labels, boxes, scores from common deployed detector output formats."""
    if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
        return outputs[0], outputs[1], outputs[2]

    if isinstance(outputs, dict):
        label_keys = ["labels", "label", "pred_labels"]
        box_keys = ["boxes", "bbox", "bboxes", "pred_boxes"]
        score_keys = ["scores", "score", "pred_scores"]

        labels = next((outputs[key] for key in label_keys if key in outputs), None)
        boxes = next((outputs[key] for key in box_keys if key in outputs), None)
        scores = next((outputs[key] for key in score_keys if key in outputs), None)

        if labels is not None and boxes is not None and scores is not None:
            return labels, boxes, scores

    if isinstance(outputs, (list, tuple)) and len(outputs) == 1 and isinstance(outputs[0], dict):
        return _extract_predictor_outputs(outputs[0])

    structure = f"type={type(outputs)}"
    if isinstance(outputs, (list, tuple)):
        structure += f", len={len(outputs)}, item_types={[type(item) for item in outputs[:5]]}"
    elif isinstance(outputs, dict):
        structure += f", keys={list(outputs.keys())}"
    raise RuntimeError(
        "Unexpected predictor output structure. Expected either (labels, boxes, scores) "
        f"or a dict containing them. Got {structure}"
    )


def _xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    """Convert one COCO xywh box to xyxy."""
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    """Convert one xyxy box to xywh."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _sort_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort predictions by descending score."""
    return sorted(predictions, key=lambda item: float(item["score"]), reverse=True)


def filter_predictions(predictions: list[dict[str, Any]], score_threshold: float, max_detections: int | None = None) -> list[dict[str, Any]]:
    """Filter by score, sort descending, and optionally keep top-k detections."""
    filtered = [dict(item) for item in predictions if float(item["score"]) >= float(score_threshold)]
    filtered = _sort_predictions(filtered)
    return filtered[:max_detections] if max_detections is not None else filtered


def load_annotations(annotations: str | Path | dict[str, Any] | list[Any] | None, use_cache: bool = True) -> Any:
    """Load annotations once and optionally cache file-based JSON."""
    if annotations is None:
        return None

    if isinstance(annotations, (dict, list)):
        return annotations

    annotations_path = Path(annotations).resolve()
    cache_key = str(annotations_path)

    if use_cache and cache_key in _ANNOTATIONS_CACHE:
        return _ANNOTATIONS_CACHE[cache_key]

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    if use_cache:
        _ANNOTATIONS_CACHE[cache_key] = annotations_data

    return annotations_data


def get_image_record(annotations: str | Path | dict[str, Any] | list[Any] | None, image_path: Path) -> dict[str, Any]:
    """Return the COCO image record for one image path."""
    data = load_annotations(annotations, use_cache=True)
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError("COCO annotations with an 'images' list are required.")

    image_name = image_path.name
    image_posix = image_path.as_posix()
    matches = []

    for image_info in data["images"]:
        file_name = str(image_info.get("file_name", "")).replace("\\", "/")
        if file_name == image_name or file_name == image_posix or file_name.endswith("/" + image_name):
            matches.append(image_info)

    if not matches:
        raise ValueError(f"Could not find image '{image_name}' in the COCO annotations file.")

    if len(matches) > 1:
        exact = [m for m in matches if str(m.get("file_name", "")).replace("\\", "/") == image_name]
        if len(exact) == 1:
            matches = exact
        else:
            raise ValueError(f"Found multiple matching image entries for '{image_name}' in the COCO annotations.")

    return dict(matches[0])


def _extract_gt_boxes_from_coco(coco: dict[str, Any], image_path: Path) -> list[list[float]]:
    """Extract GT boxes for one image from a COCO annotations dict."""
    image_record = get_image_record(coco, image_path)
    image_id = image_record["id"]
    gt_boxes = []
    for ann in coco["annotations"]:
        if ann.get("image_id") == image_id and "bbox" in ann:
            gt_boxes.append(_xywh_to_xyxy(ann["bbox"]))
    return gt_boxes


def extract_gt_boxes(annotations: str | Path | dict[str, Any] | list[Any] | None, image_path: Path) -> list[list[float]]:
    """Extract GT boxes from COCO, direct lists, or None."""
    data = load_annotations(annotations, use_cache=True)
    if data is None:
        return []

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        return _extract_gt_boxes_from_coco(data, image_path)

    if isinstance(data, list):
        gt_boxes = []
        for item in data:
            if isinstance(item, dict) and "bbox" in item:
                gt_boxes.append(_xywh_to_xyxy(item["bbox"]))
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                gt_boxes.append([float(v) for v in item])
            else:
                raise ValueError("List annotations must contain dicts with 'bbox' or direct xyxy boxes.")
        return gt_boxes

    raise ValueError("Unsupported annotations input. Use a COCO JSON path, a loaded COCO dict, or a list of boxes.")


def _box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def evaluate_predictions(predictions: list[dict[str, Any]], gt_boxes: list[list[float]], iou_threshold: float) -> dict[str, Any]:
    """Compute per-image metrics plus per-prediction and per-GT logs."""
    matched_gt_indices: set[int] = set()
    prediction_details: list[dict[str, Any]] = []
    gt_details: list[dict[str, Any]] = []

    sorted_predictions = _sort_predictions(predictions)

    gt_best_prediction_iou = [0.0 for _ in gt_boxes]
    gt_best_prediction_index = [-1 for _ in gt_boxes]
    gt_best_prediction_score = [None for _ in gt_boxes]
    gt_matched_prediction_index = [-1 for _ in gt_boxes]
    gt_matched_score = [None for _ in gt_boxes]
    gt_matched_iou = [None for _ in gt_boxes]

    tp = 0
    fp = 0

    for pred_idx, pred in enumerate(sorted_predictions):
        pred_box = [float(v) for v in pred["bbox_xyxy"]]
        pred_score = float(pred["score"])
        pred_label = int(pred["label"])

        best_iou_any_gt = 0.0
        best_iou_any_gt_idx = -1
        best_unmatched_iou = 0.0
        best_unmatched_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = _box_iou_xyxy(pred_box, gt_box)

            if iou > gt_best_prediction_iou[gt_idx]:
                gt_best_prediction_iou[gt_idx] = float(iou)
                gt_best_prediction_index[gt_idx] = pred_idx
                gt_best_prediction_score[gt_idx] = pred_score

            if iou > best_iou_any_gt:
                best_iou_any_gt = iou
                best_iou_any_gt_idx = gt_idx

            if gt_idx not in matched_gt_indices and iou > best_unmatched_iou:
                best_unmatched_iou = iou
                best_unmatched_gt_idx = gt_idx

        is_tp = best_unmatched_gt_idx >= 0 and best_unmatched_iou >= iou_threshold
        is_fp = not is_tp
        matched_gt_index = best_unmatched_gt_idx if is_tp else None
        matched_iou = float(best_unmatched_iou) if is_tp else None

        if is_tp:
            matched_gt_indices.add(best_unmatched_gt_idx)
            gt_matched_prediction_index[best_unmatched_gt_idx] = pred_idx
            gt_matched_score[best_unmatched_gt_idx] = pred_score
            gt_matched_iou[best_unmatched_gt_idx] = float(best_unmatched_iou)
            tp += 1
        else:
            fp += 1

        prediction_details.append(
            {
                "prediction_index": pred_idx,
                "label": pred_label,
                "score": pred_score,
                "bbox_xyxy": pred_box,
                "bbox_xywh": _xyxy_to_xywh(pred_box),
                "best_iou": float(best_iou_any_gt),
                "best_gt_index": best_iou_any_gt_idx if best_iou_any_gt_idx >= 0 else None,
                "matched_gt_index": matched_gt_index,
                "matched_iou": matched_iou,
                "is_tp": bool(is_tp),
                "is_fp": bool(is_fp),
            }
        )

    for gt_idx, gt_box in enumerate(gt_boxes):
        matched_prediction_index = gt_matched_prediction_index[gt_idx]
        gt_details.append(
            {
                "gt_index": gt_idx,
                "bbox_xyxy": [float(v) for v in gt_box],
                "matched_prediction_index": matched_prediction_index if matched_prediction_index >= 0 else None,
                "matched_score": gt_matched_score[gt_idx],
                "matched_iou": gt_matched_iou[gt_idx],
                "best_prediction_index": gt_best_prediction_index[gt_idx] if gt_best_prediction_index[gt_idx] >= 0 else None,
                "best_prediction_score": gt_best_prediction_score[gt_idx],
                "best_prediction_iou": float(gt_best_prediction_iou[gt_idx]),
                "is_fn": matched_prediction_index < 0,
            }
        )

    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    tp_ious = [item["matched_iou"] for item in prediction_details if item["is_tp"] and item["matched_iou"] is not None]
    tp_scores = [item["score"] for item in prediction_details if item["is_tp"]]
    fp_scores = [item["score"] for item in prediction_details if item["is_fp"]]
    pred_scores = [item["score"] for item in prediction_details]
    gt_best_ious = [item["best_prediction_iou"] for item in gt_details]
    gt_best_scores = [item["best_prediction_score"] if item["best_prediction_score"] is not None else 0.0 for item in gt_details]

    return {
        "num_gt": int(len(gt_boxes)),
        "num_predictions": int(len(sorted_predictions)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_best_iou_to_gt": float(np.mean(gt_best_ious)) if gt_best_ious else 0.0,
        "std_best_iou_to_gt": float(np.std(gt_best_ious)) if gt_best_ious else 0.0,
        "mean_best_confidence_to_gt": float(np.mean(gt_best_scores)) if gt_best_scores else 0.0,
        "std_best_confidence_to_gt": float(np.std(gt_best_scores)) if gt_best_scores else 0.0,
        "mean_tp_iou": float(np.mean(tp_ious)) if tp_ious else 0.0,
        "std_tp_iou": float(np.std(tp_ious)) if tp_ious else 0.0,
        "mean_tp_confidence": float(np.mean(tp_scores)) if tp_scores else 0.0,
        "std_tp_confidence": float(np.std(tp_scores)) if tp_scores else 0.0,
        "best_tp_confidence": float(np.max(tp_scores)) if tp_scores else 0.0,
        "mean_fp_confidence": float(np.mean(fp_scores)) if fp_scores else 0.0,
        "std_fp_confidence": float(np.std(fp_scores)) if fp_scores else 0.0,
        "mean_prediction_confidence": float(np.mean(pred_scores)) if pred_scores else 0.0,
        "std_prediction_confidence": float(np.std(pred_scores)) if pred_scores else 0.0,
        "prediction_details": prediction_details,
        "gt_details": gt_details,
    }


def _get_sorted_category_ids(annotations_data: dict[str, Any]) -> list[int]:
    """Return sorted category ids from annotations."""
    categories = annotations_data.get("categories", [])
    category_ids = sorted(int(cat["id"]) for cat in categories if "id" in cat)
    if category_ids:
        return category_ids

    inferred_ids = sorted({int(ann["category_id"]) for ann in annotations_data.get("annotations", []) if "category_id" in ann})
    return inferred_ids or [1]


def _map_prediction_label_to_category_id(pred_label: int, category_ids: list[int], label_to_category_id: dict[int, int] | None) -> int:
    """Map model labels to COCO category ids."""
    if label_to_category_id is not None and pred_label in label_to_category_id:
        return int(label_to_category_id[pred_label])

    if len(category_ids) == 1:
        return int(category_ids[0])

    if pred_label in category_ids:
        return int(pred_label)
    if (pred_label + 1) in category_ids:
        return int(pred_label + 1)
    if 0 <= pred_label < len(category_ids):
        return int(category_ids[pred_label])
    if 1 <= pred_label <= len(category_ids):
        return int(category_ids[pred_label - 1])

    return int(category_ids[0])


def compute_coco_ap_metrics(annotations_data: dict[str, Any], image_paths: list[Path], predictions_by_image: dict[str, list[dict[str, Any]]], label_to_category_id: dict[int, int] | None = None) -> dict[str, Any]:
    """Compute official COCO AP metrics with pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ImportError("pycocotools is required. Install it with: pip install pycocotools") from exc

    category_ids = _get_sorted_category_ids(annotations_data)

    if not image_paths:
        return {
            "ap50": 0.0,
            "ap50_95": 0.0,
            "per_iou_rows": [],
            "total_gt": 0,
            "total_predictions": 0,
            "gt_category_ids": category_ids,
            "prediction_category_ids": [],
        }

    image_records = [get_image_record(annotations_data, image_path) for image_path in image_paths]
    image_ids = [int(record["id"]) for record in image_records]
    image_id_set = set(image_ids)

    gt_dataset = {
        "images": [dict(record) for record in image_records],
        "annotations": [dict(ann) for ann in annotations_data.get("annotations", []) if int(ann.get("image_id", -1)) in image_id_set],
        "categories": [dict(cat) for cat in annotations_data.get("categories", [])],
    }
    if "info" in annotations_data:
        gt_dataset["info"] = annotations_data["info"]
    if "licenses" in annotations_data:
        gt_dataset["licenses"] = annotations_data["licenses"]

    detections = []
    prediction_category_ids: set[int] = set()

    for image_path, image_record in zip(image_paths, image_records):
        image_name = image_path.name
        image_id = int(image_record["id"])
        for pred in predictions_by_image.get(image_name, []):
            category_id = _map_prediction_label_to_category_id(
                pred_label=int(pred.get("label", 0)),
                category_ids=category_ids,
                label_to_category_id=label_to_category_id,
            )
            prediction_category_ids.add(int(category_id))
            bbox_xywh = pred.get("bbox_xywh") or _xyxy_to_xywh(pred["bbox_xyxy"])
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [float(v) for v in bbox_xywh],
                    "score": float(pred["score"]),
                }
            )

    total_gt = len(gt_dataset["annotations"])
    total_predictions = len(detections)

    if total_gt == 0 or total_predictions == 0:
        per_iou_rows = []
        for iou_threshold in np.linspace(0.50, 0.95, 10):
            per_iou_rows.append(
                {
                    "iou_threshold": float(round(float(iou_threshold), 2)),
                    "ap": 0.0,
                    "total_gt": int(total_gt),
                    "total_predictions": int(total_predictions),
                }
            )
        return {
            "ap50": 0.0,
            "ap50_95": 0.0,
            "per_iou_rows": per_iou_rows,
            "total_gt": int(total_gt),
            "total_predictions": int(total_predictions),
            "gt_category_ids": category_ids,
            "prediction_category_ids": sorted(prediction_category_ids),
        }

    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    precision = coco_eval.eval["precision"]
    per_iou_rows = []
    for iou_index, iou_threshold in enumerate(coco_eval.params.iouThrs):
        iou_precision = precision[iou_index, :, :, 0, 2]
        valid = iou_precision[iou_precision > -1]
        ap_value = float(np.mean(valid)) if valid.size > 0 else 0.0
        per_iou_rows.append(
            {
                "iou_threshold": float(round(float(iou_threshold), 2)),
                "ap": ap_value,
                "total_gt": int(total_gt),
                "total_predictions": int(total_predictions),
            }
        )

    return {
        "ap50": float(coco_eval.stats[1]),
        "ap50_95": float(coco_eval.stats[0]),
        "per_iou_rows": per_iou_rows,
        "total_gt": int(total_gt),
        "total_predictions": int(total_predictions),
        "gt_category_ids": category_ids,
        "prediction_category_ids": sorted(prediction_category_ids),
    }


def _render_detection_image(image_path: Path, gt_boxes: list[list[float]], prediction_details: list[dict[str, Any]], render_scale: int = 10, line_width: int = 5) -> Image.Image:
    """Render enlarged image with blue GT, green TP, red FP boxes."""
    if render_scale < 1:
        raise ValueError(f"render_scale must be >= 1, got {render_scale}")
    if line_width < 1:
        raise ValueError(f"line_width must be >= 1, got {line_width}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scaled_image = image.resize((width * render_scale, height * render_scale), resample=_NEAREST_RESAMPLE)

    draw = ImageDraw.Draw(scaled_image)

    for gt_box in gt_boxes:
        x1, y1, x2, y2 = [float(v) * render_scale for v in gt_box]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=line_width)

    for item in prediction_details:
        x1, y1, x2, y2 = [float(v) * render_scale for v in item["bbox_xyxy"]]
        color = "green" if bool(item["is_tp"]) else "red"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    return scaled_image


def save_detection_image(image: str | Path, gt_boxes: list[list[float]], prediction_details: list[dict[str, Any]], output_path: str | Path, render_scale: int = 10, line_width: int = 5) -> None:
    """Save rendered detection image."""
    image_path = Path(image).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered = _render_detection_image(
        image_path=image_path,
        gt_boxes=gt_boxes,
        prediction_details=prediction_details,
        render_scale=render_scale,
        line_width=line_width,
    )
    rendered.save(output_path)


def show_detections_for_image(image: str | Path, gt_boxes: list[list[float]], prediction_details: list[dict[str, Any]], title: str | None = None, render_scale: int = 10, line_width: int = 5) -> None:
    """Display rendered detection image."""
    image_path = Path(image).resolve()
    rendered = _render_detection_image(
        image_path=image_path,
        gt_boxes=gt_boxes,
        prediction_details=prediction_details,
        render_scale=render_scale,
        line_width=line_width,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(rendered)
    plt.title(title or image_path.name)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def prediction_details_to_rows(image_name: str, image_id: int, evaluation_scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert per-prediction details to flat spreadsheet rows."""
    rows = []
    for item in evaluation_scores.get("prediction_details", []):
        rows.append(
            {
                "image": image_name,
                "image_id": int(image_id),
                "prediction_index": item["prediction_index"],
                "label": item["label"],
                "score": item["score"],
                "bbox_x1": item["bbox_xyxy"][0],
                "bbox_y1": item["bbox_xyxy"][1],
                "bbox_x2": item["bbox_xyxy"][2],
                "bbox_y2": item["bbox_xyxy"][3],
                "best_iou": item["best_iou"],
                "best_gt_index": item["best_gt_index"],
                "matched_gt_index": item["matched_gt_index"],
                "matched_iou": item["matched_iou"],
                "is_tp": item["is_tp"],
                "is_fp": item["is_fp"],
            }
        )
    return rows


def gt_details_to_rows(image_name: str, image_id: int, evaluation_scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert per-GT details to flat spreadsheet rows."""
    rows = []
    for item in evaluation_scores.get("gt_details", []):
        rows.append(
            {
                "image": image_name,
                "image_id": int(image_id),
                "gt_index": item["gt_index"],
                "bbox_x1": item["bbox_xyxy"][0],
                "bbox_y1": item["bbox_xyxy"][1],
                "bbox_x2": item["bbox_xyxy"][2],
                "bbox_y2": item["bbox_xyxy"][3],
                "matched_prediction_index": item["matched_prediction_index"],
                "matched_score": item["matched_score"],
                "matched_iou": item["matched_iou"],
                "best_prediction_index": item["best_prediction_index"],
                "best_prediction_score": item["best_prediction_score"],
                "best_prediction_iou": item["best_prediction_iou"],
                "is_fn": item["is_fn"],
            }
        )
    return rows


def image_summary_row(image_name: str, image_id: int, evaluation_scores: dict[str, Any], score_threshold: float, iou_threshold: float) -> dict[str, Any]:
    """Create one flat per-image summary row."""
    return {
        "image": image_name,
        "image_id": int(image_id),
        "num_gt": evaluation_scores["num_gt"],
        "num_predictions": evaluation_scores["num_predictions"],
        "tp": evaluation_scores["tp"],
        "fp": evaluation_scores["fp"],
        "fn": evaluation_scores["fn"],
        "precision": evaluation_scores["precision"],
        "recall": evaluation_scores["recall"],
        "f1": evaluation_scores["f1"],
        "mean_best_iou_to_gt": evaluation_scores["mean_best_iou_to_gt"],
        "std_best_iou_to_gt": evaluation_scores["std_best_iou_to_gt"],
        "mean_best_confidence_to_gt": evaluation_scores["mean_best_confidence_to_gt"],
        "std_best_confidence_to_gt": evaluation_scores["std_best_confidence_to_gt"],
        "mean_tp_iou": evaluation_scores["mean_tp_iou"],
        "std_tp_iou": evaluation_scores["std_tp_iou"],
        "mean_tp_confidence": evaluation_scores["mean_tp_confidence"],
        "std_tp_confidence": evaluation_scores["std_tp_confidence"],
        "best_tp_confidence": evaluation_scores["best_tp_confidence"],
        "mean_fp_confidence": evaluation_scores["mean_fp_confidence"],
        "std_fp_confidence": evaluation_scores["std_fp_confidence"],
        "mean_prediction_confidence": evaluation_scores["mean_prediction_confidence"],
        "std_prediction_confidence": evaluation_scores["std_prediction_confidence"],
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
    }


def gt_sample_summary_rows(image_name: str, image_id: int, evaluation_scores: dict[str, Any], score_threshold: float, iou_threshold: float) -> list[dict[str, Any]]:
    """Create one row per GT box with best predicted IoU and image-level context."""
    rows = []
    for item in evaluation_scores.get("gt_details", []):
        sample_status = "false_negative" if bool(item["is_fn"]) else "matched_tp"
        rows.append(
            {
                "image": image_name,
                "image_id": int(image_id),
                "gt_index": item["gt_index"],
                "sample_status": sample_status,
                "best_prediction_index": item["best_prediction_index"],
                "best_prediction_score": item["best_prediction_score"],
                "best_prediction_iou": item["best_prediction_iou"],
                "matched_prediction_index": item["matched_prediction_index"],
                "matched_score": item["matched_score"],
                "matched_iou": item["matched_iou"],
                "is_fn": item["is_fn"],
                "num_gt_in_image": evaluation_scores["num_gt"],
                "num_predictions_in_image": evaluation_scores["num_predictions"],
                "num_tp_in_image": evaluation_scores["tp"],
                "num_fp_in_image": evaluation_scores["fp"],
                "num_fn_in_image": evaluation_scores["fn"],
                "image_has_fp": evaluation_scores["fp"] > 0,
                "image_has_fn": evaluation_scores["fn"] > 0,
                "mean_tp_iou_in_image": evaluation_scores["mean_tp_iou"],
                "best_tp_confidence_in_image": evaluation_scores["best_tp_confidence"],
                "score_threshold": float(score_threshold),
                "iou_threshold": float(iou_threshold),
            }
        )
    return rows


def clear_model_cache() -> None:
    """Clear cached detectors and annotations."""
    _MODEL_CACHE.clear()
    _ANNOTATIONS_CACHE.clear()


def load_detector(pth: str | Path, config: str | Path, device: str | None = None, repo_root: str | Path | None = None, use_cache: bool = True) -> LoadedDetector:
    """Load DEIMv2 once and return reusable detector."""
    pth_path = Path(pth).resolve()
    config_path = Path(config).resolve()
    repo_root_path = Path(repo_root).resolve() if repo_root is not None else None
    torch_device = _get_device(device)

    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    return _load_deimv2_predictor(
        config_path=config_path,
        weights_path=pth_path,
        device=torch_device,
        use_cache=use_cache,
        repo_root=repo_root_path,
    )


def run_loaded_detection_raw(detector: LoadedDetector, image: str | Path) -> list[dict[str, Any]]:
    """Run one image and return all raw predictions before thresholding."""
    image_path = Path(image).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    _, image_tensor, orig_size = _prepare_image_tensor(image_path=image_path, detector=detector)

    with torch.no_grad():
        outputs = detector.predictor(image_tensor, orig_size)

    labels, boxes, scores = _extract_predictor_outputs(outputs)

    labels_np = _strip_batch_dim(labels).reshape(-1)
    boxes_np = _strip_batch_dim(boxes).reshape(-1, 4)
    scores_np = _strip_batch_dim(scores).reshape(-1)

    if not (len(labels_np) == len(boxes_np) == len(scores_np)):
        raise RuntimeError(
            "Predictor outputs have inconsistent lengths after conversion. "
            f"labels={len(labels_np)}, boxes={len(boxes_np)}, scores={len(scores_np)}"
        )

    predictions: list[dict[str, Any]] = []
    for label, box, score in zip(labels_np, boxes_np, scores_np):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        predictions.append(
            {
                "label": int(label),
                "score": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            }
        )

    return _sort_predictions(predictions)


def run_loaded_detection(detector: LoadedDetector, image: str | Path, annotations: str | Path | dict[str, Any] | list[Any] | None = None, score_threshold: float = 0.25, iou_threshold: float = 0.5, show_detections: bool = False, render_scale: int = 10, line_width: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one image, threshold predictions, evaluate, and optionally visualize."""
    image_path = Path(image).resolve()
    raw_predictions = run_loaded_detection_raw(detector=detector, image=image_path)
    gt_boxes = extract_gt_boxes(annotations=annotations, image_path=image_path)
    filtered_predictions = filter_predictions(predictions=raw_predictions, score_threshold=score_threshold)
    evaluation_scores = evaluate_predictions(
        predictions=filtered_predictions,
        gt_boxes=gt_boxes,
        iou_threshold=float(iou_threshold),
    )

    evaluation_scores["image"] = str(image_path)
    evaluation_scores["checkpoint"] = str(detector.pth_path)
    evaluation_scores["config"] = str(detector.config_path)
    evaluation_scores["device"] = str(detector.device)
    evaluation_scores["score_threshold"] = float(score_threshold)
    evaluation_scores["iou_threshold"] = float(iou_threshold)

    if show_detections:
        title = (
            f"{image_path.name} | "
            f"TP={evaluation_scores['tp']} FP={evaluation_scores['fp']} FN={evaluation_scores['fn']} "
            f"| F1={evaluation_scores['f1']:.3f}"
        )
        show_detections_for_image(
            image=image_path,
            gt_boxes=gt_boxes,
            prediction_details=evaluation_scores["prediction_details"],
            title=title,
            render_scale=render_scale,
            line_width=line_width,
        )

    return filtered_predictions, evaluation_scores


def run_one_detection(pth: str | Path, config: str | Path, image: str | Path, annotations: str | Path | dict[str, Any] | list[Any] | None = None, device: str | None = None, score_threshold: float = 0.25, iou_threshold: float = 0.5, use_cache: bool = True, repo_root: str | Path | None = None, show_detections: bool = False, render_scale: int = 10, line_width: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One-shot wrapper that still reuses cached model and annotations."""
    detector = load_detector(
        pth=pth,
        config=config,
        device=device,
        repo_root=repo_root,
        use_cache=use_cache,
    )

    annotations_data = load_annotations(
        annotations=annotations,
        use_cache=use_cache,
    )

    return run_loaded_detection(
        detector=detector,
        image=image,
        annotations=annotations_data,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        show_detections=show_detections,
        render_scale=render_scale,
        line_width=line_width,
    )