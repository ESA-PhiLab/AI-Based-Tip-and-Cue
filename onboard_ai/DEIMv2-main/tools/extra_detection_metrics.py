#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_slug(s: str) -> str:
    """safe_slug(s) -> str: Filesystem-safe slug."""
    import re
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    t = re.sub(r"_+", "_", t).strip("_")
    return t or "run"


def infer_run_and_tag(out_dir: Path) -> tuple[str, str]:
    """infer_run_and_tag(out_dir) -> (run_name, tag): Infer run_name + fold tag from out_dir path."""
    parts = list(out_dir.resolve().parts)

    run_name = "run"
    tag = "unknown"

    # tag
    for p in reversed(parts):
        if p == "final_location_holdout":
            tag = "final"
            break
        if p.startswith("fold") and p[4:].isdigit():
            tag = p
            break

    # run_name: results/<RUN_NAME>/...
    for i in range(len(parts) - 1):
        if parts[i] == "results" and i + 1 < len(parts):
            run_name = parts[i + 1]
            break

    return safe_slug(run_name), safe_slug(tag)


def cleanup_base_plots(plots_dir: Path, base_names: list[str]) -> None:
    """cleanup_base_plots(plots_dir, base_names) -> None: Remove legacy base-name plots."""
    for bn in base_names:
        p = plots_dir / f"{bn}.png"
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def read_json(path: str) -> Any:
    """read_json(path) -> Any: Read JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """write_json(path, obj) -> None: Write JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def iou_xywh(a: list[float], b: list[float]) -> float:
    """iou_xywh(a, b) -> float: IoU for [x,y,w,h]."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = aw * ah + bw * bh - inter
    return 0.0 if ua <= 0 else inter / ua


def match_detections_with_thr(
    gt_by_img: dict[int, list[dict[str, Any]]],
    dt_by_img: dict[int, list[dict[str, Any]]],
    iou_thr: float,
    score_thr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, int]]]:
    """match_detections_with_thr(gt_by_img, dt_by_img, iou_thr, score_thr) -> (matches, misses, img_counts)."""
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    img_counts: dict[int, dict[str, int]] = {}

    for img_id, gts in gt_by_img.items():
        dts = dt_by_img.get(img_id, [])
        gts_used = set()

        dts_sorted = sorted(dts, key=lambda d: float(d.get("score", 0.0)), reverse=True)

        for dt in dts_sorted:
            best = None
            best_iou = -1.0
            for gi, gt in enumerate(gts):
                if gi in gts_used:
                    continue
                if int(gt.get("category_id", -1)) != int(dt.get("category_id", -2)):
                    continue
                iou = iou_xywh(gt["bbox"], dt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best = gi
            if best is not None and best_iou >= iou_thr:
                gts_used.add(best)
                matches.append(
                    {
                        "image_id": img_id,
                        "category_id": int(dt["category_id"]),
                        "score": float(dt.get("score", 0.0)),
                        "iou": float(best_iou),
                    }
                )

        for gi, gt in enumerate(gts):
            if gi not in gts_used:
                misses.append({"image_id": img_id, "category_id": int(gt.get("category_id", -1))})

        has_gt = 1 if len(gts) > 0 else 0
        has_dt = 1 if any((float(d.get("score", 0.0)) >= score_thr) for d in dts_sorted) else 0
        img_counts[img_id] = {"has_gt": has_gt, "has_dt": has_dt}

    for img_id, dts in dt_by_img.items():
        if img_id in gt_by_img:
            continue
        dts_sorted = sorted(dts, key=lambda d: float(d.get("score", 0.0)), reverse=True)
        has_dt = 1 if any((float(d.get("score", 0.0)) >= score_thr) for d in dts_sorted) else 0
        img_counts[img_id] = {"has_gt": 0, "has_dt": has_dt}

    return matches, misses, img_counts


def best_image_level_threshold_f1(gt_by_img: dict[int, list[dict[str, Any]]], dt_by_img: dict[int, list[dict[str, Any]]]) -> tuple[float, float, float, float]:
    """best_image_level_threshold_f1(gt_by_img, dt_by_img) -> (best_thr, best_f1, best_precision, best_recall)."""
    img_ids = sorted(set(gt_by_img.keys()) | set(dt_by_img.keys()))
    if not img_ids:
        return 0.0, 0.0, 0.0, 0.0

    y_true: list[int] = []
    y_score: list[float] = []
    for img_id in img_ids:
        y_true.append(1 if len(gt_by_img.get(img_id, [])) > 0 else 0)
        scores = [float(d.get("score", 0.0)) for d in dt_by_img.get(img_id, [])]
        y_score.append(max(scores) if scores else 0.0)

    thresholds = sorted(set(y_score), reverse=True)
    if not thresholds:
        return 0.0, 0.0, 0.0, 0.0
    if thresholds[-1] != 0.0:
        thresholds.append(0.0)

    best_thr = thresholds[-1]
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0

    for thr in thresholds:
        tp = fp = fn = tn = 0
        for yt, ys in zip(y_true, y_score):
            pred = 1 if ys >= thr else 0
            if pred == 1 and yt == 1:
                tp += 1
            elif pred == 1 and yt == 0:
                fp += 1
            elif pred == 0 and yt == 1:
                fn += 1
            else:
                tn += 1

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_p = p
            best_r = r

    return best_thr, best_f1, best_p, best_r


def match_detections_legacy(
    gt_by_img: dict[int, list[dict[str, Any]]],
    dt_by_img: dict[int, list[dict[str, Any]]],
    iou_thr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, int]]]:
    """match_detections_legacy(gt_by_img, dt_by_img, iou_thr) -> (matches, misses, img_counts)."""
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    img_counts: dict[int, dict[str, int]] = {}

    for img_id, gts in gt_by_img.items():
        dts = dt_by_img.get(img_id, [])
        gts_used = set()

        dts_sorted = sorted(dts, key=lambda d: float(d.get("score", 0.0)), reverse=True)

        for dt in dts_sorted:
            best = None
            best_iou = -1.0
            for gi, gt in enumerate(gts):
                if gi in gts_used:
                    continue
                if int(gt.get("category_id", -1)) != int(dt.get("category_id", -2)):
                    continue
                iou = iou_xywh(gt["bbox"], dt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best = gi
            if best is not None and best_iou >= iou_thr:
                gts_used.add(best)
                matches.append(
                    {
                        "image_id": img_id,
                        "category_id": int(dt["category_id"]),
                        "score": float(dt.get("score", 0.0)),
                        "iou": float(best_iou),
                    }
                )

        for gi, gt in enumerate(gts):
            if gi not in gts_used:
                misses.append({"image_id": img_id, "category_id": int(gt.get("category_id", -1))})

        has_gt = 1 if len(gts) > 0 else 0
        score_thr = 0.05
        has_dt = 1 if any((float(d.get("score", 0.0)) >= score_thr) for d in dts_sorted) else 0
        img_counts[img_id] = {"has_gt": has_gt, "has_dt": has_dt}

    for img_id, dts in dt_by_img.items():
        if img_id in gt_by_img:
            continue
        has_dt = 1 if len(dts) > 0 else 0
        img_counts[img_id] = {"has_gt": 0, "has_dt": has_dt}

    return matches, misses, img_counts


def pr_curve_from_matches(
    gt_by_img: dict[int, list[dict[str, Any]]],
    dt_all: list[dict[str, Any]],
    iou_thr: float,
) -> tuple[list[float], list[float], float]:
    """pr_curve_from_matches(gt_by_img, dt_all, iou_thr) -> (precisions, recalls, ap)."""
    gt_total = sum(len(v) for v in gt_by_img.values())
    if gt_total == 0:
        return [], [], float("nan")

    gt_used: dict[int, set[int]] = defaultdict(set)
    dt_sorted = sorted(dt_all, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    tps: list[int] = []
    fps: list[int] = []

    tp = 0
    fp = 0
    for dt in dt_sorted:
        img_id = int(dt["image_id"])
        cat = int(dt["category_id"])
        gts = gt_by_img.get(img_id, [])
        best_iou = -1.0
        best_gi = None
        for gi, gt in enumerate(gts):
            if gi in gt_used[img_id]:
                continue
            if int(gt.get("category_id", -1)) != cat:
                continue
            iou = iou_xywh(gt["bbox"], dt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi is not None and best_iou >= iou_thr:
            gt_used[img_id].add(best_gi)
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)

    precisions: list[float] = []
    recalls: list[float] = []
    for tp_i, fp_i in zip(tps, fps):
        precisions.append(tp_i / max(1, tp_i + fp_i))
        recalls.append(tp_i / gt_total)

    ap = 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return precisions, recalls, ap


def plot_pr(out_path: Path, precisions: list[float], recalls: list[float], title: str) -> None:
    """plot_pr(out_path, precisions, recalls, title) -> None: Save PR curve."""
    if not precisions or not recalls:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(out_path: Path, tp: int, fp: int, fn: int, tn: int, title: str) -> None:
    """plot_confusion_matrix(out_path, tp, fp, fn, tn, title) -> None: Save 2x2 confusion matrix plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = [[tp, fp], [fn, tn]]
    plt.figure()
    plt.imshow(mat)
    plt.title(title)
    plt.xticks([0, 1], ["Pred+: has box", "Pred-: no box"], rotation=20, ha="right")
    plt.yticks([0, 1], ["GT+: has box", "GT-: no box"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(mat[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def roc_image_level(gt_by_img: dict[int, list[dict[str, Any]]], dt_by_img: dict[int, list[dict[str, Any]]]) -> tuple[list[float], list[float], float]:
    """roc_image_level(gt_by_img, dt_by_img) -> (fpr, tpr, auc): Image-level ROC using max score per image."""
    img_ids = sorted(set(gt_by_img.keys()) | set(dt_by_img.keys()))
    if not img_ids:
        return [], [], float("nan")

    y_true: list[int] = []
    y_score: list[float] = []
    for img_id in img_ids:
        y_true.append(1 if len(gt_by_img.get(img_id, [])) > 0 else 0)
        scores = [float(d.get("score", 0.0)) for d in dt_by_img.get(img_id, [])]
        y_score.append(max(scores) if scores else 0.0)

    thresholds = sorted(set(y_score), reverse=True)
    if thresholds and thresholds[-1] != 0.0:
        thresholds.append(0.0)

    P = sum(y_true)
    N = len(y_true) - P
    if P == 0 or N == 0:
        return [], [], float("nan")

    tpr: list[float] = []
    fpr: list[float] = []
    for thr in thresholds:
        tp = 0
        fp = 0
        for yt, ys in zip(y_true, y_score):
            pred_pos = 1 if ys >= thr else 0
            if pred_pos == 1 and yt == 1:
                tp += 1
            if pred_pos == 1 and yt == 0:
                fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)

    pairs = sorted(zip(fpr, tpr), key=lambda x: x[0])
    auc = 0.0
    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]
        x1, y1 = pairs[i]
        auc += (x1 - x0) * (y0 + y1) * 0.5
    return [p[0] for p in pairs], [p[1] for p in pairs], auc


def plot_roc(out_path: Path, fpr: list[float], tpr: list[float], title: str) -> None:
    """plot_roc(out_path, fpr, tpr, title) -> None: Save ROC curve."""
    if not fpr or not tpr:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--score_thr", type=float, default=0.05)
    ap.add_argument("--optimize_score_thr", type=int, default=1)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--fold_tag", default="")
    args = ap.parse_args()

    gt = read_json(args.gt)
    pred_path = Path(args.pred)
    if not pred_path.exists():
        # Fallback: use rank0 file if merged file wasn't written
        rank0 = pred_path.with_name(pred_path.name + ".rank0.json")
        if rank0.exists():
            pred_path = rank0
        else:
            # fallback: any rank file
            cands = sorted(pred_path.parent.glob(pred_path.name + ".rank*.json"))
            if cands:
                pred_path = cands[0]

    pred = read_json(str(pred_path))

    gt_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt.get("annotations", []):
        gt_by_img[int(ann["image_id"])].append(ann)

    dt_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if isinstance(pred, list):
        for d in pred:
            dt_by_img[int(d["image_id"])].append(d)

    dt_all = [d for ds in dt_by_img.values() for d in ds]

    if int(args.optimize_score_thr) == 1:
        best_thr, best_f1, best_p, best_r = best_image_level_threshold_f1(gt_by_img, dt_by_img)
        score_thr_used = float(best_thr)
        score_thr_mode = "optimized_f1"
    else:
        best_thr, best_f1, best_p, best_r = None, None, None, None
        score_thr_used = float(args.score_thr)
        score_thr_mode = "fixed"

    _, _, img_counts = match_detections_with_thr(gt_by_img, dt_by_img, float(args.iou_thr), float(score_thr_used))

    TP = FP = FN = TN = 0
    for _, c in img_counts.items():
        if c["has_gt"] == 1 and c["has_dt"] == 1:
            TP += 1
        elif c["has_gt"] == 0 and c["has_dt"] == 1:
            FP += 1
        elif c["has_gt"] == 1 and c["has_dt"] == 0:
            FN += 1
        else:
            TN += 1

    precision_img = TP / max(1, TP + FP)
    recall_img = TP / max(1, TP + FN)
    f1_img = 0.0 if (precision_img + recall_img) == 0 else (2 * precision_img * recall_img / (precision_img + recall_img))

    precisions, recalls, ap_val = pr_curve_from_matches(gt_by_img, dt_all, float(args.iou_thr))
    fpr, tpr, auc = roc_image_level(gt_by_img, dt_by_img)

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Decide <model_name> and <foldn>
    if str(args.run_name).strip() and str(args.fold_tag).strip():
        model_name = safe_slug(str(args.run_name).strip())
        fold_tag = safe_slug(str(args.fold_tag).strip())
    else:
        model_name, fold_tag = infer_run_and_tag(out_dir)

    suffix = f"{model_name}_{fold_tag}"

    thr_txt = out_dir / f"best_image_level_f1_threshold_{suffix}.txt"
    thr_txt.write_text(f"{float(score_thr_used):.8f}\n", encoding="utf-8")
    print("WROTE:", str(thr_txt), flush=True)

    # Cleanup: remove legacy base-name plots and any previously suffixed plots
    stems = ["pr_curve", "confusion_matrix", "roc_curve", "score_hist", "pred_box_area_hist", "per_class_ap_topk"]
    for stem in stems:
        for p in plots_dir.glob(f"{stem}*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    plot_pr(
        plots_dir / f"pr_curve_{suffix}.png",
        precisions,
        recalls,
        f"PR curve (IoU>={args.iou_thr})",
    )
    plot_confusion_matrix(
        plots_dir / f"confusion_matrix_{suffix}.png",
        TP,
        FP,
        FN,
        TN,
        "Image-level confusion (has any box)",
    )
    plot_roc(
        plots_dir / f"roc_curve_{suffix}.png",
        fpr,
        tpr,
        f"ROC (image-level), AUC={auc:.4f}" if not math.isnan(auc) else "ROC (image-level)",
    )

    metrics_extra = {
        "iou_thr": float(args.iou_thr),
        "score_thr_used": float(score_thr_used),
        "score_thr_mode": str(score_thr_mode),
        "best_thr_f1": float(best_thr) if best_thr is not None else None,
        "best_f1": float(best_f1) if best_f1 is not None else None,
        "best_precision": float(best_p) if best_p is not None else None,
        "best_recall": float(best_r) if best_r is not None else None,
        "image_level_confusion": {"TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN)},
        "image_level_precision": float(precision_img),
        "image_level_recall": float(recall_img),
        "image_level_f1": float(f1_img),
        "pr_ap_global": float(ap_val) if not math.isnan(ap_val) else None,
        "roc_auc_image_level": float(auc) if not math.isnan(auc) else None,
    }
    write_json(str(out_dir / "metrics_extra.json"), metrics_extra)

    print("WROTE:", str(out_dir / "metrics_extra.json"), flush=True)


def cleanup_extra_metric_plots(plots_dir: Path, suffix: str) -> None:
    """cleanup_extra_metric_plots(plots_dir, suffix) -> None: Delete old ROC/PR/CM plots before re-plotting."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    stems = [
        "pr_curve",
        "confusion_matrix",
        "roc_curve",
        "score_hist",
        "pred_box_area_hist",
        "per_class_ap_topk",
    ]

    # Remove both legacy (no suffix) and any suffixed variants.
    for stem in stems:
        for p in plots_dir.glob(f"{stem}*.png"):
            try:
                p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()