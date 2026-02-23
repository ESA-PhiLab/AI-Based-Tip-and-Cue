#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def read_json(path: str) -> Any:
    """read_json(path) -> Any: Read JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    """write_json(path, obj) -> None: Write JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def coco_metrics_dict(coco_eval) -> dict[str, float]:
    """coco_metrics_dict(coco_eval) -> dict[str,float]: Convert COCOeval.stats to named AP/AR metrics safely."""
    stats_obj = getattr(coco_eval, "stats", None)
    if stats_obj is None:
        stats = []
    else:
        # stats_obj can be numpy.ndarray; never use it in boolean context.
        try:
            stats = list(stats_obj.tolist())
        except Exception:
            stats = list(stats_obj)

    if len(stats) < 12:
        return {}

    s = [float(x) for x in stats[:12]]

    return {
        "AP_precision_iou_0.50:0.95_area_all_maxdets_100": s[0],
        "AP_precision_iou_0.50_area_all_maxdets_100": s[1],
        "AP_precision_iou_0.75_area_all_maxdets_100": s[2],
        "AP_precision_iou_0.50:0.95_area_small_maxdets_100": s[3],
        "AP_precision_iou_0.50:0.95_area_medium_maxdets_100": s[4],
        "AP_precision_iou_0.50:0.95_area_large_maxdets_100": s[5],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_1": s[6],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_10": s[7],
        "AR_recall_iou_0.50:0.95_area_all_maxdets_100": s[8],
        "AR_recall_iou_0.50:0.95_area_small_maxdets_100": s[9],
        "AR_recall_iou_0.50:0.95_area_medium_maxdets_100": s[10],
        "AR_recall_iou_0.50:0.95_area_large_maxdets_100": s[11],
    }


def per_class_ap(coco_eval: COCOeval, coco_gt: COCO) -> dict[str, float]:
    """per_class_ap(coco_eval, coco_gt) -> dict[str,float]: Per-category AP (IoU=0.50:0.95, area=all, maxDets=100)."""
    p = coco_eval.eval.get("precision", None)
    if p is None:
        return {}
    maxdets = list(coco_eval.params.maxDets)
    midx = maxdets.index(100) if 100 in maxdets else len(maxdets) - 1
    aidx = 0
    out: dict[str, float] = {}
    cat_ids = list(coco_eval.params.catIds)
    for ki, cid in enumerate(cat_ids):
        pk = p[:, :, ki, aidx, midx]
        vals = pk[pk > -1]
        ap = float(vals.mean()) if vals.size > 0 else float("nan")
        name = coco_gt.loadCats([cid])[0].get("name", str(cid))
        out[f"AP_class_{name}"] = ap
    return out


def plot_topk_per_class_ap(out_path: Path, per_class: dict[str, float], top_k: int) -> None:
    """plot_topk_per_class_ap(out_path, per_class, top_k) -> None: Plot top-K per-class AP bar chart."""
    items: list[tuple[str, float]] = []
    for k, v in per_class.items():
        if not k.startswith("AP_class_"):
            continue
        if isinstance(v, float) and v == v:
            items.append((k.replace("AP_class_", ""), float(v)))
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: max(0, int(top_k))]
    if not items:
        return

    labels = [x[0] for x in items]
    vals = [x[1] for x in items]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(labels, vals)
    plt.title(f"Top-{len(items)} per-class AP (IoU=0.50:0.95)")
    plt.ylabel("AP")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_score_hist(out_path: Path, preds: list[dict[str, Any]]) -> None:
    """plot_score_hist(out_path, preds) -> None: Plot histogram of detection scores."""
    scores: list[float] = []
    for d in preds:
        try:
            scores.append(float(d.get("score", 0.0)))
        except Exception:
            continue
    if not scores:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Detection score distribution")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_box_area_hist(out_path: Path, preds: list[dict[str, Any]]) -> None:
    """plot_box_area_hist(out_path, preds) -> None: Plot histogram of predicted bbox areas (px^2)."""
    areas: list[float] = []
    for d in preds:
        b = d.get("bbox")
        if not isinstance(b, list) or len(b) != 4:
            continue
        try:
            w = float(b[2])
            h = float(b[3])
            if w > 0 and h > 0:
                areas.append(w * h)
        except Exception:
            continue
    if not areas:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(areas, bins=50)
    plt.title("Predicted bbox area distribution")
    plt.xlabel("area (px^2)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--gt_ann", default=None)
    ap.add_argument("--pred_json", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--top_k_classes", type=int, default=25)
    args = ap.parse_args()

    gt_ann = args.gt_ann if args.gt_ann is not None else args.gt
    pred_json = args.pred_json if args.pred_json is not None else args.pred

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json_detailed = str(out_dir / "metrics_detailed.json")
        out_json_main = str(out_dir / "metrics.json")
        plots_dir = out_dir / "plots"
    else:
        out_dir = None
        out_json_detailed = args.out_json
        out_json_main = args.out_json
        plots_dir = None

    if not gt_ann or not pred_json or not out_json_detailed:
        raise SystemExit("Need (--gt_ann, --pred_json, --out_json) OR (--gt, --pred, --out_dir).")

    coco_gt = COCO(gt_ann)
    preds = read_json(pred_json)
    if not isinstance(preds, list):
        raise RuntimeError("pred_json must be a list of COCO detections")

    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    m = coco_metrics_dict(coco_eval)
    per_cls = per_class_ap(coco_eval, coco_gt)
    m.update(per_cls)

    write_json(out_json_detailed, m)
    if out_json_main != out_json_detailed:
        write_json(out_json_main, m)

    print("WROTE:", out_json_main)

    if plots_dir is not None:
        plot_topk_per_class_ap(plots_dir / "per_class_ap_topk.png", per_cls, args.top_k_classes)
        plot_score_hist(plots_dir / "score_hist.png", preds)
        plot_box_area_hist(plots_dir / "pred_box_area_hist.png", preds)


if __name__ == "__main__":
    main()