"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Iterable

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..optim import ModelEMA, Warmup


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs) -> dict[str, float]:
    """train_one_epoch(...) -> dict[str,float]: One training epoch; returns global-avg meters."""
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter | None = kwargs.get("writer", None)

    ema: ModelEMA | None = kwargs.get("ema", None)
    scaler: GradScaler | None = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup | None = kwargs.get("lr_warmup_scheduler", None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = {}
                for key, value in model.state_dict().items():
                    state[key.replace("module.", "")] = value
                out_dir = kwargs.get("output_dir", None)
                if out_dir:
                    dist_utils.save_on_master(state, os.path.join(out_dir, "nan.pth"))
                raise RuntimeError("NaN or Inf detected in pred_boxes.")

            loss_dict = criterion(outputs, targets, **metas)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # Scheduler behavior:
        # - self_lr_scheduler: per-iter scheduler that returns updated optimizer
        # - else: warmup steps per-iter (no args); epoch scheduler stepping is handled outside this loop
        if self_lr_scheduler:
            if lr_scheduler is not None:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ema is not None:
            ema.update(model)

        if writer is not None and dist_utils.is_main_process():
            for k, v in loss_dict_reduced.items():
                try:
                    writer.add_scalar(f"train/{k}", float(v), global_step)
                except Exception:
                    pass
            try:
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            except Exception:
                pass

        # Do NOT step lr_scheduler here for the default (non self_lr_scheduler) case.
        # Epoch-level stepping is handled by the solver when warmup is done.
        pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module | None, postprocessor, data_loader,
             coco_evaluator: CocoEvaluator | None, device, epoch: int = -1, compute_loss: bool = True, **kwargs) -> tuple[dict[str, Any], CocoEvaluator | None]:
    """evaluate(...) -> (stats, coco_evaluator): Compatible with DEIM solver call signature + optional COCO dump."""
    model.eval()
    if criterion is not None:
        criterion.eval()
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else ()

    dump_path = (os.getenv("DEIM_DUMP_PREDICTIONS", "") or "").strip()
    do_dump = dump_path != ""
    local_dump: list[dict[str, Any]] = []

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Repo behavior for metrics: forward WITHOUT targets.
        outputs_pred = model(samples)

        # Optional: compute loss for logging (separate forward), does not affect predictions.
        if compute_loss and criterion is not None:
            metas = dict(epoch=epoch, step=step, global_step=step, epoch_step=len(data_loader))
            with torch.autocast(device_type=device.type, enabled=False):
                outputs_loss = model(samples, targets=targets)
                loss_dict = criterion(outputs_loss, targets, **metas)

            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())
            metric_logger.update(loss=loss_value, **loss_dict_reduced)

        if coco_evaluator is not None:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs_pred, orig_target_sizes)
            res = {t["image_id"].item(): o for t, o in zip(targets, results)}
            coco_evaluator.update(res)

            if do_dump:
                label_offset = int((os.getenv("DEIM_LABEL_OFFSET", "0") or "0").strip())
                for t, o in zip(targets, results):
                    image_id = int(t["image_id"].item())
                    boxes = o["boxes"].detach().cpu().tolist()
                    scores = o["scores"].detach().cpu().tolist()
                    labels = o["labels"].detach().cpu().tolist()
                    for b, s, l in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = [float(x) for x in b]
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        local_dump.append(
                            {
                                "image_id": image_id,
                                "category_id": int(l) + label_offset,
                                "bbox": [x1, y1, w, h],
                                "score": float(s),
                            }
                        )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats: dict[str, Any] = {}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    for k, meter in metric_logger.meters.items():
        if k not in stats:
            stats[k] = float(meter.global_avg)

    if do_dump:
        import json
        from pathlib import Path

        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = int(dist.get_rank())
        except Exception:
            rank = 0

        try:
            Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
            rank_path = Path(dump_path + f".rank{rank}.json")
            with rank_path.open("w", encoding="utf-8") as f:
                json.dump(local_dump, f)
        except Exception as e:
            print(f"[det_engine.evaluate] WARNING: failed to write rank predictions dump: {e}", file=sys.stderr, flush=True)

        if rank == 0:
            try:
                with open(dump_path, "w", encoding="utf-8") as f:
                    json.dump(local_dump, f)
            except Exception as e:
                print(f"[det_engine.evaluate] WARNING: failed to write predictions dump: {e}", file=sys.stderr, flush=True)

    return stats, coco_evaluator