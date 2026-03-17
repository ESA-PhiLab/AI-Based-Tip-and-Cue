"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

import os
import json
from pathlib import Path
from typing import Any


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)
    writer_log_every = kwargs.get("writer_log_every", 10)  # log interval in steps

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # ---- extra tensorboard logging (new-style tags) ----
        if writer and dist_utils.is_main_process() and (global_step % writer_log_every == 0):
            try:
                writer.add_scalar("train/loss_total", float(loss_value), global_step)
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f"train/{k}", float(v), global_step)
            except Exception as e:
                print(f"[TensorBoard] WARNING: disabling writer after train-step failure: {e}", flush=True)
                writer = None
        # ----------------------------------------------------

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            try:
                writer.add_scalar("Loss/total", float(loss_value), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Lr/pg_{j}", float(pg["lr"]), global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f"Loss/{k}", float(v), global_step)
            except Exception as e:
                print(f"[TensorBoard] WARNING: disabling writer after failure: {e}", flush=True)
                writer = None

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if writer and dist_utils.is_main_process() and "loss" in metric_logger.meters:
        try:
            writer.add_scalars("Loss/total_epoch", {"train": float(metric_logger.meters["loss"].global_avg)}, epoch)
        except Exception as e:
            print(f"[TensorBoard] WARNING: disabling writer after epoch-train failure: {e}", flush=True)
            writer = None

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device,
             epoch: int = -1, **kwargs):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    writer: SummaryWriter = kwargs.get("writer", None)

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # ---- optional COCO predictions dump (path provided by eval_one_deimv2.py) ----
    dump_path = (os.getenv("DEIM_DUMP_PREDICTIONS", "") or "").strip()
    do_dump = dump_path != ""
    local_dump: list[dict[str, Any]] = []
    label_offset = int((os.getenv("DEIM_LABEL_OFFSET", "0") or "0").strip())
    # ---------------------------------------------------------------------------

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # ---- compute eval loss (extra forward with targets) ----
        metas = dict(epoch=epoch, step=0, global_step=0, epoch_step=len(data_loader))
        with torch.autocast(device_type=str(device), enabled=False):
            outputs_loss = model(samples, targets=targets)
            loss_dict = criterion(outputs_loss, targets, **metas)

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        # -------------------------------------------------------



        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # ---- collect predictions for COCO JSON dump ----
        if do_dump:
            for t, o in zip(targets, results):
                image_id = int(t["image_id"].item())
                boxes = o["boxes"].detach().cpu().tolist()
                scores = o["scores"].detach().cpu().tolist()
                labels = o["labels"].detach().cpu().tolist()

                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = [float(x) for x in b]
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    local_dump.append({
                        "image_id": image_id,
                        "category_id": int(l) + label_offset,
                        "bbox": [x1, y1, w, h],
                        "score": float(s),
                    })
        # -----------------------------------------------

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # ---- tensorboard validation logging ----
    if writer and dist_utils.is_main_process():
        for k, meter in metric_logger.meters.items():
            try:
                writer.add_scalar(f"val/{k}", float(meter.global_avg), epoch)
            except Exception:
                pass
    # ----------------------------------------

    print("Averaged stats:", metric_logger)

    if "loss" in metric_logger.meters:
        print(f"Eval loss (global_avg): {metric_logger.meters['loss'].global_avg}")

    if writer and dist_utils.is_main_process() and "loss" in metric_logger.meters:
        try:
            writer.add_scalars("Loss/total_epoch", {"val": float(metric_logger.meters["loss"].global_avg)}, epoch)
        except Exception as e:
            print(f"[TensorBoard] WARNING: disabling writer after epoch-val failure: {e}", flush=True)
            writer = None

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    # ---- optional tensorboard logging for COCO metrics (epoch x-axis) ----
    if writer and dist_utils.is_main_process():
        if coco_evaluator is not None and "bbox" in iou_types:
            try:
                coco = coco_evaluator.coco_eval["bbox"].stats.tolist()
                writer.add_scalar("val/coco_AP", float(coco[0]), epoch)
                writer.add_scalar("val/coco_AP50", float(coco[1]), epoch)
                writer.add_scalar("val/coco_AP75", float(coco[2]), epoch)
            except Exception:
                pass
    # --------------------------------------------------------------------


    # ---- write predictions to DEIM_DUMP_PREDICTIONS (rank-safe) ----
    if do_dump:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = int(dist.get_rank())
        except Exception:
            rank = 0

        try:
            Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Always write per-rank file
        try:
            rank_path = Path(dump_path + f".rank{rank}.json")
            with rank_path.open("w", encoding="utf-8") as f:
                json.dump(local_dump, f)
        except Exception as e:
            print(f"[det_engine.evaluate] WARNING: failed writing rank predictions: {e}", file=sys.stderr, flush=True)

        # Rank 0 also writes the main file (this is what eval_one_deimv2.py expects)
        if rank == 0:
            try:
                with Path(dump_path).open("w", encoding="utf-8") as f:
                    json.dump(local_dump, f)
            except Exception as e:
                print(f"[det_engine.evaluate] WARNING: failed writing predictions: {e}", file=sys.stderr, flush=True)
    # ---------------------------------------------------------------

    return stats, coco_evaluator