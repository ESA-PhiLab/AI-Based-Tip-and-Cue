"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

from __future__ import annotations

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print(f"     ## Using Self-defined Scheduler-{args.lrsheduler} ## ")
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"number of trainable parameters: {n_trainable}")
        print(f"number of non-trainable parameters: {n_frozen}")

        top1 = 0.0
        best_stat: dict[str, float | int] = {"epoch": -1}

        # Evaluate once before resume training (matches repo behavior)
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, _ = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch=self.last_epoch,
                compute_loss=True,
            )
            v = test_stats.get("coco_eval_bbox", None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                best_stat["epoch"] = int(self.last_epoch)
                best_stat["coco_eval_bbox"] = float(v[0])
                top1 = float(v[0])
                print(f"best_stat: {best_stat}")

        start_time = time.time()
        start_epoch = self.last_epoch + 1

        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            stop_epoch = getattr(self.train_dataloader.collate_fn, "stop_epoch", None)
            if stop_epoch is not None and epoch == stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
            )

            if not self.self_lr_scheduler:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            # Keep repo behavior: only write last/checkpointXXXX during stage-1
            if self.output_dir and (stop_epoch is None or epoch < stop_epoch):
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch=epoch,
                compute_loss=True,
            )

            # Tensorboard
            if self.writer and dist_utils.is_main_process():
                for k, v in test_stats.items():
                    if isinstance(v, (list, tuple)):
                        for i, vi in enumerate(v):
                            self.writer.add_scalar(f"Test/{k}_{i}", float(vi), epoch)
                    else:
                        self.writer.add_scalar(f"Test/{k}", float(v), epoch)

            # Track best using first COCO bbox stat (AP@[.50:.95]) like upstream
            v = test_stats.get("coco_eval_bbox", None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                ap = float(v[0])
                if "coco_eval_bbox" in best_stat:
                    best_stat["epoch"] = epoch if ap > float(best_stat["coco_eval_bbox"]) else int(best_stat["epoch"])
                    best_stat["coco_eval_bbox"] = max(float(best_stat["coco_eval_bbox"]), ap)
                else:
                    best_stat["epoch"] = epoch
                    best_stat["coco_eval_bbox"] = ap

                if float(best_stat["coco_eval_bbox"]) > float(top1):
                    top1 = float(best_stat["coco_eval_bbox"])
                    if self.output_dir:
                        if stop_epoch is not None and epoch >= stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

                print(f"best_stat: {best_stat}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": int(epoch),
                "n_parameters": int(n_trainable),
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")

    def val(self, ):
        self.eval()
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=self.last_epoch,
            compute_loss=True,
        )

        if self.output_dir and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        return