"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = float("-inf")  # global best AP50
        best_stg1 = float("-inf")  # best AP50 in stage 1
        best_stg2_local = float("-inf")  # best AP50 in stage 2, even if not global best
        best_stat = {'epoch': -1}

        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device, epoch=self.last_epoch, writer=self.writer
            )
            if "coco_eval_bbox" in test_stats and len(test_stats["coco_eval_bbox"]) > 1:
                metric_name = "coco_eval_bbox_ap50"
                metric_val = float(test_stats["coco_eval_bbox"][1])  # AP50

                best_stat["epoch"] = self.last_epoch
                best_stat[metric_name] = metric_val
                top1 = metric_val

                stop_epoch = self.train_dataloader.collate_fn.stop_epoch
                if self.last_epoch < stop_epoch:
                    best_stg1 = metric_val
                else:
                    best_stg2_local = metric_val

                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

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
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device, epoch=epoch, writer=self.writer
            )

            if "coco_eval_bbox" not in test_stats or len(test_stats["coco_eval_bbox"]) <= 1:
                raise RuntimeError("Missing coco_eval_bbox AP50 in test_stats.")

            if self.writer and dist_utils.is_main_process():
                for i, v in enumerate(test_stats["coco_eval_bbox"]):
                    self.writer.add_scalar(f"Test/coco_eval_bbox_{i}", float(v), epoch)

            metric_name = "coco_eval_bbox_ap50"
            metric_val = float(test_stats["coco_eval_bbox"][1])  # AP50
            stop_epoch = self.train_dataloader.collate_fn.stop_epoch
            in_stage2 = epoch >= stop_epoch

            # Track best seen value for reporting
            if metric_name in best_stat:
                if metric_val > best_stat[metric_name]:
                    best_stat[metric_name] = metric_val
                    best_stat["epoch"] = epoch
            else:
                best_stat[metric_name] = metric_val
                best_stat["epoch"] = epoch

            best_stat_print["epoch"] = best_stat["epoch"]
            best_stat_print[metric_name] = best_stat[metric_name]

            # Stage 1 best
            if self.output_dir and not in_stage2 and metric_val > best_stg1:
                best_stg1 = metric_val
                dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

            # Stage 2 local best
            if self.output_dir and in_stage2 and metric_val > best_stg2_local:
                best_stg2_local = metric_val
                dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2_local.pth")

            # Global best AP50
            if metric_val > top1:
                top1 = metric_val
                best_stat["epoch"] = epoch
                best_stat[metric_name] = metric_val
                best_stat_print["epoch"] = epoch
                best_stat_print[metric_name] = metric_val

                if self.output_dir:
                    if in_stage2:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")

            print(f"best_stat: {best_stat_print}")


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device, epoch=self.last_epoch, writer=self.writer)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return