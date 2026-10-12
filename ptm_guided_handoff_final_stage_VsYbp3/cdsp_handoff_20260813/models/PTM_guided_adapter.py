"""PTM-guided placement plus soft old-anchor response stability.

This learner is isolated from official CDS and CDSP.  It retains the existing
nonlinear bottleneck adapter, disables APR/SCC/synthetic CA/replay, and stores
only class/layer summary statistics.
"""

import copy
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.cil_metrics import (
    class_accuracy_rows,
    forgetting_from_matrix,
    task_accuracy_row,
)
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss
from utils.prototype_classifier import pairwise_geometry_drift
from utils.ptm_guidance import (
    AdapterInputCollector,
    AnchorResponseStability,
    ClassPrototypeAccumulator,
    LayerAnchorAccumulator,
    PTMGuidanceMemory,
    active_adapter_branch,
    adapter_units,
    anchor_drift_rows,
    placement_weight,
    ptm_placement_loss,
)
from utils.ptm_metrics import PTMGuidedResultWriter
from utils.toolkit import accuracy


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class Learner(BaseLearner):
    """Exemplar-free PTM-guided adapter learner."""

    def __init__(self, args):
        super().__init__(args)
        if "adapter" not in args["convnet_type"]:
            raise NotImplementedError("PTM-guided learning requires an adapter backbone")
        self._validate_config(args)
        self.args = args
        self._network = SimpleVitNet(args, True)
        self.batch_size = int(args["batch_size"])
        self.init_lr = float(args["init_lr"])
        self.weight_decay = float(args.get("weight_decay", 0.0005))
        self.min_lr = float(args.get("min_lr", 0.0))
        self.num_workers = int(args.get("num_workers", 8))
        self.place_enabled = bool(args.get("ptm_place_enabled", True))
        self.move_enabled = bool(args.get("old_move_enabled", True))
        self.place_initial = float(args.get("ptm_place_weight", 1.0))
        self.place_final = float(args.get("ptm_place_final_weight", 0.0))
        self.place_schedule = args.get("ptm_place_schedule", "cosine")
        self.move_weight = float(args.get("old_move_weight", 1.0))
        self.smoke_test = bool(args.get("smoke_test", False))
        self.max_train_batches = int(args.get("max_train_batches", 2)) if self.smoke_test else None
        self.max_statistics_batches = (
            int(args.get("max_statistics_batches", 2)) if self.smoke_test else None
        )
        self.max_test_batches = int(args.get("max_test_batches", 2)) if self.smoke_test else None

        self.memory = PTMGuidanceMemory()
        self.movement = AnchorResponseStability(args.get("old_move_loss", "smooth_l1"))
        self.unit_names = list(adapter_units(self._network.convnet).keys())
        self.task_boundaries = []
        self.total_tasks = int(args["task_count"])
        self._task_start_time = None
        self._training_seconds = 0.0

        self.result_writer = PTMGuidedResultWriter(
            args.get("result_root", "results"),
            args["dataset"],
            args["task_count"],
            args["method"],
            args["seed"],
        )
        logging.info(
            "PTM-guided learner: method=%s adapters=%d placement=%s place=%s move=%s",
            args["method"],
            len(self.unit_names),
            args.get("adapter_placement"),
            self.place_enabled,
            self.move_enabled,
        )

    @staticmethod
    def _validate_config(args):
        forbidden = {
            "apr": args.get("apr", False),
            "scc": args.get("scc", False),
            "ca": args.get("ca", False),
            "feature": args.get("feature", False),
            "logit": args.get("logit", False),
            "project_gradient": args.get("project_gradient", False),
            "project_after_step": args.get("project_after_step", False),
        }
        enabled = [key for key, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "PTM-guided controlled experiments require APR/SCC/CA/distillation/"
                "hard projection off; enabled={}".format(enabled)
            )
        method = args.get("method")
        expected = {
            "adapter_baseline": (False, False),
            "ptm_place_only": (True, False),
            "ptm_place_move": (True, True),
        }
        if method not in expected:
            raise ValueError("Unknown PTM-guided comparison method: {}".format(method))
        actual = (
            bool(args.get("ptm_place_enabled", False)),
            bool(args.get("old_move_enabled", False)),
        )
        if actual != expected[method]:
            raise ValueError(
                "Method {} requires ptm_place_enabled={}, old_move_enabled={}; got {}".format(
                    method, expected[method][0], expected[method][1], actual
                )
            )

    def _plain_network(self):
        return self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    @staticmethod
    def _limited_batches(loader, limit):
        for batch_index, batch in enumerate(loader):
            if limit is not None and batch_index >= limit:
                break
            yield batch_index, batch

    def incremental_train(self, data_manager):
        self._task_start_time = time.perf_counter()
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.task_boundaries.append(self._total_classes)
        self._network.update_fc(task_size)

        current_classes = np.arange(self._known_classes, self._total_classes)
        logging.info(
            "PTM-guided task %d/%d: classes [%d,%d), task_size=%d",
            self._cur_task + 1,
            data_manager.nb_tasks,
            self._known_classes,
            self._total_classes,
            task_size,
        )
        train_dataset = data_manager.get_dataset(current_classes, source="train", mode="train")
        statistics_dataset = data_manager.get_dataset(current_classes, source="train", mode="test")
        test_dataset = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        statistics_loader = DataLoader(
            statistics_dataset,
            batch_size=self.batch_size,
            shuffle=self.smoke_test,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._network.to(self._device)
        branch = active_adapter_branch(self._network.convnet)
        logging.info("Task %d active nonlinear adapter branch: %d", self._cur_task, branch)

        ptm_prototypes = self._collect_ptm_prototypes(statistics_loader)
        if not self.smoke_test and set(ptm_prototypes) != set(current_classes.tolist()):
            raise RuntimeError("PTM prototype collection did not cover every current class")
        self.memory.add_ptm_prototypes(ptm_prototypes)
        snapshot_count = self.movement.prepare(self.memory, self._network.convnet)
        logging.info("Movement snapshot contains %d unit/class anchors", snapshot_count)

        if len(self._multiple_gpus) > 1:
            logging.info("Using DataParallel on logical devices %s", self._multiple_gpus)
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        train_started = time.perf_counter()
        self._train(self.train_loader)
        self._training_seconds = time.perf_counter() - train_started

        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module
        self._network.fc.backup()
        statistics = self._collect_adapted_statistics(statistics_loader)
        self.memory.add_adapted_statistics(statistics, self._network.convnet)
        logging.info(
            "Task %d training/statistics complete: elapsed=%s training=%s",
            self._cur_task,
            _format_duration(time.perf_counter() - self._task_start_time),
            _format_duration(self._training_seconds),
        )

    def _make_optimizer(self, model, epochs):
        if self._cur_task == 0:
            groups = [
                {"params": model.convnet.blocks[-1].parameters(), "lr": 0.01,
                 "weight_decay": self.weight_decay},
                {"params": model.convnet.blocks[:-1].parameters(), "lr": 0.01,
                 "weight_decay": self.weight_decay},
                {"params": model.fc.parameters(), "lr": 0.1,
                 "weight_decay": self.weight_decay},
            ]
        else:
            groups = [
                {"params": model.convnet.parameters(), "lr": 0.01,
                 "weight_decay": self.weight_decay},
                {"params": model.fc.parameters(), "lr": 0.01,
                 "weight_decay": self.weight_decay},
            ]
        if self.args["optimizer"].lower() == "sgd":
            optimizer = optim.SGD(
                groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay
            )
        elif self.args["optimizer"].lower() == "adam":
            optimizer = optim.AdamW(groups, lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unsupported optimizer {}".format(self.args["optimizer"]))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=self.min_lr
        )
        return optimizer, scheduler

    def _train(self, train_loader):
        model = self._plain_network()
        configured_epochs = self.args["init_epochs"] if self._cur_task == 0 else self.args["inc_epochs"]
        epochs = 1 if self.smoke_test else int(configured_epochs)
        optimizer, scheduler = self._make_optimizer(model, epochs)
        classification_loss = AngularPenaltySMLoss(
            loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["margin"]
        )
        task_started = time.perf_counter()
        epoch_progress = tqdm(
            range(epochs),
            desc="PTM task {}/{} epochs".format(self._cur_task + 1, self.total_tasks),
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            self._network.train()
            epoch_started = time.perf_counter()
            lambda_place = placement_weight(
                epoch, epochs, self.place_initial, self.place_final, self.place_schedule
            ) if self.place_enabled else 0.0
            sums = {"classification": 0.0, "placement": 0.0, "movement": 0.0, "total": 0.0}
            correct, total, batches = 0, 0, 0
            for _, (_, inputs, targets) in self._limited_batches(
                train_loader, self.max_train_batches
            ):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = model.forward_adapted(inputs)
                logits = outputs["logits"]
                loss_cls = classification_loss(
                    logits[:, self._known_classes:], targets - self._known_classes
                )
                loss_place = (
                    ptm_placement_loss(outputs["sfeatures"], targets, self.memory.ptm_prototypes)
                    if self.place_enabled else outputs["sfeatures"].sum() * 0.0
                )
                loss_move = (
                    self.movement.loss(model.convnet)
                    if self.move_enabled else outputs["sfeatures"].sum() * 0.0
                )
                loss = loss_cls + lambda_place * loss_place + self.move_weight * loss_move

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError("Non-finite PTM-guided training loss")

                batches += 1
                sums["classification"] += loss_cls.item()
                sums["placement"] += loss_place.item()
                sums["movement"] += loss_move.item()
                sums["total"] += loss.item()
                predictions = torch.max(logits, dim=1)[1]
                correct += predictions.eq(targets).sum().item()
                total += len(targets)
            if not batches:
                raise RuntimeError("Training loader yielded no batches")
            scheduler.step()
            means = {key: value / batches for key, value in sums.items()}
            train_accuracy = 100.0 * correct / max(total, 1)
            elapsed = time.perf_counter() - task_started
            mean_epoch = elapsed / float(epoch + 1)
            eta = mean_epoch * (epochs - epoch - 1)
            row = {
                "task": self._cur_task,
                "epoch": epoch,
                "classification_loss": means["classification"],
                "placement_loss": means["placement"],
                "movement_loss": means["movement"],
                "lambda_place": lambda_place,
                "lambda_move": self.move_weight if self.move_enabled else 0.0,
                "total_loss": means["total"],
                "train_accuracy": train_accuracy,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            self.result_writer.loss_components.append(row)
            epoch_progress.set_postfix(
                loss="{:.4f}".format(means["total"]),
                cls="{:.4f}".format(means["classification"]),
                place="{:.4f}".format(means["placement"]),
                move="{:.4f}".format(means["movement"]),
                eta=_format_duration(eta),
            )
            logging.info(
                "Task %d epoch %d/%d: cls_loss=%.6f place_loss=%.6f move_loss=%.6f "
                "lambda_place=%.6f lambda_move=%.6f total_loss=%.6f train_acc=%.2f%% "
                "epoch_time=%s elapsed=%s eta=%s",
                self._cur_task, epoch + 1, epochs, means["classification"],
                means["placement"], means["movement"], lambda_place,
                self.move_weight if self.move_enabled else 0.0, means["total"],
                train_accuracy, _format_duration(row["epoch_seconds"]),
                _format_duration(elapsed), _format_duration(eta),
            )

    def _collect_ptm_prototypes(self, loader):
        model = self._plain_network()
        model.eval()
        accumulator = ClassPrototypeAccumulator()
        started = time.perf_counter()
        with torch.no_grad():
            for _, (_, inputs, targets) in self._limited_batches(
                loader, self.max_statistics_batches
            ):
                features = model.extract_features(inputs.to(self._device), use_adapter=False)
                if features.requires_grad:
                    raise RuntimeError("Frozen PTM reference unexpectedly requires gradients")
                accumulator.update(features, targets)
        result = accumulator.finalize(normalize=True)
        logging.info(
            "Collected %d normalized PTM class prototypes in %s (no raw samples retained)",
            len(result), _format_duration(time.perf_counter() - started),
        )
        return result

    def _collect_adapted_statistics(self, loader):
        model = self._plain_network()
        model.eval()
        branch = active_adapter_branch(model.convnet)
        units = adapter_units(model.convnet, branch)
        accumulator = LayerAnchorAccumulator(units.keys())
        started = time.perf_counter()
        with AdapterInputCollector(model.convnet, branch) as collector:
            with torch.no_grad():
                for _, (_, inputs, targets) in self._limited_batches(
                    loader, self.max_statistics_batches
                ):
                    features = model.extract_features(inputs.to(self._device), use_adapter=True)
                    accumulator.update(collector.pop(), targets, features)
        result = accumulator.finalize()
        logging.info(
            "Collected adapted prototypes and %d adapter-unit anchor tables in %s",
            len(result["layer_anchors"]), _format_duration(time.perf_counter() - started),
        )
        return result

    def _evaluate(self):
        model = self._plain_network()
        model.eval()
        predictions, targets_all = [], []
        current = ClassPrototypeAccumulator()
        with torch.no_grad():
            for _, (_, inputs, targets) in self._limited_batches(
                self.test_loader, self.max_test_batches
            ):
                outputs = model.forward_adapted(inputs.to(self._device))
                k = min(self.topk, self._total_classes)
                predictions.append(torch.topk(outputs["logits"], k=k, dim=1)[1].cpu().numpy())
                targets_all.append(targets.numpy())
                current.update(outputs["sfeatures"], targets)
        if not predictions:
            raise RuntimeError("Test loader yielded no batches")
        return np.concatenate(predictions), np.concatenate(targets_all), current.finalize()

    def eval_task(self):
        model = self._plain_network()
        predicted, y_true, current_prototypes = self._evaluate()
        grouped = accuracy(
            predicted[:, 0], y_true, self._known_classes, increment=int(self.args["increment"])
        )
        topk_accuracy = float(np.around(
            (predicted.T == np.tile(y_true, (predicted.shape[1], 1))).sum()
            * 100.0 / len(y_true), decimals=2,
        ))
        task_row = task_accuracy_row(predicted[:, 0], y_true, self.task_boundaries)
        self.result_writer.task_accuracy_matrix.append(task_row)
        forgetting = forgetting_from_matrix(self.result_writer.task_accuracy_matrix)

        valid_ids = sorted(set(current_prototypes).intersection(self.memory.ptm_prototypes))
        ptm_drift_rows = []
        for class_id in valid_ids:
            reference = self.memory.ptm_prototypes[class_id].view(1, -1)
            current = current_prototypes[class_id].float().view(1, -1)
            drift = 1.0 - F.cosine_similarity(reference, current, dim=1).item()
            ptm_drift_rows.append({
                "task": self._cur_task,
                "class_id": class_id,
                "ptm_prototype_cosine_drift": float(drift),
            })
        geometry_drift = pairwise_geometry_drift(
            self.memory.ptm_prototypes, current_prototypes, valid_ids
        )
        mean_ptm_drift = float(np.mean([
            row["ptm_prototype_cosine_drift"] for row in ptm_drift_rows
        ])) if ptm_drift_rows else 0.0
        for row in ptm_drift_rows:
            row["ptm_geometry_drift"] = geometry_drift
        self.result_writer.ptm_geometry_drift.extend(ptm_drift_rows or [{
            "task": self._cur_task,
            "class_id": "",
            "ptm_prototype_cosine_drift": 0.0,
            "ptm_geometry_drift": geometry_drift,
        }])

        anchor_rows = anchor_drift_rows(self.memory, model.convnet, self._cur_task)
        self.result_writer.anchor_drift.extend(anchor_rows)
        mean_anchor_drift = float(np.mean([
            row["response_cosine_drift"] for row in anchor_rows
        ])) if anchor_rows else 0.0

        self.result_writer.class_accuracy.extend(
            class_accuracy_rows(predicted[:, 0], y_true, self._cur_task, "head")
        )
        task_elapsed = time.perf_counter() - self._task_start_time
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self._device))
            if self._device.type == "cuda" and torch.cuda.is_available() else None
        )
        task_metric = {
            "task": self._cur_task,
            "seen_classes": self._total_classes,
            "last_accuracy": float(grouped["total"]),
            "top5_accuracy": topk_accuracy,
            "old_accuracy": float(grouped["old"]),
            "new_accuracy": float(grouped["new"]),
            "incremental_accuracy": float(np.mean(
                [row["last_accuracy"] for row in self.result_writer.task_metrics]
                + [float(grouped["total"])]
            )),
            "forgetting": forgetting,
            "mean_ptm_prototype_drift": mean_ptm_drift,
            "ptm_geometry_drift": geometry_drift,
            "mean_anchor_response_drift": mean_anchor_drift,
            "training_seconds": self._training_seconds,
            "task_elapsed_seconds": task_elapsed,
            "peak_gpu_memory_bytes": peak_memory,
        }
        self.result_writer.task_metrics.append(task_metric)
        self.result_writer.flush(self._config_snapshot())
        logging.info(
            "Task %d evaluation: Last-Acc=%.2f Inc-Acc=%.2f old=%.2f new=%.2f "
            "forgetting=%.2f ptm_proto_drift=%.6f ptm_geometry=%.6f "
            "anchor_drift=%.6f elapsed=%s",
            self._cur_task, grouped["total"], task_metric["incremental_accuracy"],
            grouped["old"], grouped["new"], forgetting, mean_ptm_drift,
            geometry_drift, mean_anchor_drift, _format_duration(task_elapsed),
        )
        return {"grouped": grouped, "top1": grouped["total"], "top5": topk_accuracy}

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.convnet.flag += 1
        destination = 1 if self._network.convnet.flag % 2 == 0 else 2
        self._network.convnet.copy_adapter(destination)
        if self.args.get("save_checkpoints", True) and not self.smoke_test:
            self.save_checkpoint()

    def _config_snapshot(self):
        snapshot = copy.deepcopy(self.args)
        snapshot["device"] = [str(device) for device in self._multiple_gpus]
        return snapshot

    def save_checkpoint(self, filename=None):
        if filename is None:
            filename = os.path.join(
                self.result_writer.checkpoint_directory,
                "checkpoint_task_{:02d}.pth".format(self._cur_task),
            )
        state = {
            "model_state_dict": self._network.state_dict(),
            "current_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "task_boundaries": self.task_boundaries,
            "convnet_flag": self._network.convnet.flag,
            "guidance_memory": self.memory.state_dict(),
            "config": self._config_snapshot(),
            "result_history": {
                name: getattr(self.result_writer, name) for name in (
                    "task_metrics", "task_accuracy_matrix", "class_accuracy",
                    "ptm_geometry_drift", "anchor_drift", "loss_components",
                )
            },
        }
        torch.save(state, filename)
        logging.info("Saved resumable PTM-guided checkpoint: %s", filename)
        return filename

    def load_checkpoint(self, filename, data_manager):
        checkpoint = torch.load(filename, map_location="cpu")
        saved_config = checkpoint.get("config", {})
        for key in (
            "dataset", "seed", "init_cls", "increment", "task_count",
            "convnet_type", "ffn_num", "adapter_placement", "adapter_interval", "method",
        ):
            if key in saved_config and saved_config[key] != self.args.get(key):
                raise ValueError("Checkpoint config mismatch for {}".format(key))
        completed_task = int(checkpoint["current_task"])
        for task_index in range(completed_task + 1):
            self._network.update_fc(data_manager.get_task_size(task_index))
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._cur_task = completed_task
        self._known_classes = int(checkpoint["known_classes"])
        self._total_classes = int(checkpoint["total_classes"])
        self.task_boundaries = [int(value) for value in checkpoint["task_boundaries"]]
        self._network.convnet.flag = int(checkpoint["convnet_flag"])
        self.memory.load_state_dict(checkpoint["guidance_memory"])
        history = checkpoint.get("result_history", {})
        for name in (
            "task_metrics", "task_accuracy_matrix", "class_accuracy",
            "ptm_geometry_drift", "anchor_drift", "loss_components",
        ):
            setattr(self.result_writer, name, history.get(name, []))
        logging.info("Resumed PTM-guided checkpoint %s after task %d", filename, completed_task)
        return completed_task + 1
