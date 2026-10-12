"""Class-discriminative subspace protection diagnostic for official CDS.

CDS is a repair-after-drift method (APR/SCC/CA).  This separate learner tests
prevent-discriminative-space-drift with stored prototypes and compressed,
layer-wise statistics.  It does not alter the original CDS learner.

The ViT module is the existing nonlinear bottleneck adapter
``up_proj(ReLU(down_proj(x)))``.  This code is therefore not KeepLoRA; it only
adopts the protected-subspace philosophy and enforces ``down_proj.weight Q=0``.
"""

import copy
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.cil_metrics import (
    StructuredResultWriter,
    class_accuracy_rows,
    forgetting_from_matrix,
    task_accuracy_row,
)
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss
from utils.prototype_classifier import (
    class_means_from_sums,
    ordered_prototype_matrix,
    pairwise_geometry_drift,
    prototype_drift_rows,
    prototype_logits,
)
from utils.protected_subspace import (
    AdapterInputCollector,
    ProtectedSubspaceManager,
    StreamingLayerStatistics,
)
from utils.toolkit import accuracy


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class Learner(BaseLearner):
    """Exemplar-free prototype learner with optional protected subspaces."""

    def __init__(self, args):
        super().__init__(args)
        if "adapter" not in args["convnet_type"]:
            raise NotImplementedError("CDSP requires an Adapter backbone")
        self._validate_controlled_config(args)

        self.args = args
        self._network = SimpleVitNet(args, True)
        self.batch_size = int(args["batch_size"])
        self.init_lr = float(args["init_lr"])
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.num_workers = int(args.get("num_workers", 8))
        self.eval_classifier = args.get("eval_classifier", "prototype")
        if self.eval_classifier not in {"prototype", "head"}:
            raise ValueError("eval_classifier must be 'prototype' or 'head'")

        self.subspace = ProtectedSubspaceManager(args, self.feature_dim)
        self.subspace.resolve_layers(self._network.convnet)
        self.task_boundaries = []
        self.total_tasks = None
        self._task_start_time = None
        self._training_seconds = 0.0

        method = args.get("method", args.get("subspace_mode", "prototype_baseline"))
        self.result_writer = StructuredResultWriter(
            args.get("result_root", "results"), args["dataset"], method, args["seed"]
        )

    @staticmethod
    def _validate_controlled_config(args):
        forbidden = {
            "apr": args.get("apr", False),
            "scc": args.get("scc", False),
            "ca": args.get("ca", False),
            "feature": args.get("feature", False),
            "logit": args.get("logit", False),
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "Controlled CDSP experiments require APR/SCC/CA/distillation off; enabled: {}".format(enabled)
            )

    def _plain_network(self):
        return self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    def incremental_train(self, data_manager):
        self._task_start_time = time.perf_counter()
        self.total_tasks = data_manager.nb_tasks
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.task_boundaries.append(self._total_classes)
        self._network.update_fc(task_size)

        logging.info(
            "CDSP task %d/%d: learning classes [%d, %d), mode=%s",
            self._cur_task + 1,
            data_manager.nb_tasks,
            self._known_classes,
            self._total_classes,
            self.subspace.mode,
        )

        # Exemplar-free boundary: training and statistic collection request only
        # classes introduced by the current task.  No old train split is queried.
        current_classes = np.arange(self._known_classes, self._total_classes)
        train_dataset = data_manager.get_dataset(current_classes, source="train", mode="train")
        statistics_dataset = data_manager.get_dataset(current_classes, source="train", mode="test")
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
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
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._network.to(self._device)
        active_branch = self.subspace.prepare_for_task(self._network.convnet)
        start_rank_rows = self.subspace.rank_stats(
            self._cur_task, "task_start", self._known_classes
        )
        self._append_subspace_rows(start_rank_rows, self.subspace.violation_stats(self._network.convnet))
        logging.info("Task %d active adapter branch: adapter%d", self._cur_task, active_branch)
        self._log_rank_rows(start_rank_rows)

        if len(self._multiple_gpus) > 1:
            logging.info("Using DataParallel on logical devices %s", self._multiple_gpus)
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        train_started = time.perf_counter()
        self._train(self.train_loader)
        self._training_seconds = time.perf_counter() - train_started

        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

        self._network.fc.backup()
        trained_rank_rows = self.subspace.rank_stats(
            self._cur_task, "post_training_constraint", self._known_classes
        )
        self._append_subspace_rows(
            trained_rank_rows, self.subspace.violation_stats(self._network.convnet)
        )
        collected = self._collect_current_task_statistics(statistics_loader)
        self.subspace.update_memory(collected, current_classes.tolist())

        end_rank_rows = self.subspace.rank_stats(
            self._cur_task, "task_end_memory", self._total_classes
        )
        # This basis includes the just-finished task and is intentionally not
        # imposed until the next task starts, so a violation is not meaningful here.
        self._append_subspace_rows(end_rank_rows, [])
        self._log_rank_rows(end_rank_rows)
        logging.info(
            "Task %d training+memory complete: elapsed=%s (training=%s)",
            self._cur_task,
            _format_duration(time.perf_counter() - self._task_start_time),
            _format_duration(self._training_seconds),
        )

    def _make_optimizer(self, model, epochs):
        if self._cur_task == 0:
            param_groups = [
                {
                    "params": model.convnet.blocks[-1].parameters(),
                    "lr": 0.01,
                    "weight_decay": self.args["weight_decay"],
                },
                {
                    "params": model.convnet.blocks[:-1].parameters(),
                    "lr": 0.01,
                    "weight_decay": self.args["weight_decay"],
                },
                {
                    "params": model.fc.parameters(),
                    "lr": 0.1,
                    "weight_decay": self.args["weight_decay"],
                },
            ]
        else:
            param_groups = [
                {
                    "params": model.convnet.parameters(),
                    "lr": 0.01,
                    "weight_decay": self.args["weight_decay"],
                },
                {
                    "params": model.fc.parameters(),
                    "lr": 0.01,
                    "weight_decay": self.args["weight_decay"],
                },
            ]

        optimizer_name = self.args["optimizer"].lower()
        if optimizer_name == "sgd":
            optimizer = optim.SGD(
                param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay
            )
        elif optimizer_name == "adam":
            optimizer = optim.AdamW(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unsupported optimizer: {}".format(self.args["optimizer"]))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.min_lr
        )
        return optimizer, scheduler

    def _train(self, train_loader):
        model = self._plain_network()
        epochs = int(self.args["init_epochs"] if self._cur_task == 0 else self.args["inc_epochs"])
        optimizer, scheduler = self._make_optimizer(model, epochs)
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            eps=1e-7,
            s=self.args["scale"],
            m=self.args["margin"],
        )

        task_started = time.perf_counter()
        epoch_progress = tqdm(
            range(epochs),
            desc="Task {}/{} epochs".format(self._cur_task + 1, self.total_tasks or "?"),
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            self._network.train()
            epoch_started = time.perf_counter()
            losses = 0.0
            correct, total = 0, 0
            batch_progress = tqdm(
                train_loader,
                desc="Task {} epoch {}/{}".format(self._cur_task, epoch + 1, epochs),
                leave=False,
                dynamic_ncols=True,
            )
            for _, inputs, targets in batch_progress:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                outputs = self._network(inputs, label=targets)
                logits = outputs["logits"]
                loss = loss_cos(
                    logits[:, self._known_classes:], targets - self._known_classes
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.subspace.enforce_constraint(model.convnet, optimizer)

                losses += loss.item()
                predictions = torch.max(logits, dim=1)[1]
                correct += predictions.eq(targets).sum().item()
                total += len(targets)
                batch_progress.set_postfix(loss="{:.4f}".format(loss.item()))

            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_started
            elapsed = time.perf_counter() - task_started
            mean_epoch = elapsed / float(epoch + 1)
            eta = mean_epoch * (epochs - epoch - 1)
            train_accuracy = 100.0 * correct / max(total, 1)
            mean_loss = losses / max(len(train_loader), 1)
            violation = self.subspace.violation_stats(model.convnet)
            max_violation = max(
                (row["protected_violation_ratio"] for row in violation), default=0.0
            )
            epoch_progress.set_postfix(
                loss="{:.4f}".format(mean_loss),
                acc="{:.2f}".format(train_accuracy),
                eta=_format_duration(eta),
            )
            logging.info(
                "Task %d epoch %d/%d: loss=%.4f train_acc=%.2f%% "
                "epoch_time=%s elapsed=%s eta=%s max_violation=%.3e",
                self._cur_task,
                epoch + 1,
                epochs,
                mean_loss,
                train_accuracy,
                _format_duration(epoch_seconds),
                _format_duration(elapsed),
                _format_duration(eta),
                max_violation,
            )

    def _collect_current_task_statistics(self, loader):
        model = self._plain_network()
        model.eval()
        branch = self.subspace.active_branch(model.convnet)
        statistics = StreamingLayerStatistics(self.subspace.layer_ids, self.feature_dim)
        started = time.perf_counter()
        with AdapterInputCollector(model.convnet, branch, self.subspace.layer_ids) as collector:
            progress = tqdm(
                loader,
                desc="Task {} compressed statistics".format(self._cur_task),
                leave=False,
                dynamic_ncols=True,
            )
            with torch.no_grad():
                for _, inputs, targets in progress:
                    inputs = inputs.to(self._device)
                    final_features = model.extract_vector(inputs)
                    layer_features = collector.pop()
                    statistics.update(layer_features, targets, final_features)
        logging.info(
            "Task %d current-class statistics collected in %s; no sample features retained",
            self._cur_task,
            _format_duration(time.perf_counter() - started),
        )
        return statistics.finalize()

    def _evaluate_both_classifiers(self, loader):
        model = self._plain_network()
        model.eval()
        prototype_matrix = ordered_prototype_matrix(
            self.subspace.final_prototypes,
            self._total_classes,
            self._device,
        )
        prototype_predictions, head_predictions, targets_all = [], [], []
        current_sums, current_counts = {}, {}
        progress = tqdm(
            loader,
            desc="Task {} evaluation".format(self._cur_task),
            leave=False,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for _, inputs, targets in progress:
                inputs = inputs.to(self._device)
                outputs = model(inputs)
                features = outputs["sfeatures"]
                proto_scores = prototype_logits(features, prototype_matrix)
                k = min(self.topk, self._total_classes)
                prototype_predictions.append(torch.topk(proto_scores, k=k, dim=1)[1].cpu().numpy())
                head_predictions.append(torch.topk(outputs["logits"], k=k, dim=1)[1].cpu().numpy())
                targets_all.append(targets.numpy())

                cpu_features = features.detach().cpu().float()
                for class_id in torch.unique(targets).tolist():
                    class_id = int(class_id)
                    class_sum = cpu_features[targets == class_id].sum(dim=0)
                    current_sums[class_id] = current_sums.get(
                        class_id, torch.zeros_like(class_sum)
                    ) + class_sum
                    current_counts[class_id] = current_counts.get(class_id, 0) + int(
                        (targets == class_id).sum().item()
                    )

        return (
            np.concatenate(prototype_predictions),
            np.concatenate(head_predictions),
            np.concatenate(targets_all),
            class_means_from_sums(current_sums, current_counts),
        )

    def eval_task(self):
        prototype_pred, head_pred, y_true, current_test_prototypes = (
            self._evaluate_both_classifiers(self.test_loader)
        )
        primary_pred = prototype_pred if self.eval_classifier == "prototype" else head_pred
        grouped = accuracy(
            primary_pred[:, 0],
            y_true,
            self._known_classes,
            increment=int(self.args["increment"]),
        )
        topk_accuracy = np.around(
            (primary_pred.T == np.tile(y_true, (primary_pred.shape[1], 1))).sum()
            * 100.0
            / len(y_true),
            decimals=2,
        )

        prototype_top1 = float(np.mean(prototype_pred[:, 0] == y_true) * 100.0)
        head_top1 = float(np.mean(head_pred[:, 0] == y_true) * 100.0)
        task_row = task_accuracy_row(primary_pred[:, 0], y_true, self.task_boundaries)
        self.result_writer.task_accuracy_matrix.append(task_row)
        forgetting = forgetting_from_matrix(self.result_writer.task_accuracy_matrix)

        old_class_ids = list(range(self._known_classes))
        drift_rows = prototype_drift_rows(
            self.subspace.final_prototypes,
            current_test_prototypes,
            old_class_ids,
            self._cur_task,
        )
        pairwise_drift = pairwise_geometry_drift(
            self.subspace.final_prototypes, current_test_prototypes, old_class_ids
        )
        mean_prototype_drift = float(
            np.mean([row["prototype_cosine_drift"] for row in drift_rows])
        ) if drift_rows else 0.0
        for row in drift_rows:
            row["pairwise_geometry_drift"] = pairwise_drift
        if not drift_rows:
            drift_rows.append({
                "task": self._cur_task,
                "class_id": "",
                "prototype_cosine_drift": 0.0,
                "pairwise_geometry_drift": pairwise_drift,
            })
        self.result_writer.prototype_drift.extend(drift_rows)

        task_elapsed = time.perf_counter() - self._task_start_time
        task_metric = {
            "task": self._cur_task,
            "seen_classes": self._total_classes,
            "primary_classifier": self.eval_classifier,
            "primary_top1": float(grouped["total"]),
            "primary_top5": float(topk_accuracy),
            "old_accuracy": float(grouped["old"]),
            "new_accuracy": float(grouped["new"]),
            "prototype_top1": prototype_top1,
            "head_top1": head_top1,
            "average_incremental_accuracy": float(np.mean(
                [row["primary_top1"] for row in self.result_writer.task_metrics]
                + [float(grouped["total"])]
            )),
            "forgetting": forgetting,
            "mean_prototype_cosine_drift": mean_prototype_drift,
            "pairwise_geometry_drift": pairwise_drift,
            "training_seconds": self._training_seconds,
            "task_elapsed_seconds": task_elapsed,
        }
        self.result_writer.task_metrics.append(task_metric)
        self.result_writer.class_accuracy.extend(
            class_accuracy_rows(prototype_pred[:, 0], y_true, self._cur_task, "prototype")
        )
        self.result_writer.class_accuracy.extend(
            class_accuracy_rows(head_pred[:, 0], y_true, self._cur_task, "head")
        )
        self.result_writer.flush(self._config_snapshot(), self.eval_classifier)

        logging.info(
            "Task %d evaluation: primary(%s)=%.2f%% old=%.2f%% new=%.2f%% "
            "prototype=%.2f%% head=%.2f%% forgetting=%.2f drift=%.6f pairwise=%.6f elapsed=%s",
            self._cur_task,
            self.eval_classifier,
            grouped["total"],
            grouped["old"],
            grouped["new"],
            prototype_top1,
            head_top1,
            forgetting,
            mean_prototype_drift,
            pairwise_drift,
            _format_duration(task_elapsed),
        )
        return {
            "grouped": grouped,
            "top1": grouped["total"],
            "top{}".format(self.topk): topk_accuracy,
        }

    def _append_subspace_rows(self, rank_rows, violation_rows):
        by_layer = {row["layer"]: row["protected_violation_ratio"] for row in violation_rows}
        for row in rank_rows:
            row = dict(row)
            row["protected_violation_ratio"] = by_layer.get(row["layer"], "")
            self.result_writer.subspace_stats.append(row)

    @staticmethod
    def _log_rank_rows(rows):
        for row in rows:
            logging.info(
                "Subspace task=%d phase=%s layer=%d classes=%d inter=%d intra=%d task=%d "
                "protected=%d residual=%d ratio=%.4f",
                row["task"], row["phase"], row["layer"], row["number_of_classes"],
                row["q_inter_rank"], row["q_intra_rank"], row["q_task_rank"],
                row["protected_rank"], row["residual_dim"], row["protected_ratio"],
            )

    def after_task(self):
        self.subspace.clear_gradient_hooks()
        self._known_classes = self._total_classes
        self._network.convnet.flag += 1
        # Copy the just-trained branch into the branch that will be trained next.
        destination = 1 if self._network.convnet.flag % 2 == 0 else 2
        self._network.convnet.copy_adapter(destination)
        if self.args.get("save_checkpoints", True):
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
            "subspace_memory": self.subspace.state_dict(),
            "config": self._config_snapshot(),
            "result_history": {
                "task_metrics": self.result_writer.task_metrics,
                "task_accuracy_matrix": self.result_writer.task_accuracy_matrix,
                "class_accuracy": self.result_writer.class_accuracy,
                "subspace_stats": self.result_writer.subspace_stats,
                "prototype_drift": self.result_writer.prototype_drift,
            },
        }
        torch.save(state, filename)
        logging.info("Saved resumable CDSP checkpoint: %s", filename)
        return filename

    def load_checkpoint(self, filename, data_manager):
        checkpoint = torch.load(filename, map_location="cpu")
        saved_config = checkpoint.get("config", {})
        for key in (
            "dataset", "seed", "init_cls", "increment", "convnet_type", "ffn_num",
            "adapter_placement", "adapter_interval", "method", "subspace_mode",
        ):
            if key in saved_config and saved_config[key] != self.args.get(key):
                raise ValueError(
                    "Checkpoint config mismatch for {}: saved={!r}, current={!r}".format(
                        key, saved_config[key], self.args.get(key)
                    )
                )
        completed_task = int(checkpoint["current_task"])
        for task_index in range(completed_task + 1):
            self._network.update_fc(data_manager.get_task_size(task_index))
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self._cur_task = completed_task
        self._known_classes = int(checkpoint["known_classes"])
        self._total_classes = int(checkpoint["total_classes"])
        self.task_boundaries = [int(value) for value in checkpoint["task_boundaries"]]
        self._network.convnet.flag = int(checkpoint.get("convnet_flag", completed_task + 1))
        self.subspace.load_state_dict(checkpoint["subspace_memory"])

        history = checkpoint.get("result_history", {})
        for name in (
            "task_metrics", "task_accuracy_matrix", "class_accuracy",
            "subspace_stats", "prototype_drift",
        ):
            setattr(self.result_writer, name, history.get(name, []))
        logging.info("Resumed CDSP checkpoint %s after task %d", filename, completed_task)
        return completed_task + 1
