"""Two-view Structure training; the only branch difference is the JS term."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from oracle_audit.snapshot_learner import AuditSnapshotLearner
from utils.loss import AngularPenaltySMLoss
from utils.distill_loss import LobachevskyRelativeGeomLoss


class PairedStructureViews(Dataset):
    """Call the original Structure train transform twice on the same image."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        index1, view1, label1 = self.base[index]
        index2, view2, label2 = self.base[index]
        if index1 != index2 or label1 != label2:
            raise AssertionError("paired views must refer to the same original sample")
        return index1, view1, view2, label1


class VariationPilotLearner(AuditSnapshotLearner):
    def __init__(self, args):
        super().__init__(args)
        self.pilot_branch = str(args["pilot_branch"])
        if self.pilot_branch not in {"control", "js"}:
            raise ValueError(self.pilot_branch)
        if not args["scc"] or not args["apr"] or not args["ca"] or args["old_new_separation"]:
            raise RuntimeError("pilot requires vanilla SCC+APR+CA Structure without separation")
        self.lambda_js = 0.1 if self.pilot_branch == "js" else 0.0
        self._stats_update_count = 0
        self._apr_correction_count = 0
        self._ca_call_count = 0
        self._task_train_records = {}
        original_correct = self.adjust_distribution.correct_prototypes

        def counted_correction(*values, **options):
            self._apr_correction_count += 1
            return original_correct(*values, **options)

        self.adjust_distribution.correct_prototypes = counted_correction

    def load_experiment_checkpoint(self, path):
        # The source checkpoint is Structure_AUDIT, but all new outputs must
        # live in a distinct branch directory. Relax only the run-name key.
        experiment_id = self.args["experiment_id"]
        source_experiment_id = "Structure_AUDIT" if "Structure_AUDIT" in Path(path).parts else experiment_id
        self.args["experiment_id"] = source_experiment_id
        try:
            checkpoint = super().load_experiment_checkpoint(path)
        finally:
            self.args["experiment_id"] = experiment_id
        if checkpoint["stage"] != "task_end" or int(checkpoint["cur_task"]) not in (7, 8):
            raise RuntimeError("pilot requires T8 or own T9 task_end checkpoint")
        self._network.convnet.flag = int(checkpoint.get("network_flag_after_task", checkpoint["cur_task"] + 1))
        return checkpoint

    def _compute_class_mean(self, *args, **kwargs):
        self._stats_update_count += 1
        return super()._compute_class_mean(*args, **kwargs)

    def _stage2_compact_classifier(self, *args, **kwargs):
        self._ca_call_count += 1
        return super()._stage2_compact_classifier(*args, **kwargs)

    def incremental_train(self, data_manager):
        before = (self._stats_update_count, self._apr_correction_count, self._ca_call_count)
        super().incremental_train(data_manager)
        task = self._cur_task + 1
        after = (self._stats_update_count, self._apr_correction_count, self._ca_call_count)
        delta = tuple(end - start for start, end in zip(before, after))
        if delta != (1, 1, 1):
            raise AssertionError(f"T{task}: expected one stats/APR/CA update, got {delta}")
        record = self._task_train_records[task]
        record.update({"stats_update_count": delta[0], "apr_correction_count": delta[1],
                       "ca_call_count": delta[2], "ca_epochs": int(self.args["ca_epochs"]),
                       "ca_exposure_per_class": int(self.args["replay_samples_per_class"]),
                       "consistency_candidate_set": "current-only",
                       "js_temperature": 1.0, "lambda_js": self.lambda_js,
                       "augmentation": "original Structure train transform invoked independently twice"})
        (self.run_dir / f"pilot_task_{task:02d}_manifest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    def _train(self, train_loader, test_loader):
        task = self._cur_task + 1
        generator = torch.Generator()
        generator.manual_seed(int(self.args["seed"]) + 10000 * task)
        paired = DataLoader(
            PairedStructureViews(train_loader.dataset),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
            num_workers=int(self.args.get("num_workers", 8)), generator=generator,
            pin_memory=True,
        )
        return super()._train(paired, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        if self._cur_task < 8:
            raise RuntimeError("pilot only trains T9/T10")
        task = self._cur_task + 1
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["margin"]
        )
        fingerprints = hashlib.sha256()
        epoch_records = []
        expected_samples = len(train_loader.dataset)
        for epoch in range(start_epoch, self.tuned_epochs):
            started = time.perf_counter()
            self._network.train()
            counts = {"samples": 0, "optimizer_steps": 0}
            sums = {"classification": 0.0, "scc": 0.0, "js": 0.0, "total": 0.0}
            for indices, view1, view2, targets in train_loader:
                fingerprints.update(indices.numpy().tobytes())
                fingerprints.update(view1.numpy().tobytes())
                fingerprints.update(view2.numpy().tobytes())
                view1 = view1.to(self._device, non_blocking=True)
                view2 = view2.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                out1 = self._network(view1, label=targets)
                out2 = self._network(view2, label=targets)
                logits1 = out1["logits"][:, self._known_classes : self._total_classes]
                logits2 = out2["logits"][:, self._known_classes : self._total_classes]
                target_local = targets - self._known_classes
                classification = 0.5 * (loss_cos(logits1, target_local) + loss_cos(logits2, target_local))
                scc = 0.5 * (
                    LobachevskyRelativeGeomLoss(out1, self._class_means, self._device)
                    + LobachevskyRelativeGeomLoss(out2, self._class_means, self._device)
                )
                # logits are the original (unscaled) Structure cosine logits;
                # current-only is the same candidate set as the supervised loss.
                p1 = F.softmax(logits1, dim=1)
                p2 = F.softmax(logits2, dim=1)
                middle = 0.5 * (p1 + p2)
                js = 0.5 * (
                    (p1 * (p1.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=1).mean()
                    + (p2 * (p2.clamp_min(1e-12).log() - middle.clamp_min(1e-12).log())).sum(dim=1).mean()
                )
                loss = classification + float(self.args["scc_value"]) * scc + self.lambda_js * js
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite pilot loss T{task} epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count = len(targets)
                counts["samples"] += count
                counts["optimizer_steps"] += 1
                for key, value in (("classification", classification), ("scc", scc),
                                   ("js", js), ("total", loss)):
                    sums[key] += float(value.detach().cpu()) * count
            scheduler.step()
            if counts["samples"] != expected_samples or counts["optimizer_steps"] != len(train_loader):
                raise AssertionError(f"T{task} epoch {epoch+1}: exposure/step mismatch")
            epoch_row = {"task": task, "epoch": epoch + 1, "samples": counts["samples"],
                         "optimizer_steps": counts["optimizer_steps"],
                         **{key + "_loss": value / counts["samples"] for key, value in sums.items()},
                         "epoch_seconds": time.perf_counter() - started}
            epoch_records.append(epoch_row)
            print(f"[{self.args['protocol_id']} {self.pilot_branch}] T{task} epoch {epoch+1}/{self.tuned_epochs} loss={epoch_row['total_loss']:.4f} js={epoch_row['js_loss']:.6f}", flush=True)
            if self.args.get("checkpoint_every_epoch", False):
                self.save_experiment_checkpoint(self.run_dir / "checkpoint_latest.pt", stage="train_epoch",
                                                next_epoch=epoch + 1, optimizer=optimizer, scheduler=scheduler)
        self._task_train_records[task] = {
            "task": task, "branch": self.pilot_branch, "seed": int(self.args["seed"]),
            "view_sha256": fingerprints.hexdigest(), "num_views_per_image": 2,
            "train_samples_per_epoch": expected_samples,
            "epochs": self.tuned_epochs, "optimizer_steps": sum(row["optimizer_steps"] for row in epoch_records),
            "epoch_records": epoch_records,
        }
