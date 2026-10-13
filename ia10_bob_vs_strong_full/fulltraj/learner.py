"""Only acquisition input changes; original SCC/APR/moments/Gaussian CA remain."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from oracle_audit.snapshot_learner import AuditSnapshotLearner
from utils.loss import AngularPenaltySMLoss
from utils.distill_loss import LobachevskyRelativeGeomLoss

from .common import experiment_hash, progress, read_json, write_json
from .views import FullViews


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class FullTrajectoryLearner(AuditSnapshotLearner):
    def __init__(self, args, bank=()):
        super().__init__(args)
        self.branch = args["fulltraj_branch"]
        self.current_bank = list(bank)
        self._active_views = None

    def save_experiment_checkpoint(self, *args, **kwargs):
        self.args["fulltraj_checkpoint_network_flag"] = int(self._network.convnet.flag)
        value = super().save_experiment_checkpoint(*args, **kwargs)
        path = self.run_dir / "task_checkpoints/manifest.json"
        if path.is_file():
            manifest = read_json(path)
            manifest.update({"training_behavior_modified": True,
                             "purpose": "full cumulative matched auxiliary-image acquisition",
                             "bank_saved_in_learner_state": False})
            write_json(path, manifest)
        return value

    def load_experiment_checkpoint(self, path):
        value = super().load_experiment_checkpoint(path)
        assert value["config"]["fulltraj_branch"] == self.branch
        assert value["config"]["fulltraj_experiment_hash"] == self.args["fulltraj_experiment_hash"]
        # The legacy serializer does not store the non-tensor alternating adapter flag.
        self._network.convnet.flag = int(value["config"]["fulltraj_checkpoint_network_flag"])
        return value

    def _train(self, train_loader, test_loader):
        task = self._cur_task + 1
        assert self.data_manager.generated_statistics_count == 0
        assert set(map(str, self.stats_loader.dataset.images)) <= self.data_manager.real_paths
        views = FullViews(train_loader.dataset, self.branch, task, self.current_bank)
        self._active_views = views
        loader = DataLoader(views, batch_size=self.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator(), drop_last=False)
        return super()._train(loader, test_loader)

    def _init_train(self, loader, test_loader, optimizer, scheduler, start_epoch=0):
        task = self._cur_task + 1
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7,
                                       s=self.args["scale"], m=self.args["margin"])
        path = self.run_dir / f"matched_training_T{task:02d}.json"
        epochs = read_json(path)["epochs"][:start_epoch] if path.exists() else []
        if len(epochs) != start_epoch:
            raise RuntimeError("epoch checkpoint and matched log disagree")
        for epoch in range(start_epoch, self.tuned_epochs):
            loader.dataset.epoch = epoch
            # Explicit epoch streams allow exact restart from epoch checkpoints.
            model_seed = 1993 + task * 10000 + epoch
            random.seed(model_seed)
            np.random.seed(model_seed)
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(model_seed)
            loader.generator.manual_seed(1993 + task * 10000 + epoch)
            self._network.train()
            order_hash, view_hash, aux_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
            count, steps, total_loss = 0, 0, 0.0
            lr_before = [group["lr"] for group in optimizer.param_groups]
            for indices, anchor, auxiliary, targets, aux_ids in loader:
                order_hash.update(indices.numpy().tobytes())
                view_hash.update(anchor.numpy().tobytes())
                for identity in aux_ids:
                    aux_hash.update(identity.encode() + b"\0")
                labels = targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                output = self._network(anchor.to(self._device), label=labels)
                real_loss = 0.5 * loss_cos(output["logits"][:, self._known_classes:], labels - self._known_classes)
                if self._cur_task > 0:
                    real_loss = real_loss + self.args["scc_value"] * LobachevskyRelativeGeomLoss(
                        output, self._class_means, self._device)
                if not torch.isfinite(real_loss):
                    raise FloatingPointError("nonfinite anchor loss")
                loss_value = float(real_loss.detach())
                real_loss.backward()
                del output, real_loss
                output = self._network(auxiliary.to(self._device), label=labels)
                auxiliary_loss = 0.5 * loss_cos(output["logits"][:, self._known_classes:], labels - self._known_classes)
                if not torch.isfinite(auxiliary_loss):
                    raise FloatingPointError("nonfinite auxiliary loss")
                loss_value += float(auxiliary_loss.detach())
                auxiliary_loss.backward()
                del output, auxiliary_loss
                optimizer.step()
                count += len(labels)
                steps += 1
                total_loss += loss_value * len(labels)
            scheduler.step()
            assert count == len(loader.dataset) and steps == len(loader)
            epochs.append({"epoch": epoch + 1, "real_samples": count, "aux_samples": count,
                           "optimizer_steps": steps, "real_order_sha256": order_hash.hexdigest(),
                           "real_view_sha256": view_hash.hexdigest(), "aux_schedule_sha256": aux_hash.hexdigest(),
                           "learning_rates_before_epoch": lr_before,
                           "learning_rates_after_epoch": [g["lr"] for g in optimizer.param_groups],
                           "loss": total_loss / count})
            write_json(path, {"task": task, "branch": self.branch, "epochs": epochs,
                              "real_aux_loss_weights": [0.5, 0.5], "SCC_source": "real_anchor_only",
                              "stored_statistics_source": "real_train_only",
                              "generated_statistics_count": 0,
                              "anchor_batch_size": self.batch_size,
                              "effective_forward_images_per_full_step": self.batch_size * 2,
                              "sequential_gradient_accumulation": "two backward calls, one optimizer step",
                              "strong_transform": str(loader.dataset.strong)})
            self.save_experiment_checkpoint(self.run_dir / "checkpoint_latest.pt", stage="train_epoch",
                                            next_epoch=epoch + 1, optimizer=optimizer, scheduler=scheduler)
            progress("training_smoke" if self.args.get("fulltraj_smoke") else "training", branch=self.branch, task=task, epoch=epoch + 1,
                     total_epochs=self.tuned_epochs, loss=round(total_loss / count, 6))

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        task = self._cur_task + 1
        torch.manual_seed(1993 + task * 100000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1993 + task * 100000)
        progress("Gaussian_CA", branch=self.branch, task=task, epochs=ca_epochs,
                 replay_samples_per_class=self.args["replay_samples_per_class"])
        return super()._stage2_compact_classifier(task_size, ca_epochs)

    def release_synthetic_access(self):
        if self._active_views is not None:
            self._active_views.release()
        self._active_views = None
        self.current_bank.clear()
        assert not self.current_bank


def assert_matched_logs(left, right):
    assert left["task"] == right["task"]
    keys = ("real_samples", "aux_samples", "optimizer_steps", "real_order_sha256", "real_view_sha256",
            "learning_rates_before_epoch", "learning_rates_after_epoch")
    assert len(left["epochs"]) == len(right["epochs"])
    for a, b in zip(left["epochs"], right["epochs"]):
        for key in keys:
            assert a[key] == b[key], f"matched acquisition failed at epoch {a['epoch']}: {key}"
    assert left["generated_statistics_count"] == right["generated_statistics_count"] == 0
