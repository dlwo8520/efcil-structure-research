"""Original Structure optimizers/SCC/APR/CA with matched current auxiliary source."""
from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oracle_audit.snapshot_learner import AuditSnapshotLearner
from utils.loss import AngularPenaltySMLoss
from utils.distill_loss import LobachevskyRelativeGeomLoss
from utils.experiment_io import capture_rng_state, restore_rng_state
from data.data_manager import DummyDataset

from .common import write_json, read_json, run_dir
from .matched_views import MatchedViews
from .moments import mixed_moments


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_hash(state):
    digest = hashlib.sha256(state["torch"].numpy().tobytes())
    for value in state.get("cuda", []):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


class AugmentationLearner(AuditSnapshotLearner):
    def __init__(self, args, parents):
        super().__init__(args)
        self.parents = parents
        self.branch = args["augmentation_branch"]
        self._gm_state = None

    def save_experiment_checkpoint(self, *args, **kwargs):
        value = super().save_experiment_checkpoint(*args, **kwargs)
        manifest_path = self.run_dir / "task_checkpoints/manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifest.update({"training_behavior_modified": True,
                             "purpose": "matched auxiliary-image source pilot snapshots"})
            write_json(manifest_path, manifest)
        return value

    def load_experiment_checkpoint(self, path):
        # Only the experiment name changes; original compatibility checks remain.
        from oracle_audit.runtime import torch_load
        original_id = self.args["experiment_id"]
        checkpoint = torch_load(Path(path))
        self.args["experiment_id"] = checkpoint["config"]["experiment_id"]
        try:
            value = super().load_experiment_checkpoint(path)
        finally:
            self.args["experiment_id"] = original_id
        self._network.convnet.flag = int(value.get("network_flag_after_task", value["cur_task"] + 1))
        return value

    def _train(self, train_loader, test_loader):
        task = self._cur_task + 1
        augmented = task in (1, 9) and self.branch in ("R", "S", "G")
        own = [row for row in self.parents if int(row["task"]) == task]
        dataset = MatchedViews(train_loader.dataset, own, self.data_manager, self.branch, task, augmented)
        generator = torch.Generator().manual_seed(1993 + task * 10000)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=0, generator=generator, drop_last=False)
        return super()._train(loader, test_loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        task = self._cur_task + 1
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7,
                                       s=self.args["scale"], m=self.args["margin"])
        anchor_hash, parent_hash = hashlib.sha256(), hashlib.sha256()
        epoch_rows = []
        for epoch in range(start_epoch, self.tuned_epochs):
            train_loader.dataset.epoch = epoch
            self._network.train()
            samples, steps, forward_images, loss_sum = 0, 0, 0, 0.0
            for batch in train_loader:
                indices, anchor = batch[:2]
                if train_loader.dataset.augmented:
                    aux, targets, parent_ids = batch[2:]
                    for parent_id in parent_ids:
                        parent_hash.update(parent_id.encode() + b"\0")
                else:
                    targets = batch[2]
                    aux = None
                anchor_hash.update(indices.numpy().tobytes())
                anchor_hash.update(anchor.numpy().tobytes())
                targets = targets.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                output = self._network(anchor.to(self._device), label=targets)
                classification = loss_cos(output["logits"][:, self._known_classes:], targets - self._known_classes)
                loss = classification * (0.5 if aux is not None else 1.0)
                if self._cur_task > 0:
                    loss = loss + self.args["scc_value"] * LobachevskyRelativeGeomLoss(output, self._class_means, self._device)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite real-anchor loss")
                total = float(loss.detach())
                loss.backward()
                del output, loss
                if aux is not None:
                    output_aux = self._network(aux.to(self._device), label=targets)
                    aux_loss = 0.5 * loss_cos(output_aux["logits"][:, self._known_classes:], targets - self._known_classes)
                    if not torch.isfinite(aux_loss):
                        raise FloatingPointError("nonfinite auxiliary loss")
                    total += float(aux_loss.detach())
                    aux_loss.backward()
                    del output_aux, aux_loss
                optimizer.step()
                samples += len(targets)
                forward_images += len(targets) * (2 if aux is not None else 1)
                steps += 1
                loss_sum += total * len(targets)
            scheduler.step()
            if samples != len(train_loader.dataset) or steps != len(train_loader):
                raise AssertionError("matched update/exposure count failed")
            epoch_rows.append({"epoch": epoch + 1, "anchor_samples": samples, "forward_backward_images": forward_images,
                               "optimizer_steps": steps, "loss": loss_sum / samples,
                               "learning_rates_after_step": [g["lr"] for g in optimizer.param_groups]})
            print(f"[{self.args['protocol_id']} {self.branch}] T{task} epoch {epoch + 1}/{self.tuned_epochs} loss={loss_sum / samples:.4f}", flush=True)
        write_json(self.run_dir / f"matched_training_T{task:02d}.json",
                   {"task": task, "branch": self.branch, "anchor_sha256": anchor_hash.hexdigest(),
                    "parent_schedule_sha256": parent_hash.hexdigest(), "epochs": epoch_rows,
                    "optimizer": self.args["optimizer"], "batch_size_anchor": self.batch_size,
                    "effective_forward_batch": self.batch_size * (2 if train_loader.dataset.augmented else 1),
                    "auxiliary_transform": str(train_loader.dataset.strong if self.branch == "S" else train_loader.dataset.ordinary)
                    if train_loader.dataset.augmented else None,
                    "sequential_backward_accumulation": train_loader.dataset.augmented,
                    "SCC_source": "real anchor only", "runtime_train_epochs_completed": len(epoch_rows)})

    def mixed_statistics(self):
        means, covs = self._class_means.copy(), self._class_covs.detach().cpu().clone()
        records = []
        for label in range(self._known_classes, self._total_classes):
            real_dataset = self.data_manager.get_dataset(np.array([label]), source="train", mode="test")
            paths = [r["generated_path"] for r in self.parents if int(r["task"]) == self._cur_task + 1 and int(r["y"]) == label]
            gen_dataset = DummyDataset(np.asarray(paths), np.repeat(label, len(paths)), real_dataset.trsf, use_path=True)
            with torch.no_grad():
                real, _ = self._extract_vectors(DataLoader(real_dataset, batch_size=64, shuffle=False, num_workers=0))
                generated, _ = self._extract_vectors(DataLoader(gen_dataset, batch_size=64, shuffle=False, num_workers=0))
            if np.max(np.abs(real.mean(0) - self._class_means[label])) > 1e-5:
                raise AssertionError("mixed branch real feature mean differs from G real statistics")
            mean, covariance = mixed_moments(real, generated)
            combined = np.concatenate((real, generated)).astype(np.float64)
            covariance_error = float(np.max(np.abs(covariance - np.cov(combined, rowvar=False))))
            if covariance_error > 1e-8:
                raise AssertionError("pooled covariance missing between-source scatter")
            means[label] = mean
            covs[label] = torch.as_tensor(covariance) + torch.eye(covariance.shape[0]) * 1e-3
            records.append({"class": label, "real_n": len(real), "generated_n": len(generated),
                            "generated_fraction": len(generated) / (len(real) + len(generated)),
                            "mean_shift_l2": float(np.linalg.norm(means[label] - self._class_means[label])),
                            "covariance_shift_fro": float(torch.linalg.matrix_norm(covs[label] - self._class_covs[label].cpu())),
                            "pooled_vs_concatenated_covariance_max_abs": covariance_error})
        return means, covs, records

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        task = self._cur_task + 1
        # All branches use the same standard-normal stream shape/order and CA recipe.
        torch.manual_seed(1993 + task * 100000)
        torch.cuda.manual_seed_all(1993 + task * 100000)
        if self.branch != "G" or task != 9:
            return super()._stage2_compact_classifier(task_size, ca_epochs)
        self.save_experiment_checkpoint(self.run_dir / "pre_ca_bundle.pt", stage="pre_ca")
        initial_head = copy.deepcopy(self._network.fc.state_dict())
        initial_rng = capture_rng_state()
        real_means, real_covs = self._class_means.copy(), self._class_covs.clone()
        mixed_means, mixed_covs, records = self.mixed_statistics()
        # Feature extraction is deterministic but DataLoader advances CPU RNG.
        restore_rng_state(initial_rng)
        super()._stage2_compact_classifier(task_size, ca_epochs)
        g_head = copy.deepcopy(self._network.fc.state_dict())
        g_rng = capture_rng_state()
        self._network.fc.load_state_dict(initial_head)
        self._class_means, self._class_covs = mixed_means, mixed_covs
        restore_rng_state(initial_rng)
        # Original CA implementation; the second run differs only in current moments.
        original_dir, original_args = self.run_dir, self.args.copy()
        g_sidecar = self._audit_pre_ca_sidecar
        self.run_dir = run_dir(self.args["protocol_id"], "T9_T10", "GM")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.args = {**original_args, "experiment_id": "GM", "augmentation_branch": "GM", "run_dir": str(self.run_dir)}
        try:
            super()._stage2_compact_classifier(task_size, ca_epochs)
            gm_sidecar = self._audit_pre_ca_sidecar
        finally:
            self.run_dir, self.args = original_dir, original_args
            self._audit_pre_ca_sidecar = g_sidecar
        gm_rng = capture_rng_state()
        if rng_hash(g_rng) != rng_hash(gm_rng):
            raise AssertionError("G/GM CA random streams consumed different epsilon/order draws")
        self._gm_state = {"head": copy.deepcopy(self._network.fc.state_dict()),
                          "means": mixed_means, "covs": mixed_covs, "rng": gm_rng, "sidecar": gm_sidecar}
        self._network.fc.load_state_dict(g_head)
        self._class_means, self._class_covs = real_means, real_covs
        restore_rng_state(g_rng)
        write_json(self.run_dir / "statistics_comparison.json", {"task": task,
                   "same_pre_CA_head_sha256": state_hash(initial_head),
                   "same_epsilon_and_permutation_RNG_before_sha256": rng_hash(initial_rng),
                   "G_GM_rng_after_equal": True, "old_moments_identical": True,
                   "APR_displacement_source": "real", "current_class_changes": records})

    def prepare_t1_gm(self):
        means, covs, records = self.mixed_statistics()
        self._gm_state = {"head": copy.deepcopy(self._network.fc.state_dict()),
                          "means": means, "covs": covs, "rng": capture_rng_state()}
        write_json(self.run_dir / "statistics_comparison.json", {"task": 1,
                   "CA_skipped_as_in_original_T1": True, "same_representation_and_head": True,
                   "current_class_changes": records})
