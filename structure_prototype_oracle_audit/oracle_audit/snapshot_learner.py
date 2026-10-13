from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from models.CDS_adapter import Learner as StructureLearner
from utils.experiment_io import capture_rng_state, utc_now_iso, write_json


class AuditSnapshotLearner(StructureLearner):
    """Vanilla Structure plus immutable audit observations.

    Training, APR, SCC, and classifier-adaptation math are unchanged.  The
    subclass only records the pre-CA head/RNG and retains immutable task-end
    checkpoints; the actual CA call is delegated to the original method.
    """

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        """Observe the exact pre-CA head/RNG without changing CA behavior."""
        network = self._network.module if hasattr(self._network, "module") else self._network
        sidecar_dir = self.run_dir / "pre_ca_classifier"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        target = sidecar_dir / f"task_{self._cur_task + 1:02d}.pt"
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "format_version": 1,
                "purpose": "post-hoc oracle-head fair initialization; not training input",
                "task_id": self._cur_task + 1,
                "cur_task": self._cur_task,
                "known_classes": self._known_classes,
                "total_classes": self._total_classes,
                "task_sizes": list(self.task_sizes),
                "task_size_argument": int(task_size),
                "classifier_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in network.fc.state_dict().items()
                },
                "rng_state_before_ca": capture_rng_state(),
                "ca_epochs": int(ca_epochs),
                "replay_samples_per_class": int(self.args["replay_samples_per_class"]),
                "init_lr": float(self.init_lr),
                "weight_decay": float(self.weight_decay),
                "scale": float(self.args["scale"]),
            },
            temporary,
        )
        os.replace(temporary, target)
        self._audit_pre_ca_sidecar = target
        return super()._stage2_compact_classifier(task_size, ca_epochs)

    def save_experiment_checkpoint(self, path, stage, next_epoch=0, optimizer=None, scheduler=None):
        state_id = super().save_experiment_checkpoint(
            path,
            stage=stage,
            next_epoch=next_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if stage != "task_end":
            return state_id

        source = Path(path).resolve()
        snapshot_dir = self.run_dir / "task_checkpoints"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        task_id = self._cur_task + 1
        target = snapshot_dir / f"task_{task_id:02d}.pt"
        if target.exists():
            previous = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
            if previous.get("state_id") != state_id:
                raise RuntimeError(f"refusing to replace mismatched task snapshot: {target}")
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)

        manifest_path = snapshot_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "format_version": 1,
                "purpose": "read-only post-hoc oracle audit",
                "training_behavior_modified": False,
                "entries": [],
            }
        entry = {
            "task_id": task_id,
            "state_id": state_id,
            "path": str(target),
            "bytes": target.stat().st_size,
            "network_flag_after_task": int(self._network.convnet.flag),
            "pre_ca_classifier_sidecar": (
                str(self._audit_pre_ca_sidecar)
                if getattr(self, "_audit_pre_ca_sidecar", None) is not None
                else None
            ),
            "created_at_utc": utc_now_iso(),
        }
        manifest["entries"] = [
            row for row in manifest["entries"] if int(row["task_id"]) != task_id
        ] + [entry]
        manifest["entries"].sort(key=lambda row: int(row["task_id"]))
        write_json(manifest_path, manifest)
        return state_id
