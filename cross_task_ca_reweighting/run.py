"""One-shot T10 cross-task CA objective pilot; no backbone or source-file updates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = ROOT.parent
AUDIT_ROOT = HOME / "structure_prototype_oracle_audit"
TRAJECTORY_ROOT = HOME / "error_trajectory_diagnostic"
PROTOCOLS = ("IA10", "CAR10")
SEED = 1993
EXPECTED_B1 = {"IA10": 126, "CAR10": 473}
COEFFICIENTS = {"control": 1, "intervention": 2}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_paths(protocol: str) -> dict[str, Path]:
    training = AUDIT_ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{SEED}"
    return {
        "checkpoint": training / "task_checkpoints/task_10.pt",
        "sidecar": training / "pre_ca_classifier/task_10.pt",
        "training_config": training / "resolved_config.json",
        "protocol_manifest": training / "protocol_manifest.json",
        "class_order": training / "class_order.json",
        "task_summary": training / "task_summary.csv",
        "snapshot_manifest": training / "task_checkpoints/manifest.json",
        "feature_cache": AUDIT_ROOT / "ten_task_hypothesis_revalidation" / protocol /
                         ".cache/model_10_on_task_10.npz",
        "trajectory": TRAJECTORY_ROOT / protocol / "sample_trajectory.csv",
        "trajectory_manifest": TRAJECTORY_ROOT / "run_manifest.json",
    }


def preflight(protocol: str) -> tuple[dict[str, Path], dict]:
    paths = source_paths(protocol)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{protocol}: required {name} missing: {path}")
    config = json.loads(paths["training_config"].read_text(encoding="utf-8"))
    protocol_manifest = json.loads(paths["protocol_manifest"].read_text(encoding="utf-8"))
    order = json.loads(paths["class_order"].read_text(encoding="utf-8"))
    snapshot = json.loads(paths["snapshot_manifest"].read_text(encoding="utf-8"))
    trajectory = json.loads(paths["trajectory_manifest"].read_text(encoding="utf-8"))
    if trajectory["protocols"][protocol]["status"] != "complete":
        raise ValueError(f"{protocol}: fixed trajectory diagnostic is incomplete")
    if int(config["seed"]) != SEED or config["protocol_id"] != protocol or config["method"] != "Structure":
        raise ValueError(f"{protocol}: source is not seed-1993 vanilla Structure")
    if config["experiment_id"] != "Structure_AUDIT" or config["ca"] is not True:
        raise ValueError(f"{protocol}: source is not CA-enabled Structure_AUDIT")
    if len(config["task_partition"]) != 10 or int(config["expected_tasks"]) != 10:
        raise ValueError(f"{protocol}: not a ten-task source")
    if list(map(int, protocol_manifest["task_partition"])) != list(map(int, config["task_partition"])):
        raise ValueError(f"{protocol}: protocol manifest and resolved config partitions differ")
    if list(map(int, protocol_manifest["class_order_original_ids"])) != list(map(
        int, order["class_order_original_ids"]
    )):
        raise ValueError(f"{protocol}: source class order records disagree")
    if snapshot.get("training_behavior_modified") is not False:
        raise ValueError(f"{protocol}: source snapshot altered training behavior")
    entries = snapshot.get("entries", [])
    if len(entries) != 10 or [int(row["task_id"]) for row in entries] != list(range(1, 11)):
        raise ValueError(f"{protocol}: incomplete T1..T10 source sequence")
    last = entries[-1]
    if (last["state_id"] != "task_end:task_10" or
            Path(last["path"]).resolve() != paths["checkpoint"].resolve() or
            Path(last["pre_ca_classifier_sidecar"]).resolve() != paths["sidecar"].resolve()):
        raise ValueError(f"{protocol}: T10 checkpoint/sidecar do not belong to source baseline")
    return paths, config


def load_inputs(protocol: str, paths: dict[str, Path], config: dict):
    import numpy as np
    import torch
    from analysis import read_csv
    from oracle_audit.oracle_head import checkpoint_classifier_weights, load_classifier_initialization
    from oracle_audit.runtime import torch_load

    checkpoint = torch_load(paths["checkpoint"])
    with paths["task_summary"].open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    original_t10 = [row for row in original_rows if int(row["task_id"]) == 10 and
                    row["eval_stage"] == "task_final"]
    if len(original_t10) != 1:
        raise ValueError("source task summary has no unique original T10 evaluation")
    partition = [int(value) for value in config["task_partition"]]
    classes = sum(partition)
    if checkpoint.get("stage") != "task_end" or checkpoint.get("state_id") != "task_end:task_10":
        raise ValueError("T10 checkpoint is not the task-end state")
    if int(checkpoint["total_classes"]) != classes or list(map(int, checkpoint["task_sizes"])) != partition:
        raise ValueError("checkpoint classes do not match T10 task partition")
    for key in ("seed", "protocol_id", "experiment_id", "method", "task_partition"):
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"checkpoint config {key} differs from resolved source config")
    if checkpoint["config"].get("logit") not in (None, False):
        raise ValueError("unexpected logit-normalization branch; original CA loss differs")
    initial_weights, sidecar, source = load_classifier_initialization(
        checkpoint, paths["checkpoint"], np.arange(classes), 10
    )
    if sidecar is None or source != "exact_pre_ca_sidecar":
        raise ValueError("exact T10 pre-CA classifier sidecar is required")
    if int(sidecar["task_id"]) != 10 or list(map(int, sidecar["task_sizes"])) != partition:
        raise ValueError("pre-CA sidecar has a different task layout")
    if int(sidecar["replay_samples_per_class"]) != 256 or int(sidecar["ca_epochs"]) != 5:
        raise ValueError("unexpected Structure CA exposure/epochs")
    if "rng_state_before_ca" not in sidecar:
        raise ValueError("pre-CA RNG state required to replay the original Gaussian stream")
    with np.load(paths["feature_cache"], allow_pickle=False) as cache:
        for key in ("test_features", "test_labels", "test_sample_ids", "model_task", "class_task"):
            if key not in cache.files:
                raise ValueError(f"T10 feature cache lacks {key}")
        if int(cache["model_task"].item()) != 10 or int(cache["class_task"].item()) != 10:
            raise ValueError("feature cache is not final-T10-on-T10")
        test_features = np.asarray(cache["test_features"], dtype=np.float32).copy()
        test_labels = np.asarray(cache["test_labels"], dtype=np.int64).copy()
        test_ids = np.asarray(cache["test_sample_ids"]).copy()
    if test_features.ndim != 2 or test_features.shape[0] != len(test_labels):
        raise ValueError("invalid real-test feature matrix")
    if len(test_ids) != len(test_labels) or len(set(map(str, test_ids))) != len(test_ids):
        raise ValueError("test sample IDs absent or duplicated")
    if int(original_t10[0]["num_samples"]) != len(test_ids):
        raise ValueError("original vanilla T10 sample count differs from real test cache")
    if np.any(~np.isfinite(test_features)) or test_labels.min() < 0 or test_labels.max() >= classes:
        raise ValueError("invalid test features or class labels")
    trajectory_rows = read_csv(paths["trajectory"])
    if len(trajectory_rows) != len(test_ids):
        raise ValueError("trajectory and final feature cache have different test sample counts")
    baseline_by_id = {row["sample_id"]: row for row in trajectory_rows}
    if len(baseline_by_id) != len(test_ids):
        raise ValueError("duplicate trajectory sample ID")
    if set(baseline_by_id) != set(map(str, test_ids)):
        raise ValueError("T10 feature cache and trajectory IDs do not match exactly")
    b1_mask = np.zeros(len(test_ids), dtype=bool)
    trajectory_prediction = np.empty(len(test_ids), dtype=np.int64)
    for index, (sample_id, label) in enumerate(zip(test_ids, test_labels)):
        row = baseline_by_id[str(sample_id)]
        if int(row["true_label"]) != int(label) or row["protocol"] != protocol:
            raise ValueError(f"trajectory label/protocol mismatch for {sample_id}")
        b1_mask[index] = (row["error_type"] == "B_birth_correct_final_wrong" and
                          row["birth_restriction_restored"] == "1")
        trajectory_prediction[index] = int(row["pred_T10"])
    if int(b1_mask.sum()) != EXPECTED_B1[protocol]:
        raise AssertionError(f"{protocol}: fixed B1 count {b1_mask.sum()} != {EXPECTED_B1[protocol]}")
    means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float64).copy()
    covariances = [value.detach().cpu().clone() for value in checkpoint["class_covs"][:classes]]
    final_weights = checkpoint_classifier_weights(checkpoint)
    if initial_weights.shape != final_weights.shape or initial_weights.shape[0] != classes:
        raise ValueError("pre-CA/final head row layout mismatch")
    if test_features.shape[1] != initial_weights.shape[1] or means.shape != tuple(initial_weights.shape):
        raise ValueError("feature/head/statistics dimensionality mismatch")
    if len(covariances) != classes:
        raise ValueError("stored Gaussian covariance missing for a class")
    del checkpoint
    return {
        "partition": partition, "initial_weights": initial_weights,
        "final_weights": final_weights, "sidecar": sidecar,
        "means": means, "covariances": covariances,
        "test_features": test_features, "test_labels": test_labels, "test_ids": test_ids,
        "b1_mask": b1_mask, "trajectory_prediction": trajectory_prediction,
        "original_vanilla_correct_n": int(original_t10[0]["num_correct"]),
        "original_vanilla_accuracy": float(original_t10[0]["num_correct"]) / len(test_ids),
    }


def _epoch_bank(data: dict, device):
    """Mirror vanilla Structure base.py Gaussian draw + CPU randperm exactly."""
    import torch

    classes = sum(data["partition"])
    sidecar = data["sidecar"]
    cap = int(sidecar["replay_samples_per_class"])
    task_size_argument = int(sidecar["task_size_argument"])
    task_index = 9
    pieces = []
    for class_id in range(classes):
        # The division by *final task size* follows Structure's CA implementation,
        # including CAR10's unequal task sizes. Loss groups use actual partition.
        task_id_for_mean_scale = class_id // task_size_argument
        mean_scale = 0.9 + (task_id_for_mean_scale + 1) / (task_index + 1) * 0.1
        mean = torch.tensor(data["means"][class_id], dtype=torch.float64).to(device) * mean_scale
        covariance = data["covariances"][class_id].to(device)
        distribution = torch.distributions.MultivariateNormal(mean.float(), covariance.float())
        pieces.append(distribution.sample((cap,)))
    features = torch.cat(pieces, dim=0).float().to(device)
    labels = torch.arange(classes, device=device).repeat_interleave(cap)
    permutation = torch.randperm(features.size(0))  # original CPU-generated permutation
    features = features[permutation]
    labels = labels[permutation]
    return features, labels


def _tensor_hash(tensor) -> str:
    import numpy as np

    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def train_pair(data: dict, device) -> tuple[dict, list[dict], list[dict]]:
    import torch
    from oracle_audit.oracle_head import DiagnosticCosineHead, _capture_rng, _restore_rng
    from losses import assert_ce_identity, ca_loss, class_birth_tasks

    sidecar = data["sidecar"]
    epochs = int(sidecar["ca_epochs"])
    cap = int(sidecar["replay_samples_per_class"])
    classes = sum(data["partition"])
    birth = class_birth_tasks(data["partition"], device=device)
    heads = {
        name: DiagnosticCosineHead(data["initial_weights"], task_sizes=data["partition"]).to(device)
        for name in COEFFICIENTS
    }
    if not torch.equal(heads["control"].weight.detach(), heads["intervention"].weight.detach()):
        raise AssertionError("the two CA branches did not receive the same initial head")
    optimizers = {
        name: torch.optim.SGD(head.parameters(), lr=float(sidecar["init_lr"]),
                              momentum=0.9, weight_decay=float(sidecar["weight_decay"]))
        for name, head in heads.items()
    }
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for name, optimizer in optimizers.items()
    }
    for head in heads.values():
        head.eval()  # original CA runs the network in eval mode
    fixed_logits = torch.linspace(-2, 2, steps=classes * 3, device=device).view(3, classes)
    fixed_targets = torch.tensor([0, classes // 2, classes - 1], dtype=torch.long, device=device)
    initial_identity_delta = assert_ce_identity(fixed_logits, fixed_targets, birth)
    saved_rng = _capture_rng()
    trace: list[dict] = []
    bank_rows: list[dict] = []
    total_steps = {name: 0 for name in COEFFICIENTS}
    first_real_identity_delta = None
    try:
        original_rng = sidecar["rng_state_before_ca"]
        if torch.cuda.is_available() and len(original_rng.get("cuda", [])) != torch.cuda.device_count():
            raise ValueError("pre-CA CUDA RNG device count differs from current visible devices")
        _restore_rng(original_rng)
        for epoch in range(epochs):
            inputs, targets = _epoch_bank(data, device)
            if inputs.shape != (classes * cap, data["initial_weights"].shape[1]):
                raise AssertionError("Gaussian bank shape changed")
            label_counts = torch.bincount(targets, minlength=classes)
            if not torch.all(label_counts == cap):
                raise AssertionError("Gaussian exposure is not 256 per class")
            bank_rows.append({
                "epoch": epoch + 1, "sample_n": int(targets.numel()),
                "features_sha256": _tensor_hash(inputs), "labels_sha256": _tensor_hash(targets),
                "shared_by_branches": True,
            })
            running = {name: {"loss": 0.0, "within": 0.0, "cross": 0.0} for name in COEFFICIENTS}
            for step in range(classes):
                start, stop = step * cap, (step + 1) * cap
                batch_features, batch_labels = inputs[start:stop], targets[start:stop]
                if epoch == 0 and step == 0:
                    with torch.no_grad():
                        first_logits = float(sidecar["scale"]) * heads["control"](batch_features)
                        first_real_identity_delta = assert_ce_identity(first_logits, batch_labels, birth)
                for name, coefficient in COEFFICIENTS.items():
                    logits = float(sidecar["scale"]) * heads[name](batch_features)
                    loss, components = ca_loss(logits, batch_labels, birth, coefficient)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite {name} CA loss at epoch {epoch+1} step {step+1}")
                    optimizers[name].zero_grad()
                    loss.backward()
                    optimizers[name].step()
                    running[name]["loss"] += float(loss.detach().cpu())
                    running[name]["within"] += components["within"]
                    running[name]["cross"] += components["cross"]
                    total_steps[name] += 1
            for name in COEFFICIENTS:
                schedulers[name].step()
                trace.append({
                    "branch": name, "epoch": epoch + 1, "optimizer_steps_cumulative": total_steps[name],
                    "loss_mean": running[name]["loss"] / classes,
                    "within_mean": running[name]["within"] / classes,
                    "cross_mean": running[name]["cross"] / classes,
                    "learning_rate_after_epoch": float(optimizers[name].param_groups[0]["lr"]),
                    "bank_features_sha256": bank_rows[-1]["features_sha256"],
                    "bank_labels_sha256": bank_rows[-1]["labels_sha256"],
                })
            if optimizers["control"].param_groups[0]["lr"] != optimizers["intervention"].param_groups[0]["lr"]:
                raise AssertionError("branch learning-rate schedules diverged")
            print(f"CA epoch {epoch+1}/{epochs}: Control {running['control']['loss']/classes:.5f}, "
                  f"Intervention {running['intervention']['loss']/classes:.5f}", flush=True)
            # Vanilla CA evaluates the (deterministic) test DataLoader after
            # every epoch. Creating that iterator consumes one CPU torch RNG
            # draw for its worker base seed, before the next epoch's randperm.
            # We reproduce only the RNG side effect, not test-based feedback.
            bank_rows[-1]["original_eval_loader_base_seed_draw"] = int(
                torch.empty((), dtype=torch.int64).random_().item()
            )
            del inputs, targets
        expected_steps = classes * epochs
        if any(value != expected_steps for value in total_steps.values()):
            raise AssertionError("branch optimizer step counts differ")
        if not all(torch.isfinite(head.weight).all() for head in heads.values()):
            raise FloatingPointError("nonfinite trained classifier weights")
        result = {
            name: head.weight.detach().cpu().float().numpy().copy()
            for name, head in heads.items()
        }
        result["identity_synthetic_max_abs"] = initial_identity_delta
        result["identity_first_real_max_abs"] = first_real_identity_delta
        result["optimizer_steps"] = total_steps
        return result, trace, bank_rows
    finally:
        _restore_rng(saved_rng)


def score_head(features, weights, task_sizes, device, batch_size: int):
    import numpy as np
    import torch
    from oracle_audit.oracle_head import DiagnosticCosineHead

    head = DiagnosticCosineHead(torch.as_tensor(weights), task_sizes=task_sizes).to(device).eval()
    score_blocks = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            score_blocks.append(head(batch).detach().cpu().numpy())
    scores = np.concatenate(score_blocks)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite real-test scores")
    return scores


def evaluate_protocol(protocol: str, paths: dict[str, Path], config: dict, device,
                      options, manifest: dict) -> None:
    import numpy as np
    import torch
    from analysis import make_tables, write_csv

    output = ROOT / protocol
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    manifest["protocols"][protocol] = {
        "status": "running", "started_utc": utc_now(), "source_paths": {k: str(v) for k, v in paths.items()},
        "source_sha256": source_hashes, "source_experiment_id": config["experiment_id"],
        "source_training_git_commit": config.get("git_commit"),
        "source_snapshot_hash_at_training": json.loads(
            paths["protocol_manifest"].read_text(encoding="utf-8")
        ).get("source_snapshot_hash"),
        "task_partition": list(map(int, config["task_partition"])),
        "seed": SEED, "bank_mode": "original_all_class_Gaussian_per_epoch",
        "coefficients": COEFFICIENTS, "test_used_for_selection": False,
    }
    write_json(ROOT / "run_manifest.json", manifest)
    data = load_inputs(protocol, paths, config)
    print(f"[{protocol}] fixed B1 IDs: {int(data['b1_mask'].sum())}; "
          f"real test N={len(data['test_ids'])}", flush=True)
    trained, trace, bank_rows = train_pair(data, device)
    write_csv(output / "ca_epoch_trace.csv", [{"protocol": protocol, **row} for row in trace])
    write_json(output / "bank_manifest.json", {
        "bank_mode": "original_all_class_Gaussian_per_epoch", "rng_source": str(paths["sidecar"]),
        "note": "One bank was sampled per epoch and fed to both heads lockstep; "
                "bank hashes, not old feature exemplars, are persisted.",
        "epochs": bank_rows,
    })
    np.savez_compressed(output / "head_weights.npz",
                        initial=data["initial_weights"].numpy(),
                        control=trained["control"], intervention=trained["intervention"])
    baseline_scores = score_head(data["test_features"], data["final_weights"],
                                 data["partition"], device, options.batch_size)
    baseline_prediction = np.argmax(baseline_scores, axis=1)
    baseline_mismatch = int(np.sum(baseline_prediction != data["trajectory_prediction"]))
    if baseline_mismatch > 2:
        raise AssertionError(f"{protocol}: cached final features disagree with trajectory T10 on "
                             f"{baseline_mismatch} samples (>2); stop before outcome analysis")
    control_scores = score_head(data["test_features"], trained["control"],
                                data["partition"], device, options.batch_size)
    intervention_scores = score_head(data["test_features"], trained["intervention"],
                                     data["partition"], device, options.batch_size)
    paired = make_tables(protocol, data["test_ids"], data["test_labels"],
                         control_scores, intervention_scores, baseline_scores,
                         data["b1_mask"], data["partition"], output)
    original_final = data["final_weights"].numpy()
    head_abs_delta = float(np.max(np.abs(trained["control"] - original_final)))
    paired["audit"] = {
        "original_training_task_summary_correct_n": data["original_vanilla_correct_n"],
        "original_training_task_summary_accuracy": data["original_vanilla_accuracy"],
        "cached_vanilla_vs_original_training_accuracy_delta_pp": 100 * (
            paired["branch"]["vanilla_cached"]["accuracy"] - data["original_vanilla_accuracy"]),
        "trajectory_baseline_vs_cached_final_head_prediction_mismatch_n": baseline_mismatch,
        "matched_control_vs_vanilla_cached_accuracy_delta_pp": 100 * (
            paired["branch"]["control"]["accuracy"] - paired["branch"]["vanilla_cached"]["accuracy"]),
        "matched_control_vs_original_final_head_weight_max_abs": head_abs_delta,
        "ce_identity_synthetic_max_abs": trained["identity_synthetic_max_abs"],
        "ce_identity_first_real_max_abs": trained["identity_first_real_max_abs"],
        "optimizer_steps": trained["optimizer_steps"],
        "task_partition": data["partition"],
        "ca_epochs": int(data["sidecar"]["ca_epochs"]),
        "samples_per_class_per_epoch": int(data["sidecar"]["replay_samples_per_class"]),
        "original_ca_mean_scale_task_size_argument": int(data["sidecar"]["task_size_argument"]),
    }
    if sha256(paths["checkpoint"]) != source_hashes["checkpoint"]:
        raise RuntimeError("source T10 checkpoint changed during the pilot")
    write_json(output / "summary.json", paired)
    manifest["protocols"][protocol]["status"] = "complete"
    manifest["protocols"][protocol]["completed_utc"] = utc_now()
    manifest["protocols"][protocol]["summary_path"] = str(output / "summary.json")
    write_json(ROOT / "run_manifest.json", manifest)
    print(f"[{protocol}] complete: Control {paired['overall']['control_accuracy']:.4%}, "
          f"Intervention {paired['overall']['intervention_accuracy']:.4%}", flush=True)
    del data, trained
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int, help="physical GPU ID chosen by the user")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="frozen real-test feature scoring batch size; CA batch remains fixed at 256")
    options = parser.parse_args()
    if options.gpu < 0 or options.batch_size < 1:
        parser.error("GPU ID must be nonnegative and scoring batch size positive")
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before GPU isolation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
    sys.path.insert(0, str(AUDIT_ROOT))
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"physical GPU {options.gpu} unavailable")
    preflighted = {protocol: preflight(protocol) for protocol in PROTOCOLS}
    device = torch.device("cuda:0")
    manifest = {
        "experiment": "Cross-Task CA Reweighting Pilot", "seed": SEED,
        "physical_gpu": options.gpu, "created_utc": utc_now(),
        "protocol_order": list(PROTOCOLS), "protocols": {},
        "single_fixed_coefficient": 2, "full_method_ready": False,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "run_manifest.json", manifest)
    try:
        for protocol in PROTOCOLS:
            paths, config = preflighted[protocol]
            try:
                evaluate_protocol(protocol, paths, config, device, options, manifest)
            except Exception as error:
                record = manifest["protocols"].setdefault(protocol, {})
                record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                write_json(ROOT / "run_manifest.json", manifest)
                raise
        from report import create_report
        create_report(ROOT, manifest)
        print(f"Final report: {ROOT / 'final_report.md'}", flush=True)
    finally:
        write_json(ROOT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
