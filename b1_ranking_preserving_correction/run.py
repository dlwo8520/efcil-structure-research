"""Final-T10 ranking-preserving correction pilot; runs IA10 then CAR10 on --gpu."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME_ROOT = ROOT.parent
SOURCE_ROOT = HOME_ROOT / "cross_task_ca_reweighting"
AUDIT_ROOT = HOME_ROOT / "structure_prototype_oracle_audit"
PROTOCOLS = ("IA10", "CAR10")
EXPECTED_B1 = {"IA10": 126, "CAR10": 473}
SEED = 1993
FIT_SEED = 20261008
SAMPLES_PER_CLASS = 256


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def class_gaussian_logits(data: dict, device, bank_seed: int, chunk_size: int = 256):
    """Original Structure Gaussian law, independently drawn fit/check, no raw replay saved."""
    import numpy as np
    import torch
    from oracle_audit.oracle_head import DiagnosticCosineHead

    classes = sum(data["partition"])
    dimensions = data["final_weights"].shape[1]
    scale = float(data["sidecar"]["scale"])
    task_size_argument = int(data["sidecar"]["task_size_argument"])
    head = DiagnosticCosineHead(data["final_weights"], data["partition"]).to(device).eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    fit_logits = np.empty((classes * SAMPLES_PER_CLASS, classes), dtype=np.float32)
    check_logits = np.empty_like(fit_logits)
    labels = np.repeat(np.arange(classes, dtype=np.int64), SAMPLES_PER_CLASS)
    fit_hash, check_hash = hashlib.sha256(), hashlib.sha256()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(bank_seed)
        torch.cuda.manual_seed_all(bank_seed)
        with torch.inference_mode():
            for class_id in range(classes):
                mean_scale = 0.9 + (class_id // task_size_argument + 1) / 10 * 0.1
                mean = torch.as_tensor(data["means"][class_id], dtype=torch.float64, device=device)
                covariance = data["covariances"][class_id].to(device)
                if covariance.shape != (dimensions, dimensions):
                    raise ValueError(f"class {class_id}: bad covariance shape")
                distribution = torch.distributions.MultivariateNormal(
                    (mean * mean_scale).float(), covariance.float()
                )
                # One independent fit bank and one independent holdout bank.
                for target, digest in ((fit_logits, fit_hash), (check_logits, check_hash)):
                    start = class_id * SAMPLES_PER_CLASS
                    for offset in range(0, SAMPLES_PER_CLASS, chunk_size):
                        count = min(chunk_size, SAMPLES_PER_CLASS - offset)
                        features = distribution.sample((count,)).float()
                        digest.update(features.cpu().contiguous().numpy().tobytes())
                        scores = scale * head(features)
                        target[start + offset:start + offset + count] = scores.cpu().numpy()
                if (class_id + 1) % 20 == 0 or class_id + 1 == classes:
                    print(f"  Gaussian logit bank: {class_id + 1}/{classes} classes", flush=True)
    del head
    return fit_logits, check_logits, labels, {
        "seed": bank_seed, "samples_per_class_fit": SAMPLES_PER_CLASS,
        "samples_per_class_check": SAMPLES_PER_CLASS,
        "source": "checkpoint class_means/class_covs, original CA mean scaling; new classes Gaussian too",
        "fit_features_sha256": fit_hash.hexdigest(),
        "check_features_sha256": check_hash.hexdigest(),
        "raw_features_saved": False,
    }


def score_test(data: dict, device, batch_size: int):
    import numpy as np
    from cross_task_run import score_head

    cosine = score_head(data["test_features"], data["final_weights"],
                        data["partition"], device, batch_size)
    scores = np.asarray(cosine, dtype=np.float64) * float(data["sidecar"]["scale"])
    predictions = scores.argmax(axis=1)
    mismatch = int((predictions != data["trajectory_prediction"]).sum())
    if mismatch:
        raise AssertionError(f"cached vanilla vs original post-CA checkpoint: {mismatch} prediction mismatches")
    if int((predictions[data["b1_mask"]] == data["test_labels"][data["b1_mask"]]).sum()):
        raise AssertionError("C0 must be wrong on every fixed B1 sample")
    return scores, mismatch


def run_protocol(protocol: str, device, batch_size: int, run_config: dict) -> dict:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from cross_task_run import preflight, load_inputs
    from b1_ranking_preserving_correction.analysis import analyze, read_csv, birth_groups
    from b1_ranking_preserving_correction.core import (
        Correction, descriptors, fit, invariant, normalize_apply, normalize_fit,
    )

    start = time.perf_counter()
    source_paths, source_config = preflight(protocol)
    data = load_inputs(protocol, source_paths, source_config)
    if int(data["b1_mask"].sum()) != EXPECTED_B1[protocol]:
        raise AssertionError("fixed B1 count changed")
    baseline, mismatch = score_test(data, device, batch_size)
    classes = sum(data["partition"])
    print(f"[{protocol}] C0 parity exact; N={len(baseline)}, B1={EXPECTED_B1[protocol]}", flush=True)

    fit_bank, check_bank, labels, bank_audit = class_gaussian_logits(
        data, device, FIT_SEED + (0 if protocol == "IA10" else 1))
    birth_zero = torch.as_tensor(birth_groups(data["partition"]) - 1, dtype=torch.long, device=device)
    fit_scores = torch.as_tensor(fit_bank, dtype=torch.float64, device=device)
    check_scores = torch.as_tensor(check_bank, dtype=torch.float64, device=device)
    target = torch.as_tensor(labels, dtype=torch.long, device=device)
    raw_fit = descriptors(fit_scores, data["partition"])
    phi_fit, mean, std = normalize_fit(raw_fit)
    phi_check = normalize_apply(descriptors(check_scores, data["partition"]), mean, std)
    test_scores = torch.as_tensor(baseline, dtype=torch.float64, device=device)
    phi_test = normalize_apply(descriptors(test_scores, data["partition"]), mean, std)
    if not torch.isfinite(phi_fit).all() or not torch.isfinite(phi_check).all():
        raise FloatingPointError("nonfinite Gaussian descriptor")
    zero = Correction(len(data["partition"]), conditional=True).to(device)
    with torch.inference_mode():
        zero_test = zero(test_scores, phi_test, birth_zero)
        if not torch.equal(zero_test, test_scores):
            raise AssertionError("C2 zero correction differs from checkpoint C0 logits")
        zero_prediction = zero_test.argmax(1).cpu().numpy()
        if not np.array_equal(zero_prediction, baseline.argmax(1)):
            raise AssertionError("C2 zero correction differs from checkpoint C0 prediction")
    output = ROOT / protocol
    output.mkdir(parents=True, exist_ok=True)
    score_arrays = {"C0": baseline}
    fitting, checks, correction_stats = {}, {}, {}
    model_weights = {}
    for name, conditional in (("C1", False), ("C2", True)):
        clock = time.perf_counter()
        model, status = fit(fit_scores, phi_fit, target, birth_zero, conditional)
        with torch.inference_mode():
            fit_corrected = model(fit_scores, phi_fit, birth_zero)
            check_corrected = model(check_scores, phi_check, birth_zero)
            test_corrected = model(test_scores, phi_test, birth_zero)
            fit_ce = float(F.cross_entropy(fit_corrected, target))
            check_ce = float(F.cross_entropy(check_corrected, target))
            offsets = model.offsets(phi_test)
            correction_stats[name] = {"test_offset_max_abs": float(offsets.abs().max()),
                                      "test_offset_rms": float(offsets.square().mean().sqrt())}
            score_arrays[name] = test_corrected.cpu().numpy()
            invariant_info = invariant(test_scores, test_corrected, data["partition"])
        status.update({"fit_seconds": time.perf_counter() - clock, **invariant_info})
        fitting[name] = status
        checks[name] = {"fit_Gaussian_CE": fit_ce, "check_Gaussian_CE": check_ce}
        model_weights[name + "_b"] = model.b.detach().cpu().numpy()
        if conditional:
            model_weights[name + "_A"] = model.A.detach().cpu().numpy()
        print(f"[{protocol}] {name}: objective {status['initial_objective']:.6f} -> "
              f"{status['final_objective']:.6f}, closures {status['closure_calls']}, "
              f"converged={status['converged']}", flush=True)
    checks["C0"] = {"fit_Gaussian_CE": float(F.cross_entropy(fit_scores, target)),
                    "check_Gaussian_CE": float(F.cross_entropy(check_scores, target))}
    trajectory = read_csv(source_paths["trajectory"])
    metrics = analyze(protocol, data["test_ids"], data["test_labels"], data["partition"],
                      score_arrays, trajectory, output, EXPECTED_B1[protocol])
    weights_bytes = {}
    actual_weights_bytes = {}
    for name in ("C1", "C2"):
        coefficients = [model_weights[name + "_b"]]
        if name == "C2":
            coefficients.append(model_weights[name + "_A"])
        parameter_count = sum(array.size for array in coefficients) + 2 * mean.numel()
        weights_bytes[name] = 4 * parameter_count
        actual_weights_bytes[name] = 8 * parameter_count
    np.savez_compressed(output / "correction_parameters.npz",
                        normalization_mean=mean.detach().cpu().numpy(),
                        normalization_std=std.detach().cpu().numpy(),
                        birth_task_zero=birth_zero.cpu().numpy(),
                        **model_weights)
    metrics.update({
        "source_checkpoint": str(source_paths["checkpoint"]),
        "source_checkpoint_sha256": sha256(source_paths["checkpoint"]),
        "source_feature_cache_sha256": sha256(source_paths["feature_cache"]),
        "source_trajectory_sha256": sha256(source_paths["trajectory"]),
        "source_seed": SEED, "task_partition": data["partition"],
        "C0_cached_vanilla_prediction_mismatch_n": mismatch,
        "C2_zero_correction_exact_logit_parity": True,
        "bank": bank_audit, "fit": fitting, "gaussian_CE": checks,
        "correction_size": correction_stats,
        "additional_inference_storage_bytes_float32": weights_bytes,
        "saved_inference_storage_bytes_float64": actual_weights_bytes,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "runtime_seconds": time.perf_counter() - start,
        "test_used_for_fitting_or_selection": False,
    })
    from b1_ranking_preserving_correction.report import protocol_decision
    metrics["decision"] = protocol_decision(metrics)
    save_json(output / "metrics.json", metrics)
    print(f"[{protocol}] complete: C0={metrics['branches']['C0']['accuracy']:.4%}, "
          f"C1={metrics['branches']['C1']['accuracy']:.4%}, "
          f"C2={metrics['branches']['C2']['accuracy']:.4%}", flush=True)
    del data, fit_scores, check_scores, test_scores
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU number; set by the user")
    parser.add_argument("--batch-size", type=int, default=512, help="frozen real-test scoring batch")
    args = parser.parse_args()
    if args.gpu < 0 or args.batch_size < 1:
        parser.error("GPU must be nonnegative and batch size positive")
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before CUDA_VISIBLE_DEVICES isolation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(HOME_ROOT))
    sys.path.insert(0, str(SOURCE_ROOT))
    sys.path.insert(0, str(AUDIT_ROOT))
    # Give the existing runner an unambiguous name without importing it as
    # run.py (this script itself also has that basename).
    import importlib.util
    specification = importlib.util.spec_from_file_location("cross_task_run", SOURCE_ROOT / "run.py")
    if specification is None or specification.loader is None:
        raise ImportError("could not load source preflight/feature-cache helpers")
    module = importlib.util.module_from_spec(specification)
    sys.modules["cross_task_run"] = module
    specification.loader.exec_module(module)
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"physical GPU {args.gpu} is not available or not isolated")
    device = torch.device("cuda:0")
    # Check all source paths before creating any result file.
    for protocol in PROTOCOLS:
        module.preflight(protocol)
    config = {"experiment": "B1 ranking-preserving task correction", "seed": SEED,
              "physical_gpu": args.gpu, "protocols": list(PROTOCOLS),
              "fit_bank_seed_base": FIT_SEED, "samples_per_class_per_bank": SAMPLES_PER_CLASS,
              "branches": {"C0": "identity", "C1": "centered task bias b",
                           "C2": "centered linear task correction A phi + b"},
              "descriptor": "per-task [max, top1-top2 gap, logsumexp]; fit-bank z-score; clip [-5,5]",
              "regularization": {"lambda": 1.0, "eta": 0.001},
              "optimizer": {"name": "full-batch LBFGS", "lr": 1, "max_iter": 100,
                            "max_eval": 150, "history_size": 20, "line_search_fn": "strong_wolfe"},
              "fit_dtype": "float64", "inference": "global argmax over all classes",
              "new_classes": "Gaussian", "test_used_for_selection": False,
              "CA_rerun": False}
    ROOT.mkdir(parents=True, exist_ok=True)
    save_json(ROOT / "config.json", config)
    manifest = {"created_utc": timestamp(), "physical_gpu": args.gpu,
                "protocols": {key: {"status": "pending"} for key in PROTOCOLS}}
    save_json(ROOT / "run_manifest.json", manifest)
    results = {}
    try:
        for protocol in PROTOCOLS:
            manifest["protocols"][protocol] = {"status": "running", "started_utc": timestamp()}
            save_json(ROOT / "run_manifest.json", manifest)
            try:
                results[protocol] = run_protocol(protocol, device, args.batch_size, config)
            except Exception as error:
                manifest["protocols"][protocol].update({"status": "failed",
                    "error": f"{type(error).__name__}: {error}"})
                save_json(ROOT / "run_manifest.json", manifest)
                raise
            manifest["protocols"][protocol].update({"status": "complete",
                                                     "finished_utc": timestamp()})
            save_json(ROOT / "run_manifest.json", manifest)
        from b1_ranking_preserving_correction.report import render_report
        render_report(ROOT, results)
        manifest["status"] = "complete"
        manifest["finished_utc"] = timestamp()
        save_json(ROOT / "run_manifest.json", manifest)
        print(f"Final report: {ROOT / 'final_report.md'}", flush=True)
    finally:
        save_json(ROOT / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
