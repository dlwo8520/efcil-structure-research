"""Birth-time support association from existing frozen feature caches."""
from __future__ import annotations

import numpy as np

from .common import (AUDIT, ROOT, checkpoint_path, source_config, read_csv, write_csv,
                     write_json, sha256, data_manager)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def top_support(query, bank, k, exclude_diagonal=False):
    similarity = query @ bank.T
    if exclude_diagonal:
        np.fill_diagonal(similarity, -np.inf)
    return np.partition(similarity, -k, axis=1)[:, -k:].mean(axis=1)


def actual_train_bank(protocol, task, lower, upper, device):
    """Birth cache currently holds test only; extract real current train locally."""
    from oracle_audit.runtime import torch_load, load_frozen_learner, extract_split, release_learner
    target = ROOT / "audit" / protocol / ".cache" / f"birth_train_{task:02d}.npz"
    fingerprint = sha256(checkpoint_path(protocol, task))
    if target.exists():
        with np.load(target, allow_pickle=False) as values:
            if str(values["source_checkpoint_sha256"].item()) != fingerprint:
                raise RuntimeError("birth train cache source checkpoint changed")
            return values["train_features"], values["train_labels"]
    manager = data_manager(protocol)
    model = load_frozen_learner(torch_load(checkpoint_path(protocol, task)), device,
                                ROOT / "audit" / protocol / ".scratch" / f"T{task:02d}")
    features, labels, _ = extract_split(model, manager, np.arange(lower, upper), "train", 64, 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, train_features=features, train_labels=labels,
                        source_checkpoint_sha256=np.asarray(fingerprint))
    release_learner(model)
    return features, labels


def run_audit(protocol, device):
    import torch
    from oracle_audit.runtime import torch_load
    from oracle_audit.oracle_head import DiagnosticCosineHead, checkpoint_classifier_weights

    config = source_config(protocol)
    partition = config["task_partition"]
    trajectory = read_csv(ROOT.parent / "error_trajectory_diagnostic" / protocol / "sample_trajectory.csv")
    fixed_a = [r for r in trajectory if r["error_type"] == "A_persistent_wrong" and int(r["birth_task"]) < 10]
    expected = {"IA10": 327, "CAR10": 907}[protocol]
    if len(fixed_a) != expected:
        raise AssertionError(f"fixed retention Type A: {len(fixed_a)} != {expected}")
    by_id = {r["sample_id"]: r for r in trajectory}
    output, comparisons, manifests = [], [], []
    for task in range(1, 10):
        lower, upper = sum(partition[:task - 1]), sum(partition[:task])
        cache = AUDIT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / f"model_{task:02d}_on_task_{task:02d}.npz"
        with np.load(cache, allow_pickle=False) as values:
            for name in ("test_features", "test_labels", "test_sample_ids"):
                if name not in values:
                    raise RuntimeError(f"{cache} lacks {name}")
            if {"train_features", "train_labels"}.issubset(values.files):
                train_x, train_y = values["train_features"], values["train_labels"]
            else:
                train_x, train_y = actual_train_bank(protocol, task, lower, upper, device)
            test_x, test_y, test_ids = values["test_features"], values["test_labels"], values["test_sample_ids"]
        selected = (test_y >= lower) & (test_y < upper)
        x, y, ids = test_x[selected], test_y[selected], test_ids[selected]
        checkpoint = torch_load(checkpoint_path(protocol, task))
        head = DiagnosticCosineHead(checkpoint_classifier_weights(checkpoint), partition[:task]).to(device).eval()
        with torch.no_grad():
            scores = np.concatenate([head(torch.as_tensor(x[i:i + 512], device=device)).cpu().numpy()
                                     for i in range(0, len(x), 512)])
        local = lower + scores[:, lower:upper].argmax(1)
        pred = scores.argmax(1)
        records = [by_id[str(sample_id)] for sample_id in ids]
        if any(int(r["true_label"]) != int(label) or int(r["pred_Tbirth"]) != int(p)
               for r, label, p in zip(records, y, pred)):
            raise AssertionError(f"{protocol} T{task}: birth head/cache/trajectory parity failed")
        is_a = np.array([r["error_type"] == "A_persistent_wrong" for r in records])
        birth_correct = pred == y
        if task == 1 and (is_a & (local == y)).any():
            raise AssertionError("T1 cannot contain A-cross")
        all_rival = scores.copy()
        all_rival[np.arange(len(y)), y] = -np.inf
        within_rival = scores[:, lower:upper].copy()
        within_rival[np.arange(len(y)), y - lower] = -np.inf
        within_margin = scores[np.arange(len(y)), y] - within_rival.max(1)
        all_margin = scores[np.arange(len(y)), y] - all_rival.max(1)
        n = min(64, min(int((train_y == c).sum()) for c in range(lower, upper)))
        k = min(5, n - 1)
        for seed in (1993, 1994, 1995):
            if n < 2:
                for class_id in range(lower, upper):
                    comparisons.append({"protocol": protocol, "task": task, "y": class_id,
                                        "subsample_seed": seed, "support_valid": False,
                                        "A_n": int((is_a & (y == class_id)).sum()),
                                        "birth_correct_n": int((birth_correct & (y == class_id)).sum()),
                                        "has_class_matched_comparison": bool((birth_correct & (y == class_id)).any()),
                                        "A_minus_correct_own_support": None})
                for i in np.flatnonzero(is_a | birth_correct):
                    output.append({"protocol": protocol, "task": task, "sample_id": str(ids[i]),
                                   "y": int(y[i]), "group": "A" if is_a[i] else "birth_correct",
                                   "A_location": ("A-cross" if local[i] == y[i] else "A-within") if is_a[i] else "",
                                   "subsample_seed": seed, "support_valid": False, "bank_n_per_class": n, "k": max(0, k),
                                   "local_pred": int(local[i]), "all_pred": int(pred[i]),
                                   "head_margin_within": float(within_margin[i]), "head_margin_all": float(all_margin[i])})
                continue
            rng = np.random.RandomState(seed + task * 100)
            banks = [unit(train_x[rng.choice(np.flatnonzero(train_y == c), n, replace=False)])
                     for c in range(lower, upper)]
            normalized = unit(x)
            supports = np.stack([top_support(normalized, bank, k) for bank in banks], axis=1)
            own = supports[np.arange(len(y)), y - lower]
            rival_support = supports.copy()
            rival_support[np.arange(len(y)), y - lower] = -np.inf
            rival = rival_support.max(1)
            for class_id, bank in zip(range(lower, upper), banks):
                a_mask = is_a & (y == class_id)
                correct_mask = birth_correct & (y == class_id)
                loo = top_support(bank, bank, k, exclude_diagonal=True)
                comparisons.append({"protocol": protocol, "task": task, "y": class_id,
                                    "subsample_seed": seed, "support_valid": True, "A_n": int(a_mask.sum()),
                                    "birth_correct_n": int(correct_mask.sum()),
                                    "has_class_matched_comparison": bool(correct_mask.any()),
                                    "A_own_support": float(own[a_mask].mean()) if a_mask.any() else None,
                                    "correct_own_support": float(own[correct_mask].mean()) if correct_mask.any() else None,
                                    "A_minus_correct_own_support": float(own[a_mask].mean() - own[correct_mask].mean())
                                    if a_mask.any() and correct_mask.any() else None,
                                    "LOO_train_support_mean": float(loo.mean()),
                                    "LOO_train_support_q05": float(np.quantile(loo, .05)),
                                    "LOO_train_support_q50": float(np.quantile(loo, .5)),
                                    "LOO_train_support_q95": float(np.quantile(loo, .95)),
                                    **{f"{kind}_minus_correct_{field}": float(values[a_mask & (local == y if kind == 'A_cross' else local != y)].mean() - values[correct_mask].mean())
                                       if (a_mask & (local == y if kind == 'A_cross' else local != y)).any() and correct_mask.any() else None
                                       for kind in ("A_within", "A_cross")
                                       for field, values in (("own_support", own), ("rival_support", rival), ("support_margin", own - rival))}})
            for i in np.flatnonzero(is_a | birth_correct):
                output.append({"protocol": protocol, "task": task, "sample_id": str(ids[i]),
                               "y": int(y[i]), "group": "A" if is_a[i] else "birth_correct",
                               "A_location": ("A-cross" if local[i] == y[i] else "A-within") if is_a[i] else "",
                               "subsample_seed": seed, "support_valid": True, "bank_n_per_class": n, "k": k,
                               "own_support": float(own[i]), "rival_support": float(rival[i]),
                               "support_margin_within": float(own[i] - rival[i]),
                               "head_margin_within": float(within_margin[i]), "head_margin_all": float(all_margin[i]),
                               "local_pred": int(local[i]), "all_pred": int(pred[i])})
        manifests.append({"task": task, "cache": str(cache), "source_checkpoint_sha256": sha256(checkpoint_path(protocol, task)),
                          "bank_n_per_class": n, "k": max(0, k), "support_valid": n >= 2,
                          "A_n": int(is_a.sum()), "A_cross_n": int((is_a & (local == y)).sum()),
                          "A_within_n": int((is_a & (local != y)).sum()), "birth_prediction_mismatch": 0})
        print(f"[{protocol} birth audit] T{task}: A={is_a.sum()}, within={(is_a & (local != y)).sum()}, cross={(is_a & (local == y)).sum()}", flush=True)
        del head, checkpoint
    target = ROOT / "audit" / protocol
    if sum(r["A_n"] for r in manifests) != expected:
        raise AssertionError("birth caches do not cover every fixed Type A sample")
    write_csv(target / "birth_support_audit.csv", output)
    write_csv(target / "class_matched_support.csv", comparisons)
    valid = [r for r in comparisons if r["A_minus_correct_own_support"] is not None]
    summary = {"protocol": protocol, "fixed_A_n": expected, "tasks": manifests,
               "support_difference_macro_by_subsample_seed": {
                   str(seed): float(np.mean([r["A_minus_correct_own_support"] for r in valid if r["subsample_seed"] == seed]))
                   if any(r["subsample_seed"] == seed for r in valid) else None for seed in (1993, 1994, 1995)},
               "A_classes_without_birth_correct": sorted({r["y"] for r in comparisons if r["A_n"] and not r["has_class_matched_comparison"]}),
               "diagnostic_only": True, "generation_selection_uses_this_audit": False}
    summary["by_error_location"] = {}
    for kind in ("A-within", "A-cross"):
        own = [r for r in output if r.get("A_location") == kind and r["subsample_seed"] == 1993]
        valid_own = [r for r in own if r["support_valid"]]
        summary["by_error_location"][kind] = {"n": len(own), "support_valid_n": len(valid_own),
                "positive_support_margin_local_head_wrong_n": sum(r["support_margin_within"] > 0 and r["local_pred"] != r["y"] for r in valid_own)}
    write_json(target / "summary.json", summary)
    lines = [f"# {protocol}: Birth error location + real feature support audit", "",
             f"Fixed vanilla retention Type A: {expected}. Birth post-CA head; deterministic evaluation transform. "
             "Support is current-task actual train only (no Gaussian/old actual bank). Head margins are raw cosine scores. "
             "A-within/A-cross and class-matched support are associations, not proof of image coverage causality. "
             "This audit never selects generation parents or prompts.", "",
             "| Task | A | A-within | A-cross | Balanced n/class | k | Support valid |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for row in manifests:
        lines.append(f"| {row['task']} | {row['A_n']} | {row['A_within_n']} | {row['A_cross_n']} | {row['bank_n_per_class']} | {row['k']} | {row['support_valid']} |")
    lines += ["", "Macro class-matched A−birth-correct own support by subsample seed: "
              f"`{summary['support_difference_macro_by_subsample_seed']}`.", "",
              "Classes containing A but no birth-correct comparator (not silently counted as comparable): "
              f"`{summary['A_classes_without_birth_correct']}`.", "",
              "Error-location counts / positive-support versus local-head disagreement: "
              f"`{summary['by_error_location']}`.", "",
              "[Sample measurements](birth_support_audit.csv) · [Class-matched and leave-one-out reference](class_matched_support.csv) · [Summary](summary.json)"]
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
