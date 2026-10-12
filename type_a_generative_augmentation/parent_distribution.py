"""Train-only feature cluster diagnostic; never a parent-selection criterion."""
from __future__ import annotations

import numpy as np

from .common import ROOT, AUDIT, PROTOCOLS, data_manager, sample_identity, write_csv, write_json
from .audit import unit


def report_parent_distribution(parents):
    rows = []
    for protocol in PROTOCOLS:
        manager = data_manager(protocol)
        for task in (1, 9):
            lower, upper = sum(manager._increments[:task - 1]), sum(manager._increments[:task])
            local_cache = ROOT / "audit" / protocol / ".cache" / f"birth_train_{task:02d}.npz"
            source_cache = AUDIT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / f"model_{task:02d}_on_task_{task:02d}.npz"
            cache = local_cache if local_cache.exists() else source_cache
            with np.load(cache, allow_pickle=False) as bank:
                if not {"train_features", "train_labels"}.issubset(bank.files):
                    raise RuntimeError("run --stage audit before parent distribution; no new selection from this diagnostic")
                features, labels = bank["train_features"], bank["train_labels"]
            current = (labels >= lower) & (labels < upper)
            features, labels = features[current], labels[current]
            ids, expected = sample_identity(manager, np.arange(lower, upper), "train")
            if not np.array_equal(labels, expected):
                raise AssertionError("parent distribution feature/identity ordering differs")
            for label in range(lower, upper):
                mask = labels == label
                x, class_ids = unit(features[mask]), ids[mask]
                selected = {r["parent_id"] for r in parents if r["protocol"] == protocol and int(r["task"]) == task and int(r["y"]) == label}
                if not selected.issubset(set(map(str, class_ids))):
                    raise AssertionError("parent ID not in current real train")
                parent_mask = np.array([str(key) in selected for key in class_ids])
                k = min(4, len(x))
                rng = np.random.RandomState(1993 + task * 1000 + label)
                centers = x[rng.choice(len(x), k, replace=False)].copy()
                for _ in range(15):
                    assignments = (x @ centers.T).argmax(1)
                    for cluster in range(k):
                        if (assignments == cluster).any():
                            centers[cluster] = unit(x[assignments == cluster].mean(0, keepdims=True))[0]
                assignments = (x @ centers.T).argmax(1)
                for cluster in range(k):
                    cluster_mask = assignments == cluster
                    train_n, parent_n = int(cluster_mask.sum()), int((cluster_mask & parent_mask).sum())
                    rows.append({"protocol": protocol, "task": task, "y": label, "cluster": cluster,
                                 "real_class_n": len(x), "parent_class_n": int(parent_mask.sum()),
                                 "real_cluster_n": train_n, "parent_cluster_n": parent_n,
                                 "real_cluster_fraction": train_n / len(x), "parent_cluster_fraction": parent_n / parent_mask.sum(),
                                 "fraction_difference": parent_n / parent_mask.sum() - train_n / len(x)})
    write_csv(ROOT / "parent_distribution.csv", rows)
    write_json(ROOT / "parent_distribution_metadata.json", {"diagnostic_only": True, "used_for_selection": False,
               "feature_source": "vanilla birth post-CA current actual train", "clustering": "spherical kmeans K<=4, 15 fixed iterations",
               "interpretation": "descriptive coverage/overrepresentation, not a reason to replace parents or tune recipe"})
