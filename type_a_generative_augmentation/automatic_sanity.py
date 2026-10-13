"""Fixed, pair-only technical/duplicate/gross-deviation checks; NOT label QC."""
from __future__ import annotations

from collections import Counter
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import ROOT, DEFAULT_GENERATOR, PROTOCOLS, read_json, read_csv, write_json, write_csv, sha256


LIMITATION = ("Exploratory pilot without human QC. Generated labels are inherited from parents, not verified. "
              "Automatic similarity/structure checks cannot guarantee CAR model/year identity or complete object retention. "
              "No individual human choices or classifier confidence are used. Fixed-rule filtering may still introduce distribution bias. "
              "Downstream G-S evaluates this fixed generator + automatic-sanity recipe, not all generative augmentation.")


def write_report(summary):
    lines = ["# Automatic sanity — exploratory pilot", "", LIMITATION, "", "Status: " + summary["status"], "",
             "All candidates are assessed by the same frozen rules. Human review file and contact sheets do not affect selection. "
             "Excluded images remain on disk as artifacts; only their training eligibility changes. R/S/G share the same accepted parent set.", "",
             "| Protocol | Generated candidates | Accepted | Excluded | Reasons (non-exclusive) |", "|---|---:|---:|---:|---|"]
    for p, record in summary["protocols"].items():
        lines.append(f"| {p} | {record['total']} | {record['accepted']} | {record['excluded']} | {record['reason_counts_nonexclusive']} |")
    lines += ["", "## Pair similarity statistics", "",
              "Statistics include safety-blocked/black images in the all-decodable subset. High CLIP cosine does not certify fine-grained identity. "
              "Zero flags for a rule mean no threshold crossing, not that all objects are preserved.", "",
              "| Protocol | Subset | N | Global cosine mean | Global q05 | Center cosine mean | Block SSIM mean | Pixel MAE mean |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if "mean" not in x else f"{x['mean']:.5f}"
    for p, record in summary["protocols"].items():
        for name, stats in record["similarity"].items():
            q05 = "—" if "q05" not in stats["global_cosine"] else f"{stats['global_cosine']['q05']:.5f}"
            lines.append(f"| {p} | {name} | {stats['global_cosine']['n']} | {fmt(stats['global_cosine'])} | {q05} | "
                         f"{fmt(stats['center_cosine'])} | {fmt(stats['block_ssim'])} | {fmt(stats['mean_absolute_error'])} |")
    lines += ["", "Thresholds are exploratory heuristics fixed before downstream outcomes, not calibrated label/segmentation guarantees. "
              "A conservative global+center CLIP mismatch or center edge collapse may flag gross changes; subtler object deletion/identity changes can pass. "
              "Block SSIM is a custom non-overlapping 8x8 RGB score, not standard Gaussian-window SSIM.", "",
              "[Rules](sanity_policy.json) · [Per-image measurements](automatic_sanity.csv) · [Similarity statistics](automatic_sanity_summary.json) · "
              "[Class counts](automatic_parent_counts.csv) · [Contact sheets (artifact only)](contact_sheets/)"]
    (ROOT / "automatic_sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def settings():
    mode = read_json(ROOT / "exploratory_config.json")
    policy = read_json(ROOT / "sanity_policy.json")
    if mode["mode"] != "exploratory_automatic_sanity" or mode["label_fidelity_verified"]:
        raise RuntimeError("invalid exploratory mode")
    return mode, policy


def block_ssim(left, right):
    """Mean RGB SSIM of non-overlapping 8x8 blocks, data range 1; no Gaussian window."""
    a = left.reshape(64, 8, 64, 8, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 64).astype(np.float64)
    b = right.reshape(64, 8, 64, 8, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 64).astype(np.float64)
    am, bm = a.mean(1), b.mean(1)
    ac, bc = a - am[:, None], b - bm[:, None]
    av, bv, covariance = (ac * ac).mean(1), (bc * bc).mean(1), (ac * bc).mean(1)
    value = ((2 * am * bm + .01**2) * (2 * covariance + .03**2)) / ((am**2 + bm**2 + .01**2) * (av + bv + .03**2))
    return float(value.mean())


def edge_energy(array):
    gray = array.mean(2)
    return float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2)


def center(image, fraction):
    size = int(image.width * fraction)
    left = (image.width - size) // 2
    return image.crop((left, left, left + size, left + size))


def pixel_metrics(parent, generated, policy):
    p, g = np.asarray(parent, np.float32) / 255, np.asarray(generated, np.float32) / 255
    pc, gc = center(parent, policy["center_crop_fraction"]), center(generated, policy["center_crop_fraction"])
    pca, gca = np.asarray(pc, np.float32) / 255, np.asarray(gc, np.float32) / 255
    def dhash(image):
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        return gray[:, 1:] > gray[:, :-1]
    return {"mean_absolute_error": float(np.abs(p - g).mean()),
            "pixel_rmse": float(np.sqrt(((p - g)**2).mean())), "block_ssim": block_ssim(p, g),
            "dhash_distance": int(np.count_nonzero(dhash(parent) != dhash(generated))),
            "generated_channel_std": float(g.std(axis=(0, 1)).mean()), "generated_edge_energy": edge_energy(g),
            "parent_center_edge_energy": edge_energy(pca), "generated_center_edge_energy": edge_energy(gca),
            "center_edge_ratio": edge_energy(gca) / max(edge_energy(pca), 1e-8),
            "center_block_ssim": block_ssim(np.asarray(pc.resize((512, 512)), np.float32) / 255,
                                            np.asarray(gc.resize((512, 512)), np.float32) / 255)}


def exclusion_rules(metrics, policy):
    reasons = []
    duplicate = policy["near_duplicate"]
    if (metrics["block_ssim"] >= duplicate["block_ssim_min"] and
            metrics["mean_absolute_error"] <= duplicate["mean_absolute_error_max"] and
            metrics["dhash_distance"] <= duplicate["dhash_distance_max"]):
        reasons.append("near_duplicate")
    collapse = policy["collapsed_image"]
    if metrics["generated_channel_std"] <= collapse["channel_std_max"] and metrics["generated_edge_energy"] <= collapse["edge_energy_max"]:
        reasons.append("collapsed_image")
    semantic = policy["gross_semantic_deviation"]
    if metrics["global_cosine"] < semantic["global_cosine_below"] and metrics["center_cosine"] < semantic["center_cosine_below"]:
        reasons.append("gross_semantic_deviation_proxy")
    structure = policy["extreme_structure_loss"]
    if (metrics["center_cosine"] < structure["center_cosine_below"] and
            metrics["center_edge_ratio"] < structure["center_edge_ratio_below"] and
            metrics["center_block_ssim"] < structure["center_block_ssim_below"] and
            metrics["parent_center_edge_energy"] >= structure["parent_center_edge_min"]):
        reasons.append("extreme_structure_loss_proxy")
    return reasons


class FrozenPairEncoder:
    def __init__(self, device, policy):
        import torch
        from transformers import CLIPImageProcessor
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        self.device, self.policy = device, policy
        self.processor = CLIPImageProcessor.from_pretrained(str(DEFAULT_GENERATOR / "feature_extractor"), local_files_only=True)
        self.model = StableDiffusionSafetyChecker.from_pretrained(str(DEFAULT_GENERATOR / "safety_checker"),
                          torch_dtype=torch.float32, local_files_only=True).to(device).eval().requires_grad_(False)

    def score(self, pairs):
        import torch
        images = []
        for parent, generated in pairs:
            images.extend((parent, generated, center(parent, self.policy["center_crop_fraction"]), center(generated, self.policy["center_crop_fraction"])))
        embeddings = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                pixels = self.processor(images=images[start:start + 16], return_tensors="pt")["pixel_values"].to(self.device)
                output = self.model.vision_model(pixel_values=pixels).pooler_output
                embeddings.append(torch.nn.functional.normalize(self.model.visual_projection(output).float(), dim=1).cpu())
        values = torch.cat(embeddings).reshape(len(pairs), 4, -1)
        return [{"global_cosine": float((v[0] * v[1]).sum()), "center_cosine": float((v[2] * v[3]).sum())} for v in values]


def fingerprints():
    return {"generation_manifest_sha256": sha256(ROOT / "generation_manifest.csv"),
            "generation_config_sha256": sha256(ROOT / "config.json"),
            "mode_sha256": sha256(ROOT / "exploratory_config.json"), "policy_sha256": sha256(ROOT / "sanity_policy.json"),
            "vision_weights_sha256": sha256(DEFAULT_GENERATOR / "safety_checker/model.safetensors"),
            "vision_config_sha256": sha256(DEFAULT_GENERATOR / "safety_checker/config.json"),
            "processor_config_sha256": sha256(DEFAULT_GENERATOR / "feature_extractor/preprocessor_config.json")}


def load_accepted():
    settings()
    summary = read_json(ROOT / "automatic_sanity_summary.json")
    if summary["fingerprints"] != fingerprints():
        raise RuntimeError("automatic sanity inputs/policy changed; never reuse stale exclusions")
    for filename, expected in summary["output_sha256"].items():
        if sha256(ROOT / filename) != expected:
            raise RuntimeError("automatic sanity output changed: " + filename)
    if summary["status"] != "READY_EXPLORATORY":
        raise RuntimeError("automatic sanity HOLD: " + str(summary["empty_current_classes"]))
    accepted = read_csv(ROOT / "accepted_parent_manifest.csv")
    for row in accepted:
        if row["acceptance"] != "fixed_automatic_rules" or row["label_fidelity_verified"] != "false":
            raise RuntimeError("invalid acceptance provenance")
        for prefix in ("parent", "generated"):
            if sha256(row[prefix + "_path"]) != row[prefix + "_sha256"]:
                raise RuntimeError("accepted image changed: " + row["record_id"])
    return accepted


def run_sanity(device, encoder=None):
    mode, policy = settings()
    provenance = fingerprints()
    target = ROOT / "automatic_sanity_summary.json"
    if target.exists():
        accepted = load_accepted()
        write_report(read_json(target))
        print(f"Reuse fixed automatic sanity: {len(accepted)} accepted; label fidelity NOT verified", flush=True)
        return accepted
    if any((ROOT / "training").glob("*/*/*/start_manifest.json")):
        raise RuntimeError("cannot introduce/change automatic selection after training starts")
    started = time.perf_counter()
    import torch
    if device.type == "cuda":
        # PyTorch 2.4 reset_peak_memory_stats needs an initialized CUDA allocator.
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    encoder = encoder if encoder is not None else FrozenPairEncoder(device, policy)
    rows = read_csv(ROOT / "generation_manifest.csv")
    measurements = []
    for offset in range(0, len(rows), 16):
        chunk, pairs, pair_indices = [], [], []
        for row in rows[offset:offset + 16]:
            record = {"record_id": row["record_id"], "protocol": row["protocol"], "task": int(row["task"]),
                      "y": int(row["y"]), "parent_id": row["parent_id"], "parent_path": row["parent_path"],
                      "generated_path": row["generated_path"], "label_fidelity_verified": False}
            reasons = []
            if sha256(row["parent_path"]) != row["parent_sha256"]:
                raise RuntimeError("real train parent changed; repair real dataset before training")
            try:
                with Image.open(row["parent_path"]) as source:
                    source.load()
                    parent = ImageOps.pad(source.convert("RGB"), (512, 512), color=(127, 127, 127))
            except (OSError, ValueError) as error:
                raise RuntimeError("real train parent unreadable; anchors would also fail") from error
            if row["generation_status"] != "generated":
                reasons.append("generation_failed_or_safety_blocked")
            try:
                path = Path(row["generated_path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                if sha256(path) != row["generated_sha256"]:
                    reasons.append("generated_artifact_hash_mismatch")
                with Image.open(path) as source:
                    source.load()
                    generated = source.convert("RGB")
                if list(generated.size) != policy["expected_generated_size"]:
                    reasons.append("unexpected_generated_dimensions")
                generated = generated.resize((512, 512), Image.Resampling.LANCZOS)
                record.update(pixel_metrics(parent, generated, policy))
                pairs.append((parent, generated))
                pair_indices.append(len(chunk))
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                reasons.append("missing_or_undecodable_image")
                record["decode_error"] = type(error).__name__
            record["reasons"] = reasons
            chunk.append(record)
        if pairs:
            similarities = encoder.score(pairs)
            if len(similarities) != len(pairs):
                raise AssertionError("pair encoder output count mismatch")
            for index, similarity in zip(pair_indices, similarities):
                if not all(np.isfinite(value) for value in similarity.values()):
                    raise FloatingPointError("nonfinite sanity embeddings; do not filter by infrastructure failures")
                chunk[index].update(similarity)
                chunk[index]["reasons"].extend(exclusion_rules(chunk[index], policy))
        for record in chunk:
            record["accepted"] = not record["reasons"]
            record["exclusion_reason"] = ";".join(record.pop("reasons"))
        measurements.extend(chunk)
        print(f"Automatic sanity {min(offset + 16, len(rows))}/{len(rows)}", flush=True)
    accepted_ids = {r["record_id"] for r in measurements if r["accepted"]}
    accepted = [{**row, "acceptance": "fixed_automatic_rules", "label_fidelity_verified": "false",
                 "qc_mode": mode["mode"]} for row in rows if row["record_id"] in accepted_ids]
    groups = {(r["protocol"], int(r["task"]), int(r["y"])) for r in rows}
    empty = [f"{p}_T{t}_C{y}" for p, t, y in sorted(groups) if not any(r["protocol"] == p and int(r["task"]) == t and int(r["y"]) == y for r in accepted)]
    protocol_summary = {}
    fields = ("global_cosine", "center_cosine", "block_ssim", "mean_absolute_error", "pixel_rmse", "dhash_distance", "center_edge_ratio")
    for protocol in PROTOCOLS:
        own = [r for r in measurements if r["protocol"] == protocol]
        protocol_summary[protocol] = {"total": len(own), "accepted": sum(r["accepted"] for r in own),
              "excluded": sum(not r["accepted"] for r in own),
              "reason_counts_nonexclusive": dict(Counter(reason for r in own for reason in r["exclusion_reason"].split(";") if reason)),
              "similarity": {}}
        for subset in ("all_decodable", "accepted", "excluded_decodable"):
            chosen = own if subset == "all_decodable" else [r for r in own if r["accepted"] == (subset == "accepted")]
            protocol_summary[protocol]["similarity"][subset] = {}
            for field in fields:
                values = [r[field] for r in chosen if field in r]
                protocol_summary[protocol]["similarity"][subset][field] = {"n": len(values), "mean": float(np.mean(values)),
                     "min": float(np.min(values)), "q05": float(np.quantile(values, .05)), "median": float(np.median(values)),
                     "q95": float(np.quantile(values, .95)), "max": float(np.max(values))} if values else {"n": 0}
    counts = [{"protocol": p, "task": t, "y": y,
               "candidate_n": sum(r["protocol"] == p and int(r["task"]) == t and int(r["y"]) == y for r in rows),
               "accepted_common_parent_n": sum(r["protocol"] == p and int(r["task"]) == t and int(r["y"]) == y for r in accepted)} for p, t, y in sorted(groups)]
    write_csv(ROOT / "automatic_sanity.csv", measurements)
    write_csv(ROOT / "automatic_parent_counts.csv", counts)
    if accepted:
        write_csv(ROOT / "accepted_parent_manifest.csv", accepted)
    outputs = ["automatic_sanity.csv", "automatic_parent_counts.csv"] + (["accepted_parent_manifest.csv"] if accepted else [])
    summary = {"status": "HOLD_EMPTY_CURRENT_CLASS" if empty else "READY_EXPLORATORY", "mode": mode,
               "label_fidelity_verified": False, "limitation": LIMITATION,
               "fingerprints": provenance, "protocols": protocol_summary, "empty_current_classes": empty,
               "output_sha256": {name: sha256(ROOT / name) for name in outputs},
               "resource_usage": {"seconds": time.perf_counter() - started, "device": str(device),
                    "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    "additional_downloads": False, "additional_weights": "existing pinned generator CLIP component reused"}}
    write_json(target, summary)
    write_report(summary)
    if empty:
        raise RuntimeError("automatic sanity left an empty class; HOLD: " + str(empty))
    print(f"Automatic sanity READY_EXPLORATORY: {len(accepted)}/{len(rows)} used; label fidelity NOT verified", flush=True)
    return accepted
