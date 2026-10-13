"""Train-only parent bank and SD1.5 editing; human or explicit exploratory gate."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import (ROOT, PROTOCOLS, DEFAULT_GENERATOR, data_manager, sample_identity,
                     ensure_config, read_csv, write_csv, write_json, sha256)


def prepare_parents():
    config = ensure_config()
    target = ROOT / "generation_manifest.csv"
    if target.exists():
        return read_csv(target)
    rows = []
    for protocol in PROTOCOLS:
        manager = data_manager(protocol)
        for task in (1, 9):
            start, stop = sum(manager._increments[:task - 1]), sum(manager._increments[:task])
            dataset = manager.get_dataset(np.arange(start, stop), source="train", mode="test")
            ids, identity_labels = sample_identity(manager, np.arange(start, stop), "train")
            if not np.array_equal(dataset.labels, identity_labels):
                raise AssertionError("parent train identity mismatch")
            rng = np.random.RandomState(config["parent_seed"] + task + (0 if protocol == "IA10" else 1000))
            for label in range(start, stop):
                candidates = np.flatnonzero(dataset.labels == label)
                indices = np.sort(rng.choice(candidates, min(16, len(candidates)), replace=False))
                for position, index in enumerate(indices):
                    parent = Path(str(dataset.images[index])).resolve()
                    train_root = Path(manager.split_roots["train"]).resolve()
                    if not parent.is_relative_to(train_root):
                        raise AssertionError("parent source is not real train")
                    key = f"{protocol}_T{task:02d}_C{label:03d}_P{position:02d}"
                    variation = ("soft overcast daylight", "warm natural evening light",
                                 "a different neutral outdoor background", "a different indoor background")[position % 4]
                    identity = "car model, year, body, grille and lights" if protocol == "CAR10" else "object identity, shape and markings"
                    prompt = f"Photograph of the exact same object as the input, {variation}. Preserve the exact {identity}, pose and composition."
                    rows.append({"record_id": key, "protocol": protocol, "task": task,
                                 "y": label, "original_class_id": int(manager._class_order[label]),
                                 "class_name": manager.class_names[manager._class_order[label]],
                                 "real_train_class_n": len(candidates), "selected_parent_class_n": len(indices),
                                 "parent_id": str(ids[index]), "parent_path": str(parent),
                                 "parent_sha256": sha256(parent), "prompt": prompt,
                                 "seed": config["parent_seed"] + len(rows),
                                 "generated_path": str(ROOT / "generated" / protocol / f"T{task:02d}" / f"{key}.png"),
                                 "generation_status": "pending", "generated_sha256": "",
                                 "generation_seconds": "", "review_sample": 0})
        current = [r for r in rows if r["protocol"] == protocol]
        rng = np.random.RandomState(config["parent_seed"] + (0 if protocol == "IA10" else 1000))
        selected = []
        for label in sorted({r["y"] for r in current}):
            group = [r for r in current if r["y"] == label]
            selected.append(group[rng.randint(len(group))]["record_id"])
        other = [r["record_id"] for r in current if r["record_id"] not in selected]
        remaining = min(100 - len(selected), len(other))
        selected.extend(rng.choice(other, remaining, replace=False).tolist())
        for row in current:
            row["review_sample"] = int(row["record_id"] in selected)
    write_csv(target, rows)
    return rows


def generate(device):
    import importlib.metadata
    import torch
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as error:
        raise RuntimeError("generation dependencies missing; use .venv_generator and requirements-generator.txt") from error
    config = ensure_config()
    rows = prepare_parents()
    weights = sorted(path for path in DEFAULT_GENERATOR.rglob("*") if path.is_file())
    if not weights or not (DEFAULT_GENERATOR / "model_index.json").is_file():
        raise FileNotFoundError(f"local pinned generator not available: {DEFAULT_GENERATOR}")
    manifest_path = ROOT / "generator_resource.json"
    from .common import read_json
    weight_records = [
            {"relative_path": str(path.relative_to(DEFAULT_GENERATOR)), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in weights]
    if manifest_path.exists() and read_json(manifest_path)["files"] != weight_records:
        raise RuntimeError("pinned generator files changed after first invocation")
    if not manifest_path.exists():
        write_json(manifest_path, {"model": config["generator"], "files": weight_records,
            "packages": {name: importlib.metadata.version(name) for name in ("diffusers", "transformers", "torch")}})
    started = time.perf_counter()
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(DEFAULT_GENERATOR), torch_dtype=torch.float16, local_files_only=True).to(device)
    pipeline.set_progress_bar_config(disable=True)
    for component in pipeline.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval().requires_grad_(False)
    write_json(ROOT / "generator_runtime_config.json", {"scheduler_class": type(pipeline.scheduler).__name__,
               "scheduler_config": dict(pipeline.scheduler.config), "frozen": True})
    torch.cuda.reset_peak_memory_stats(device)
    for index, row in enumerate(rows):
        path = Path(row["generated_path"])
        if row["generation_status"] in ("generated", "technical_reject") and path.is_file():
            if sha256(path) != row["generated_sha256"]:
                raise RuntimeError(f"generated artifact changed: {path}")
            continue
        if sha256(row["parent_path"]) != row["parent_sha256"]:
            raise RuntimeError("parent image changed after recipe preparation")
        clock = time.perf_counter()
        with Image.open(row["parent_path"]) as source:
            parent = ImageOps.pad(source.convert("RGB"), (512, 512), color=(127, 127, 127))
        with torch.inference_mode():
            result = pipeline(prompt=row["prompt"], image=parent, strength=0.2,
                              num_inference_steps=30, guidance_scale=7.5,
                              generator=torch.Generator(device=device).manual_seed(int(row["seed"])))
        path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(path)
        flagged = result.nsfw_content_detected and result.nsfw_content_detected[0]
        row.update({"generation_status": "technical_reject" if flagged else "generated",
                    "generated_sha256": sha256(path), "generation_seconds": time.perf_counter() - clock})
        write_csv(ROOT / "generation_manifest.csv", rows)
        print(f"Generate {index + 1}/{len(rows)}: {row['record_id']} {row['generation_status']}", flush=True)
    review = ROOT / "quality_review.csv"
    if not review.exists():
        write_csv(review, [{"record_id": row["record_id"], "protocol": row["protocol"],
                           "task": row["task"], "y": row["y"], "review_sample": row["review_sample"],
                           "class_name": row["class_name"], "original_class_id": row["original_class_id"],
                           "review_status": "reject" if row["generation_status"] != "generated" else "pending",
                           "label_fidelity_verified": "no", "variation": "pending", "reason": "",
                           "parent_path": row["parent_path"], "generated_path": row["generated_path"]} for row in rows])
    make_contact_sheets(rows)
    write_json(ROOT / "generation_resource_usage.json", {"invocation_seconds": time.perf_counter() - started,
               "generation_seconds_sum": sum(float(r["generation_seconds"] or 0) for r in rows),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "generated_image_bytes": sum(Path(r["generated_path"]).stat().st_size for r in rows if Path(r["generated_path"]).exists()),
               "image_count": len(rows), "weights_bytes": sum(p.stat().st_size for p in weights)})
    if (ROOT / "exploratory_config.json").exists():
        print("Generation complete. Run automatic sanity before training; contact sheets are artifacts only.", flush=True)
    else:
        print("Generation complete. Review quality_review.csv against contact_sheets before --stage train.", flush=True)


def make_contact_sheets(rows):
    output = ROOT / "contact_sheets"
    output.mkdir(exist_ok=True)
    for protocol in PROTOCOLS:
        sampled = [r for r in rows if r["protocol"] == protocol and int(r["review_sample"])]
        for page in range(0, len(sampled), 10):
            canvas = Image.new("RGB", (620, 10 * 184), "white")
            draw = ImageDraw.Draw(canvas)
            for slot, row in enumerate(sampled[page:page + 10]):
                draw.text((4, slot * 184), row["record_id"] + " | parent / generated", fill="black")
                for col, field in enumerate(("parent_path", "generated_path")):
                    if Path(row[field]).is_file():
                        with Image.open(row[field]) as image:
                            preview = ImageOps.contain(image.convert("RGB"), (300, 160))
                        canvas.paste(preview, (col * 310, slot * 184 + 20))
            canvas.save(output / f"{protocol}_{page // 10 + 1:02d}.jpg")


def quality_gate(write_outputs=True):
    config = ensure_config()
    if (ROOT / "exploratory_config.json").exists():
        from .automatic_sanity import load_accepted
        if not (ROOT / "automatic_sanity_summary.json").exists():
            raise RuntimeError("run --stage auto-sanity first; exploratory mode does not fabricate human QC")
        return load_accepted()
    rows = read_csv(ROOT / "generation_manifest.csv")
    reviews = {r["record_id"]: r for r in read_csv(ROOT / "quality_review.csv")}
    if set(reviews) != {r["record_id"] for r in rows}:
        raise RuntimeError("review IDs differ from fixed generation bank")
    for row in rows:
        if row["generation_status"] not in ("generated", "technical_reject"):
            raise RuntimeError("generation bank incomplete; finish generation before human gate")
        for prefix in ("parent", "generated"):
            if sha256(row[prefix + "_path"]) != row[prefix + "_sha256"]:
                raise RuntimeError(f"{prefix} image changed after generation: {row['record_id']}")
    summary, accepted = {}, []
    for protocol in PROTOCOLS:
        own = [r for r in rows if r["protocol"] == protocol]
        sampled = [r for r in own if int(r["review_sample"])]
        incomplete = []
        bad, duplicate = 0, 0
        for row in sampled:
            review = reviews[row["record_id"]]
            if review["review_status"] not in ("accept", "reject", "uncertain"):
                incomplete.append(row["record_id"])
            if review["review_status"] != "accept" or review["label_fidelity_verified"] != "yes":
                bad += 1
            if review["review_status"] == "accept" and review["variation"] not in ("meaningful", "near_duplicate"):
                incomplete.append(row["record_id"])
            duplicate += int(review["variation"] == "near_duplicate")
        passes = bool(sampled) and not incomplete and bad / len(sampled) <= 0.05 and duplicate / len(sampled) <= 0.5
        summary[protocol] = {"status": "PASS" if passes else "HOLD", "review_sample_n": len(sampled),
                             "bad_or_uncertain_n": bad, "near_duplicate_n": duplicate,
                             "incomplete_ids": incomplete,
                             "review_status_counts": {status: sum(reviews[r["record_id"]]["review_status"] == status for r in own)
                                                      for status in ("accept", "reject", "uncertain", "pending")},
                             "unsampled_images_individually_verified": False}
        if passes:
            for row in own:
                review = reviews[row["record_id"]]
                if row["generation_status"] != "generated" or review["review_status"] in ("reject", "uncertain"):
                    continue
                if review["review_status"] == "accept" and review["label_fidelity_verified"] != "yes":
                    continue
                if review["variation"] == "near_duplicate":
                    continue
                accepted.append({**row, "acceptance": "individual_review" if review["review_status"] == "accept" else "recipe_sample_gate"})
        summary[protocol]["adopted_n"] = sum(r["protocol"] == protocol for r in accepted)
        summary[protocol]["excluded_n"] = len(own) - summary[protocol]["adopted_n"]
        summary[protocol]["adopted_per_current_class"] = {f"T{task}_C{label}": sum(r["protocol"] == protocol and int(r["task"]) == task and int(r["y"]) == label for r in accepted)
                   for task, label in sorted({(int(r["task"]), int(r["y"])) for r in own})}
    if write_outputs:
        write_json(ROOT / "generation_quality.json", summary)
        lines = ["# Generation quality", "", "Review is train-only, before pilot test outcomes. "
                 "Sample gate does not verify every generated image individually.", ""]
        for protocol, record in summary.items():
            lines.append(f"- {protocol}: {record['status']}; sample={record['review_sample_n']}, "
                         f"bad/uncertain={record['bad_or_uncertain_n']}, near-duplicate={record['near_duplicate_n']}.")
            lines.append(f"  Adopted={record['adopted_n']}, excluded={record['excluded_n']}; review statuses: `{record['review_status_counts']}`.")
        (ROOT / "generation_quality.md").write_text("\n".join(lines) + "\n")
    if any(record["status"] != "PASS" for record in summary.values()):
        raise RuntimeError("generation quality HOLD; complete human review or stop this recipe")
    for protocol in PROTOCOLS:
        for task in (1, 9):
            own = [r for r in rows if r["protocol"] == protocol and int(r["task"]) == task]
            for label in {int(r["y"]) for r in own}:
                if not any(r["protocol"] == protocol and int(r["task"]) == task and int(r["y"]) == label for r in accepted):
                    raise RuntimeError(f"no shared usable parent for {protocol} T{task} class {label}; training HOLD")
    if write_outputs:
        write_csv(ROOT / "accepted_parent_manifest.csv", accepted)
    return accepted
