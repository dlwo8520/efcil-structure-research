"""Isolated generator worker: official LoRA and text-to-image BOB, never img2img."""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

from .common import (ROOT, VENDOR, digest, now, progress, read_csv, read_json, sha256,
                     spec, task_dir, write_csv, write_json)
from .inputs import export_official_inputs, validate_captions
from .logging_compat import compatible_tracker_config


def load_official(relative, name):
    path = VENDOR / relative
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def assets():
    from huggingface_hub import HfApi, snapshot_download
    cfg = spec()["bob"]
    path = ROOT / "assets/model_lock.json"
    if path.exists():
        lock = read_json(path)
        assert lock["caption_model_id"] == cfg["caption_model_id"]
        assert Path(lock["caption_local_path"]).is_dir()
        assert Path(lock["base_local_path"]).is_dir()
        return lock
    base = Path(cfg["base_local_path"])
    if not (base / "model_index.json").is_file():
        raise FileNotFoundError(f"pinned SD1.5 local snapshot is missing: {base}")
    revision = HfApi().model_info(cfg["caption_model_id"]).sha
    local = snapshot_download(cfg["caption_model_id"], revision=revision)
    lock = {"created_at": now(), "caption_model_id": cfg["caption_model_id"],
            "caption_revision": revision, "caption_local_path": str(local),
            "base_model_id": cfg["base_model_id"], "base_revision": cfg["base_revision"],
            "base_local_path": str(base), "base_model_index_sha256": sha256(base / "model_index.json")}
    write_json(path, lock)
    return lock


def usage(task, stage, started, target=None, **fields):
    import torch
    value = {"task": task, "stage": stage, "seconds": time.perf_counter() - started,
             "visible_physical_gpu": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
             "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0), **fields}
    target = task_dir(task) if target is None else Path(target)
    write_json(target / f"resource_{stage}.json", value)


def caption(task, target=None):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    cfg = spec()["bob"]
    lock = read_json(ROOT / "assets/model_lock.json")
    folder = task_dir(task) if target is None else Path(target)
    records = read_csv(folder / "real_train.csv")
    output_path = folder / "captions.csv"
    rows = read_csv(output_path) if output_path.exists() else []
    done = {row["source_id"]: row for row in rows}
    assert len(done) == len(rows)
    for row in records:
        if row["source_id"] in done:
            assert done[row["source_id"]]["source_sha256"] == row["source_sha256"]
    pending = [r for r in records if r["source_id"] not in done]
    if not pending:
        export_official_inputs(task, folder)
        print(f"T{task}: preserve completed context captions", flush=True)
        return
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        lock["caption_local_path"], torch_dtype=dtype, attn_implementation="sdpa", local_files_only=True,
    ).to("cuda:0").eval()
    processor = AutoProcessor.from_pretrained(lock["caption_local_path"], local_files_only=True,
                                             max_pixels=cfg["caption_max_pixels"])
    prompts = {field: f"Describe the {field} of the object in as few words as possible. Refer to the object as simply 'an object'."
               for field in ("background", "pose")}
    for row in pending:
        assert int(row["task"]) == task
        result = {"task": task, "source_id": row["source_id"], "source_sha256": row["source_sha256"]}
        for field, query in prompts.items():
            # Deliberately no class name, test ID, future label or hardness input.
            messages = [{"role": "user", "content": [
                {"type": "image", "image": row["source_path"]}, {"type": "text", "text": query}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=cfg["caption_max_new_tokens"], do_sample=False)
            answer = processor.batch_decode([output[0, inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            answer = " ".join(answer.split()).rstrip(".")
            if not answer:
                raise RuntimeError(f"empty {field} caption: {row['source_id']}")
            result[field] = answer
        rows.append(result)
        write_csv(output_path, rows)
        progress("caption", task=task, completed=len(rows), total=len(records))
    usage(task, "caption", started, target=folder, real_images=len(records), images_captioned_this_attempt=len(pending),
          model_revision=lock["caption_revision"], model_dtype=str(dtype))
    export_official_inputs(task, folder)


def lora(task, target=None, smoke=False):
    import torch
    cfg = spec()["bob"]
    target = task_dir(task) if target is None else Path(target)
    export_official_inputs(task, target)
    identity = digest({"bob": cfg, "inputs": read_json(target / "input_manifest.json"),
                       "captions_sha256": sha256(target / "captions.csv"), "smoke": smoke})
    success = target / "lora_complete.json"
    if success.exists():
        prior = read_json(success)
        assert prior["identity"] == identity
        assert prior["lora_sha256"] == sha256(target / "lora/pytorch_lora_weights.safetensors")
        print(f"T{task}: preserve completed official BOB LoRA", flush=True)
        return
    sys.path.insert(0, str(VENDOR / "finetune"))
    official = load_official("finetune/finetune_t2i.py", "official_bob_finetune")
    arguments = ["official_bob_finetune", "--yaml_file", str(target / "official_config.yaml"),
                 "--seed", str(1993 + task), "--train_batch_size", str(2 if smoke else cfg["finetune_batch_size"]),
                 "--num_train_epochs", str(1 if smoke else cfg["finetune_epochs"]), "--rank", str(cfg["rank"]),
                 "--learning_rate", str(cfg["learning_rate"]), "--lr_scheduler", cfg["lr_scheduler"],
                 "--lr_warmup_steps", str(cfg["warmup_steps"]), "--mixed_precision", cfg["mixed_precision"],
                 "--gradient_accumulation_steps", "1", "--dataloader_num_workers", "0",
                 "--checkpointing_steps", "5000", "--report_to", "tensorboard"]
    if cfg["gradient_checkpointing"]:
        arguments.append("--gradient_checkpointing")
    if list((target / "lora").glob("checkpoint-*")):
        arguments += ["--resume_from_checkpoint", "latest"]
    original_argv = sys.argv
    sys.argv = arguments
    try:
        args = official.parse_args()
    finally:
        sys.argv = original_argv
    # Upstream config.py omits this attribute, although the training/hooks use it.
    # Both attention LoRAs are already created/trained by upstream; no algorithm change.
    args.train_text_encoder = True
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    progress("official_lora_smoke" if smoke else "official_lora", task=task, epochs=1 if smoke else cfg["finetune_epochs"])
    with compatible_tracker_config(official, target / "lora" / args.logging_dir / "full_tracker_config.json"):
        official.main(args)
    weight = target / "lora/pytorch_lora_weights.safetensors"
    assert weight.is_file()
    usage(task, "lora", started, target=target, official_repo_commit=cfg["repo_commit"],
          base_model=cfg["base_model_id"], real_images=len(read_csv(target / "real_train.csv")),
          epochs=1 if smoke else cfg["finetune_epochs"], smoke_only=smoke, command=arguments,
          resume_replays_partial_epoch_as_upstream=True)
    write_json(success, {"identity": identity, "lora_sha256": sha256(weight), "completed_at": now()})


def image_qc(path, pixel_hashes, safety=False, policy=None):
    """Technical rules only. No CLIP confidence, subtype or human selection."""
    import hashlib
    import numpy as np
    from PIL import Image
    policy = policy or spec()["bob"]["technical_QC"]
    try:
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
            if min(rgb.size) < policy["min_side"]:
                return False, "corrupt_or_too_small", ""
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + array.tobytes()).hexdigest()
            if safety:
                return False, "generation_safety_failure", pixel_hash
            if float(array.std()) <= policy["blank_std_max"]:
                return False, "blank", pixel_hash
            if policy["reject_exact_pixel_duplicates"] and pixel_hash in pixel_hashes:
                return False, "exact_duplicate", pixel_hash
            pixel_hashes.add(pixel_hash)
            return True, "", pixel_hash
    except (OSError, ValueError):
        return False, "decode_failure", ""


def generation_plan(task, real_records, captions, count=None):
    cfg = spec()["bob"]
    count = cfg["images_per_class"] if count is None else count
    classes = sorted({(int(r["class_id"]), r["class_name"]) for r in real_records})
    rows = []
    for label, name in classes:
        for number in range(count):
            seed = cfg["generation_seed"] + task * 1_000_000 + label * 1000 + number
            context = real_records[random.Random(seed).randrange(len(real_records))]
            assert int(context["task"]) == task
            caption = captions[context["source_id"]]
            rows.append({"task": task, "class_id": label, "class_name": name,
                         "record_id": f"IA10_T{task:02d}_C{label:03d}_N{number:03d}",
                         "context_source_id": context["source_id"], "context_source_path": context["source_path"],
                         "seed": seed, "background": caption["background"], "pose": caption["pose"],
                         "prompt": cfg["template"].format(name, caption["background"], caption["pose"]),
                         "generated_path": str(task_dir(task) / "generated" / f"C{label:03d}" / f"{number:03d}.png"),
                         "bob_config_sha256": digest(cfg)})
    return rows


def generate(task):
    import torch
    cfg = spec()["bob"]
    target = task_dir(task)
    records, captions = validate_captions(task)
    assert (target / "lora_complete.json").is_file()
    plan = generation_plan(task, records, captions)
    manifest_path = target / "generation_manifest.csv"
    existing = read_csv(manifest_path) if manifest_path.exists() else []
    done = {r["record_id"]: r for r in existing}
    assert len(done) == len(existing)
    pixel_hashes = {r["pixel_sha256"] for r in existing if r["accepted"] == "1"}
    for row in plan:
        if row["record_id"] in done:
            prior = done[row["record_id"]]
            assert prior["bob_config_sha256"] == row["bob_config_sha256"] and prior["prompt"] == row["prompt"]
            assert prior["file_sha256"] == sha256(prior["generated_path"])
    pending = [r for r in plan if r["record_id"] not in done]
    if not pending:
        verify_bank(task)
        return
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    official = load_official("generation/generate.py", "official_bob_generation")
    pipe = official.get_pipe(cfg["base_local_path"], str(target / "lora"), torch.device("cuda:0"), False)
    worker = official.GenerateImage(pipe, torch.device("cuda:0"), cfg["guidance_scale"], cfg["inference_steps"],
                                   cfg["images_per_class"], str(target / "generated"), 1,
                                   str(target / "captions"), cfg["template"])
    for row in pending:
        # Official inference has no generator argument, so reseed all streams per record.
        official.set_seed(int(row["seed"]))
        output = worker.run_pipe([row["prompt"]])
        path = Path(row["generated_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        output.images[0].resize((cfg["image_size"], cfg["image_size"])).save(path)
        flags = getattr(output, "nsfw_content_detected", None)
        accepted, reason, pixel_hash = image_qc(path, pixel_hashes, safety=bool(flags and flags[0]))
        row.update({"accepted": int(accepted), "technical_failure": reason,
                    "pixel_sha256": pixel_hash, "file_sha256": sha256(path),
                    "official_repo_commit": cfg["repo_commit"],
                    "base_model_revision": cfg["base_revision"],
                    "lora_sha256": read_json(target / "lora_complete.json")["lora_sha256"]})
        existing.append(row)
        write_csv(manifest_path, existing)
        progress("bob_generation", task=task, generated=len(existing), total=len(plan),
                 accepted=sum(int(r["accepted"]) for r in existing))
    usage(task, "generation", started, images_generated_this_attempt=len(pending), images_total=len(existing),
          accepted=sum(int(r["accepted"]) for r in existing),
          disk_image_bytes=sum(Path(r["generated_path"]).stat().st_size for r in existing))
    verify_bank(task)


def verify_bank(task):
    rows = read_csv(task_dir(task) / "generation_manifest.csv")
    cfg = spec()["bob"]
    assert len(rows) == 20 * cfg["images_per_class"], "incomplete candidate bank"
    accepted = [r for r in rows if r["accepted"] == "1"]
    assert {int(r["class_id"]) for r in accepted} == set(range((task - 1) * 20, task * 20)), "empty class after technical QC"
    for row in accepted:
        assert int(row["task"]) == task and Path(row["generated_path"]).is_file()
        assert row["file_sha256"] == sha256(row["generated_path"])
    completed_path = task_dir(task) / "bank_complete.json"
    manifest_hash = sha256(task_dir(task) / "generation_manifest.csv")
    if completed_path.exists():
        assert read_json(completed_path)["manifest_sha256"] == manifest_hash, "fixed accepted bank changed after completion"
    write_json(completed_path, {
        "task": task, "candidate_count": len(rows), "accepted_count": len(accepted),
        "manifest_sha256": manifest_hash,
        "technical_QC_only": True, "human_QC": False, "label_fidelity_verified": False,
        "learner_access": "current task only; bank not included in checkpoints"})
    return accepted


def feasibility():
    """One class, two train images, one LoRA update, one generated image. No test."""
    import gc
    import torch
    from .common import experiment_hash
    target = ROOT / "feasibility/bob"
    complete = target / "complete.json"
    identity = experiment_hash()
    if complete.exists():
        assert read_json(complete)["experiment_hash"] == identity
        return
    all_real = read_csv(task_dir(1) / "real_train.csv")
    label = all_real[0]["class_id"]
    records = [dict(r) for r in all_real if r["class_id"] == label][:2]
    assert len(records) == 2
    for row in records:
        link = target / "real" / row["class_name"] / Path(row["bob_input_path"]).name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(row["source_path"])
        row["bob_input_path"] = str(link)
    write_csv(target / "real_train.csv", records)
    write_json(target / "input_manifest.json", {"purpose": "technical_smoke_only", "records_hash": digest(records), "test_images": 0})
    caption(1, target)
    gc.collect(); torch.cuda.empty_cache()
    lora(1, target, smoke=True)
    gc.collect(); torch.cuda.empty_cache()
    cfg = spec()["bob"]
    official = load_official("generation/generate.py", "official_bob_feasibility_generation")
    pipe = official.get_pipe(cfg["base_local_path"], str(target / "lora"), torch.device("cuda:0"), False)
    worker = official.GenerateImage(pipe, torch.device("cuda:0"), cfg["guidance_scale"], cfg["inference_steps"],
                                   1, str(target), 1, str(target / "captions"), cfg["template"])
    caps = read_csv(target / "captions.csv")[0]
    prompt = cfg["template"].format(records[0]["class_name"], caps["background"], caps["pose"])
    official.set_seed(1993)
    output = worker.run_pipe([prompt])
    path = target / "smoke_generated.png"
    output.images[0].save(path)
    flags = getattr(output, "nsfw_content_detected", None)
    accepted, reason, pixel_hash = image_qc(path, set(), safety=bool(flags and flags[0]))
    if not accepted:
        raise RuntimeError(f"one-class technical smoke image failed QC: {reason}; no full trajectory started")
    write_csv(target / "smoke_bank.csv", [{"task": 1, "class_id": label, "accepted": "1", "record_id": "smoke_only",
                                          "generated_path": str(path), "prompt": prompt, "pixel_sha256": pixel_hash}])
    write_json(complete, {"experiment_hash": identity, "technical_smoke_only": True,
                          "real_images": 2, "lora_updates": 1, "generated_images": 1,
                          "test_accuracy_computed": False, "generated_sha256": sha256(path)})


def logging_check():
    """Real installed Accelerator/TensorBoard + official args; no models or GPU."""
    import os
    import tempfile
    # This stage is a separate process, so a CPU Accelerator cannot accidentally
    # change the state/device of a subsequent GPU training Accelerator.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from accelerate.utils import ProjectConfiguration
    from torch.utils.tensorboard.summary import hparams
    sys.path.insert(0, str(VENDOR / "finetune"))
    official = load_official("finetune/finetune_t2i.py", "official_bob_logging_check")
    with tempfile.TemporaryDirectory(prefix="bob_logging_check_") as temporary:
        folder = Path(temporary)
        config_path = folder / "official_config.yaml"
        write_json(config_path, {
            "model": spec()["bob"]["base_local_path"], "lora_path": str(folder / "lora"),
            "data_path": str(folder / "unused_real"), "caption_path": str(folder / "unused_captions"),
            "template": spec()["bob"]["template"], "classnames": ["logging_check_only"],
        })
        argv = sys.argv
        sys.argv = ["official_bob_logging_check", "--yaml_file", str(config_path),
                    "--mixed_precision", "no", "--report_to", "tensorboard"]
        try:
            args = official.parse_args()
        finally:
            sys.argv = argv
        args.train_text_encoder = True
        original = dict(vars(args))
        try:
            hparams(original, {})
        except ValueError:
            reproduced = True
        else:
            raise AssertionError("expected the original unsupported hparam types")
        full_config = folder / "full_tracker_config.json"
        with compatible_tracker_config(official, full_config):
            accelerator = official.Accelerator(cpu=True, log_with="tensorboard", project_config=ProjectConfiguration(
                project_dir=str(folder), logging_dir=str(folder / "logs")))
            accelerator.init_trackers("bob_logging_check", config=vars(args))
            accelerator.log({"smoke_scalar": 1.0}, step=0)
            accelerator.end_training()
        assert vars(args) == original and read_json(full_config)["classnames"] == original["classnames"]
        assert read_json(full_config)["revision"] is None
        assert not torch.cuda.is_initialized(), "logging regression must not initialize CUDA"
        assert list(folder.rglob("events.out.tfevents.*")), "no TensorBoard event was written"
    print({"logging_check": "PASS", "original_error_reproduced": reproduced,
           "actual_Accelerator_TensorBoard": True, "training_args_unchanged": True,
           "CUDA_initialized": False, "training_executed": False}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("assets", "doctor", "logging-check", "feasibility", "caption", "lora", "generate"))
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.stage == "assets":
        assets()
    elif args.stage == "logging-check":
        logging_check()
    elif args.stage == "feasibility":
        feasibility()
    elif args.stage == "doctor":
        import importlib.metadata
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        sys.path.insert(0, str(VENDOR / "finetune"))
        load_official("finetune/finetune_t2i.py", "official_bob_import_check")
        load_official("generation/generate.py", "official_bob_generation_check")
        assert torch.cuda.is_available(), "BOB generator cannot see the selected GPU"
        lock = read_json(ROOT / "assets/model_lock.json")
        assert Path(lock["caption_local_path"]).is_dir()
        assert lock["base_model_index_sha256"] == sha256(Path(lock["base_local_path"]) / "model_index.json")
        print({"generator_environment": {p: importlib.metadata.version(p) for p in
              ("torch", "diffusers", "transformers", "accelerate", "peft")}, "CUDA": True}, flush=True)
    else:
        if args.task is None or not 1 <= args.task <= 10:
            parser.error("--task 1..10 required")
        {"caption": caption, "lora": lora, "generate": generate}[args.stage](args.task)


if __name__ == "__main__":
    main()
