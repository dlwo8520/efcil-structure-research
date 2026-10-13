import sys
import logging
import copy
import torch
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import time


def CDS_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    experiment_started = time.perf_counter()
    for seed_index, seed in enumerate(seed_list):
        seed_started = time.perf_counter()
        args["seed"] = seed
        args["device"] = device
        print("[Progress] Seed {}/{} ({}) started".format(seed_index + 1, len(seed_list), seed))
        _train(args)
        seed_elapsed = time.perf_counter() - seed_started
        completed = seed_index + 1
        eta = seed_elapsed * (len(seed_list) - completed)
        print(
            "[Progress] Seed {}/{} finished in {}; estimated remaining {}".format(
                completed, len(seed_list), _format_duration(seed_elapsed), _format_duration(eta)
            )
        )
    print("[Timing] Experiment elapsed: {}".format(
        _format_duration(time.perf_counter() - experiment_started)
    ))


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    # Each seed must own a separate fresh log. ``basicConfig`` is otherwise a
    # no-op after the first seed in the same Python process.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        handler.close()
        root_logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            # A fresh invocation gets a clean log instead of mixing earlier
            # failed/smoke-run output with the real experiment.
            logging.FileHandler(filename=logfilename + ".log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log file: %s.log", logfilename)

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    _reset_peak_cuda_stats(args["device"])


    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        dataset_root=args.get("dataset_root"),
        expected_num_classes=args.get("expected_num_classes"),
        expected_tasks=args.get("task_count"),
    )
    model = model_factory.get_model(args["model_name"], args)

    start_task = 0
    resume_checkpoint = args.get("resume_checkpoint")
    if resume_checkpoint:
        if not hasattr(model, "load_checkpoint"):
            raise ValueError("resume_checkpoint is supported only by resumable learners")
        start_task = model.load_checkpoint(resume_checkpoint, data_manager)

    print()
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    completed_task_seconds = []
    stop_task = data_manager.nb_tasks
    if args.get("smoke_test", False):
        stop_task = min(stop_task, start_task + 1)
        logging.info("Smoke test enabled: running one task only")
    for task in range(start_task, stop_task):
        task_started = time.perf_counter()
        logging.info("Task %d/%d started", task + 1, data_manager.nb_tasks)
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        task_seconds = time.perf_counter() - task_started
        completed_task_seconds.append(task_seconds)
        mean_task_seconds = sum(completed_task_seconds) / len(completed_task_seconds)
        remaining_tasks = data_manager.nb_tasks - task - 1
        logging.info(
            "Task %d/%d finished: elapsed=%s, estimated experiment remaining=%s",
            task + 1,
            data_manager.nb_tasks,
            _format_duration(task_seconds),
            _format_duration(mean_task_seconds * remaining_tasks),
        )
    _log_peak_cuda_stats(args["device"])


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1 or str(device).lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def _format_bytes(num_bytes):
    if num_bytes is None:
        return "N/A"
    num_bytes = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if num_bytes < 1024 or unit == units[-1]:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def _get_cuda_index(dev):
    if isinstance(dev, torch.device):
        if dev.type != "cuda":
            return None
        return dev.index if dev.index is not None else torch.cuda.current_device()
    if isinstance(dev, int):
        return dev if dev >= 0 else None
    if isinstance(dev, str) and dev.isdigit():
        return int(dev)
    return None


def _reset_peak_cuda_stats(devices):
    if not torch.cuda.is_available():
        return
    for dev in devices:
        dev_index = _get_cuda_index(dev)
        if dev_index is None:
            continue
        try:
            # Newer PyTorch builds can reject peak-stat resets before the CUDA
            # context for this logical device has been initialized.
            torch.empty(0, device=torch.device("cuda", dev_index))
            torch.cuda.reset_peak_memory_stats(dev_index)
        except Exception as exc:
            logging.warning("Failed to reset peak memory stats on cuda:%s (%s)", dev_index, exc)


def _log_peak_cuda_stats(devices):
    if not torch.cuda.is_available():
        print("Peak GPU memory: N/A (CUDA not available)")
        logging.info("Peak GPU memory: N/A (CUDA not available)")
        return
    for dev in devices:
        dev_index = _get_cuda_index(dev)
        if dev_index is None:
            continue
        try:
            torch.cuda.synchronize(dev_index)
            allocated = torch.cuda.max_memory_allocated(dev_index)
            reserved = torch.cuda.max_memory_reserved(dev_index)
        except Exception as exc:
            logging.warning("Failed to query peak memory on cuda:%s (%s)", dev_index, exc)
            continue
        msg = (
            f"Peak GPU memory on cuda:{dev_index}: "
            f"allocated={_format_bytes(allocated)}, "
            f"reserved={_format_bytes(reserved)}"
        )
        print(msg)
        logging.info(msg)
