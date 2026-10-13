import copy
import logging
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
import math

from utils.adjust_distribution import LobachevskyPrototypeCorrector
from utils.distribution_regions import (
    SeparationCache,
    build_regions_for_class,
    build_single_region,
    clone_regions,
    compare_mixture_to_class_statistics,
    flatten_regions,
    rectify_local_regions,
    region_storage_bytes,
    summarize_regions,
)
from utils.distill_loss import LobachevskyRelativeGeomLoss, features_distill_loss, \
    logits_distill_loss
from utils.experiment_io import (
    append_jsonl,
    capture_rng_state,
    restore_rng_state,
    update_json,
    upsert_csv_rows,
    utc_now_iso,
    write_json,
)

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None

        self.adjust_distribution = LobachevskyPrototypeCorrector(self._device, self.args['gamma_weight'],
                                                                 self.args['cov'])
        self._local_stats = {}
        self._single_regions = {}
        self._task_source_means = None
        self._task_source_covs = None
        self._task_source_local_stats = {}
        self._sep_reference_means = None
        self._sep_reference_covs = None
        self._separation_cache = None
        self._resume_stage = None
        self._resume_next_epoch = 0
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self.task_sizes = []
        self.run_dir = Path(self.args.get("run_dir", "outputs/unresolved"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.convnet.flag += 1

        flag = self._network.convnet.flag % 2

        if flag == 0:
            self._network.convnet.copy_adapter(1)
        else:
            self._network.convnet.copy_adapter(2)

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        resuming_current_task = self._resume_stage == "train_epoch"
        if not resuming_current_task:
            self._cur_task += 1
            task_size = data_manager.get_task_size(self._cur_task)
            self._total_classes = self._known_classes + task_size
            self._network.update_fc(task_size)
            self.task_sizes.append(task_size)
        else:
            task_size = data_manager.get_task_size(self._cur_task)
            if self._total_classes != self._known_classes + task_size:
                raise RuntimeError("resume checkpoint task dimensions do not match the data protocol")
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        workers = int(self.args.get("num_workers", num_workers))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=workers)
        stats_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test"
        )
        self.stats_loader = DataLoader(
            stats_dataset,
            batch_size=int(self.args["stats_batch_size"]),
            shuffle=False,
            num_workers=workers,
        )

        if not isinstance(self._network, nn.DataParallel):
            self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0 and not resuming_current_task:
            self._task_source_means = self._class_means.copy()
            self._task_source_covs = self._class_covs.detach().cpu().clone()
            self._task_source_local_stats = clone_regions(self._local_stats)
            now_task_old_class_mean, now_task_old_class_covs = self._conpute_now_task_old_class_mean(data_manager)
            self.now_task_mean = now_task_old_class_mean.copy()
            self.now_task_covs = now_task_old_class_covs
            self._sep_reference_means = now_task_old_class_mean.copy()
            self._sep_reference_covs = now_task_old_class_covs.detach().cpu().clone()

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

        new_means_after_np_val = self._class_means[self._known_classes:self._total_classes].copy()
        new_covs_after_np_val = self._class_covs[self._known_classes:self._total_classes].detach().cpu().clone()

        if self._cur_task > 0:

            old_means_np_val = self._class_means[:self._total_classes - data_manager.get_task_size(self._cur_task)]
            old_covs_np_val = self._class_covs[:self._total_classes - data_manager.get_task_size(self._cur_task)]
            new_means_before_np_val = self.now_task_mean
            new_covs_before_np_val = self.now_task_covs
            new_means_after_np_val = self._class_means[
                                     self._total_classes - data_manager.get_task_size(self._cur_task):]
            new_covs_after_np_val = self._class_covs[self._total_classes - data_manager.get_task_size(self._cur_task):]
            corrected_means_np_val, corrected_covs_np_val = self.adjust_distribution.correct_prototypes(
                old_means_np_val,
                old_covs_np_val,
                new_means_before_np_val,
                new_covs_before_np_val,
                new_means_after_np_val,
                new_covs_after_np_val
                )

            apr_rows = []
            for class_id in range(self._known_classes):
                class_metadata = data_manager.get_class_metadata(class_id)
                mean_shift = torch.as_tensor(corrected_means_np_val[class_id]) - torch.as_tensor(
                    old_means_np_val[class_id]
                )
                covariance_change = corrected_covs_np_val[class_id].cpu() - old_covs_np_val[class_id].cpu()
                apr_rows.append(
                    {
                        "experiment_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task_id": self._cur_task + 1,
                        "stage": "task_final_apr",
                        **class_metadata,
                        "apr_active": bool(self.args["apr"]),
                        "mean_shift_l2": (
                            float(torch.linalg.vector_norm(mean_shift)) if self.args["apr"] else None
                        ),
                        "covariance_change_frobenius": (
                            float(torch.linalg.matrix_norm(covariance_change))
                            if self.args["apr"] else None
                        ),
                        "source_feature_space": f"task_{self._cur_task}:task_final",
                        "target_feature_space": self._feature_space_id("task_final"),
                    }
                )
            upsert_csv_rows(
                self.run_dir / "apr_diagnostics.csv",
                apr_rows,
                ("experiment_id", "seed", "task_id", "mapped_class_id", "stage"),
            )

            if self.args['apr']:
                self._class_covs[
                :self._total_classes - data_manager.get_task_size(self._cur_task)] = corrected_covs_np_val

                self._class_means[
                :self._total_classes - data_manager.get_task_size(self._cur_task)] = corrected_means_np_val
                print("mean_and_cov_update")

        self._update_region_statistics(
            data_manager,
            new_means_after_np_val,
            new_covs_after_np_val,
        )

        if self._cur_task > 0 and self.args['ca_epochs'] > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
        self._resume_stage = None
        self._resume_next_epoch = 0
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None

    def _train(self, train_loader, test_loader):
        model = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': model.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': model.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},

                {'params': model.fc.parameters(), 'lr': 0.1, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._restore_training_optimizers(optimizer, scheduler)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self._resume_next_epoch)

        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []

            param_groups.append(
                {'params': model.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})
            param_groups.append(
                {'params': model.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs,
                                                             eta_min=self.min_lr)
            self._restore_training_optimizers(optimizer, scheduler)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self._resume_next_epoch)

    def head_train(self, train_loader, test_loader):
        model = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        head_epochs = 10
        param_groups = []

        param_groups.append(
            {'params': model.convnet.parameters(), 'lr': 0.0, 'weight_decay': self.args['weight_decay']})
        param_groups.append(
            {'params': model.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']})

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs,
                                                         eta_min=self.min_lr)
        prog_bar = tqdm(range(head_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        for _, epoch in enumerate(prog_bar):
            model.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                outputs = model(inputs, label=targets)
                logits = outputs["logits"]

                loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                loss = loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(model, test_loader)
            info = "Task {}, turning head process ,Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)

        logging.info(info)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        prog_bar = tqdm(range(start_epoch, self.tuned_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        for epoch in prog_bar:
            epoch_started = time.perf_counter()
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            if self._separation_is_active() and epoch % int(self.args["sep_refresh_epochs"]) == 0:
                self._refresh_separation_cache(epoch)
            self._network.train()
            weighted_sums = {
                "classification": 0.0,
                "scc": 0.0,
                "feature_distillation": 0.0,
                "logit_distillation": 0.0,
                "separation": 0.0,
                "weighted_separation": 0.0,
                "original_structure": 0.0,
                "total": 0.0,
                "feature_norm": 0.0,
            }
            nearest_d2 = []
            nearest_rho = []
            intrusions = []
            sep_gradient_norm = None
            correct, processed_samples, optimizer_steps = 0, 0, 0
            scc_active = self._cur_task > 0 and bool(self.args["scc"])
            feature_active = self._cur_task > 0 and not scc_active and bool(self.args["feature"])
            logit_active = (
                self._cur_task > 0 and not scc_active and not feature_active and bool(self.args["logit"])
            )
            separation_active = self._separation_is_active()

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                current_batch_size = int(targets.shape[0])
                loss_dtl_scc = None
                loss_dtl_feature = None
                loss_dtl_logit = None

                if self._cur_task == 0:
                    outputs = self._network(inputs, label=targets)
                    logits = outputs["logits"]

                    loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                    original_loss = loss_cls
                else:
                    outputs = self._network(inputs, label=targets)
                    logits = outputs["logits"]

                    loss_cls = loss_cos(logits[:, self._known_classes:], targets - self._known_classes)
                    loss_dtl_scc = LobachevskyRelativeGeomLoss(outputs, self._class_means, self._device)
                    loss_dtl_feature = features_distill_loss(outputs)
                    loss_dtl_logit = logits_distill_loss(outputs, self._class_means, self._network, T=4.0)

                    if self.args['scc']:
                        # loss = loss_cls + 5.0 * loss_dtl_scc
                        original_loss = loss_cls + self.args['scc_value'] * loss_dtl_scc
                    elif self.args['feature']:
                        original_loss = loss_cls + 0.1 * loss_dtl_feature
                    elif self.args['logit']:
                        original_loss = loss_cls + 5.0 * loss_dtl_logit
                    else:
                        original_loss = loss_cls

                separation_loss = outputs["sfeatures"].sum() * 0.0
                if self._separation_is_active():
                    if self._separation_cache is None:
                        raise RuntimeError("separation is active but its epoch cache is missing")
                    separation_loss, separation_diagnostics = self._separation_cache.loss(outputs["sfeatures"])
                    nearest_d2.append(separation_diagnostics.nearest_d2.cpu())
                    nearest_rho.append(separation_diagnostics.nearest_rho.cpu())
                    intrusions.append(separation_diagnostics.intrusion.float().cpu())
                    if i == 0 and self.args.get("sep_grad_diagnostics", True):
                        trainable = [parameter for parameter in self._network.parameters() if parameter.requires_grad]
                        gradients = torch.autograd.grad(
                            separation_loss,
                            trainable,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        squared_norm = sum(
                            gradient.detach().float().square().sum()
                            for gradient in gradients
                            if gradient is not None
                        )
                        sep_gradient_norm = float(torch.sqrt(squared_norm).item()) if not isinstance(squared_norm, int) else 0.0
                weighted_separation = float(self.args["sep_lambda"]) * separation_loss
                loss = original_loss + weighted_separation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at task={self._cur_task + 1}, epoch={epoch + 1}, batch={i}"
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                weighted_sums["classification"] += float(loss_cls.item()) * current_batch_size
                if scc_active:
                    weighted_sums["scc"] += float(loss_dtl_scc.item()) * current_batch_size
                if feature_active:
                    weighted_sums["feature_distillation"] += (
                        float(loss_dtl_feature.item()) * current_batch_size
                    )
                if logit_active:
                    weighted_sums["logit_distillation"] += (
                        float(loss_dtl_logit.item()) * current_batch_size
                    )
                if separation_active:
                    weighted_sums["separation"] += float(separation_loss.item()) * current_batch_size
                    weighted_sums["weighted_separation"] += (
                        float(weighted_separation.item()) * current_batch_size
                    )
                weighted_sums["original_structure"] += float(original_loss.item()) * current_batch_size
                weighted_sums["total"] += float(loss.item()) * current_batch_size
                weighted_sums["feature_norm"] += float(
                    outputs["sfeatures"].detach().norm(dim=1).sum().item()
                )
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                processed_samples += current_batch_size
                optimizer_steps += 1

            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_started
            checkpoint_enabled = bool(self.args.get("checkpoint_every_epoch", True))
            checkpoint_state_id = (
                f"train_epoch:task_{self._cur_task + 1}:next_epoch_{epoch + 1}"
                if checkpoint_enabled else None
            )
            diagnostics = {
                "experiment_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task_id": self._cur_task + 1,
                "stage": "feature_training",
                "epoch": epoch + 1,
                "classification_active": True,
                "classification_raw": weighted_sums["classification"] / processed_samples,
                "classification_coefficient": 1.0,
                "classification_weighted": weighted_sums["classification"] / processed_samples,
                "scc_active": scc_active,
                "scc_raw": weighted_sums["scc"] / processed_samples if scc_active else None,
                "scc_coefficient": float(self.args["scc_value"]) if scc_active else None,
                "scc_weighted": (
                    weighted_sums["scc"] * float(self.args["scc_value"]) / processed_samples
                    if scc_active else None
                ),
                "feature_distillation_active": feature_active,
                "feature_distillation_raw": (
                    weighted_sums["feature_distillation"] / processed_samples
                    if feature_active else None
                ),
                "feature_distillation_coefficient": 0.1 if feature_active else None,
                "feature_distillation_weighted": (
                    weighted_sums["feature_distillation"] * 0.1 / processed_samples
                    if feature_active else None
                ),
                "logit_distillation_active": logit_active,
                "logit_distillation_raw": (
                    weighted_sums["logit_distillation"] / processed_samples if logit_active else None
                ),
                "logit_distillation_coefficient": 5.0 if logit_active else None,
                "logit_distillation_weighted": (
                    weighted_sums["logit_distillation"] * 5.0 / processed_samples
                    if logit_active else None
                ),
                "separation_active": separation_active,
                "separation_raw": (
                    weighted_sums["separation"] / processed_samples if separation_active else None
                ),
                "separation_coefficient": (
                    float(self.args["sep_lambda"]) if separation_active else None
                ),
                "separation_weighted": (
                    weighted_sums["weighted_separation"] / processed_samples
                    if separation_active else None
                ),
                "original_structure_loss": weighted_sums["original_structure"] / processed_samples,
                "total_loss": weighted_sums["total"] / processed_samples,
                "student_feature_norm": weighted_sums["feature_norm"] / processed_samples,
                "separation_parameter_gradient_norm_first_batch": sep_gradient_norm,
                "learning_rate": learning_rates[0],
                "learning_rates": learning_rates,
                "processed_samples": processed_samples,
                "optimizer_steps": optimizer_steps,
                "epoch_seconds": epoch_seconds,
                "nan_or_inf_detected": False,
                "numerical_fallback_count": (
                    sum(
                        int(bool(details.get("fallback_applied", False)))
                        for details in self._separation_cache.factorization_details
                    )
                    if separation_active else 0
                ),
                "checkpoint_path": (
                    str(self.run_dir / "checkpoint_latest.pt") if checkpoint_enabled else None
                ),
                "checkpoint_state_id": checkpoint_state_id,
            }
            if nearest_rho:
                rho = torch.cat(nearest_rho)
                d2 = torch.cat(nearest_d2)
                intrusion = torch.cat(intrusions)
                diagnostics.update(
                    {
                        "nearest_old_d2_mean": float(d2.mean()),
                        "nearest_old_d2_p05": float(torch.quantile(d2, 0.05)),
                        "nearest_old_d2_p50": float(torch.quantile(d2, 0.50)),
                        "nearest_old_d2_p95": float(torch.quantile(d2, 0.95)),
                        "nearest_old_rho_mean": float(rho.mean()),
                        "nearest_old_rho_p05": float(torch.quantile(rho, 0.05)),
                        "nearest_old_rho_p50": float(torch.quantile(rho, 0.50)),
                        "nearest_old_rho_p95": float(torch.quantile(rho, 0.95)),
                        "intrusion_ratio": float(intrusion.mean()),
                    }
                )
            append_jsonl(self.run_dir / "training_diagnostics.jsonl", diagnostics)
            upsert_csv_rows(
                self.run_dir / "epoch_metrics.csv",
                [diagnostics],
                ("experiment_id", "seed", "task_id", "stage", "epoch"),
            )

            train_acc = np.around(tensor2numpy(correct) * 100 / processed_samples, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                diagnostics["total_loss"],
                train_acc,
                test_acc,
            )

            prog_bar.set_description(info)
            if checkpoint_enabled:
                self.save_experiment_checkpoint(
                    self.run_dir / "checkpoint_latest.pt",
                    stage="train_epoch",
                    next_epoch=epoch + 1,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            metadata_updates = {
                "updated_at_utc": utc_now_iso(),
                "last_recorded_stage": "feature_training",
                "last_recorded_task": self._cur_task + 1,
                "last_recorded_epoch": epoch + 1,
            }
            if checkpoint_enabled:
                metadata_updates.update(
                    {
                        "last_checkpoint_state_id": checkpoint_state_id,
                        "last_checkpoint_path": str(self.run_dir / "checkpoint_latest.pt"),
                    }
                )
            update_json(self.run_dir / "run_metadata.json", metadata_updates)

        if start_epoch >= self.tuned_epochs:
            logging.info("Training epochs already complete in resume checkpoint; continuing task finalization")

    def _separation_is_active(self):
        return (
            self._cur_task > 0
            and bool(self.args.get("old_new_separation", False))
            and float(self.args.get("sep_lambda", 0.0)) > 0.0
        )

    def _feature_space_id(self, stage):
        model = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        student_adapter = 2 if model.convnet.flag % 2 == 1 else 1
        return f"task_{self._cur_task}:{stage}:student_adapter_{student_adapter}:raw_768d"

    def _collect_current_features(self, feature_key="sfeatures", return_sample_ids=False):
        was_training = self._network.training
        self._network.eval()
        features, labels, sample_ids = [], [], []
        with torch.no_grad():
            for indices, inputs, targets in self.stats_loader:
                outputs = self._network(inputs.to(self._device))
                features.append(outputs[feature_key].detach().cpu().float())
                labels.append(targets.detach().cpu().long())
                if return_sample_ids:
                    for index in indices.tolist():
                        source = self.stats_loader.dataset.images[int(index)]
                        if self.stats_loader.dataset.use_path:
                            source = Path(source)
                            train_root = Path(self.data_manager.split_roots.get("train", source.parent))
                            try:
                                source = source.relative_to(train_root)
                            except ValueError:
                                pass
                            sample_ids.append(source.as_posix())
                        else:
                            sample_ids.append(str(index))
        self._network.train(was_training)
        collected = (torch.cat(features, dim=0), torch.cat(labels, dim=0))
        return (*collected, sample_ids) if return_sample_ids else collected

    def _statistics_from_features(self, features, labels):
        class_count = self._total_classes - self._known_classes
        means = np.zeros((class_count, features.shape[1]), dtype=np.float32)
        covariances = torch.zeros((class_count, features.shape[1], features.shape[1]), dtype=torch.float32)
        for offset, class_id in enumerate(range(self._known_classes, self._total_classes)):
            class_features = features[labels == class_id]
            if class_features.shape[0] == 0:
                raise RuntimeError(f"no current-task features found for class {class_id}")
            means[offset] = class_features.mean(dim=0).numpy()
            centered = class_features - class_features.mean(dim=0, keepdim=True)
            if class_features.shape[0] > 1:
                covariance = centered.T @ centered / float(class_features.shape[0] - 1)
            else:
                covariance = torch.zeros((features.shape[1], features.shape[1]), dtype=torch.float32)
            covariance = 0.5 * (covariance + covariance.T)
            covariance.diagonal().add_(float(self.args["stats_cov_jitter"]))
            covariances[offset] = covariance
        return means, covariances

    def _refresh_separation_cache(self, epoch):
        if self._task_source_means is None or self._sep_reference_means is None:
            raise RuntimeError("epoch APR source/reference statistics are missing")
        rng_state = capture_rng_state()
        was_training = self._network.training
        started = time.perf_counter()
        try:
            target_features, target_labels = self._collect_current_features("sfeatures")
            target_means, target_covariances = self._statistics_from_features(target_features, target_labels)
            feature_space = self._feature_space_id(f"epoch_{epoch}_cache")
            if self.args.get("apr", True):
                corrected_means, corrected_covariances = self.adjust_distribution.correct_prototypes(
                    self._task_source_means,
                    self._task_source_covs,
                    self._sep_reference_means,
                    self._sep_reference_covs,
                    target_means,
                    target_covariances,
                )
                mean_shifts, covariance_additions = self.adjust_distribution.class_level_corrections(
                    self._task_source_means,
                    self._sep_reference_means,
                    self._sep_reference_covs,
                    target_means,
                    target_covariances,
                )
            else:
                corrected_means = self._task_source_means.copy()
                corrected_covariances = self._task_source_covs.clone()
                mean_shifts = torch.zeros_like(torch.from_numpy(self._task_source_means)).to(self._device)
                covariance_additions = torch.zeros_like(self._task_source_covs).to(self._device)

            if self.args.get("multi_center", False):
                if self.args.get("apr", True):
                    cache_regions_by_class = rectify_local_regions(
                        self._task_source_local_stats,
                        mean_shifts,
                        covariance_additions,
                        feature_space,
                        self.adjust_distribution.epsilon_cov_reg,
                    )
                else:
                    cache_regions_by_class = clone_regions(self._task_source_local_stats)
                    for regions in cache_regions_by_class.values():
                        for region in regions:
                            region["feature_space"] = feature_space
            else:
                cache_regions_by_class = {}
                for class_id in range(self._known_classes):
                    boundary = self._single_regions[class_id][0]
                    region = dict(boundary)
                    region["mean"] = torch.as_tensor(corrected_means[class_id]).detach().cpu().float()
                    region["covariance"] = torch.as_tensor(corrected_covariances[class_id]).detach().cpu().float()
                    region["tau_by_covariance"] = dict(boundary["tau_by_covariance"])
                    region["regularization"] = dict(boundary.get("regularization", {}))
                    region["feature_space"] = feature_space
                    cache_regions_by_class[class_id] = [region]

            regions = flatten_regions(cache_regions_by_class)
            self._separation_cache = SeparationCache(
                regions,
                self.args["separation_covariance"],
                feature_space,
                self.args,
            )
            elapsed = time.perf_counter() - started
            cache_record = {
                "task": self._cur_task,
                "epoch": epoch,
                "source_model": "previous-task final student copied to frozen teacher adapter",
                "source_feature_space": sorted(
                    {region["feature_space"] for region in flatten_regions(
                        self._task_source_local_stats if self.args.get("multi_center", False) else self._single_regions
                    )}
                ),
                "target_feature_space": feature_space,
                "boundary_policy": self.args["sep_boundary_policy"],
                "timing": self.args["sep_timing"],
                "covariance_view": self.args["separation_covariance"],
                "region_count": self._separation_cache.region_count,
                "cache_storage_bytes": self._separation_cache.storage_bytes,
                "refresh_seconds": elapsed,
                "factorization_fallbacks": sum(
                    bool(details.get("fallback_applied", False))
                    for details in self._separation_cache.factorization_details
                ),
            }
            append_jsonl(self.run_dir / "separation_cache.jsonl", cache_record)
            logging.info("Separation cache: %s", cache_record)
        finally:
            restore_rng_state(rng_state)
            self._network.train(was_training)

    def _update_region_statistics(self, data_manager, new_means_after, new_covariances_after):
        needs_regions = bool(self.args.get("multi_center", False) or self.args.get("old_new_separation", False))
        if not needs_regions:
            return
        rng_state = capture_rng_state()
        try:
            features, labels, sample_ids = self._collect_current_features(
                "sfeatures", return_sample_ids=True
            )
        finally:
            restore_rng_state(rng_state)
        final_feature_space = self._feature_space_id("task_final")
        new_regions = {}
        kmeans_records = []
        cluster_summary_rows = []
        cluster_detail_rows = []
        cluster_assignment_rows = []
        for offset, class_id in enumerate(range(self._known_classes, self._total_classes)):
            class_mask = labels == class_id
            class_features = features[class_mask]
            if self.args.get("multi_center", False):
                regions, kmeans = build_regions_for_class(
                    class_features,
                    class_id=class_id,
                    birth_task=self._cur_task,
                    feature_space=final_feature_space,
                    num_prototypes=int(self.args["num_prototypes"]),
                    seed=int(self.args["seed"]) + class_id * 1009,
                    config=self.args,
                )
                new_regions[class_id] = regions
                kmeans_records.append(
                    {
                        "class_id": class_id,
                        **{key: value for key, value in kmeans.items() if key != "assignments"},
                    }
                )
                class_metadata = data_manager.get_class_metadata(class_id)
                clustering_run_id = (
                    f"{self.args['experiment_id']}:seed_{self.args['seed']}:task_{self._cur_task + 1}:"
                    f"task_final:class_{class_id}"
                )
                common = {
                    "experiment_id": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task_id": self._cur_task + 1,
                    "clustering_run_id": clustering_run_id,
                    "stage": "task_final_statistics",
                    "epoch": None,
                    **class_metadata,
                    "class_sample_count": int(class_features.shape[0]),
                    "requested_k": int(kmeans["requested_k"]),
                    "actual_k": int(kmeans["actual_k"]),
                    "fallback_applied": bool(kmeans["fallback_applied"]),
                    "fallback_reasons": kmeans["fallback_reasons"],
                    "small_cluster_policy": kmeans["small_cluster_policy"],
                    "feature_space": final_feature_space,
                    "feature_checkpoint_key": f"task_{self._cur_task + 1}:feature_training_complete",
                    "statistics_checkpoint_path": str(self.run_dir / "checkpoint_latest.pt"),
                }
                cluster_summary_rows.append(
                    {
                        **common,
                        "cluster_counts": [int(region["count"]) for region in regions],
                        "cluster_ratios": [float(region["weight"]) for region in regions],
                    }
                )
                for region in regions:
                    cluster_detail_rows.append(
                        {
                            **common,
                            "cluster_id": int(region["component_id"]),
                            "cluster_sample_count": int(region["count"]),
                            "cluster_ratio": float(region["weight"]),
                            "covariance_rank": int(region["rank"]),
                            "covariance_fallback": region["fallback"],
                            "tau_isotropic": region["tau_by_covariance"]["isotropic"],
                            "tau_diagonal": region["tau_by_covariance"]["diagonal"],
                            "tau_full": region["tau_by_covariance"]["full"],
                            "statistics_key": (
                                f"local_stats[{class_id}][{region['component_id']}]"
                            ),
                        }
                    )
                class_sample_ids = [
                    sample_id for sample_id, selected in zip(sample_ids, class_mask.tolist()) if selected
                ]
                for sample_id, cluster_id in zip(class_sample_ids, kmeans["assignments"]):
                    cluster_assignment_rows.append(
                        {
                            "experiment_id": self.args["experiment_id"],
                            "seed": self.args["seed"],
                            "task_id": self._cur_task + 1,
                            "clustering_run_id": clustering_run_id,
                            **class_metadata,
                            "sample_id": sample_id,
                            "cluster_id": int(cluster_id),
                        }
                    )
            else:
                new_regions[class_id] = [
                    build_single_region(
                        class_features,
                        class_id=class_id,
                        birth_task=self._cur_task,
                        feature_space=final_feature_space,
                        mean=torch.as_tensor(new_means_after[offset]),
                        covariance=new_covariances_after[offset],
                        config=self.args,
                    )
                ]

        if cluster_summary_rows:
            upsert_csv_rows(
                self.run_dir / "cluster_summary.csv",
                cluster_summary_rows,
                ("clustering_run_id",),
            )
            upsert_csv_rows(
                self.run_dir / "cluster_details.csv",
                cluster_detail_rows,
                ("clustering_run_id", "cluster_id"),
            )
            upsert_csv_rows(
                self.run_dir / "cluster_assignments.csv",
                cluster_assignment_rows,
                ("clustering_run_id", "sample_id"),
            )

        old_regions = {}
        if self._cur_task > 0:
            if self.args.get("multi_center", False):
                if self.args.get("apr", True):
                    mean_shifts, covariance_additions = self.adjust_distribution.class_level_corrections(
                        self._task_source_means,
                        self.now_task_mean,
                        self.now_task_covs,
                        new_means_after,
                        new_covariances_after,
                    )
                    old_regions = rectify_local_regions(
                        self._task_source_local_stats,
                        mean_shifts,
                        covariance_additions,
                        final_feature_space,
                        self.adjust_distribution.epsilon_cov_reg,
                    )
                else:
                    old_regions = clone_regions(self._task_source_local_stats)
                    for regions in old_regions.values():
                        for region in regions:
                            region["feature_space"] = final_feature_space
            else:
                for class_id in range(self._known_classes):
                    boundary = self._single_regions[class_id][0]
                    region = dict(boundary)
                    region["mean"] = torch.as_tensor(self._class_means[class_id]).detach().cpu().float()
                    region["covariance"] = self._class_covs[class_id].detach().cpu().float().clone()
                    region["tau_by_covariance"] = dict(boundary["tau_by_covariance"])
                    region["regularization"] = dict(boundary.get("regularization", {}))
                    region["feature_space"] = final_feature_space
                    old_regions[class_id] = [region]

        merged = {**old_regions, **new_regions}
        if self.args.get("multi_center", False):
            self._local_stats = merged
        else:
            self._single_regions = merged
        write_json(
            self.run_dir / f"region_statistics_task_{self._cur_task}.json",
            {
                "task": self._cur_task,
                "multi_center": bool(self.args.get("multi_center", False)),
                "storage_bytes": region_storage_bytes(merged),
                "regions": summarize_regions(merged),
                "kmeans": kmeans_records,
                "mixture_vs_class_moments": compare_mixture_to_class_statistics(
                    merged, self._class_means, self._class_covs
                ),
            },
        )
        if (
            self.args.get("old_new_separation", False)
            and float(self.args.get("sep_lambda", 0.0)) > 0.0
            and self._cur_task > 0
        ):
            evaluation_cache = SeparationCache(
                flatten_regions(old_regions),
                self.args["separation_covariance"],
                final_feature_space,
                self.args,
            )
            d2_values, rho_values, intrusion_values = [], [], []
            with torch.no_grad():
                for start in range(0, features.shape[0], int(self.args["stats_batch_size"])):
                    batch = features[start : start + int(self.args["stats_batch_size"])].to(self._device)
                    separation_loss, diagnostics = evaluation_cache.loss(batch)
                    d2_values.append(diagnostics.nearest_d2.cpu())
                    rho_values.append(diagnostics.nearest_rho.cpu())
                    intrusion_values.append(diagnostics.intrusion.float().cpu())
            d2_values = torch.cat(d2_values)
            rho_values = torch.cat(rho_values)
            intrusion_values = torch.cat(intrusion_values)
            append_jsonl(
                self.run_dir / "separation_task_end_diagnostics.jsonl",
                {
                    "stage": "task_end_eval",
                    "task": self._cur_task,
                    "feature_space": final_feature_space,
                    "covariance_view": self.args["separation_covariance"],
                    "region_count": evaluation_cache.region_count,
                    "nearest_old_d2_mean": float(d2_values.mean()),
                    "nearest_old_d2_p05": float(torch.quantile(d2_values, 0.05)),
                    "nearest_old_d2_p50": float(torch.quantile(d2_values, 0.50)),
                    "nearest_old_d2_p95": float(torch.quantile(d2_values, 0.95)),
                    "nearest_old_rho_mean": float(rho_values.mean()),
                    "nearest_old_rho_p05": float(torch.quantile(rho_values, 0.05)),
                    "nearest_old_rho_p50": float(torch.quantile(rho_values, 0.50)),
                    "nearest_old_rho_p95": float(torch.quantile(rho_values, 0.95)),
                    "intrusion_ratio": float(intrusion_values.mean()),
                },
            )
            self._separation_cache = evaluation_cache

    def _restore_training_optimizers(self, optimizer, scheduler):
        if self._resume_stage != "train_epoch":
            return
        if self._resume_optimizer_state is not None:
            optimizer.load_state_dict(self._resume_optimizer_state)
        if self._resume_scheduler_state is not None:
            scheduler.load_state_dict(self._resume_scheduler_state)

    def _checkpoint_network(self):
        return self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    def save_experiment_checkpoint(self, path, stage, next_epoch=0, optimizer=None, scheduler=None):
        state_id = (
            f"train_epoch:task_{self._cur_task + 1}:next_epoch_{int(next_epoch)}"
            if stage == "train_epoch"
            else f"{stage}:task_{self._cur_task + 1}"
        )
        payload = {
            "format_version": 1,
            "state_id": state_id,
            "stage": stage,
            "next_epoch": int(next_epoch),
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "task_sizes": list(self.task_sizes),
            "network_state_dict": self._checkpoint_network().state_dict(),
            "class_means": getattr(self, "_class_means", None),
            "class_covs": getattr(self, "_class_covs", None),
            "radius": getattr(self, "radius", None),
            "local_stats": self._local_stats,
            "single_regions": self._single_regions,
            "task_source_means": self._task_source_means,
            "task_source_covs": self._task_source_covs,
            "task_source_local_stats": self._task_source_local_stats,
            "sep_reference_means": self._sep_reference_means,
            "sep_reference_covs": self._sep_reference_covs,
            "now_task_mean": getattr(self, "now_task_mean", None),
            "now_task_covs": getattr(self, "now_task_covs", None),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": capture_rng_state(),
            "config": self.args,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        write_json(
            self.run_dir / "checkpoint_manifest.json",
            {
                "checkpoint_path": str(target.resolve()),
                "checkpoint_state_id": state_id,
                "stage": stage,
                "task_id": self._cur_task + 1,
                "next_epoch": int(next_epoch),
                "updated_at_utc": utc_now_iso(),
                "note": "checkpoint_latest.pt is replaced atomically and represents only this latest state",
            },
        )
        return state_id

    def load_experiment_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        checkpoint_config = checkpoint.get("config", {})
        compatibility_keys = (
            "experiment_id",
            "dataset",
            "seed",
            "init_cls",
            "increment",
            "expected_tasks",
            "multi_center",
            "num_prototypes",
            "old_new_separation",
            "separation_covariance",
            "sep_lambda",
            "sep_percentile",
            "sep_timing",
            "sep_boundary_policy",
        )
        mismatches = {
            key: {"checkpoint": checkpoint_config.get(key), "current": self.args.get(key)}
            for key in compatibility_keys
            if checkpoint_config.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"checkpoint/config mismatch: {mismatches}")
        checkpoint_sizes = list(checkpoint["task_sizes"])
        if self._checkpoint_network().fc is not None:
            raise RuntimeError("load_experiment_checkpoint must be called on a fresh learner")
        for size in checkpoint_sizes:
            self._checkpoint_network().update_fc(size)
        self._checkpoint_network().load_state_dict(checkpoint["network_state_dict"])
        self.task_sizes = checkpoint_sizes
        self._cur_task = int(checkpoint["cur_task"])
        self._known_classes = int(checkpoint["known_classes"])
        self._total_classes = int(checkpoint["total_classes"])
        if checkpoint["class_means"] is not None:
            self._class_means = checkpoint["class_means"]
            self._class_covs = checkpoint["class_covs"]
        self.radius = checkpoint.get("radius")
        self._local_stats = checkpoint.get("local_stats", {})
        self._single_regions = checkpoint.get("single_regions", {})
        self._task_source_means = checkpoint.get("task_source_means")
        self._task_source_covs = checkpoint.get("task_source_covs")
        self._task_source_local_stats = checkpoint.get("task_source_local_stats", {})
        self._sep_reference_means = checkpoint.get("sep_reference_means")
        self._sep_reference_covs = checkpoint.get("sep_reference_covs")
        self.now_task_mean = checkpoint.get("now_task_mean")
        self.now_task_covs = checkpoint.get("now_task_covs")
        self._resume_stage = checkpoint["stage"]
        self._resume_next_epoch = int(checkpoint.get("next_epoch", 0))
        self._resume_optimizer_state = checkpoint.get("optimizer_state_dict")
        self._resume_scheduler_state = checkpoint.get("scheduler_state_dict")
        restore_rng_state(checkpoint["rng_state"])
        logging.info(
            "Loaded checkpoint %s (stage=%s task=%d next_epoch=%d)",
            path,
            self._resume_stage,
            self._cur_task,
            self._resume_next_epoch,
        )
        return checkpoint

    def compute_irr_ratio(self):
        block_len = self._update_grads[self._cur_task]
        finetune_block = []
        ratio_list = []
        for block in self._update_grads[self._cur_task].keys():
            ratio = self._update_grads[self._cur_task][block] / self._update_grads[self._cur_task - 1][block]
            ratio_list.append(ratio)
            if ratio >= 0.9 and ratio <= 1.1:
                finetune_block.append(block)

        print("ratio", ratio_list)
        return block

    def cnt_match_block(self, old_blocks, new_blocks):
        finetune_block = []
        for nb in new_blocks:
            is_match = False
            for ob in old_blocks:
                if nb == ob:
                    is_match = True
                    break
            if is_match is False:
                finetune_block.append(nb)
        return finetune_block

    def compute_sentive(self):
        self._network.eval()
        sentive_network = copy.deepcopy(self._network)
        param_groups = [
            {'params': sentive_network.convnet.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']},
            {'params': sentive_network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)

        update_magnitudes = {}
        for i, (_, inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = sentive_network(inputs)["logits"]
            loss = F.cross_entropy(logits[:, self._known_classes:], targets - self._known_classes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for j, (name, param) in enumerate(sentive_network.named_parameters()):
                if "adapt" in name:
                    if name in update_magnitudes:
                        update_magnitudes[name] += (param.grad ** 2)
                    else:
                        update_magnitudes[name] = (param.grad ** 2)
        grad_shapes = {}
        grad_shapes_int = {}
        for key in update_magnitudes.keys():
            grad_shapes[key] = update_magnitudes[key].shape
            grad_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
        large_tensor = torch.cat([update_magnitudes[key].flatten() for key in grad_shapes.keys()])
        _, indexes = large_tensor.topk(math.ceil(0.0001 * large_tensor.shape[0]))
        print(indexes)

        tmp_large_tensor = torch.zeros_like(large_tensor, device=self._device)
        tmp_large_tensor[indexes] = 1.

        tmp_large_tensor_list = tmp_large_tensor.split([shape for shape in grad_shapes_int.values()])

        structured_param_num = 0
        structured_names = []
        tuned_vectors = []

        unstructured_param_num = 0
        unstructured_name_shapes = {}
        unstructured_name_shapes_int = {}
        unstructured_grad_mask = {}
        grad_sum_dict = {}
        for i, key in enumerate(grad_shapes.keys()):
            grad_sum = tmp_large_tensor_list[i].view(grad_shapes[key]).sum()
            grad_sum_dict[key] = grad_sum
            cur_param_num = grad_sum.item()

            unstructured_param_num += grad_sum.item()
            unstructured_name_shapes[key] = tmp_large_tensor_list[i].view(grad_shapes[key]).shape
            unstructured_name_shapes_int[key] = np.cumprod(list(update_magnitudes[key].shape))[-1]
            unstructured_grad_mask[key] = tmp_large_tensor_list[i].view(grad_shapes[key])

        return unstructured_grad_mask

    def distill_loss(self, outputs):
        cls_token = outputs["sfeatures"]
        l2_d_st = torch.norm(outputs["sfmaps"] - outputs["tfmaps"], p=2, dim=-1)
        l2_d = torch.norm(outputs["sfmaps"] - cls_token.unsqueeze(1), p=1, dim=-1)
        l2_d = 1 / (l2_d + 1e-8)
        l2_d = l2_d / l2_d.max(dim=-1, keepdims=True).values
        distill_loss = torch.mean(l2_d * l2_d_st) / l2_d_st.shape[-1]
        cls_loss = torch.mean(torch.norm(outputs["sfeatures"] - outputs["tfeatures"], p=2, dim=-1))
        return distill_loss + 0.1 * cls_loss

    def distill_loss_cos(self, outputs, normalize_input_features=True, cls_loss_weight=0.1, detach_weights=False):
        s_cls_feats = outputs["sfeatures"]
        t_cls_feats = outputs["tfeatures"]
        s_patch_feats = outputs["sfmaps"]
        t_patch_feats = outputs["tfmaps"]
        if normalize_input_features:
            s_cls_feats = F.normalize(s_cls_feats, p=2, dim=-1)
            t_cls_feats = F.normalize(t_cls_feats, p=2, dim=-1)
            s_patch_feats = F.normalize(s_patch_feats, p=2, dim=-1)
            t_patch_feats = F.normalize(t_patch_feats, p=2, dim=-1)

        cos_sim_ps_pt = torch.cosine_similarity(s_patch_feats, t_patch_feats, dim=-1)
        cos_dist_ps_pt = 1.0 - cos_sim_ps_pt

        cos_sim_ps_sc = torch.cosine_similarity(s_patch_feats, s_cls_feats.unsqueeze(1), dim=-1)
        cos_dist_ps_sc = 1.0 - cos_sim_ps_sc

        weights = 1.0 / (cos_dist_ps_sc + 1e-8)
        weights = weights / (weights.max(dim=-1, keepdim=True).values + 1e-8)
        if detach_weights:
            weights = weights.detach()

        patch_distill_loss = torch.mean(weights * cos_dist_ps_pt)
        cos_sim_cs_ct = torch.cosine_similarity(s_patch_feats, t_patch_feats, dim=-1)
        cos_dict_cs_ct = 1.0 - cos_sim_cs_ct
        cls_token_distill_loss = torch.mean(cos_dict_cs_ct)

        total_loss = patch_distill_loss + 0. * cls_loss_weight * cls_token_distill_loss
        return total_loss


def pairwise_euclidean_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    a_squared = np.sum(a ** 2, axis=1, keepdims=True)
    b_squared = np.sum(b ** 2, axis=1, keepdims=True).T
    ab = a @ b.T
    dist = np.sqrt(a_squared - 2 * ab + b_squared + 1e-8)
    return dist


def cos(old_task_mean, new_task_mean_old_model, new_task_mean_new_model):
    old_task_mean = np.asarray(old_task_mean)
    new_task_mean_old_model = np.asarray(new_task_mean_old_model)
    new_task_mean_new_model = np.asarray(new_task_mean_new_model)

    gap = new_task_mean_new_model - new_task_mean_old_model

    old_expand = old_task_mean[:, np.newaxis, :]
    new_expand = new_task_mean_old_model[np.newaxis, :, :]

    vec = old_expand - new_expand

    gap_expand = gap[np.newaxis, :, :]

    dot_product = np.sum(vec * gap_expand, axis=2)
    vec_norm = np.linalg.norm(vec, axis=2)
    gap_norm = np.linalg.norm(gap_expand, axis=2)

    gap_norm = np.broadcast_to(gap_norm, vec_norm.shape)

    cos_theta = dot_product / (vec_norm * gap_norm + 1e-8)
    return cos_theta


def compute_D(d, cos_theta, k):
    exponent = d + k * cos_theta
    D = np.exp(-exponent)
    row_sum = np.sum(D, axis=1, keepdims=True)
    D = D / (row_sum + 1e-8)
    return D


def correct_covariance_strategy_2(
        gap: np.ndarray,
        old_class_covs: torch.Tensor,
        flag=0
) -> torch.Tensor:
    epsilon_norm = 1e-9
    epsilon_eig = 1e-6

    if not isinstance(gap, np.ndarray):
        raise TypeError("gap must be a NumPy ndarray.")
    if not isinstance(old_class_covs, torch.Tensor):
        raise TypeError("old_class_covs must be a PyTorch Tensor.")
    if gap.ndim != 3:
        raise ValueError(f"gap must be a 3D array (N, M, D), got shape {gap.shape}")
    if old_class_covs.ndim != 3:
        raise ValueError(f"old_class_covs must be a 3D tensor (N, D, D), got shape {old_class_covs.shape}")
    if gap.shape[0] != old_class_covs.shape[0]:
        raise ValueError(
            f"Number of classes (N) mismatch: gap has {gap.shape[0]}, "
            f"old_class_covs has {old_class_covs.shape[0]}."
        )
    if gap.shape[2] != old_class_covs.shape[1]:
        raise ValueError(
            f"Feature dimension (D) mismatch: gap has {gap.shape[2]}, "
            f"old_class_covs has {old_class_covs.shape[1]}."
        )
    if old_class_covs.shape[1] != old_class_covs.shape[2]:
        raise ValueError(
            f"Covariance matrices must be square (D, D), got D1={old_class_covs.shape[1]}, D2={old_class_covs.shape[2]}"
        )

    N = gap.shape[0]
    M = gap.shape[1]
    D = gap.shape[2]

    device = old_class_covs.device
    dtype = old_class_covs.dtype

    gap_tensor = torch.from_numpy(gap).to(device=device, dtype=dtype)

    corrected_covs_list = []
    all_coeffs_log = []

    for i in range(N):
        sigma_c_old = old_class_covs[i]

        total_final_correction_for_class_i = torch.zeros_like(sigma_c_old)

        current_class_coeffs_log = []

        for m in range(M):
            delta_mu_c_m = gap_tensor[i, m, :]

            norm_delta_mu_c_m = torch.linalg.norm(delta_mu_c_m)

            if norm_delta_mu_c_m < epsilon_norm:
                var_c_m = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                v_c_m = delta_mu_c_m / norm_delta_mu_c_m
                var_c_m = v_c_m @ sigma_c_old @ v_c_m

            var_c_m = torch.clamp(var_c_m, min=0.0)

            epsilon_var = 1e-9
            alpha_base = 10.
            if flag == 0:
                ratio = (torch.sum(delta_mu_c_m ** 2)) / (var_c_m + epsilon_var)
                coeff_c_m = alpha_base * torch.log1p(ratio)
            elif flag == 1:
                coeff_c_m = alpha_base / (var_c_m + epsilon_var)
            elif flag == 2:
                avg_std_dev_class_c = torch.sqrt(torch.trace(sigma_c_old) / D + epsilon_norm)
                norm_delta_mu = torch.linalg.norm(delta_mu_c_m)

                relative_shift_strength = norm_delta_mu / (avg_std_dev_class_c + epsilon_norm)
                coeff_c_m = alpha_base * relative_shift_strength
            else:
                raise ValueError('flag input error')
            current_class_coeffs_log.append(coeff_c_m.item())

            raw_correction_term_m = delta_mu_c_m.unsqueeze(1) @ delta_mu_c_m.unsqueeze(0)

            final_correction_term_m = coeff_c_m * raw_correction_term_m

            total_final_correction_for_class_i += final_correction_term_m

        all_coeffs_log.append(current_class_coeffs_log)

        sigma_c_tilde = sigma_c_old + total_final_correction_for_class_i

        corrected_covs_list.append(sigma_c_tilde)

    corrected_covs_tensor = torch.stack(corrected_covs_list, dim=0)

    return corrected_covs_tensor
