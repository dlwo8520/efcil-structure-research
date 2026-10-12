"""SGDS Eq. (4)-(10) activation guidance for the official PCDS learner.

Only semantic strategy formulation, semantic exploration, activation
compaction, and the associated counters are implemented.  SGDS adapter-bank
retrieval, fusion, entropy selection, MOS alignment, and its classifier are
intentionally excluded.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


KNOWLEDGE_REUSE = "KNOWLEDGE_REUSE"
NEW_SUBSPACE_ALLOCATION = "NEW_SUBSPACE_ALLOCATION"


class SGDSActivationGuidance(nn.Module):
    """Training-only SGDS mask with persistent global/class counters."""

    def __init__(
        self,
        feature_dim,
        num_classes,
        num_epochs,
        target_layer,
        paper_k=0.6,
        beta=0.5,
        gamma=1.0,
        exploration_ratio=0.5,
        mask_seed=100003,
        epsilon=1.0e-12,
    ):
        super().__init__()
        if not 0.0 < paper_k <= 1.0:
            raise ValueError("paper_k must be in (0, 1]")
        if not 0.0 < exploration_ratio < 1.0:
            raise ValueError("exploration_ratio must be in (0, 1)")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.num_epochs = int(num_epochs)
        self.target_layer = int(target_layer)
        self.paper_k = float(paper_k)
        # Algorithm 1 applies TopK(floor(kN)); k is therefore retained here.
        self.retain_dimensions = max(1, math.floor(self.paper_k * self.feature_dim))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.exploration_ratio = float(exploration_ratio)
        self.mask_seed = int(mask_seed)
        self.exploration_epochs = int(math.ceil(self.num_epochs * self.exploration_ratio))
        self.epsilon = float(epsilon)

        self.register_buffer("global_counter", torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer(
            "class_counter", torch.zeros(num_classes, feature_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "semantic_prototypes", torch.zeros(num_classes, feature_dim, dtype=torch.float32)
        )
        self.register_buffer("semantic_available", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer(
            "semantic_probabilities", torch.zeros(num_classes, num_classes, dtype=torch.float32)
        )
        self.register_buffer("strategy_code", torch.full((num_classes,), -1, dtype=torch.int8))
        self.register_buffer("score_old", torch.zeros(num_classes, dtype=torch.float32))
        self.register_buffer("score_new", torch.zeros(num_classes, dtype=torch.float32))

        self._batch_labels = None  # type: Optional[torch.Tensor]
        self._epoch = None  # type: Optional[int]
        self._current_task = None  # type: Optional[int]
        self._current_begin = 0
        self._current_end = 0
        self._mask_generators = {}
        self._reset_task_diagnostics()

    @property
    def requested_zero_ratio(self):
        return 1.0 - self.retain_dimensions / self.feature_dim

    def _reset_task_diagnostics(self):
        self._task_total_decisions = 0
        self._task_active_decisions = 0
        self._task_probability_sum = 0.0
        self._task_probability_count = 0
        self._task_class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self._task_class_active = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def prepare_task(self, task_id, begin_class, current_prototypes, num_epochs=None):
        """Store frozen-PTM prototypes and implement semantic Eq. (4)-(6)."""
        if num_epochs is not None:
            self.num_epochs = int(num_epochs)
            self.exploration_epochs = int(
                math.ceil(self.num_epochs * self.exploration_ratio)
            )
        begin_class = int(begin_class)
        current = current_prototypes.detach().to(
            device=self.semantic_prototypes.device, dtype=torch.float32
        )
        classes = int(current.shape[0])
        end_class = begin_class + classes
        if current.shape != (classes, self.feature_dim):
            raise ValueError("Invalid semantic prototype shape {}".format(tuple(current.shape)))
        if begin_class and not bool(self.semantic_available[:begin_class].all()):
            raise RuntimeError("Old semantic prototypes are missing")
        if bool(self.semantic_available[begin_class:end_class].any()):
            raise RuntimeError("Current semantic prototypes were already registered")

        self.semantic_prototypes[begin_class:end_class].copy_(current)
        self.semantic_available[begin_class:end_class] = True
        seen = F.normalize(self.semantic_prototypes[:end_class], dim=1)
        cosine = F.normalize(current, dim=1).matmul(seen.transpose(0, 1))
        conditional = F.softmax(cosine, dim=1)  # Eq. (4)
        rows = []  # type: List[Dict[str, Any]]
        for local_id in range(classes):
            class_id = begin_class + local_id
            self.semantic_probabilities[class_id, :end_class].copy_(conditional[local_id])
            old_score = float(conditional[local_id, :begin_class].sum().item())
            # Eq. (6) and Algorithm 1 include the class itself.
            new_score = float(conditional[local_id, begin_class:end_class].sum().item())
            reuse = old_score > new_score
            self.score_old[class_id] = old_score
            self.score_new[class_id] = new_score
            self.strategy_code[class_id] = 1 if reuse else 0
            old_cosine = cosine[local_id, :begin_class]
            rows.append({
                "task_id": int(task_id),
                "class_id": class_id,
                "old_classes": begin_class,
                "new_classes": classes,
                "s_old": old_score,
                "s_new": new_score,
                "strategy": KNOWLEDGE_REUSE if reuse else NEW_SUBSPACE_ALLOCATION,
                "eq6_includes_self": 1,
                "max_old_cosine_similarity": (
                    float(old_cosine.max().item()) if old_cosine.numel() else float("nan")
                ),
                "mean_old_cosine_similarity": (
                    float(old_cosine.mean().item()) if old_cosine.numel() else float("nan")
                ),
            })
        self._current_task = int(task_id)
        self._current_begin = begin_class
        self._current_end = end_class
        self._reset_task_diagnostics()
        return rows

    def set_batch_context(self, labels, epoch):
        labels = labels.detach().long()
        if labels.ndim != 1:
            raise ValueError("SGDS labels must be a vector")
        if self._current_task is None:
            raise RuntimeError("prepare_task must be called first")
        if bool((labels < self._current_begin).any()) or bool(
            (labels >= self._current_end).any()
        ):
            raise RuntimeError("SGDS received a class outside the current task")
        self._batch_labels = labels
        self._epoch = int(epoch)

    def clear_batch_context(self):
        self._batch_labels = None
        self._epoch = None

    def _normalize_counter(self, counter):
        maximum = counter.max(dim=-1, keepdim=True).values
        return counter / maximum.clamp_min(self.epsilon)

    def _probabilities(self, labels):
        result = torch.empty(
            labels.shape[0], self.feature_dim,
            device=self.global_counter.device, dtype=torch.float32,
        )
        for value in labels.unique(sorted=True):
            class_id = int(value.item())
            selected = labels == class_id
            if self._epoch < self.exploration_epochs:
                if int(self.strategy_code[class_id].item()) == 1:
                    # Eq. (7): probability-weighted reuse of old class counters.
                    normalized = self._normalize_counter(
                        self.class_counter[:self._current_begin]
                    )
                    weights = self.semantic_probabilities[
                        class_id, :self._current_begin
                    ].to(dtype=normalized.dtype)
                    exponent = (weights[:, None] * normalized).sum(dim=0)
                    probability = 1.0 - torch.exp(-exponent)
                else:
                    # Eq. (8): unused dimensions receive larger probabilities.
                    normalized = self._normalize_counter(self.global_counter)
                    probability = torch.exp(-self.beta * normalized)
            else:
                # Eq. (10): compact around dimensions used by this class.
                normalized = self._normalize_counter(self.class_counter[class_id])
                probability = 1.0 - torch.exp(-self.gamma * normalized)
            result[selected] = probability.float()
        return result.clamp_(0.0, 1.0)

    def _mask_generator(self, device):
        key = str(device)
        if key not in self._mask_generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.mask_seed)
            self._mask_generators[key] = generator
        return self._mask_generators[key]

    def forward(self, activations):
        if not self.training or self._batch_labels is None:
            return activations
        if activations.ndim not in (2, 3) or activations.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected activation shape {}".format(tuple(activations.shape)))
        labels = self._batch_labels.to(activations.device)
        if activations.shape[0] != labels.shape[0]:
            raise RuntimeError("Activation and label batch sizes differ")

        probabilities = self._probabilities(labels).to(activations.device)
        random_values = torch.rand(
            probabilities.shape,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=self._mask_generator(probabilities.device),
        )
        bernoulli = random_values < probabilities
        expanded = bernoulli if activations.ndim == 2 else bernoulli[:, None, :]
        candidates = activations * expanded.to(activations.dtype)
        magnitudes = candidates.abs()
        if candidates.ndim == 3:
            magnitudes = magnitudes.mean(dim=1)
        top_indices = magnitudes.topk(
            k=self.retain_dimensions, dim=1, largest=True, sorted=False
        ).indices
        top_mask = torch.zeros_like(bernoulli)
        top_mask.scatter_(1, top_indices, True)
        final_mask = bernoulli & top_mask
        final_expanded = final_mask if activations.ndim == 2 else final_mask[:, None, :]
        output = activations * final_expanded.to(activations.dtype)

        # Counters use only dimensions surviving Bernoulli and Top-K.
        nonzero = output.ne(0) if output.ndim == 2 else output.ne(0).any(dim=1)
        with torch.no_grad():
            counts = nonzero.to(self.global_counter.dtype)
            self.global_counter.add_(counts.sum(dim=0))
            self.class_counter.index_add_(0, labels, counts)
            active = nonzero.sum(dim=1).detach().cpu().long()
            labels_cpu = labels.detach().cpu().long()
            self._task_total_decisions += int(labels.shape[0] * self.feature_dim)
            self._task_active_decisions += int(active.sum().item())
            self._task_probability_sum += float(probabilities.sum().item())
            self._task_probability_count += int(probabilities.numel())
            self._task_class_total.index_add_(
                0, labels_cpu, torch.full_like(labels_cpu, self.feature_dim)
            )
            self._task_class_active.index_add_(0, labels_cpu, active)
        return output

    @torch.no_grad()
    def semantic_task_summary(self):
        ids = range(self._current_begin, self._current_end)
        reuse = sum(int(self.strategy_code[index].item()) == 1 for index in ids)
        classes = self._current_end - self._current_begin
        return {
            "task_id": int(self._current_task),
            "old_classes": self._current_begin,
            "new_classes": classes,
            "knowledge_reuse_classes": reuse,
            "new_subspace_allocation_classes": classes - reuse,
            "knowledge_reuse_ratio": reuse / classes,
            "mean_s_old": float(self.score_old[self._current_begin:self._current_end].mean().item()),
            "mean_s_new": float(self.score_new[self._current_begin:self._current_end].mean().item()),
            "eq6_includes_self": 1,
            "task_epochs": self.num_epochs,
            "exploration_epochs": self.exploration_epochs,
        }

    @torch.no_grad()
    def activation_task_summary(self):
        total = self._task_total_decisions
        active = self._task_active_decisions
        ratio = active / total if total else float("nan")
        current = self.class_counter[self._current_begin:self._current_end]
        task_row = {
            "task_id": int(self._current_task),
            "target_layer": self.target_layer,
            "paper_k": self.paper_k,
            "k_interpretation": "TopK retain ratio",
            "requested_retain_ratio": self.retain_dimensions / self.feature_dim,
            "requested_zero_ratio": self.requested_zero_ratio,
            "actual_nonzero_ratio": ratio,
            "actual_zero_ratio": 1.0 - ratio,
            "mean_active_dimensions": ratio * self.feature_dim,
            "mean_activation_probability": (
                self._task_probability_sum / self._task_probability_count
                if self._task_probability_count else float("nan")
            ),
            "global_counter_max": float(self.global_counter.max().item()),
            "global_counter_mean": float(self.global_counter.mean().item()),
            "class_counter_max": float(current.max().item()),
            "class_counter_mean": float(current.mean().item()),
            "training_only_mask": 1,
            "token_reduction": "mean absolute magnitude",
        }
        class_rows = []
        for class_id in range(self._current_begin, self._current_end):
            class_total = int(self._task_class_total[class_id].item())
            class_active = int(self._task_class_active[class_id].item())
            class_rows.append({
                "task_id": int(self._current_task),
                "class_id": class_id,
                "active_dimension_decisions": class_active,
                "total_dimension_decisions": class_total,
                "actual_nonzero_ratio": class_active / class_total if class_total else float("nan"),
                "mean_active_dimensions": (
                    class_active / (class_total / self.feature_dim)
                    if class_total else float("nan")
                ),
                "class_counter_max": float(self.class_counter[class_id].max().item()),
                "class_counter_mean": float(self.class_counter[class_id].mean().item()),
            })
        return task_row, class_rows
