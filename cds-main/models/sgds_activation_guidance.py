"""SGDS Eq. (4)-(10) activation-space guidance for the PCDS adapter.

This module intentionally excludes the SGDS adapter bank, retrieval, fusion,
entropy selection, prototype classifier, and MOS alignment.  It is a training-
only activation transform for the final PCDS FFN-output adapter.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


KNOWLEDGE_REUSE = "KNOWLEDGE_REUSE"
NEW_SUBSPACE_ALLOCATION = "NEW_SUBSPACE_ALLOCATION"


class SGDSActivationGuidance(nn.Module):
    """Paper Eq. (4)-(10), with persistent global and per-class counters."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_epochs: int,
        target_layer: int,
        paper_k: float = 0.6,
        beta: float = 0.5,
        gamma: float = 1.0,
        exploration_ratio: float = 0.5,
        epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if not 0.0 < paper_k <= 1.0:
            raise ValueError("SGDS paper_k must be in (0, 1]")
        if not 0.0 < exploration_ratio < 1.0:
            raise ValueError("SGDS exploration_ratio must be in (0, 1)")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.num_epochs = int(num_epochs)
        self.target_layer = int(target_layer)
        self.paper_k = float(paper_k)
        self.retain_dimensions = max(1, math.floor(paper_k * feature_dim))
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.exploration_ratio = float(exploration_ratio)
        self.exploration_epochs = int(math.ceil(num_epochs * exploration_ratio))
        self.epsilon = float(epsilon)

        self.register_buffer(
            "global_counter", torch.zeros(feature_dim, dtype=torch.float64)
        )
        self.register_buffer(
            "class_counter",
            torch.zeros(num_classes, feature_dim, dtype=torch.float64),
        )
        self.register_buffer(
            "semantic_prototypes",
            torch.zeros(num_classes, feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "semantic_available", torch.zeros(num_classes, dtype=torch.bool)
        )
        self.register_buffer(
            "semantic_probabilities",
            torch.zeros(num_classes, num_classes, dtype=torch.float32),
        )
        self.register_buffer(
            "strategy_code", torch.full((num_classes,), -1, dtype=torch.int8)
        )
        self.register_buffer(
            "score_old", torch.zeros(num_classes, dtype=torch.float32)
        )
        self.register_buffer(
            "score_new", torch.zeros(num_classes, dtype=torch.float32)
        )

        self._batch_labels: torch.Tensor | None = None
        self._epoch: int | None = None
        self._current_task: int | None = None
        self._current_begin: int = 0
        self._current_end: int = 0
        self._reset_runtime_diagnostics()

    @property
    def requested_zero_ratio(self) -> float:
        # The paper calls k a sparsity ratio but Algorithm 1 retains floor(kN).
        return 1.0 - self.retain_dimensions / self.feature_dim

    def _reset_runtime_diagnostics(self) -> None:
        self._task_total_dimension_decisions = 0
        self._task_active_dimension_decisions = 0
        self._task_probability_sum = 0.0
        self._task_probability_count = 0
        self._task_class_total = torch.zeros(self.num_classes, dtype=torch.long)
        self._task_class_active = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def prepare_task(
        self,
        task_id: int,
        begin_class: int,
        current_semantic_prototypes: torch.Tensor,
        num_epochs: int | None = None,
    ) -> list[dict[str, Any]]:
        """Store current frozen-PTM prototypes and implement Eq. (4)-(6)."""
        if num_epochs is not None:
            self.num_epochs = int(num_epochs)
            if self.num_epochs <= 0:
                raise ValueError("SGDS task epochs must be positive")
            self.exploration_epochs = int(
                math.ceil(self.num_epochs * self.exploration_ratio)
            )
        begin_class = int(begin_class)
        current = current_semantic_prototypes.detach().to(
            device=self.semantic_prototypes.device, dtype=torch.float32
        )
        classes = int(current.shape[0])
        end_class = begin_class + classes
        if current.shape != (classes, self.feature_dim):
            raise ValueError(f"Invalid semantic prototype shape {tuple(current.shape)}")
        if begin_class < 0 or end_class > self.num_classes:
            raise ValueError("Semantic prototype class range is invalid")
        if begin_class and not bool(self.semantic_available[:begin_class].all()):
            raise RuntimeError("Past semantic prototypes are missing at a task boundary")
        if bool(self.semantic_available[begin_class:end_class].any()):
            raise RuntimeError("Current semantic prototypes were already registered")

        self.semantic_prototypes[begin_class:end_class].copy_(current)
        self.semantic_available[begin_class:end_class] = True
        seen = F.normalize(self.semantic_prototypes[:end_class], dim=1)
        current_normalized = F.normalize(current, dim=1)

        # Eq. (4): denominator is every old/current class.  The paper Eq. (6)
        # and Algorithm 1 include c itself in the current-class sum.
        cosine = current_normalized @ seen.transpose(0, 1)
        conditional = F.softmax(cosine, dim=1)
        rows: list[dict[str, Any]] = []
        for local_id in range(classes):
            class_id = begin_class + local_id
            self.semantic_probabilities[class_id, :end_class].copy_(
                conditional[local_id]
            )
            old_score = float(conditional[local_id, :begin_class].sum().item())
            new_score = float(conditional[local_id, begin_class:end_class].sum().item())
            reuse = old_score > new_score
            self.score_old[class_id] = old_score
            self.score_new[class_id] = new_score
            self.strategy_code[class_id] = 1 if reuse else 0
            old_cosine = cosine[local_id, :begin_class]
            rows.append({
                "task_id": int(task_id),
                "class_id": class_id,
                "s_old": old_score,
                "s_new": new_score,
                "strategy": KNOWLEDGE_REUSE if reuse else NEW_SUBSPACE_ALLOCATION,
                "eq6_includes_self": 1,
                "semantic_denominator_classes": end_class,
                "old_classes": begin_class,
                "new_classes": classes,
                "max_old_cosine_similarity": (
                    float(old_cosine.max().item())
                    if old_cosine.numel()
                    else float("nan")
                ),
                "mean_old_cosine_similarity": (
                    float(old_cosine.mean().item())
                    if old_cosine.numel()
                    else float("nan")
                ),
            })
        self._current_task = int(task_id)
        self._current_begin = begin_class
        self._current_end = end_class
        self._reset_runtime_diagnostics()
        return rows

    def set_batch_context(self, labels: torch.Tensor, epoch: int) -> None:
        labels = labels.detach().long()
        if labels.ndim != 1:
            raise ValueError("SGDS labels must be a vector")
        if self._current_task is None:
            raise RuntimeError("SGDS task strategy was not prepared")
        if bool((labels < self._current_begin).any()) or bool(
            (labels >= self._current_end).any()
        ):
            raise RuntimeError(
                "SGDS received a label outside the current task; refusing counter corruption"
            )
        self._batch_labels = labels
        self._epoch = int(epoch)

    def clear_batch_context(self) -> None:
        self._batch_labels = None
        self._epoch = None

    def _normalized_counter(self, counter: torch.Tensor) -> torch.Tensor:
        maximum = counter.max(dim=-1, keepdim=True).values
        return counter / maximum.clamp_min(self.epsilon)

    def _probabilities(self, labels: torch.Tensor) -> torch.Tensor:
        """Eq. (7)-(10) probabilities for one batch of class IDs."""
        result = torch.empty(
            labels.shape[0],
            self.feature_dim,
            device=self.global_counter.device,
            dtype=torch.float32,
        )
        for class_id_tensor in labels.unique(sorted=True):
            class_id = int(class_id_tensor.item())
            selected = labels == class_id
            if self._epoch is None:
                raise RuntimeError("SGDS epoch context is missing")
            if self._epoch < self.exploration_epochs:
                if int(self.strategy_code[class_id].item()) == 1:
                    # Eq. (7), weighted by Eq. (4) probabilities over old classes.
                    old_counter = self.class_counter[: self._current_begin]
                    normalized = self._normalized_counter(old_counter)
                    weights = self.semantic_probabilities[
                        class_id, : self._current_begin
                    ].to(dtype=normalized.dtype)
                    exponent = (weights[:, None] * normalized).sum(dim=0)
                    probability = 1.0 - torch.exp(-exponent)
                else:
                    # Eq. (8). Empty history naturally produces p=1.
                    normalized = self._normalized_counter(self.global_counter)
                    probability = torch.exp(-self.beta * normalized)
            else:
                # Eq. (10). Exploration has already populated this class row.
                normalized = self._normalized_counter(
                    self.class_counter[class_id]
                )
                probability = 1.0 - torch.exp(-self.gamma * normalized)
            result[selected] = probability.to(dtype=torch.float32)
        return result.clamp_(0.0, 1.0)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # SGDS Algorithm 1 defines stochastic guidance only while training.
        if not self.training or self._batch_labels is None:
            return activations
        if activations.ndim not in (2, 3) or activations.shape[-1] != self.feature_dim:
            raise ValueError(f"Unexpected SGDS activation shape {tuple(activations.shape)}")
        labels = self._batch_labels.to(device=activations.device)
        if activations.shape[0] != labels.shape[0]:
            raise RuntimeError("SGDS activation batch and label batch differ")

        probabilities = self._probabilities(labels).to(
            device=activations.device, dtype=torch.float32
        )
        bernoulli = torch.bernoulli(probabilities).to(dtype=torch.bool)
        expanded = bernoulli if activations.ndim == 2 else bernoulli[:, None, :]
        probabilistic = activations * expanded.to(dtype=activations.dtype)

        # The ViT adapter receives [B,tokens,D].  SGDS defines one D-vector per
        # sample, so token magnitudes are averaged and one dimension mask is
        # broadcast to all tokens for that sample.
        magnitudes = (
            probabilistic.abs()
            if probabilistic.ndim == 2
            else probabilistic.abs().mean(dim=1)
        )
        top_indices = magnitudes.topk(
            k=self.retain_dimensions, dim=1, largest=True, sorted=False
        ).indices
        top_mask = torch.zeros_like(bernoulli)
        top_mask.scatter_(1, top_indices, True)
        final_dimension_mask = bernoulli & top_mask
        final_expanded = (
            final_dimension_mask
            if activations.ndim == 2
            else final_dimension_mask[:, None, :]
        )
        output = activations * final_expanded.to(dtype=activations.dtype)

        # Count only dimensions that remain truly non-zero after Bernoulli and
        # deterministic Top-K, once per sample as required by Algorithm 1.
        nonzero = (
            output.ne(0)
            if output.ndim == 2
            else output.ne(0).any(dim=1)
        )
        with torch.no_grad():
            counts = nonzero.to(dtype=self.global_counter.dtype)
            self.global_counter.add_(counts.sum(dim=0))
            self.class_counter.index_add_(0, labels, counts)
            active_per_sample = nonzero.sum(dim=1).detach().cpu().long()
            labels_cpu = labels.detach().cpu().long()
            self._task_total_dimension_decisions += int(
                labels.shape[0] * self.feature_dim
            )
            self._task_active_dimension_decisions += int(active_per_sample.sum().item())
            self._task_probability_sum += float(probabilities.sum().item())
            self._task_probability_count += int(probabilities.numel())
            self._task_class_total.index_add_(
                0,
                labels_cpu,
                torch.full_like(labels_cpu, self.feature_dim),
            )
            self._task_class_active.index_add_(0, labels_cpu, active_per_sample)
        return output

    @torch.no_grad()
    def semantic_task_summary(self) -> dict[str, Any]:
        ids = range(self._current_begin, self._current_end)
        reuse = sum(int(self.strategy_code[class_id].item()) == 1 for class_id in ids)
        classes = self._current_end - self._current_begin
        old_scores = self.score_old[self._current_begin : self._current_end]
        new_scores = self.score_new[self._current_begin : self._current_end]
        return {
            "task_id": int(self._current_task),
            "new_classes": classes,
            "knowledge_reuse_classes": reuse,
            "new_subspace_allocation_classes": classes - reuse,
            "knowledge_reuse_ratio": reuse / classes,
            "old_classes": self._current_begin,
            "mean_s_old": float(old_scores.mean().item()),
            "mean_s_new": float(new_scores.mean().item()),
            "eq6_includes_self": 1,
            "task_epochs": self.num_epochs,
            "exploration_epochs": self.exploration_epochs,
        }

    @torch.no_grad()
    def activation_task_summary(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        total = self._task_total_dimension_decisions
        active = self._task_active_dimension_decisions
        active_ratio = active / total if total else float("nan")
        global_counter = self.global_counter
        current_counter = self.class_counter[self._current_begin : self._current_end]
        task_row = {
            "task_id": int(self._current_task),
            "target_layer": self.target_layer,
            "paper_k": self.paper_k,
            "k_interpretation": "retain_ratio_from_TopK_floor_kN",
            "requested_retain_ratio": self.retain_dimensions / self.feature_dim,
            "requested_zero_ratio": self.requested_zero_ratio,
            "actual_nonzero_ratio": active_ratio,
            "actual_zero_ratio": 1.0 - active_ratio,
            "mean_active_dimensions": active_ratio * self.feature_dim,
            "mean_activation_probability": (
                self._task_probability_sum / self._task_probability_count
                if self._task_probability_count
                else float("nan")
            ),
            "global_counter_max": float(global_counter.max().item()),
            "global_counter_mean": float(global_counter.mean().item()),
            "class_counter_max": float(current_counter.max().item()),
            "class_counter_mean": float(current_counter.mean().item()),
            "training_only_mask": 1,
            "token_reduction": "mean_absolute_magnitude",
        }
        class_rows: list[dict[str, Any]] = []
        for class_id in range(self._current_begin, self._current_end):
            class_total = int(self._task_class_total[class_id].item())
            class_active = int(self._task_class_active[class_id].item())
            counter = self.class_counter[class_id]
            class_rows.append({
                "task_id": int(self._current_task),
                "class_id": class_id,
                "active_dimension_decisions": class_active,
                "total_dimension_decisions": class_total,
                "actual_nonzero_ratio": (
                    class_active / class_total if class_total else float("nan")
                ),
                "mean_active_dimensions": (
                    class_active / (class_total / self.feature_dim)
                    if class_total
                    else float("nan")
                ),
                "class_counter_max": float(counter.max().item()),
                "class_counter_mean": float(counter.mean().item()),
            })
        return task_row, class_rows
