"""Layer-wise protected-subspace memory for the CDS adapter diagnostic.

CDS repairs prototypes after representation drift.  The diagnostic implemented
here instead restricts the *input-sensitive directions* of CDS's nonlinear
bottleneck adapter.  It is inspired by protected-subspace methods, but it is
not a LoRA implementation: the CDS adapter computes ``U(ReLU(Dx))`` and the
hard constraint imposed here is ``D Q = 0``.

Only compressed statistics from a class's first task are retained.  Individual
features and old training examples are never stored by this module.
"""

import logging
import time
from collections import defaultdict

import torch


_EPS = 1e-12


def _empty_basis(feature_dim):
    return torch.empty(feature_dim, 0, dtype=torch.float32)


def _energy_rank(values, threshold):
    """Return the smallest rank reaching ``threshold`` of non-negative energy."""
    if values.numel() == 0:
        return 0
    values = torch.clamp(values.detach().cpu().float(), min=0.0)
    total = values.sum()
    if not torch.isfinite(total) or total <= _EPS:
        return 0
    threshold = min(max(float(threshold), 0.0), 1.0)
    cumulative = torch.cumsum(values, dim=0) / total
    rank = int(torch.searchsorted(cumulative, torch.tensor(threshold)).item()) + 1
    return min(rank, values.numel())


def basis_from_columns(matrix, energy, feature_dim):
    """SVD basis for a [feature_dim, samples] matrix, with safe fallbacks."""
    if matrix is None or matrix.numel() == 0 or matrix.shape[1] == 0:
        return _empty_basis(feature_dim)
    matrix = matrix.detach().cpu().float()
    try:
        left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
        rank = _energy_rank(singular_values.square(), energy)
        return left[:, :rank].contiguous() if rank else _empty_basis(feature_dim)
    except RuntimeError as exc:
        logging.warning("SVD failed; using symmetric eigendecomposition fallback: %s", exc)
        return basis_from_scatter(matrix @ matrix.T, energy, feature_dim)


def basis_from_scatter(scatter, energy, feature_dim):
    """Basis from a streaming scatter matrix (equivalent to left-vector SVD)."""
    if scatter is None or scatter.numel() == 0:
        return _empty_basis(feature_dim)
    scatter = scatter.detach().cpu().float()
    scatter = 0.5 * (scatter + scatter.T)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = torch.clamp(eigenvalues[order], min=0.0)
        eigenvectors = eigenvectors[:, order]
        rank = _energy_rank(eigenvalues, energy)
        return eigenvectors[:, :rank].contiguous() if rank else _empty_basis(feature_dim)
    except RuntimeError as exc:
        logging.warning("Eigendecomposition failed; using SVD fallback: %s", exc)
        try:
            left, singular_values, _ = torch.linalg.svd(scatter, full_matrices=False)
            rank = _energy_rank(singular_values, energy)
            return left[:, :rank].contiguous() if rank else _empty_basis(feature_dim)
        except RuntimeError as svd_exc:
            logging.error("Subspace decomposition failed; using an empty basis: %s", svd_exc)
            return _empty_basis(feature_dim)


def orthonormal_union(bases, feature_dim):
    """Return an orthonormal basis for the union without inventing null directions."""
    valid = [basis.detach().cpu().float() for basis in bases
             if basis is not None and basis.numel() and basis.shape[1] > 0]
    if not valid:
        return _empty_basis(feature_dim)
    joined = torch.cat(valid, dim=1)
    try:
        left, singular_values, _ = torch.linalg.svd(joined, full_matrices=False)
        tolerance = max(joined.shape) * torch.finfo(joined.dtype).eps * singular_values.max()
        rank = int((singular_values > tolerance).sum().item())
        return left[:, :min(rank, feature_dim)].contiguous()
    except RuntimeError as exc:
        logging.warning("Basis-union SVD failed; using QR fallback: %s", exc)
        q, r = torch.linalg.qr(joined, mode="reduced")
        diagonal = torch.abs(torch.diagonal(r))
        if diagonal.numel() == 0:
            return _empty_basis(feature_dim)
        tolerance = max(joined.shape) * torch.finfo(joined.dtype).eps * diagonal.max()
        rank = int((diagonal > tolerance).sum().item())
        return q[:, :min(rank, feature_dim)].contiguous()


class StreamingLayerStatistics:
    """Accumulate class sums and scatter without retaining sample features."""

    def __init__(self, layer_ids, feature_dim):
        self.layer_ids = list(layer_ids)
        self.feature_dim = int(feature_dim)
        self.counts = defaultdict(int)
        self.layer_sums = {layer_id: {} for layer_id in self.layer_ids}
        self.layer_second = {
            layer_id: torch.zeros(feature_dim, feature_dim, dtype=torch.float32)
            for layer_id in self.layer_ids
        }
        self.final_sums = {}

    @staticmethod
    def _add_class_sums(destination, features, labels):
        for class_id in torch.unique(labels).tolist():
            class_id = int(class_id)
            class_sum = features[labels == class_id].sum(dim=0)
            if class_id in destination:
                destination[class_id] += class_sum
            else:
                destination[class_id] = class_sum.clone()

    def update(self, layer_features, labels, final_features):
        labels = labels.detach().cpu().long()
        final_features = final_features.detach().cpu().float()
        for class_id in torch.unique(labels).tolist():
            self.counts[int(class_id)] += int((labels == class_id).sum().item())
        self._add_class_sums(self.final_sums, final_features, labels)

        for layer_id in self.layer_ids:
            features = layer_features[layer_id].detach().cpu().float()
            if features.ndim != 2 or features.shape[1] != self.feature_dim:
                raise ValueError(
                    "Layer {} adapter input must have shape [batch, {}], got {}".format(
                        layer_id, self.feature_dim, tuple(features.shape)
                    )
                )
            self._add_class_sums(self.layer_sums[layer_id], features, labels)
            self.layer_second[layer_id] += features.T @ features

    def finalize(self):
        if not self.counts:
            raise RuntimeError("No samples were observed while collecting subspace statistics")

        layer_prototypes = {layer_id: {} for layer_id in self.layer_ids}
        final_prototypes = {}
        for class_id, count in self.counts.items():
            final_prototypes[class_id] = self.final_sums[class_id] / float(count)
            for layer_id in self.layer_ids:
                layer_prototypes[layer_id][class_id] = (
                    self.layer_sums[layer_id][class_id] / float(count)
                )

        total_count = float(sum(self.counts.values()))
        task_scatter = {}
        intra_scatter = {}
        for layer_id in self.layer_ids:
            total_sum = sum(self.layer_sums[layer_id].values())
            task_mean = total_sum / total_count
            task_value = self.layer_second[layer_id] - total_count * torch.outer(task_mean, task_mean)

            intra_value = self.layer_second[layer_id].clone()
            for class_id, prototype in layer_prototypes[layer_id].items():
                intra_value -= float(self.counts[class_id]) * torch.outer(prototype, prototype)

            task_scatter[layer_id] = 0.5 * (task_value + task_value.T)
            intra_scatter[layer_id] = 0.5 * (intra_value + intra_value.T)

        return {
            "counts": dict(self.counts),
            "layer_prototypes": layer_prototypes,
            "final_prototypes": final_prototypes,
            "task_scatter": task_scatter,
            "intra_scatter": intra_scatter,
        }


class AdapterInputCollector:
    """Capture the exact tensors entering active MLP adapter down projections."""

    def __init__(self, convnet, branch, layer_ids):
        self.convnet = convnet
        self.branch = int(branch)
        self.layer_ids = list(layer_ids)
        self.handles = []
        self.pending = {}

    def _capture(self, layer_id):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError("Adapter down projection received no input")
            tensor = inputs[0]
            if tensor.ndim != 3:
                raise ValueError(
                    "Expected adapter input [batch, tokens, dim], got {}".format(tuple(tensor.shape))
                )
            self.pending[layer_id] = tensor[:, 0, :].detach().cpu()
        return hook

    def __enter__(self):
        for layer_id in self.layer_ids:
            block = self.convnet.blocks[layer_id]
            adapter = getattr(block, "mlp_adapter{}".format(self.branch))
            self.handles.append(adapter.down_proj.register_forward_pre_hook(self._capture(layer_id)))
        return self

    def pop(self):
        missing = [layer_id for layer_id in self.layer_ids if layer_id not in self.pending]
        if missing:
            raise RuntimeError("Adapter hooks did not observe layers: {}".format(missing))
        captured = self.pending
        self.pending = {}
        return captured

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.pending = {}

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ProtectedSubspaceManager:
    """Own compressed memory and enforce ``down_proj.weight @ Q == 0``."""

    MODES = {"none", "task", "inter", "inter_intra"}

    def __init__(self, args, feature_dim):
        self.mode = args.get("subspace_mode", "none")
        if self.mode not in self.MODES:
            raise ValueError("Unknown subspace_mode: {}".format(self.mode))
        self.feature_dim = int(feature_dim)
        self.inter_energy = float(args.get("inter_energy", 0.95))
        self.intra_energy = float(args.get("intra_energy", 0.90))
        self.task_energy = float(args.get("task_energy", self.inter_energy))
        self.protected_layers_config = args.get("protected_layers", "all")
        self.project_gradient_enabled = bool(args.get("project_gradient", True))
        self.project_after_step_enabled = bool(args.get("project_after_step", True))
        self.freeze_adapter_bias = bool(args.get("freeze_adapter_bias", self.mode != "none"))

        self.layer_ids = []
        self.layer_prototypes = {}
        self.final_prototypes = {}
        self.task_bases = {}
        self.intra_bases = {}
        self.inter_bases = {}
        self.protected_bases = {}
        self._gradient_handles = []
        self._active_device_bases = {}

    @staticmethod
    def active_branch(convnet):
        # VisionTransformer.forward uses branch 1 when flag is even and branch 2 when odd.
        return 1 if convnet.flag % 2 == 0 else 2

    def resolve_layers(self, convnet):
        placement = getattr(convnet.tuning_config, "adapter_placement", None)
        if placement != "mlp":
            raise ValueError(
                "CDSP currently supports adapter_placement='mlp' only; got {!r}".format(placement)
            )
        interval = int(getattr(convnet.tuning_config, "adapter_interval", 1))
        eligible = [
            index for index, block in enumerate(convnet.blocks)
            if index % interval == 0 and block.mlp_adapter1 is not None and block.mlp_adapter2 is not None
        ]
        if self.protected_layers_config == "all":
            selected = eligible
        elif isinstance(self.protected_layers_config, (list, tuple)):
            requested = [int(layer_id) for layer_id in self.protected_layers_config]
            invalid = sorted(set(requested) - set(eligible))
            if invalid:
                raise ValueError("protected_layers contains inactive/invalid MLP layers: {}".format(invalid))
            selected = sorted(set(requested))
        else:
            raise ValueError("protected_layers must be 'all' or a list of layer indices")
        if not selected:
            raise ValueError("No active MLP adapter layers were selected")
        self.layer_ids = selected
        for layer_id in selected:
            self.layer_prototypes.setdefault(layer_id, {})
            self.task_bases.setdefault(layer_id, _empty_basis(self.feature_dim))
            self.intra_bases.setdefault(layer_id, _empty_basis(self.feature_dim))
            self.inter_bases.setdefault(layer_id, _empty_basis(self.feature_dim))
            self.protected_bases.setdefault(layer_id, _empty_basis(self.feature_dim))
        return selected

    def _adapter(self, convnet, layer_id, branch):
        return getattr(convnet.blocks[layer_id], "mlp_adapter{}".format(branch))

    def _build_inter_basis(self, layer_id):
        prototypes = self.layer_prototypes.get(layer_id, {})
        if len(prototypes) < 2:
            return _empty_basis(self.feature_dim)
        ordered = torch.stack([prototypes[class_id] for class_id in sorted(prototypes)], dim=1)
        centered = ordered - ordered.mean(dim=1, keepdim=True)
        return basis_from_columns(centered, self.inter_energy, self.feature_dim)

    def rebuild_protected_bases(self):
        for layer_id in self.layer_ids:
            self.inter_bases[layer_id] = self._build_inter_basis(layer_id)
            if self.mode == "task":
                protected = self.task_bases[layer_id]
            elif self.mode == "inter":
                protected = self.inter_bases[layer_id]
            elif self.mode == "inter_intra":
                protected = orthonormal_union(
                    [self.inter_bases[layer_id], self.intra_bases[layer_id]], self.feature_dim
                )
            else:
                protected = _empty_basis(self.feature_dim)
            self.protected_bases[layer_id] = protected

    @staticmethod
    def _remove_basis_component(matrix, basis):
        if basis is None or basis.numel() == 0 or basis.shape[1] == 0:
            return matrix
        return matrix - (matrix @ basis) @ basis.T

    def clear_gradient_hooks(self):
        for handle in self._gradient_handles:
            handle.remove()
        self._gradient_handles = []
        self._active_device_bases = {}

    def _apply_bias_policy(self, adapter):
        if not self.freeze_adapter_bias:
            return
        for bias in (adapter.down_proj.bias, adapter.up_proj.bias):
            if bias is not None:
                with torch.no_grad():
                    bias.zero_()
                bias.requires_grad = False

    def prepare_for_task(self, convnet):
        """Project and register hooks on the single branch trained by this task."""
        self.resolve_layers(convnet)
        self.rebuild_protected_bases()
        self.clear_gradient_hooks()
        branch = self.active_branch(convnet)
        for layer_id in self.layer_ids:
            adapter = self._adapter(convnet, layer_id, branch)
            self._apply_bias_policy(adapter)
            basis = self.protected_bases[layer_id].to(
                device=adapter.down_proj.weight.device,
                dtype=adapter.down_proj.weight.dtype,
            )
            self._active_device_bases[layer_id] = basis
            with torch.no_grad():
                projected = self._remove_basis_component(adapter.down_proj.weight, basis)
                adapter.down_proj.weight.copy_(projected)
            if self.project_gradient_enabled and basis.shape[1] > 0:
                handle = adapter.down_proj.weight.register_hook(
                    lambda gradient, q=basis: self._remove_basis_component(gradient, q)
                )
                self._gradient_handles.append(handle)
        return branch

    def enforce_constraint(self, convnet, optimizer=None):
        if not self.project_after_step_enabled:
            return
        branch = self.active_branch(convnet)
        for layer_id in self.layer_ids:
            adapter = self._adapter(convnet, layer_id, branch)
            parameter = adapter.down_proj.weight
            basis = self._active_device_bases.get(layer_id)
            if basis is None or basis.device != parameter.device or basis.dtype != parameter.dtype:
                basis = self.protected_bases[layer_id].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                self._active_device_bases[layer_id] = basis
            with torch.no_grad():
                parameter.copy_(self._remove_basis_component(parameter, basis))
                if optimizer is not None and parameter in optimizer.state:
                    state = optimizer.state[parameter]
                    # SGD momentum and Adam's signed first moment must obey the same tangent constraint.
                    for state_name in ("momentum_buffer", "exp_avg"):
                        value = state.get(state_name)
                        if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                            value.copy_(self._remove_basis_component(value, basis))
            self._apply_bias_policy(adapter)

    def violation_stats(self, convnet):
        branch = self.active_branch(convnet)
        rows = []
        for layer_id in self.layer_ids:
            weight = self._adapter(convnet, layer_id, branch).down_proj.weight.detach()
            basis = self._active_device_bases.get(layer_id)
            if basis is None or basis.device != weight.device or basis.dtype != weight.dtype:
                basis = self.protected_bases[layer_id].to(device=weight.device, dtype=weight.dtype)
            if basis.shape[1] == 0:
                ratio = 0.0
            else:
                ratio = (torch.linalg.norm(weight @ basis) / (torch.linalg.norm(weight) + _EPS)).item()
            rows.append({"layer": layer_id, "protected_violation_ratio": float(ratio)})
        return rows

    def rank_stats(self, task_index, phase, number_of_classes):
        rows = []
        for layer_id in self.layer_ids:
            protected_rank = int(self.protected_bases[layer_id].shape[1])
            rows.append({
                "task": int(task_index),
                "phase": phase,
                "layer": int(layer_id),
                "number_of_classes": int(number_of_classes),
                "q_inter_rank": int(self.inter_bases[layer_id].shape[1]),
                "q_intra_rank": int(self.intra_bases[layer_id].shape[1]),
                "q_task_rank": int(self.task_bases[layer_id].shape[1]),
                "protected_rank": protected_rank,
                "residual_dim": self.feature_dim - protected_rank,
                "residual_rank": self.feature_dim - protected_rank,
                "protected_ratio": protected_rank / float(self.feature_dim),
            })
        return rows

    def update_memory(self, collected, expected_classes):
        update_started = time.perf_counter()
        expected_classes = set(int(class_id) for class_id in expected_classes)
        observed_classes = set(int(class_id) for class_id in collected["counts"])
        if observed_classes != expected_classes:
            raise RuntimeError(
                "Current-task collector observed classes {}, expected {}".format(
                    sorted(observed_classes), sorted(expected_classes)
                )
            )

        for layer_id in self.layer_ids:
            layer_started = time.perf_counter()
            memory = self.layer_prototypes[layer_id]
            overlap = expected_classes.intersection(memory)
            if overlap:
                raise RuntimeError("Refusing to overwrite stored old-class prototypes: {}".format(sorted(overlap)))
            for class_id, prototype in collected["layer_prototypes"][layer_id].items():
                memory[int(class_id)] = prototype.detach().cpu().float()

            if self.mode == "task":
                task_basis = basis_from_scatter(
                    collected["task_scatter"][layer_id], self.task_energy, self.feature_dim
                )
                self.task_bases[layer_id] = orthonormal_union(
                    [self.task_bases[layer_id], task_basis], self.feature_dim
                )

            if self.mode == "inter_intra":
                intra_basis = basis_from_scatter(
                    collected["intra_scatter"][layer_id], self.intra_energy, self.feature_dim
                )
                self.intra_bases[layer_id] = orthonormal_union(
                    [self.intra_bases[layer_id], intra_basis], self.feature_dim
                )
            logging.info(
                "Compressed subspace memory updated for layer %d in %.2fs",
                layer_id,
                time.perf_counter() - layer_started,
            )

        overlap = expected_classes.intersection(self.final_prototypes)
        if overlap:
            raise RuntimeError("Refusing to overwrite stored final prototypes: {}".format(sorted(overlap)))
        for class_id, prototype in collected["final_prototypes"].items():
            self.final_prototypes[int(class_id)] = prototype.detach().cpu().float()
        self.rebuild_protected_bases()
        logging.info(
            "Compressed subspace update finished in %.2fs", time.perf_counter() - update_started
        )

    def state_dict(self):
        return {
            "mode": self.mode,
            "feature_dim": self.feature_dim,
            "inter_energy": self.inter_energy,
            "intra_energy": self.intra_energy,
            "task_energy": self.task_energy,
            "project_gradient": self.project_gradient_enabled,
            "project_after_step": self.project_after_step_enabled,
            "freeze_adapter_bias": self.freeze_adapter_bias,
            "layer_ids": list(self.layer_ids),
            "layer_prototypes": self.layer_prototypes,
            "final_prototypes": self.final_prototypes,
            "task_bases": self.task_bases,
            "intra_bases": self.intra_bases,
            "inter_bases": self.inter_bases,
            "protected_bases": self.protected_bases,
        }

    def load_state_dict(self, state):
        if state["mode"] != self.mode or int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("Checkpoint subspace configuration does not match the current config")
        for name, current in (
            ("inter_energy", self.inter_energy),
            ("intra_energy", self.intra_energy),
            ("task_energy", self.task_energy),
        ):
            if abs(float(state.get(name, current)) - current) > 1e-12:
                raise ValueError("Checkpoint {} does not match the current config".format(name))
        for name, current in (
            ("project_gradient", self.project_gradient_enabled),
            ("project_after_step", self.project_after_step_enabled),
            ("freeze_adapter_bias", self.freeze_adapter_bias),
        ):
            if bool(state.get(name, current)) != current:
                raise ValueError("Checkpoint {} does not match the current config".format(name))
        checkpoint_layers = [int(layer_id) for layer_id in state["layer_ids"]]
        if self.layer_ids and checkpoint_layers != self.layer_ids:
            raise ValueError("Checkpoint protected layers do not match the current config")
        self.layer_ids = checkpoint_layers
        self.layer_prototypes = {
            int(layer_id): {int(class_id): value.cpu().float() for class_id, value in values.items()}
            for layer_id, values in state["layer_prototypes"].items()
        }
        self.final_prototypes = {
            int(class_id): value.cpu().float() for class_id, value in state["final_prototypes"].items()
        }
        for name in ("task_bases", "intra_bases", "inter_bases", "protected_bases"):
            loaded = {int(layer_id): value.cpu().float() for layer_id, value in state[name].items()}
            setattr(self, name, loaded)
