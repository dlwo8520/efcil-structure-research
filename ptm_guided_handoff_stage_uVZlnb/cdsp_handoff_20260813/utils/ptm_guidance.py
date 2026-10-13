"""Frozen-PTM placement guidance and soft old-anchor response stability.

The trainable module is the repository's nonlinear bottleneck adapter
``up_proj(ReLU(down_proj(x)))``.  This module never stores raw images or
individual old features; only class/layer means and adapter responses are kept.
"""

import math
from collections import defaultdict, OrderedDict

import torch
from torch.nn import functional as F


_EPS = 1e-12


def placement_weight(epoch, epochs, initial, final=0.0, schedule="cosine"):
    """Return the epoch-level PTM guidance weight."""
    epochs = max(int(epochs), 1)
    epoch = min(max(int(epoch), 0), epochs - 1)
    initial, final = float(initial), float(final)
    if epochs == 1:
        progress = 0.0
    else:
        progress = epoch / float(epochs - 1)
    if schedule == "linear":
        factor = 1.0 - progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError("ptm_place_schedule must be 'linear' or 'cosine'")
    return final + (initial - final) * factor


def class_means(features, labels, normalize=False):
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features/labels must have shapes [B,D] and [B]")
    means = {}
    for class_id in torch.unique(labels).tolist():
        value = features[labels == int(class_id)].mean(dim=0)
        if normalize:
            value = F.normalize(value, p=2, dim=0)
        means[int(class_id)] = value
    return means


def ptm_placement_loss(current_features, labels, ptm_prototypes):
    """Cosine loss between differentiable batch class means and PTM anchors."""
    if not ptm_prototypes:
        return current_features.sum() * 0.0
    current = class_means(current_features, labels, normalize=True)
    losses = []
    for class_id, current_mean in current.items():
        if class_id not in ptm_prototypes:
            continue
        reference = ptm_prototypes[class_id].to(
            device=current_mean.device, dtype=current_mean.dtype
        )
        if reference.ndim != 1 or reference.shape != current_mean.shape:
            raise ValueError("PTM prototype shape mismatch for class {}".format(class_id))
        losses.append(1.0 - F.cosine_similarity(
            current_mean.view(1, -1), reference.view(1, -1), dim=1
        ).squeeze(0))
    return torch.stack(losses).mean() if losses else current_features.sum() * 0.0


def active_adapter_branch(convnet):
    return convnet.active_adapter_branch() if hasattr(convnet, "active_adapter_branch") else (
        1 if convnet.flag % 2 == 0 else 2
    )


def adapter_units(convnet, branch=None):
    """Return every active adapter as a stable name -> module mapping."""
    branch = int(branch or active_adapter_branch(convnet))
    placement = getattr(convnet.tuning_config, "adapter_placement", None)
    interval = int(getattr(convnet.tuning_config, "adapter_interval", 1))
    units = OrderedDict()
    for layer_id, block in enumerate(convnet.blocks):
        if layer_id % interval != 0:
            continue
        if placement in {"attention", "both"}:
            units["layer_{:02d}_attn_k".format(layer_id)] = getattr(
                block.attn, "k_mlp{}".format(branch)
            )
            units["layer_{:02d}_attn_v".format(layer_id)] = getattr(
                block.attn, "v_mlp{}".format(branch)
            )
        if placement in {"mlp", "both"}:
            units["layer_{:02d}_mlp".format(layer_id)] = getattr(
                block, "mlp_adapter{}".format(branch)
            )
        if placement == "block_end":
            units["layer_{:02d}_block_end".format(layer_id)] = getattr(
                block, "block_end_adapter{}".format(branch)
            )
    if not units:
        raise ValueError("PTM-guided learner found no active adapters for placement={!r}".format(placement))
    return units


def deterministic_adapter_response(adapter, anchors):
    """Evaluate the nonlinear residual branch without adapter dropout."""
    if anchors.ndim != 2 or anchors.shape[1] != adapter.down_proj.in_features:
        raise ValueError(
            "anchors must have shape [N,{}], got {}".format(
                adapter.down_proj.in_features, tuple(anchors.shape)
            )
        )
    value = anchors
    if adapter.adapter_layernorm_option == "in":
        value = adapter.adapter_layer_norm_before(value)
    value = F.linear(value, adapter.down_proj.weight, adapter.down_proj.bias)
    value = F.relu(value)
    value = F.linear(value, adapter.up_proj.weight, adapter.up_proj.bias)
    value = value * adapter.scale
    if adapter.adapter_layernorm_option == "out":
        value = adapter.adapter_layer_norm_before(value)
    return value


class AdapterInputCollector:
    """Collect per-batch CLS inputs to all active nonlinear adapters."""

    def __init__(self, convnet, branch=None):
        self.units = adapter_units(convnet, branch)
        self.handles = []
        self.pending = {}

    def _capture(self, name):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError("Adapter {} received no input".format(name))
            value = inputs[0]
            if value.ndim != 3:
                raise ValueError("Adapter {} input must be [B,N,D]".format(name))
            self.pending[name] = value[:, 0, :].detach().cpu()
        return hook

    def __enter__(self):
        for name, adapter in self.units.items():
            self.handles.append(
                adapter.down_proj.register_forward_pre_hook(self._capture(name))
            )
        return self

    def pop(self):
        missing = sorted(set(self.units) - set(self.pending))
        if missing:
            raise RuntimeError("Adapter hooks did not observe: {}".format(missing))
        result, self.pending = self.pending, {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles, self.pending = [], {}

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ClassPrototypeAccumulator:
    """Streaming class means without retaining sample features."""

    def __init__(self):
        self.sums = {}
        self.counts = defaultdict(int)

    def update(self, features, labels):
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()
        for class_id in torch.unique(labels).tolist():
            class_id = int(class_id)
            mask = labels == class_id
            value = features[mask].sum(dim=0)
            self.sums[class_id] = self.sums.get(class_id, torch.zeros_like(value)) + value
            self.counts[class_id] += int(mask.sum().item())

    def finalize(self, normalize=False):
        if not self.counts:
            raise RuntimeError("No samples observed while computing class prototypes")
        result = {}
        for class_id, count in self.counts.items():
            value = self.sums[class_id] / float(count)
            result[class_id] = F.normalize(value, p=2, dim=0) if normalize else value
        return result


class LayerAnchorAccumulator:
    """Streaming final-feature and adapter-input class means."""

    def __init__(self, unit_names):
        self.final = ClassPrototypeAccumulator()
        self.unit_sums = {name: {} for name in unit_names}
        self.counts = defaultdict(int)

    def update(self, unit_inputs, labels, final_features):
        labels = labels.detach().cpu().long()
        self.final.update(final_features, labels)
        for class_id in torch.unique(labels).tolist():
            self.counts[int(class_id)] += int((labels == int(class_id)).sum().item())
        for name in self.unit_sums:
            if name not in unit_inputs:
                raise RuntimeError("Missing captured unit {}".format(name))
            features = unit_inputs[name].detach().cpu().float()
            if features.ndim != 2:
                raise ValueError("Captured adapter input must be [B,D]")
            for class_id in torch.unique(labels).tolist():
                class_id = int(class_id)
                value = features[labels == class_id].sum(dim=0)
                current = self.unit_sums[name].get(class_id, torch.zeros_like(value))
                self.unit_sums[name][class_id] = current + value

    def finalize(self):
        anchors = {}
        for name, sums in self.unit_sums.items():
            anchors[name] = {
                class_id: value / float(self.counts[class_id])
                for class_id, value in sums.items()
            }
        return {"final_prototypes": self.final.finalize(), "layer_anchors": anchors}


class PTMGuidanceMemory:
    """Compressed, exemplar-free memory used by the PTM-guided learner."""

    def __init__(self):
        self.ptm_prototypes = {}
        self.adapted_prototypes = {}
        self.layer_anchors = {}
        self.anchor_reference_responses = {}

    @staticmethod
    def _ensure_new_classes(destination, values, label):
        overlap = sorted(set(destination).intersection(values))
        if overlap:
            raise RuntimeError("Refusing to overwrite {} for classes {}".format(label, overlap))

    def add_ptm_prototypes(self, values):
        self._ensure_new_classes(self.ptm_prototypes, values, "PTM prototypes")
        for class_id, value in values.items():
            self.ptm_prototypes[int(class_id)] = F.normalize(
                value.detach().cpu().float(), p=2, dim=0
            )

    def add_adapted_statistics(self, statistics, convnet):
        final = statistics["final_prototypes"]
        self._ensure_new_classes(self.adapted_prototypes, final, "adapted prototypes")
        for class_id, value in final.items():
            self.adapted_prototypes[int(class_id)] = value.detach().cpu().float()

        units = adapter_units(convnet)
        for name, values in statistics["layer_anchors"].items():
            memory = self.layer_anchors.setdefault(name, {})
            references = self.anchor_reference_responses.setdefault(name, {})
            self._ensure_new_classes(memory, values, "layer anchors")
            if name not in units:
                raise RuntimeError("Collected adapter unit {} is no longer active".format(name))
            adapter = units[name]
            for class_id, value in values.items():
                anchor = value.detach().cpu().float()
                memory[int(class_id)] = anchor
                with torch.no_grad():
                    response = deterministic_adapter_response(
                        adapter, anchor.to(adapter.down_proj.weight.device).view(1, -1)
                    ).squeeze(0)
                references[int(class_id)] = response.detach().cpu().float()

    def state_dict(self):
        return {
            "ptm_prototypes": self.ptm_prototypes,
            "adapted_prototypes": self.adapted_prototypes,
            "layer_anchors": self.layer_anchors,
            "anchor_reference_responses": self.anchor_reference_responses,
        }

    def load_state_dict(self, state):
        self.ptm_prototypes = _cpu_nested(state.get("ptm_prototypes", {}))
        self.adapted_prototypes = _cpu_nested(state.get("adapted_prototypes", {}))
        self.layer_anchors = _cpu_nested(state.get("layer_anchors", {}))
        self.anchor_reference_responses = _cpu_nested(
            state.get("anchor_reference_responses", {})
        )


def _cpu_nested(value):
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            normalized_key = int(key) if str(key).isdigit() else key
            converted[normalized_key] = _cpu_nested(item)
        return converted
    return value.detach().cpu().float() if isinstance(value, torch.Tensor) else value


class AnchorResponseStability:
    """Snapshot previous-task responses and compute soft functional movement loss."""

    def __init__(self, loss_type="smooth_l1"):
        if loss_type not in {"smooth_l1", "mse"}:
            raise ValueError("old_move_loss must be 'smooth_l1' or 'mse'")
        self.loss_type = loss_type
        self.snapshot = {}

    def prepare(self, memory, convnet):
        self.snapshot = {}
        units = adapter_units(convnet)
        for name, class_anchors in memory.layer_anchors.items():
            if not class_anchors or name not in units:
                continue
            class_ids = sorted(class_anchors)
            anchors = torch.stack([class_anchors[c] for c in class_ids])
            adapter = units[name]
            device_anchors = anchors.to(
                device=adapter.down_proj.weight.device, dtype=adapter.down_proj.weight.dtype
            )
            with torch.no_grad():
                targets = deterministic_adapter_response(adapter, device_anchors)
            self.snapshot[name] = {
                "class_ids": class_ids,
                "anchors": anchors,
                "targets": targets.detach().cpu(),
            }
        return sum(len(item["class_ids"]) for item in self.snapshot.values())

    def loss(self, convnet):
        units = adapter_units(convnet)
        losses = []
        for name, item in self.snapshot.items():
            adapter = units[name]
            anchors = item["anchors"].to(
                device=adapter.down_proj.weight.device, dtype=adapter.down_proj.weight.dtype
            )
            targets = item["targets"].to(device=anchors.device, dtype=anchors.dtype)
            current = deterministic_adapter_response(adapter, anchors)
            if self.loss_type == "smooth_l1":
                losses.append(F.smooth_l1_loss(current, targets))
            else:
                losses.append(F.mse_loss(current, targets))
        if losses:
            return torch.stack(losses).mean()
        parameter = next(iter(units.values())).down_proj.weight
        return parameter.sum() * 0.0


def anchor_drift_rows(memory, convnet, task_index):
    units = adapter_units(convnet)
    rows = []
    for name, class_anchors in memory.layer_anchors.items():
        if name not in units:
            continue
        adapter = units[name]
        for class_id in sorted(class_anchors):
            anchor = class_anchors[class_id].to(
                device=adapter.down_proj.weight.device, dtype=adapter.down_proj.weight.dtype
            ).view(1, -1)
            reference = memory.anchor_reference_responses[name][class_id].to(
                device=anchor.device, dtype=anchor.dtype
            ).view(1, -1)
            with torch.no_grad():
                current = deterministic_adapter_response(adapter, anchor)
                cosine_drift = 1.0 - F.cosine_similarity(current, reference, dim=1).item()
                smooth_l1 = F.smooth_l1_loss(current, reference).item()
            rows.append({
                "task": int(task_index),
                "adapter_unit": name,
                "class_id": int(class_id),
                "response_cosine_drift": float(cosine_drift),
                "response_smooth_l1": float(smooth_l1),
            })
    return rows
