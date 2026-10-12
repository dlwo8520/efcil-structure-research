from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .lora import QVFixedBasisLoRA, install_qv_lora


class ExpandingCosineClassifier(nn.Module):
    """One classifier composed of frozen old row blocks and one trainable block."""

    def __init__(self, feature_dim: int, scale: float) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale = float(scale)
        # The classifier starts with no parameters, so Module.to(device) would
        # otherwise have nothing to move.  Keep a non-persistent device anchor
        # so row blocks appended after the model was moved land on that device.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.weight_blocks = nn.ParameterList()

    @property
    def num_classes(self) -> int:
        return sum(int(block.shape[0]) for block in self.weight_blocks)

    def add_classes(self, initial_weights: torch.Tensor) -> None:
        if initial_weights.ndim != 2 or initial_weights.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected [classes, {self.feature_dim}] weights, got {initial_weights.shape}"
            )
        for block in self.weight_blocks:
            block.requires_grad_(False)
        normalized = F.normalize(
            initial_weights.detach().to(
                device=self._device_anchor.device,
                dtype=torch.float32,
            ),
            dim=1,
        )
        self.weight_blocks.append(nn.Parameter(normalized, requires_grad=True))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.weight_blocks:
            raise RuntimeError("Classifier has no classes")
        weights = F.normalize(torch.cat(list(self.weight_blocks), dim=0), dim=1)
        return self.scale * (F.normalize(features, dim=1) @ weights.transpose(0, 1))

    def current_parameters(self) -> list[nn.Parameter]:
        if not self.weight_blocks:
            return []
        return [self.weight_blocks[-1]]

    def freeze_all(self) -> list[bool]:
        states = [parameter.requires_grad for parameter in self.weight_blocks]
        for parameter in self.weight_blocks:
            parameter.requires_grad_(False)
        return states

    def restore_trainable_states(self, states: list[bool]) -> None:
        if len(states) != len(self.weight_blocks):
            raise ValueError("Classifier state length mismatch")
        for parameter, requires_grad in zip(self.weight_blocks, states):
            parameter.requires_grad_(requires_grad)

    def export_weights(self) -> list[torch.Tensor]:
        return [block.detach().cpu() for block in self.weight_blocks]

    def load_weights(self, blocks: list[torch.Tensor], device: torch.device) -> None:
        self.weight_blocks = nn.ParameterList()
        for index, block in enumerate(blocks):
            parameter = nn.Parameter(
                block.to(device=device, dtype=torch.float32),
                requires_grad=index == len(blocks) - 1,
            )
            self.weight_blocks.append(parameter)


class ContinualViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        classifier_scale: float,
        lora_rank: int,
        lora_alpha: float,
        lora_targets: Iterable[str],
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.backbone.num_features)
        self.lora_modules: list[QVFixedBasisLoRA] = install_qv_lora(
            self.backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            targets=lora_targets,
        )
        self.classifier = ExpandingCosineClassifier(self.feature_dim, classifier_scale)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError(f"Backbone returned unexpected feature shape {features.shape}")
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(images)
        return {"features": features, "logits": self.classifier(features)}

    def lora_B_parameters(self) -> list[nn.Parameter]:
        result: list[nn.Parameter] = []
        for module in self.lora_modules:
            result.extend(module.trainable_B_parameters())
        return result

    def merge_all_lora(self) -> None:
        for module in self.lora_modules:
            module.merge_and_reset()

    def set_capture_callbacks(self, callback) -> None:
        for module in self.lora_modules:
            module.capture_callback = callback

    def clear_capture_callbacks(self) -> None:
        for module in self.lora_modules:
            module.capture_callback = None
