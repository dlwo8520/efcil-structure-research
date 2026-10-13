from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
from torch import nn

from .keeplora_lora import KeepLoRAOutputProjection, KeepLoRAQKV
from .model import ExpandingCosineClassifier


class KeepLoRAContinualViT(nn.Module):
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
        target_set = set(lora_targets)
        if not target_set or not target_set.issubset({"q", "k", "v", "o"}):
            raise ValueError(f"Invalid KeepLoRA targets: {target_set}")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.backbone.num_features)
        self.lora_modules: list[KeepLoRAQKV | KeepLoRAOutputProjection] = []
        qkv_targets = tuple(target for target in ("q", "k", "v") if target in target_set)
        for layer_id, block in enumerate(self.backbone.blocks):
            if qkv_targets:
                qkv = KeepLoRAQKV(
                    block.attn.qkv,
                    layer_id=layer_id,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    targets=qkv_targets,
                )
                block.attn.qkv = qkv
                self.lora_modules.append(qkv)
            if "o" in target_set:
                output = KeepLoRAOutputProjection(
                    block.attn.proj,
                    layer_id=layer_id,
                    rank=lora_rank,
                    alpha=lora_alpha,
                )
                block.attn.proj = output
                self.lora_modules.append(output)
        if not self.lora_modules:
            raise RuntimeError("No KeepLoRA modules were installed")
        self.classifier = ExpandingCosineClassifier(self.feature_dim, classifier_scale)

    def features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError(f"Backbone returned unexpected feature shape {features.shape}")
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(images)
        return {"features": features, "logits": self.classifier(features)}

    def projection_sites(self):
        for module in self.lora_modules:
            for target in module.targets:
                yield module, target

    def site_keys(self) -> list[str]:
        return [module.site_key(target) for module, target in self.projection_sites()]

    def feature_group_for_site(self, site_key: str) -> str:
        for module, target in self.projection_sites():
            if module.site_key(target) == site_key:
                return module.capture_group
        raise KeyError(site_key)

    def lora_B_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self.lora_modules:
            parameters.extend(module.trainable_B_parameters())
        return parameters

    def merge_all_lora(self) -> None:
        for module in self.lora_modules:
            module.merge_and_reset()

    def set_capture_callbacks(self, callback) -> None:
        for module in self.lora_modules:
            module.capture_callback = callback

    def clear_capture_callbacks(self) -> None:
        for module in self.lora_modules:
            module.capture_callback = None
