from __future__ import annotations

import math

import timm
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import maybe_add_mask, resolve_self_attn_mask

from .model import ExpandingCosineClassifier


class PCDSAdapter(nn.Module):
    """Official CDS nonlinear bottleneck adapter: LN-down-ReLU-dropout-up."""

    def __init__(self, dim: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.down_proj = nn.Linear(dim, bottleneck)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck, dim)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(
            self.dropout(self.activation(self.down_proj(self.layer_norm(x))))
        )


class MLPWithPCDSAdapter(nn.Module):
    """Place the adapter on the ViT MLP output, matching official CDS placement."""

    def __init__(
        self,
        base_mlp: nn.Module,
        feature_dim: int,
        bottleneck: int,
        dropout: float,
        activation_guidance: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter = PCDSAdapter(feature_dim, bottleneck, dropout)
        self.activation_guidance = activation_guidance
        self.adapter_enabled = True
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_mlp(x)
        if not self.adapter_enabled:
            return output
        adapter_input = (
            self.activation_guidance(output)
            if self.activation_guidance is not None
            else output
        )
        return output + self.adapter(adapter_input)


class AttentionWithPCDSAdapters(nn.Module):
    """Official Cars CDS placement: parallel bottleneck adapters on K and V."""

    def __init__(
        self,
        base_attention: nn.Module,
        feature_dim: int,
        bottleneck: int,
        dropout: float,
        activation_guidance: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.base_attention = base_attention
        self.key_adapter = PCDSAdapter(feature_dim, bottleneck, dropout)
        self.value_adapter = PCDSAdapter(feature_dim, bottleneck, dropout)
        self.activation_guidance = activation_guidance
        self.adapter_enabled = True
        for parameter in self.base_attention.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attention = self.base_attention
        batch, tokens, channels = x.shape
        qkv = attention.qkv(x).reshape(
            batch, tokens, 3, attention.num_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = attention.q_norm(query), attention.k_norm(key)

        if self.adapter_enabled:
            adapter_input = (
                self.activation_guidance(x)
                if self.activation_guidance is not None
                else x
            )
            key_residual = self.key_adapter(adapter_input).reshape(
                batch, tokens, attention.num_heads, attention.head_dim
            ).permute(0, 2, 1, 3)
            value_residual = self.value_adapter(adapter_input).reshape(
                batch, tokens, attention.num_heads, attention.head_dim
            ).permute(0, 2, 1, 3)
            key = key + key_residual
            value = value + value_residual

        if attention.fused_attn:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=attention.attn_drop.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            query = query * attention.scale
            weights = query @ key.transpose(-2, -1)
            bias = resolve_self_attn_mask(tokens, weights, attn_mask, is_causal)
            weights = maybe_add_mask(weights, bias).softmax(dim=-1)
            output = attention.attn_drop(weights) @ value

        output = output.transpose(1, 2).reshape(batch, tokens, attention.attn_dim)
        output = attention.norm(output)
        output = attention.proj(output)
        return attention.proj_drop(output)


class PCDSContinualViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        classifier_scale: float,
        adapter_bottleneck: int,
        adapter_dropout: float,
        adapter_placement: str = "mlp",
        activation_guidance: nn.Module | None = None,
        activation_guidance_layer: int = -1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.feature_dim = int(self.backbone.num_features)
        self.adapter_placement = str(adapter_placement)
        if self.adapter_placement not in {"mlp", "attention"}:
            raise ValueError("adapter_placement must be 'mlp' or 'attention'")
        self.adapters = nn.ModuleList()
        self.adapter_wrappers: list[nn.Module] = []
        target_layer = int(activation_guidance_layer)
        if target_layer < 0:
            target_layer += len(self.backbone.blocks)
        if activation_guidance is not None and not 0 <= target_layer < len(
            self.backbone.blocks
        ):
            raise ValueError(f"Invalid activation guidance layer {activation_guidance_layer}")
        self.activation_guidance_layer = target_layer
        for layer_id, block in enumerate(self.backbone.blocks):
            guidance = activation_guidance if layer_id == target_layer else None
            if self.adapter_placement == "mlp":
                wrapped = MLPWithPCDSAdapter(
                    block.mlp,
                    feature_dim=self.feature_dim,
                    bottleneck=int(adapter_bottleneck),
                    dropout=float(adapter_dropout),
                    activation_guidance=guidance,
                )
                block.mlp = wrapped
                self.adapters.append(wrapped.adapter)
            else:
                wrapped = AttentionWithPCDSAdapters(
                    block.attn,
                    feature_dim=self.feature_dim,
                    bottleneck=int(adapter_bottleneck),
                    dropout=float(adapter_dropout),
                    activation_guidance=guidance,
                )
                block.attn = wrapped
                self.adapters.extend([wrapped.key_adapter, wrapped.value_adapter])
            self.adapter_wrappers.append(wrapped)
        self.classifier = ExpandingCosineClassifier(
            self.feature_dim, float(classifier_scale)
        )

    @property
    def activation_guidance(self) -> nn.Module | None:
        if not 0 <= self.activation_guidance_layer < len(self.adapter_wrappers):
            return None
        return self.adapter_wrappers[
            self.activation_guidance_layer
        ].activation_guidance

    def features(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> torch.Tensor:
        guidance = self.activation_guidance
        if guidance is not None and labels is not None:
            guidance.set_batch_context(labels, int(epoch))
        try:
            features = self.backbone(images)
        finally:
            if guidance is not None:
                guidance.clear_batch_context()
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected backbone feature shape {features.shape}")
        return features

    def base_features(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen PTM features with every adapter bypassed (SGDS Eq. 4)."""
        states = [wrapper.adapter_enabled for wrapper in self.adapter_wrappers]
        try:
            for wrapper in self.adapter_wrappers:
                wrapper.adapter_enabled = False
            features = self.backbone(images)
        finally:
            for wrapper, enabled in zip(self.adapter_wrappers, states):
                wrapper.adapter_enabled = enabled
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(f"Unexpected frozen-backbone feature shape {features.shape}")
        return features

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        epoch: int | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.features(images, labels=labels, epoch=epoch)
        return {"features": features, "logits": self.classifier(features)}

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapter in self.adapters
            for parameter in adapter.parameters()
            if parameter.requires_grad
        ]

    def classifier_all_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier.weight_blocks)

    def train_all_classifier_rows(self) -> list[bool]:
        states = [parameter.requires_grad for parameter in self.classifier.weight_blocks]
        for parameter in self.classifier.weight_blocks:
            parameter.requires_grad_(True)
        return states
