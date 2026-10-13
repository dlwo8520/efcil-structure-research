import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from network.vision_transformer_adapter import Adapter, VisionTransformer
from utils.ptm_guidance import (
    AnchorResponseStability,
    PTMGuidanceMemory,
    placement_weight,
    ptm_placement_loss,
)
from utils.ptm_metrics import PTMGuidedResultWriter


def adapter_config(d_model=4, bottleneck=2):
    return SimpleNamespace(
        d_model=d_model,
        attn_bn=bottleneck,
        ffn_num=bottleneck,
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="1.0",
        ffn_adapter_layernorm_option="none",
        adapter_placement="mlp",
        adapter_interval=1,
        vpt_on=False,
        vpt_num=0,
    )


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        config = adapter_config()
        self.mlp_adapter1 = Adapter(config=config, d_model=4, bottleneck=2,
                                    init_option="lora", adapter_scalar="1.0",
                                    adapter_layernorm_option="none")
        self.mlp_adapter2 = Adapter(config=config, d_model=4, bottleneck=2,
                                    init_option="lora", adapter_scalar="1.0",
                                    adapter_layernorm_option="none")


class TinyConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.tuning_config = adapter_config()
        self.blocks = nn.ModuleList([TinyBlock()])
        self.flag = 0

    def active_adapter_branch(self):
        return 1 if self.flag % 2 == 0 else 2


class PTMGuidanceTests(unittest.TestCase):
    def test_placement_loss_and_gradient(self):
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 0])
        same = {0: torch.tensor([1.0, 0.0])}
        different = {0: torch.tensor([0.0, 1.0])}
        zero_loss = ptm_placement_loss(features, labels, same)
        positive_loss = ptm_placement_loss(features, labels, different)
        self.assertLess(abs(zero_loss.item()), 1e-6)
        self.assertGreater(positive_loss.item(), 0.9)
        positive_loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_schedule_reaches_configured_endpoints(self):
        self.assertAlmostEqual(placement_weight(0, 5, 1.0, 0.0, "cosine"), 1.0)
        self.assertAlmostEqual(placement_weight(4, 5, 1.0, 0.0, "cosine"), 0.0)
        self.assertAlmostEqual(placement_weight(2, 5, 1.0, 0.0, "linear"), 0.5)

    def test_movement_loss_zero_then_positive_and_finite(self):
        convnet = TinyConvnet()
        adapter = convnet.blocks[0].mlp_adapter1
        with torch.no_grad():
            adapter.down_proj.weight.fill_(0.25)
            adapter.down_proj.bias.fill_(0.1)
            adapter.up_proj.weight.fill_(0.2)
            adapter.up_proj.bias.zero_()
        memory = PTMGuidanceMemory()
        memory.layer_anchors = {"layer_00_mlp": {0: torch.ones(4)}}
        stability = AnchorResponseStability("smooth_l1")
        self.assertEqual(stability.prepare(memory, convnet), 1)
        self.assertLess(abs(stability.loss(convnet).item()), 1e-8)
        with torch.no_grad():
            adapter.up_proj.weight.add_(0.2)
        moved = stability.loss(convnet)
        self.assertGreater(moved.item(), 0.0)
        moved.backward()
        self.assertTrue(torch.isfinite(adapter.up_proj.weight.grad).all())

    def test_task_zero_empty_movement_is_zero(self):
        convnet = TinyConvnet()
        stability = AnchorResponseStability()
        self.assertEqual(stability.prepare(PTMGuidanceMemory(), convnet), 0)
        loss = stability.loss(convnet)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(convnet.blocks[0].mlp_adapter1.down_proj.weight.grad).all())

    def test_adapter_free_forward_is_explicit_and_no_grad(self):
        config = adapter_config(d_model=8, bottleneck=2)
        model = VisionTransformer(
            img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2,
            mlp_ratio=2, qkv_bias=True, num_classes=0, tuning_config=config,
        )
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            before = model.extract_features(images, use_adapter=False)["features"]
            model.blocks[0].mlp_adapter1.up_proj.weight.add_(10.0)
            after = model.extract_features(images, use_adapter=False)["features"]
        self.assertFalse(before.requires_grad)
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))

    def test_result_writer_creates_required_files(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = PTMGuidedResultWriter(temp, "synthetic", 1, "ptm_place_move", 1993)
            writer.task_metrics.append({"last_accuracy": 50.0})
            writer.task_accuracy_matrix.append([50.0])
            writer.class_accuracy.append({"task": 0, "class_id": 0, "accuracy": 50.0})
            writer.ptm_geometry_drift.append({"task": 0, "ptm_geometry_drift": 0.1})
            writer.anchor_drift.append({"task": 0, "response_cosine_drift": 0.0})
            writer.loss_components.append({"task": 0, "total_loss": 1.0})
            writer.flush({
                "dataset": "synthetic", "task_count": 1, "method": "ptm_place_move",
                "seed": 1993, "smoke_test": False,
            })
            output = Path(writer.directory)
            expected = {
                "metrics.json", "task_metrics.csv", "task_accuracy_matrix.csv",
                "class_accuracy.csv", "ptm_geometry_drift.csv", "anchor_drift.csv",
                "loss_components.csv",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})
            self.assertEqual(json.loads((output / "metrics.json").read_text())["status"], "COMPLETE")

    def test_result_writer_marks_partial_runs_in_progress(self):
        with tempfile.TemporaryDirectory() as temp:
            writer = PTMGuidedResultWriter(temp, "synthetic", 2, "ptm_place_move", 1993)
            writer.task_metrics.append({"last_accuracy": 50.0})
            writer.flush({
                "dataset": "synthetic", "task_count": 2, "method": "ptm_place_move",
                "seed": 1993, "smoke_test": False,
            })
            metrics = Path(writer.directory, "metrics.json")
            self.assertEqual(json.loads(metrics.read_text())["status"], "IN_PROGRESS")


if __name__ == "__main__":
    unittest.main()
