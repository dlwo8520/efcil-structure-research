import unittest

import torch
from easydict import EasyDict

from models.sgds_activation_guidance import SGDSActivationGuidance
from network.vision_transformer_adapter import VisionTransformer


def tiny_model():
    config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=4,
        d_model=16,
        vpt_on=False,
        vpt_num=0,
        adapter_placement="attention",
        adapter_interval=1,
    )
    return VisionTransformer(
        img_size=32, patch_size=16, embed_dim=16, depth=2, num_heads=4,
        mlp_ratio=2, qkv_bias=True, tuning_config=config, num_classes=0,
    )


class VitSGDSIntegrationTest(unittest.TestCase):
    def test_sgds_off_matches_original_forward_path(self):
        torch.manual_seed(11)
        model = tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)
        actual = model(images)
        with torch.no_grad():
            teacher = model.forward_features(images, flag=2)
        student = model.forward_features(images, flag=1)
        self.assertTrue(torch.equal(actual["tfeatures"], teacher["features"]))
        self.assertTrue(torch.equal(actual["sfeatures"], student["features"]))
        self.assertIsNone(model.get_sgds_guidance())

    def test_only_final_block_is_guided_and_eval_has_no_mask(self):
        model = tiny_model()
        guidance = SGDSActivationGuidance(
            feature_dim=16, num_classes=2, num_epochs=2, target_layer=1,
            paper_k=0.5, beta=0.5, gamma=1.0, exploration_ratio=0.5,
        )
        model.install_sgds_guidance(guidance, 1)
        self.assertIsNone(model.blocks[0].attn.sgds_guidance)
        self.assertIs(model.blocks[1].attn.sgds_guidance, guidance)

        guidance.prepare_task(0, 0, torch.eye(2, 16))
        guidance.train()
        model.train()
        images = torch.randn(2, 3, 32, 32)
        guidance.set_batch_context(torch.tensor([0, 1]), epoch=0)
        model(images)
        guidance.clear_batch_context()
        self.assertGreater(float(guidance.global_counter.sum()), 0.0)

        counter = guidance.global_counter.clone()
        model.eval()
        first = model(images)["sfeatures"]
        second = model(images)["sfeatures"]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(counter, guidance.global_counter))

    def test_frozen_semantic_path_bypasses_both_adapter_banks(self):
        model = tiny_model().eval()
        for block in model.blocks:
            with torch.no_grad():
                block.attn.k_mlp1.up_proj.weight.fill_(1.0)
                block.attn.v_mlp1.up_proj.weight.fill_(1.0)
                block.attn.k_mlp2.up_proj.weight.fill_(2.0)
                block.attn.v_mlp2.up_proj.weight.fill_(2.0)
        images = torch.randn(1, 3, 32, 32)
        frozen_first = model.forward_frozen_features(images)["features"]
        model.flag = 1
        frozen_second = model.forward_frozen_features(images)["features"]
        self.assertTrue(torch.equal(frozen_first, frozen_second))


if __name__ == "__main__":
    unittest.main()
