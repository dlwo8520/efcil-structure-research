import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from type_a_generative_augmentation import automatic_sanity as sanity
from type_a_generative_augmentation.common import ROOT, read_json, read_csv, write_json, write_csv, sha256


class FakeEncoder:
    def score(self, pairs):
        return [{"global_cosine": .9, "center_cosine": .9} for _ in pairs]


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        self.policy = read_json(ROOT / "sanity_policy.json")

    def test_identity_detected_and_small_semantic_similarity_not_label_proof(self):
        rng = np.random.RandomState(10)
        image = Image.fromarray(rng.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        metrics = sanity.pixel_metrics(image, image, self.policy)
        metrics.update(global_cosine=.99, center_cosine=.99)
        self.assertIn("near_duplicate", sanity.exclusion_rules(metrics, self.policy))
        changed = metrics.copy()
        changed.update(block_ssim=.7, mean_absolute_error=.1, dhash_distance=10)
        self.assertEqual(sanity.exclusion_rules(changed, self.policy), [])

    def test_gross_failure_proxies_and_blank(self):
        base = {"block_ssim": .4, "mean_absolute_error": .2, "dhash_distance": 12,
                "generated_channel_std": .2, "generated_edge_energy": .04,
                "global_cosine": .1, "center_cosine": .1, "center_edge_ratio": .05,
                "parent_center_edge_energy": .1, "center_block_ssim": .1}
        reasons = sanity.exclusion_rules(base, self.policy)
        self.assertIn("gross_semantic_deviation_proxy", reasons)
        self.assertIn("extreme_structure_loss_proxy", reasons)
        base.update(global_cosine=.9, center_cosine=.9, generated_channel_std=0., generated_edge_energy=0.)
        self.assertIn("collapsed_image", sanity.exclusion_rules(base, self.policy))

    def test_all_survivors_used_manual_file_ignored_and_broken_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mode = read_json(ROOT / "exploratory_config.json")
            write_json(root / "exploratory_config.json", mode)
            write_json(root / "sanity_policy.json", self.policy)
            rng = np.random.RandomState(22)
            rows = []
            for protocol in ("IA10", "CAR10"):
                parent = Image.fromarray(rng.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                generated = Image.fromarray(rng.randint(0, 255, (512, 512, 3), dtype=np.uint8))
                p, g, broken = root / f"{protocol}_parent.png", root / f"{protocol}_gen.png", root / f"{protocol}_broken.png"
                parent.save(p)
                generated.save(g)
                # Existing non-image file stands in for a corrupt image, without editing any production image.
                broken = root / "exploratory_config.json"
                for kind, image_path in (("good", g), ("duplicate", p), ("broken", broken)):
                    rows.append({"record_id": protocol + kind, "protocol": protocol, "task": 1, "y": 0,
                                 "parent_id": "parent", "parent_path": str(p), "parent_sha256": sha256(p),
                                 "generated_path": str(image_path), "generated_sha256": sha256(image_path), "generation_status": "generated"})
            write_csv(root / "generation_manifest.csv", rows)
            # Human exclusions are intentionally ignored in exploratory mode.
            write_csv(root / "quality_review.csv", [{"record_id": r["record_id"], "review_status": "reject"} for r in rows])
            with patch.object(sanity, "ROOT", root), patch.object(sanity, "fingerprints", return_value={"synthetic": True}):
                accepted = sanity.run_sanity(torch.device("cpu"), FakeEncoder())
                self.assertEqual({r["record_id"] for r in accepted}, {"IA10good", "CAR10good"})
                verified = sanity.load_accepted()
                self.assertTrue(all(r["label_fidelity_verified"] == "false" for r in verified))
                qc = read_csv(root / "automatic_sanity.csv")
                self.assertIn("missing_or_undecodable_image", next(r for r in qc if r["record_id"] == "IA10broken")["exclusion_reason"])
                self.assertEqual(sanity.run_sanity(torch.device("cpu"), FakeEncoder()), verified)
                policy = copy.deepcopy(self.policy)
                policy["version"] = 99
                # Artifact mutation cannot silently change the common bank.
                write_csv(root / "accepted_parent_manifest.csv", verified[:1])
                with self.assertRaisesRegex(RuntimeError, "output changed"):
                    sanity.load_accepted()


if __name__ == "__main__":
    unittest.main()
