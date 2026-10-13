import json
import unittest
from pathlib import Path


class ConfigMatrixTests(unittest.TestCase):
    def test_e0_e7_factorial_and_protocol(self):
        root = Path(__file__).resolve().parents[1] / "exps_jsons" / "imageneta_5task"
        configs = [json.loads((root / f"E{i}.json").read_text(encoding="utf-8")) for i in range(8)]
        expected = [
            (False, False, "full"),
            (True, False, "full"),
            (False, True, "isotropic"),
            (False, True, "diagonal"),
            (False, True, "full"),
            (True, True, "isotropic"),
            (True, True, "diagonal"),
            (True, True, "full"),
        ]
        self.assertEqual(
            [(c["multi_center"], c["old_new_separation"], c["separation_covariance"]) for c in configs],
            expected,
        )
        for index, config in enumerate(configs):
            self.assertEqual(config["experiment_id"], f"E{index}")
            self.assertEqual(config["seed"], [1993, 1996, 1997])
            self.assertEqual(config["init_cls"], 40)
            self.assertEqual(config["increment"], 40)
            self.assertEqual(config["expected_tasks"], 5)
            self.assertEqual(config["replay_samples_per_class"], 256)


if __name__ == "__main__":
    unittest.main()
