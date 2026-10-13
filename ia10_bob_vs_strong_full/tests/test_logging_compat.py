"""No CUDA/dependency imports: scoped logger conversion regression tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fulltraj.common import read_json
from fulltraj.logging_compat import compatible_tracker_config, tensorboard_config


class FakeAccelerator:
    def init_trackers(self, project_name, config=None, init_kwargs=None):
        assert all(isinstance(v, (bool, int, float, str)) for v in config.values())
        self.received = (project_name, config, init_kwargs)
        return "initialized"


class LoggingCompatibilityTests(unittest.TestCase):
    def test_scalars_and_nested_values_without_mutating_training_config(self):
        config = {"classes": ["box turtle", "other"], "revision": None, "nested": {"x": [1, None]},
                  "rank": 16, "lr": .0001, "frozen": True, "path": Path("/some/path"), "name": "run"}
        original = copy.deepcopy(config)
        safe = tensorboard_config(config)
        self.assertEqual(config, original)
        self.assertEqual(safe["rank"], 16)
        self.assertEqual(safe["lr"], .0001)
        self.assertIs(safe["frozen"], True)
        self.assertEqual(json.loads(safe["classes"]), config["classes"])
        self.assertIsNone(json.loads(safe["revision"]))
        self.assertEqual(json.loads(safe["nested"]), config["nested"])
        self.assertEqual(json.loads(safe["path"]), str(config["path"]))

    def test_full_json_keeps_lists_nulls_and_original_accelerator_is_restored(self):
        official = SimpleNamespace(Accelerator=FakeAccelerator)
        config = {"classnames": ["box turtle"], "revision": None, "epochs": 400}
        with tempfile.TemporaryDirectory(prefix="bob_tracking_test_") as temporary:
            path = Path(temporary) / "full_config.json"
            with compatible_tracker_config(official, path):
                logger = official.Accelerator()
                self.assertEqual(logger.init_trackers("test", config=config), "initialized")
                self.assertEqual(read_json(path), config)
                self.assertEqual(logger.received[1]["revision"], "null")
            self.assertIs(official.Accelerator, FakeAccelerator)
            self.assertEqual(config["classnames"], ["box turtle"])
            self.assertIsNone(config["revision"])

    def test_restore_on_training_exception(self):
        official = SimpleNamespace(Accelerator=FakeAccelerator)
        with self.assertRaisesRegex(RuntimeError, "simulated training error"):
            with compatible_tracker_config(official, Path("unused_config.json")):
                raise RuntimeError("simulated training error")
        self.assertIs(official.Accelerator, FakeAccelerator)
