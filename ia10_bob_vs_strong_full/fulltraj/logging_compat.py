"""Adapt tracker metadata only; never modify arguments used by BOB training."""
from __future__ import annotations

import json
from contextlib import contextmanager

from .common import write_json


def tensorboard_config(config):
    """TensorBoard hparams cannot accept argparse None/list/dict values."""
    return {
        key: value if isinstance(value, (bool, int, float, str))
        else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for key, value in config.items()
    }


@contextmanager
def compatible_tracker_config(official, full_config_path):
    """Scope the integration shim to this loaded official module, not Accelerate."""
    original = official.Accelerator

    class ScalarConfigAccelerator(original):
        def init_trackers(self, project_name, config=None, init_kwargs=None):
            if config is not None:
                # Keep lists/nulls in a separate complete JSON artifact. The
                # official Dataset and optimizer still see the unmodified args.
                raw = json.loads(json.dumps(config, ensure_ascii=False, default=str))
                write_json(full_config_path, raw)
                config = tensorboard_config(config)
            return super().init_trackers(project_name, config=config, init_kwargs=init_kwargs or {})

    official.Accelerator = ScalarConfigAccelerator
    try:
        yield
    finally:
        official.Accelerator = original
