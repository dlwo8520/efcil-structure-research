from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rp.learner import RPProtoCLADLearner
from rp.losses import rp_proto_clad_relation
from rp.projection import make_projection, project_relu, projection_checksum
from stored_apr_rp.learner import StoredAPRRPProtoCLADLearner
from stored_apr_rp.losses import stored_apr_rp_relation
from stored_apr_rp.prototypes import apr_corrected_prototypes, birth_prototypes


def test_birth_is_samplewise_projected_mean_not_projected_mean():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0])
    matrix = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    record = birth_prototypes(features, labels, [0], matrix)[0]
    exact = project_relu(features, matrix).mean(dim=0)
    oldstyle = project_relu(features.mean(dim=0, keepdim=True), matrix)[0]
    assert torch.equal(record["q_birth"], exact)
    assert torch.equal(record["mu_birth"], features.mean(dim=0))
    assert not torch.equal(record["q_birth"], oldstyle)
    assert torch.equal(record["q_birth"], torch.tensor([0.5, 0.5]))


def test_zero_and_nonzero_apr_drift_use_exact_formula():
    matrix = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    records = {0: {
        "mu_birth": torch.tensor([0.5, 0.5]),
        "q_birth": torch.tensor([0.5, 0.5]),
    }}
    unchanged, oldstyle, _, projected_birth = apr_corrected_prototypes(
        records, torch.tensor([[0.5, 0.5]]), matrix,
    )
    assert torch.equal(unchanged, records[0]["q_birth"][None])
    assert torch.equal(oldstyle, projected_birth)
    apr = torch.tensor([[1.0, 0.0]])
    current, oldstyle, birth, projected_birth = apr_corrected_prototypes(records, apr, matrix)
    assert torch.equal(current, birth + oldstyle - projected_birth)
    assert torch.equal(current - oldstyle, birth - projected_birth)
    assert torch.equal(current, torch.tensor([[1.5, 0.5]]))


def test_original_r_checksum_and_frozen_regression():
    matrix = make_projection(768, 2048, 1993)
    assert matrix.requires_grad is False
    assert projection_checksum(matrix) == (
        "849337f236a4df355a5d74c27d015c45a80c27cd4f7fc0e32957a02c7619aa2d"
    )


def test_loss_matches_old_rp_when_given_oldstyle_and_has_same_selection():
    matrix = make_projection(3, 8, 1993)
    cache = SimpleNamespace(
        covariance_view="diagonal", region_count=2,
        means=torch.zeros(2, 3), factors=torch.ones(2, 3),
        taus=torch.ones(2),
    )
    features = torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.2, 0.1]], requires_grad=True)
    targets = torch.tensor([2, 3])
    old = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    current = torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.2, 0.1]])
    similarities = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    args = (features, targets, cache, old, current, similarities, 2, 0.5, matrix)
    old_result = rp_proto_clad_relation(*args)
    new_result = stored_apr_rp_relation(
        features, targets, cache, old, current, project_relu(old, matrix),
        similarities, 2, 0.5, matrix,
    )
    assert torch.allclose(new_result.loss, old_result.loss)
    assert torch.equal(
        new_result.diagnostics["selected_mask"], old_result.diagnostics["selected_mask"]
    )
    new_result.loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert matrix.grad is None


def test_first_task_birth_hook_stores_class_level_statistic():
    with tempfile.TemporaryDirectory(prefix="stored_apr_birth_") as temporary:
        model = StoredAPRRPProtoCLADLearner.__new__(StoredAPRRPProtoCLADLearner)
        model.args = {
            "protocol_id": "IA5", "experiment_id": "StoredAPR_RP_C_P020_L030", "seed": 1993,
            "rp_seed": 1993, "rp_input_dim": 2, "rp_dim": 2,
            "rp_matrix_shape": [2, 2], "rp_matrix_checksum": "test-sha",
        }
        model.run_dir = Path(temporary)
        model._device = torch.device("cpu")
        model._rp_matrix_cpu = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        model._rp_device_cache = {}
        model._birth_records = {}
        model._known_classes = 0
        model._total_classes = 1
        model._cur_task = 0
        model._feature_space_id = lambda stage: "task_0:" + stage
        sample_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        sample_labels = torch.tensor([0, 0])

        def fake_parent_update(instance, *_args):
            instance._collect_current_features("sfeatures", return_sample_ids=True)

        with patch.object(RPProtoCLADLearner, "_update_region_statistics", fake_parent_update), patch.object(
            RPProtoCLADLearner, "_collect_current_features",
            return_value=(sample_features, sample_labels, ["s0", "s1"]),
        ):
            model._update_region_statistics(None, None, None)
        assert torch.equal(model._birth_records[0]["q_birth"], torch.tensor([0.5, 0.5]))
        assert model._birth_records[0]["birth_task"] == 1
        assert (model.run_dir / "stored_rp_birth_prototypes.pt").is_file()
        assert (model.run_dir / "stored_rp_prototype_diagnostics.csv").is_file()
        assert model._birth_features is None and model._birth_labels is None


def test_checkpoint_restores_identical_birth_class_statistic():
    with tempfile.TemporaryDirectory(prefix="stored_apr_checkpoint_") as temporary:
        model = StoredAPRRPProtoCLADLearner.__new__(StoredAPRRPProtoCLADLearner)
        model.args = {
            "rp_seed": 1993, "rp_input_dim": 2, "rp_dim": 2,
            "rp_matrix_shape": [2, 2], "rp_matrix_checksum": "test-sha",
            "rp_projection": {"seed": 1993}, "expected_tasks": 5,
        }
        model.run_dir = Path(temporary)
        model._birth_records = {0: {
            "class_id": 0, "birth_task": 1,
            "mu_birth": torch.tensor([0.5, 0.5]),
            "q_birth": torch.tensor([0.5, 0.5]),
            "sample_count": 2, "rp_checksum": "test-sha",
            "rp_seed": 1993, "rp_shape": [2, 2],
            "feature_dim": 2, "rp_dim": 2,
        }}
        model._cur_task = 0
        model._known_classes = 1
        model._total_classes = 1
        model.task_sizes = [1]
        model._checkpoint_network = lambda: SimpleNamespace(state_dict=lambda: {})
        model._class_means = None
        model._class_covs = None
        model._local_stats = {}
        model._single_regions = {}
        model._task_source_means = None
        model._task_source_covs = None
        model._task_source_local_stats = {}
        model._sep_reference_means = None
        model._sep_reference_covs = None
        target = Path(temporary) / "checkpoint_latest.pt"
        model.save_experiment_checkpoint(target, stage="task_end")
        original = model._birth_records[0]["q_birth"].clone()
        model._birth_records = {}
        with patch.object(RPProtoCLADLearner, "load_experiment_checkpoint", return_value={}):
            model.load_experiment_checkpoint(target)
        assert torch.equal(model._birth_records[0]["q_birth"], original)
        assert model._birth_records[0]["rp_checksum"] == "test-sha"
