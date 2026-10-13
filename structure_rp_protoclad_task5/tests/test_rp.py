from types import SimpleNamespace

import torch

from rp.losses import rp_proto_clad_relation
from rp.projection import make_projection, project_relu, projection_checksum


def _cache():
    return SimpleNamespace(
        covariance_view="diagonal",
        region_count=4,
        means=torch.zeros(4, 3),
        factors=torch.ones(4, 3),
        taus=torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )


def _inputs():
    features = torch.tensor(
        [[0.2, 0.1, 0.3], [0.4, 0.2, 0.1]], dtype=torch.float32, requires_grad=True
    )
    targets = torch.tensor([4, 5])
    similarities = torch.tensor([[0.9, 0.7, 0.2, -0.1], [0.3, 0.8, 0.6, 0.1]])
    old = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    current = torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.2, 0.1]])
    return features, targets, similarities, old, current


def test_projection_is_frozen_reproducible_and_seed_sensitive():
    first = make_projection(768, 2048, 1993)
    second = make_projection(768, 2048, 1993)
    other = make_projection(768, 2048, 1994)
    assert first.shape == (768, 2048)
    assert first.requires_grad is False
    assert torch.equal(first, second)
    assert projection_checksum(first) == projection_checksum(second)
    assert projection_checksum(first) != projection_checksum(other)
    # N(0,1/d) implies standard deviation close to 1/sqrt(d).
    assert abs(float(first.std()) - 768 ** -0.5) < 0.002


def test_rp_proto_clad_uses_original_selection_and_rp_loss_only():
    features, targets, similarities, old, current = _inputs()
    matrix = make_projection(3, 8, 1993)
    result = rp_proto_clad_relation(
        features,
        targets,
        _cache(),
        old,
        current,
        similarities,
        known_classes=4,
        top_p=0.5,
        rp_matrix=matrix,
    )
    assert result.diagnostics["selected_mask"].tolist() == [
        [True, True, False, False],
        [False, True, True, False],
    ]
    selected = result.diagnostics["selected_mask"]
    expected = torch.where(
        selected, 1.0 + result.diagnostics["rp_cosine"], 0.0
    ).sum(dim=1).div(2).mean()
    assert torch.allclose(result.loss.detach(), expected)
    assert not torch.equal(
        result.diagnostics["original_cosine"], result.diagnostics["rp_cosine"]
    )
    result.loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert matrix.grad is None and matrix.requires_grad is False


def test_relu_is_before_normalization_and_has_no_bias():
    features = torch.tensor([[1.0, -2.0, 0.5]])
    matrix = make_projection(3, 16, 1993)
    projected = project_relu(features, matrix)
    assert torch.equal(projected, torch.relu(features @ matrix))
    assert (projected >= 0).all()
    normalized = torch.nn.functional.normalize(projected, dim=1)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(1), atol=1e-6)

