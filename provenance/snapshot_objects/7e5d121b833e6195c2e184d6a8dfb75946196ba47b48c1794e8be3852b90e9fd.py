from types import SimpleNamespace

import torch

from selective.losses import (
    enlarged_boundary_relation,
    proto_clad_relation,
    selective_am_relation,
    top_p_selection,
)
from similarity.losses import legacy_absolute_relation


def cache():
    return SimpleNamespace(
        covariance_view="diagonal",
        region_count=4,
        means=torch.zeros(4, 3),
        factors=torch.ones(4, 3),
        taus=torch.tensor([2.0, 3.0, 4.0, 5.0]),
    )


def inputs():
    features = torch.tensor([[0.2, 0.1, 0.3], [0.4, 0.2, 0.1]], requires_grad=True)
    targets = torch.tensor([4, 5])
    similarities = torch.tensor([[0.9, 0.7, 0.2, -0.1], [0.3, 0.8, 0.6, 0.1]])
    prototypes = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return features, targets, similarities, prototypes


def test_top_p_is_per_class_exact_and_tie_stable():
    similarities = torch.tensor([[0.5, 0.5, 0.4, 0.3], [0.1, 0.2, 0.3, 0.4]])
    selected, ranks, cutoff, count = top_p_selection(similarities, 0.25)
    assert count == 1
    assert selected[0].tolist() == [True, False, False, False]
    assert ranks[0].tolist() == [1, 2, 3, 4]
    assert torch.allclose(cutoff, torch.tensor([0.5, 0.4]))


def test_all_three_losses_are_finite_and_differentiable():
    for name in ("A", "B", "C"):
        features, targets, similarities, prototypes = inputs()
        if name == "A":
            result = selective_am_relation(features, targets, cache(), similarities, 4, 0.5)
        elif name == "B":
            result = enlarged_boundary_relation(features, targets, cache(), similarities, 4, 0.5, 0.5)
        else:
            result = proto_clad_relation(features, targets, cache(), prototypes, similarities, 4, 0.5)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert features.grad is not None and torch.isfinite(features.grad).all()
        assert result.diagnostics["selected_mask"].sum(dim=1).tolist() == [2, 2]


def test_c0_is_exact_inherited_am_drift():
    features, _, _, _ = inputs()
    first = legacy_absolute_relation(features, cache()).loss
    second = legacy_absolute_relation(features, cache()).loss
    assert torch.equal(first, second)
