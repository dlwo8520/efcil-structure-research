"""CPU-only numerical invariants; no model training or dataset access."""
import torch

from losses import assert_ce_identity, ca_loss, class_birth_tasks, loss_components


def test_ce_identity_irregular_car_partition():
    torch.manual_seed(1993)
    partition = [20] * 6 + [19] * 4
    birth = class_birth_tasks(partition)
    logits = torch.randn(257, 196, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 196, (257,))
    assert assert_ce_identity(logits, labels, birth, atol=1e-12) < 1e-12
    within, cross, _ = loss_components(logits, labels, birth)
    control, _ = ca_loss(logits, labels, birth, 1)
    assert torch.allclose((within + cross).mean(), control, rtol=0, atol=1e-12)


def test_intervention_changes_only_cross_term():
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]], dtype=torch.float64)
    labels = torch.tensor([0])
    birth = class_birth_tasks([2, 2])
    within, cross, _ = loss_components(logits, labels, birth)
    control, _ = ca_loss(logits, labels, birth, 1)
    intervention, _ = ca_loss(logits, labels, birth, 2)
    assert torch.allclose(intervention - control, cross.mean(), rtol=0, atol=1e-12)
