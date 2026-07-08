import torch
import torch.nn.functional as F

from tools.losses import DMDLoss


def _toy_inputs():
    logits = torch.randn(6, 3, requires_grad=True)
    target = torch.randint(0, 3, (6,))
    mask = torch.ones(6, dtype=torch.bool)
    return logits, target, mask


def test_default_osq_is_noop_and_matches_task_plus_sparsity():
    logits, target, mask = _toy_inputs()
    alpha = torch.rand(6)
    criterion = DMDLoss(sparsity_weight=0.1)

    out = criterion(logits, target, mask, {"alpha": alpha})

    expected_task = F.cross_entropy(logits[mask], target[mask])
    expected_total = expected_task + 0.1 * alpha.mean()

    assert torch.allclose(out.total, expected_total)
    assert out.osq.item() == 0.0


def test_denser_alpha_increases_sparsity_and_total_loss():
    logits, target, mask = _toy_inputs()
    criterion = DMDLoss(sparsity_weight=0.1)

    out_sparse = criterion(logits, target, mask, {"alpha": torch.zeros(6)})
    out_dense = criterion(logits, target, mask, {"alpha": torch.ones(6)})

    assert out_dense.sparsity.item() > out_sparse.sparsity.item()
    assert out_dense.total.item() > out_sparse.total.item()


def test_osq_fn_only_invoked_when_weight_positive():
    logits, target, mask = _toy_inputs()
    alpha = torch.rand(6)
    calls = {"count": 0}

    def dummy_osq_fn(structure):
        calls["count"] += 1
        return structure["alpha"].sum()

    criterion_off = DMDLoss(sparsity_weight=0.0, osq_weight=0.0, osq_fn=dummy_osq_fn)
    criterion_off(logits, target, mask, {"alpha": alpha})
    assert calls["count"] == 0

    criterion_on = DMDLoss(sparsity_weight=0.0, osq_weight=0.5, osq_fn=dummy_osq_fn)
    out_on = criterion_on(logits, target, mask, {"alpha": alpha})
    assert calls["count"] == 1
    assert out_on.osq.item() == alpha.sum().item()


def test_gradient_reaches_logits_and_alpha():
    logits, target, mask = _toy_inputs()
    alpha = torch.rand(6, requires_grad=True)
    criterion = DMDLoss(sparsity_weight=0.1)

    out = criterion(logits, target, mask, {"alpha": alpha})
    out.total.backward()

    assert logits.grad is not None
    assert alpha.grad is not None
