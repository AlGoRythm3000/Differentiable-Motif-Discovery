import torch

from models.weight_assignment import GumbelSigmoidSelector


def test_hard_output_is_exactly_binary():
    torch.manual_seed(0)
    scores = torch.randn(20)
    selector = GumbelSigmoidSelector(tau=0.5, hard=True)
    selector.train()
    alpha = selector(scores)
    assert torch.all((alpha == 0.0) | (alpha == 1.0))


def test_soft_output_is_continuous():
    torch.manual_seed(0)
    scores = torch.randn(20)
    selector = GumbelSigmoidSelector(tau=0.5, hard=False)
    selector.train()
    alpha = selector(scores)
    assert torch.all(alpha > 0.0) and torch.all(alpha < 1.0)


def test_eval_mode_is_deterministic():
    scores = torch.randn(20)
    selector = GumbelSigmoidSelector(tau=0.5, hard=False)
    selector.eval()
    alpha_1 = selector(scores)
    alpha_2 = selector(scores)
    assert torch.allclose(alpha_1, alpha_2)


def test_straight_through_gradient_bypasses_hard_threshold():
    """
    Critical contract: even with hard=True (forward value is 0./1.), the
    gradient must flow to `scores` via the soft relaxation, not be zero.
    """
    torch.manual_seed(0)
    scores = torch.randn(50, requires_grad=True)
    selector = GumbelSigmoidSelector(tau=0.5, hard=True)
    selector.train()

    alpha = selector(scores)
    alpha.sum().backward()

    assert scores.grad is not None
    assert torch.any(scores.grad != 0)


def test_invalid_tau_raises():
    try:
        GumbelSigmoidSelector(tau=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
