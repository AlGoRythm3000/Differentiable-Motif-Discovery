import torch

from models.weight_assignment import GumbelSigmoidSelector, KSubsetSelector, REINFORCESelector


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


# --------------------------------------------------------------------------
# KSubsetSelector (feat/rich-bricks)
# --------------------------------------------------------------------------

def test_ksubset_soft_output_sums_to_k():
    torch.manual_seed(0)
    scores = torch.randn(20)
    selector = KSubsetSelector(k=5, tau_init=1.0)
    selector.train()
    alpha = selector(scores)
    assert torch.allclose(alpha.sum(), torch.tensor(5.0), atol=1e-3)


def test_ksubset_hard_output_selects_exactly_k():
    torch.manual_seed(0)
    scores = torch.randn(20)
    selector = KSubsetSelector(k=5, tau_init=1.0, hard=True)
    selector.train()
    alpha = selector(scores)
    assert int((alpha.detach() == 1.0).sum().item()) == 5


def test_ksubset_gradient_flows_to_scores():
    torch.manual_seed(0)
    scores = torch.randn(20, requires_grad=True)
    selector = KSubsetSelector(k=4, tau_init=1.0)
    selector.train()
    alpha = selector(scores)
    alpha.sum().backward()
    assert scores.grad is not None
    assert torch.any(scores.grad != 0)


def test_ksubset_clamps_k_to_available_cells():
    scores = torch.randn(3)
    selector = KSubsetSelector(k=10)
    selector.eval()
    alpha = selector(scores)
    assert alpha.shape == (3,)


def test_ksubset_set_temperature_respects_floor():
    selector = KSubsetSelector(k=2, tau_init=1.0, tau_min=0.2)
    selector.set_temperature(0.01)
    assert selector.tau == 0.2


# --------------------------------------------------------------------------
# REINFORCESelector (feat/rich-bricks)
# --------------------------------------------------------------------------

def test_reinforce_train_output_is_binary_and_has_log_prob():
    torch.manual_seed(0)
    scores = torch.randn(15)
    selector = REINFORCESelector()
    selector.train()
    alpha = selector(scores)
    assert torch.all((alpha == 0.0) | (alpha == 1.0))
    assert selector.last_log_prob is not None
    assert selector.last_log_prob.shape == scores.shape


def test_reinforce_forward_value_carries_no_gradient_to_scores():
    """
    Gradient-stop contract: `alpha` itself must not backprop to `scores` -
    the differentiable signal lives entirely in `last_log_prob`, consumed by
    tools/losses.py::DMDLoss.
    """
    torch.manual_seed(0)
    scores = torch.randn(15, requires_grad=True)
    selector = REINFORCESelector()
    selector.train()
    alpha = selector(scores)
    alpha.sum().backward()
    assert scores.grad is not None
    assert torch.all(scores.grad == 0)


def test_reinforce_log_prob_carries_gradient_to_scores():
    torch.manual_seed(0)
    scores = torch.randn(15, requires_grad=True)
    selector = REINFORCESelector()
    selector.train()
    selector(scores)
    selector.last_log_prob.sum().backward()
    assert scores.grad is not None
    assert torch.any(scores.grad != 0)


def test_reinforce_eval_mode_is_deterministic_and_clears_log_prob():
    scores = torch.randn(15)
    selector = REINFORCESelector()
    selector.eval()
    alpha_1 = selector(scores)
    alpha_2 = selector(scores)
    assert torch.equal(alpha_1, alpha_2)
    assert selector.last_log_prob is None


def test_reinforce_update_baseline_tracks_ema():
    selector = REINFORCESelector(baseline_decay=0.5)
    selector.update_baseline(torch.tensor(1.0))
    assert selector.baseline.item() == 1.0  # first call initializes, not blends
    selector.update_baseline(torch.tensor(3.0))
    assert abs(selector.baseline.item() - 2.0) < 1e-6  # 0.5*1.0 + 0.5*3.0


def test_invalid_baseline_decay_raises():
    try:
        REINFORCESelector(baseline_decay=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
