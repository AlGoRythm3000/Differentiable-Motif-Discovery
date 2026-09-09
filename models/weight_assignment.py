# Step 4 : weight assignment to motifs to learn a graph embedding from the motif distribution

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Numerical floor used wherever a log() is taken of a quantity that can be
# exactly 0 (a fully-suppressed softmax mass, a saturated sigmoid).
_EPS = 1e-10


class GumbelSigmoidSelector(nn.Module):
    """
    Stage 4 (simple brick): differentiable accept/reject decision on Stage 2's
    per-cell proposal scores, via a binary Gumbel-softmax (Concrete/relaxed
    Bernoulli) relaxation with an optional straight-through estimator.

    Gradient-stop contract (read before touching this class):
      - `alpha_soft = sigmoid((scores + gumbel_noise) / tau)` is fully
        differentiable w.r.t. `scores`.
      - If `hard=False`, the forward value IS `alpha_soft` - nothing to document.
      - If `hard=True`, the forward value is `alpha_hard = (alpha_soft > 0.5)`,
        which is a `.detach()`'d boolean cast and carries ZERO gradient on its
        own. The straight-through trick
            `alpha = alpha_hard.detach() - alpha_soft.detach() + alpha_soft`
        makes the forward *value* hard (0./1.) while `d(alpha)/d(scores)`
        computed by autograd equals `d(alpha_soft)/d(scores)` exactly - i.e.
        gradient bypasses the hard threshold entirely. This is what lets
        gradients from Stage 6's objective (task, sparsity and the
        oversquashing term) reach `phi(z_C)` (== `scores` here).

    At eval time (`self.training == False`) sampling noise is dropped and the
    module is deterministic: `alpha_soft = sigmoid(scores / tau)`.
    """

    def __init__(self, tau: float = 0.5, hard: bool = True):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self.hard = hard
        self.last_log_prob = None  # Stage 4 selectors that need it (REINFORCE) set this; None otherwise.

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores: [num_cells] raw proposal logits from Stage 2.
        Returns alpha: [num_cells] in [0, 1] (soft) or {0., 1.} (hard, via STE).
        """
        if self.training:
            uniform = torch.rand_like(scores).clamp(min=1e-8, max=1 - 1e-8)
            gumbel_noise = torch.log(uniform) - torch.log1p(-uniform)
            alpha_soft = torch.sigmoid((scores + gumbel_noise) / self.tau)
        else:
            alpha_soft = torch.sigmoid(scores / self.tau)

        if not self.hard:
            return alpha_soft

        alpha_hard = (alpha_soft > 0.5).float()
        return alpha_hard.detach() - alpha_soft.detach() + alpha_soft


class KSubsetSelector(nn.Module):
    """
    Stage 4 rich brick: relaxed top-k via successive softmax without
    replacement (Xie & Ermon 2019, "Reparameterizable Subset Sampling via
    Continuous Relaxation" - the `SubsetOperator`). Unlike independent
    per-cell Bernoulli gating (`GumbelSigmoidSelector`), this gives explicit
    cardinality control: the output sums to ~k regardless of how many cells
    were proposed, instead of a k that only emerges statistically from `tau`
    and the score distribution.

    Mechanism: `k` sequential softmax draws, each one reweighted by the log of
    the *remaining* mass `(1 - already_selected_soft_mass)` before the next
    softmax - this is what "without replacement" means in the relaxed
    (differentiable) setting, as opposed to hard sampling-without-replacement
    which has no useful gradient. Every step is a plain softmax, so the whole
    thing differentiates through `scores` with no custom backward.

    Gradient-stop contract: identical shape to `GumbelSigmoidSelector`. Soft
    (`hard=False`, default) returns the accumulated khot vector directly -
    fully differentiable, entries in [0, 1] summing to ~k. Hard (`hard=True`)
    thresholds it to the true top-k indices via `torch.topk` (an exact k, not
    an approximate one) and applies the same straight-through trick: the
    forward *value* is the hard k-hot vector, but `d(alpha)/d(scores)` is
    `d(khot)/d(scores)` computed by autograd on the soft path.

    Temperature: `tau` starts at `tau_init` and is intended to be annealed
    towards `tau_min` over training via `set_temperature` (the grid runner
    calls this once per epoch and logs the schedule - see
    `tools/experiment_grid.py`); a fixed `tau_init` is used for the lifetime
    of the module if the caller never calls it.
    """

    def __init__(self, k: int, tau_init: float = 1.0, tau_min: float = 0.1,
                 anneal_rate: float = 1e-4, hard: bool = False):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        if tau_init <= 0 or tau_min <= 0:
            raise ValueError("tau_init and tau_min must be > 0")
        self.k = k
        self.tau_min = tau_min
        self.anneal_rate = anneal_rate
        self.tau = tau_init
        self.hard = hard
        self.last_log_prob = None

    def set_temperature(self, tau: float) -> None:
        """Called by the training loop to anneal `self.tau` (never below `tau_min`)."""
        self.tau = max(float(tau), self.tau_min)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores: [num_cells]. Returns alpha: [num_cells] in [0, 1], summing to
        ~k (soft) or exactly k ones (hard, via straight-through).
        """
        num_cells = scores.numel()
        k = min(self.k, num_cells)
        if k == 0:
            return torch.zeros_like(scores)

        if self.training:
            uniform = torch.rand_like(scores).clamp(min=_EPS, max=1 - _EPS)
            gumbel_noise = -torch.log(-torch.log(uniform))
            perturbed = (scores + gumbel_noise) / self.tau
        else:
            perturbed = scores / self.tau

        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        working = perturbed
        for _ in range(k):
            remaining_mass = torch.clamp(1.0 - onehot_approx, min=_EPS)
            working = working + torch.log(remaining_mass)
            onehot_approx = F.softmax(working, dim=-1)
            khot = khot + onehot_approx

        if not self.hard:
            return khot

        _, top_indices = torch.topk(khot.detach(), k=k)
        alpha_hard = torch.zeros_like(khot)
        alpha_hard[top_indices] = 1.0
        return alpha_hard.detach() - khot.detach() + khot


class REINFORCESelector(nn.Module):
    """
    Stage 4 rich brick: score-function (REINFORCE) estimator for the
    accept/reject decision - the "honest, expected-to-lose" comparison arm
    (spec 4.4). Sampling `alpha_c ~ Bernoulli(sigmoid(scores_c))` is a
    genuinely unbiased estimator of the gradient of the expected loss w.r.t.
    `scores`, unlike the biased-but-low-variance relaxations above, and it is
    included precisely so that trade-off is measured rather than assumed.

    Gradient-stop contract, deliberately different in *kind* from the other
    two selectors: `forward` itself returns a hard-sampled `alpha` that
    carries NO gradient to `scores` (`Bernoulli.sample()` is not
    differentiable, and there is no straight-through trick here - that would
    defeat the point of comparing against the relaxations). Instead:
      - `self.last_log_prob` [num_cells] holds `log p(alpha_c | scores_c)` for
        the sample just drawn - THIS is differentiable w.r.t. `scores`.
      - `DMDModel.forward` copies it into `structure["selector_log_prob"]`.
      - `tools/losses.py::DMDLoss`, when constructed with
        `reinforce_module=this selector`, reads it back and adds the
        score-function correction `((reward - baseline) * log_prob).mean()`
        to the total loss, where `reward = loss_task.detach()` (lower loss ==
        better, so this pushes `log_prob` up for actions that reduced the
        task loss) and `baseline` is this module's own EMA buffer, updated via
        `update_baseline` from the same `DMDLoss.forward` call. The selector
        owns the baseline because it is the one piece of state a REINFORCE
        estimator needs to persist across steps; nothing here has to know the
        details of the surrounding pipeline.

    At eval time (`self.training == False`) sampling is replaced by the
    deterministic threshold `sigmoid(scores) > 0.5`, `last_log_prob` is set to
    `None` (no training signal is meaningful when nothing was sampled), and
    the baseline is left untouched.
    """

    def __init__(self, baseline_decay: float = 0.9):
        super().__init__()
        if not 0.0 <= baseline_decay < 1.0:
            raise ValueError("baseline_decay must be in [0, 1)")
        self.baseline_decay = baseline_decay
        self.register_buffer("baseline", torch.zeros(()))
        self.register_buffer("baseline_initialized", torch.zeros((), dtype=torch.bool))
        self.last_log_prob = None

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # `torch.bernoulli` on CUDA validates 0 <= p <= 1 inside the kernel, so a
        # NaN probability raises a DEVICE-SIDE ASSERT: an asynchronous, unrecoverable
        # error that poisons the CUDA context for the rest of the process, surfaces at
        # whatever unrelated line next synchronizes, and makes every subsequent run in
        # the same process die at `torch.manual_seed`. That is exactly how 36 runs of
        # the feat/rich-bricks grid were lost - one diverged A7 run, 35 collateral.
        # `clamp` does NOT help: it propagates NaN rather than clipping it.
        #
        # This is the one brick that can hand a non-finite value to a kernel that
        # asserts on it (REINFORCE's score-function gradient is unbounded in variance,
        # so `scores` genuinely can blow up), so it is the one brick that checks. The
        # check costs one small device sync per forward and turns an unrecoverable
        # context poisoning into an ordinary Python exception the grid can catch,
        # record, and continue past.
        if not torch.isfinite(scores).all():
            raise RuntimeError(
                "REINFORCESelector received non-finite proposal scores "
                f"({int((~torch.isfinite(scores)).sum())} of {scores.numel()} entries). "
                "Training has diverged - the score-function estimator's variance is "
                "unbounded, so this needs gradient clipping or a lower learning rate. "
                "Raising here on purpose: torch.bernoulli would otherwise trigger a "
                "device-side assert and poison the CUDA context for every later run."
            )

        probs = torch.sigmoid(scores)

        if not self.training:
            self.last_log_prob = None
            return (probs > 0.5).float()

        probs = probs.clamp(min=_EPS, max=1 - _EPS)
        sample = torch.bernoulli(probs)
        self.last_log_prob = (sample * torch.log(probs) + (1 - sample) * torch.log1p(-probs))
        return sample

    @torch.no_grad()
    def update_baseline(self, reward: torch.Tensor) -> None:
        """
        EMA update of the variance-reduction baseline. `reward` is a scalar
        (typically the task loss, detached) - lower is better here, matching
        `last_log_prob`'s sign convention in `tools/losses.py::DMDLoss`.
        """
        reward = reward.detach().to(self.baseline.dtype)
        if not bool(self.baseline_initialized):
            self.baseline.copy_(reward)
            self.baseline_initialized.fill_(True)
        else:
            self.baseline.mul_(self.baseline_decay).add_(reward, alpha=1.0 - self.baseline_decay)
