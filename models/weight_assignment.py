# Step 4 : weight assignment to motifs to learn a graph embedding from the motif distribution

import torch
import torch.nn as nn


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
        gradients reach `phi(z_C)` (== `scores` here) as required by CLAUDE §6.

    At eval time (`self.training == False`) sampling noise is dropped and the
    module is deterministic: `alpha_soft = sigmoid(scores / tau)`.
    """

    def __init__(self, tau: float = 0.5, hard: bool = True):
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.tau = tau
        self.hard = hard

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
