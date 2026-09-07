# pour les fonctions de perte (cross-entropy, et celle parsity aware, etc.)

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class DMDLossOutput:
    total: torch.Tensor
    task: torch.Tensor
    sparsity: torch.Tensor
    osq: torch.Tensor
    reinforce: torch.Tensor


class DMDLoss(nn.Module):
    """
    Stage 6: task loss + sparsity loss on the differentiable cell-acceptance
    weights `alpha`, plus the oversquashing term.

        L = L_task + mu * L_sparsity + gamma * L_osq

    The OSq term stays a no-op by default (`osq_weight=0.0`, `osq_fn=None`) so
    the scaffold never depends on it. Turning it on is purely a matter of
    constructor arguments - `forward`'s call site never changes:

        from tools.osq_proxies import get_proxy
        DMDLoss(sparsity_weight=mu, osq_weight=gamma, osq_fn=get_proxy("r_bar"))

    `gamma = 0` short-circuits `osq_fn` entirely, so it reproduces the
    proxy-free objective bit for bit whatever proxy is configured.

    Never set `sparsity_weight = 0` together with a positive `osq_weight`: every
    OSq proxy is minimized by the complete graph, so the sparsity term is what
    makes the objective non-degenerate, not decoration.

    `reinforce_module` (feat/rich-bricks, Stage 4's `reinforce` brick only):
    when Stage 4 is a `models.weight_assignment.REINFORCESelector`, pass that
    same module instance here. Its sampled action carries no gradient to
    `scores` on its own (see the class docstring) - the score-function
    correction `((reward - baseline) * log_prob).mean()` is what supplies one,
    added to `total` alongside the other terms. `reward = loss_task.detach()`
    (lower task loss is a better action) and `baseline` is the selector's own
    EMA buffer, updated here via `update_baseline` every step this is
    constructed with a module. Mirrors how `osq_fn` was added without ever
    changing what's passed to `forward`: `None` (default) is a total no-op.
    """

    def __init__(self, sparsity_weight: float = 0.01, osq_weight: float = 0.0,
                 osq_fn: Optional[Callable[..., torch.Tensor]] = None,
                 task_loss_fn: Optional[nn.Module] = None,
                 reinforce_module: Optional[nn.Module] = None):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.osq_weight = osq_weight
        self.osq_fn = osq_fn
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()
        self.reinforce_module = reinforce_module

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                structure: dict) -> DMDLossOutput:
        """
        logits: [N, C] predictions
        target: [N] ground-truth classes
        mask:   [N] boolean mask selecting which nodes to score (train/val/test)
        structure: dict returned by DMDModel.forward's second output (exposes
                   `alpha` plus whatever else `osq_fn` may need later, e.g.
                   rewired edges / candidate cells) - this is the stable call
                   site: adding the OSq term in Phase 1 only requires passing
                   `osq_weight` / `osq_fn` at construction time, never changing
                   what's passed to `forward`.
        """
        alpha = structure["alpha"]
        loss_task = self.task_loss_fn(logits[mask], target[mask])

        # L1-style density penalty: alpha already lives in [0, 1], so its mean
        # IS the normalized L1 norm - discourages accepting too many cells.
        loss_sparsity = alpha.mean()

        if self.osq_weight > 0 and self.osq_fn is not None:
            loss_osq = self.osq_fn(structure)
        else:
            loss_osq = torch.zeros((), device=logits.device)

        loss_reinforce = torch.zeros((), device=logits.device)
        log_prob = structure.get("selector_log_prob")
        if self.reinforce_module is not None and log_prob is not None:
            reward = loss_task.detach()
            baseline = self.reinforce_module.baseline.to(reward.device)
            loss_reinforce = ((reward - baseline) * log_prob).mean()
            self.reinforce_module.update_baseline(reward)

        total = (loss_task + self.sparsity_weight * loss_sparsity
                 + self.osq_weight * loss_osq + loss_reinforce)
        return DMDLossOutput(total=total, task=loss_task, sparsity=loss_sparsity,
                              osq=loss_osq, reinforce=loss_reinforce)
