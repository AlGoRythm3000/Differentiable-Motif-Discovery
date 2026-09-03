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
    """

    def __init__(self, sparsity_weight: float = 0.01, osq_weight: float = 0.0,
                 osq_fn: Optional[Callable[..., torch.Tensor]] = None,
                 task_loss_fn: Optional[nn.Module] = None):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.osq_weight = osq_weight
        self.osq_fn = osq_fn
        self.task_loss_fn = task_loss_fn or nn.CrossEntropyLoss()

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

        total = loss_task + self.sparsity_weight * loss_sparsity + self.osq_weight * loss_osq
        return DMDLossOutput(total=total, task=loss_task, sparsity=loss_sparsity, osq=loss_osq)
