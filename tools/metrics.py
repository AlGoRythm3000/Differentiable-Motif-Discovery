# pour les métriques d'évaluation (precision, recall, f1-score, etc.)

import torch


def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """
    logits: [N, C] predictions
    y:      [N] ground-truth classes
    mask:   [N] boolean mask selecting which nodes to score
    """
    pred = logits[mask].argmax(dim=-1)
    correct = pred.eq(y[mask]).sum().item()
    return correct / mask.sum().item()
