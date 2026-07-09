# Step 2 : stochastic sampling (using gumbel softmax ?) of motifs from the motif distribution

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CandidateCells:
    """
    Flat representation of candidate higher-order cells proposed by Stage 2.

    One cell is proposed per anchor node: the anchor plus its top-k most
    similar nodes. Members are stored flat (like PyG's batch convention) so
    Stage 3's DeepSetsEncoder can consume them directly via its existing
    `forward(x, batch_index)` interface.

    node_index:   [total_slots] node id of each (cell, member) slot, includes the anchor
    cell_batch:   [total_slots] which cell each slot belongs to, in [0, num_cells)
    anchor_index: [num_cells] node id that "owns" each cell
    scores:       [num_cells] differentiable proposal logit for each cell (NOT a
                  probability - Stage 4 turns this into an accept/reject weight)
    """
    node_index: torch.Tensor
    cell_batch: torch.Tensor
    anchor_index: torch.Tensor
    scores: torch.Tensor


class MotifProposal(nn.Module):
    """
    Stage 2 (simple brick): similarity top-k candidate cell proposal.

    Computes a learned bilinear similarity logit for every node pair
    (Z @ W) @ Z.T, then picks each anchor's top-k neighbors by that logit to
    form one candidate cell per node. `torch.topk` index selection is a hard,
    non-differentiable choice (documented here per project convention) - only
    `scores` (aggregated from the retained logit values) carries gradient back
    to Z / W. Membership itself is NOT learned in this simple brick.

    Stage 4 (models/weight_assignment.py) is responsible for the differentiable
    accept/reject decision on top of `scores` - this module does not sample or
    threshold anything.
    """

    def __init__(self, latent_dim: int, top_k: int = 4, include_anchor: bool = True):
        super().__init__()
        self.top_k = top_k
        self.include_anchor = include_anchor
        self.W = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, Z: torch.Tensor, exclude_self: bool = True,
                batch: Optional[torch.Tensor] = None) -> CandidateCells:
        """
        Z: [N, d] node embeddings
        batch: optional [N] graph-id per node (PyG batching convention). When
        several graphs are batched together, similarity is computed over all
        N nodes at once - without this mask, top-k would happily recruit
        members from an unrelated graph in the same batch. Pass it whenever
        Z may contain more than one graph.
        Returns a CandidateCells with one cell per node (num_cells == N).
        """
        num_nodes = Z.size(0)

        logits = (Z @ self.W) @ Z.t()  # [N, N]
        if exclude_self:
            logits = logits.masked_fill(torch.eye(num_nodes, dtype=torch.bool, device=Z.device), float("-inf"))

        if batch is None:
            max_candidates = num_nodes - 1 if exclude_self else num_nodes
        else:
            cross_graph = batch.unsqueeze(0) != batch.unsqueeze(1)  # [N, N]
            logits = logits.masked_fill(cross_graph, float("-inf"))
            graph_sizes = torch.bincount(batch)
            max_candidates = int(graph_sizes.min().item()) - (1 if exclude_self else 0)

        k = min(self.top_k, max(max_candidates, 0))
        if k == 0:
            raise ValueError(
                "top_k leaves no valid same-graph candidate: the smallest graph in this "
                "batch has too few nodes for the requested --top-k."
            )

        topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)  # both [N, k]

        anchor_index = torch.arange(num_nodes, device=Z.device)
        scores = topk_values.mean(dim=-1)  # [N], differentiable w.r.t. Z / W

        if self.include_anchor:
            member_index = torch.cat([anchor_index.unsqueeze(-1), topk_indices], dim=-1)  # [N, k+1]
        else:
            member_index = topk_indices  # [N, k]

        cell_batch = anchor_index.unsqueeze(-1).expand_as(member_index).reshape(-1)
        node_index = member_index.reshape(-1)

        return CandidateCells(
            node_index=node_index,
            cell_batch=cell_batch,
            anchor_index=anchor_index,
            scores=scores,
        )
