# Step 2 : stochastic sampling (using gumbel softmax ?) of motifs from the motif distribution

from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-10


@dataclass
class CandidateCells:
    """
    Flat representation of candidate higher-order cells proposed by Stage 2.

    Every anchor-based brick (`topk`, `autoregressive`) proposes one cell per
    node (`num_cells == num_nodes`). `cycle_basis` is graph-global instead -
    one cell per independent cycle - so `num_cells` can be anything from 0 up,
    and is NOT tied to the node count. Anything outside Stage 2/3/4 must not
    assume `num_cells == num_nodes` without checking which proposal produced
    this object.

    node_index:   [total_slots] node id of each (cell, member) slot, includes the anchor
    cell_batch:   [total_slots] which cell each slot belongs to, in [0, num_cells)
    anchor_index: [num_cells] node id that "owns" each cell (for `cycle_basis`,
                  an arbitrary-but-consistent member of the cycle - there is no
                  natural anchor for a cycle)
    scores:       [num_cells] differentiable proposal logit for each cell (NOT a
                  probability - Stage 4 turns this into an accept/reject weight)
    """
    node_index: torch.Tensor
    cell_batch: torch.Tensor
    anchor_index: torch.Tensor
    scores: torch.Tensor


# ---------------------------------------------------------------------------
# masking shared by every anchor-based proposal brick (topk, autoregressive):
# never recruit a member from a different graph in the batch, never pick the
# anchor itself as its own member. Extracted from what was originally
# MotifProposal.forward's inline logic so `AutoregressiveProposal` reuses the
# exact same masking RULES instead of re-deriving them.
# ---------------------------------------------------------------------------

def _max_same_graph_candidates(num_nodes: int, exclude_self: bool,
                                batch: Optional[torch.Tensor]) -> int:
    """Largest same-graph candidate count any anchor in this batch can supply."""
    if batch is None:
        return num_nodes - 1 if exclude_self else num_nodes
    graph_sizes = torch.bincount(batch)
    return int(graph_sizes.min().item()) - (1 if exclude_self else 0)


def _apply_candidate_mask(logits: torch.Tensor, num_nodes: int, exclude_self: bool,
                           batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Masks a [N, N] anchor-by-candidate logits matrix to -inf where invalid."""
    if exclude_self:
        logits = logits.masked_fill(torch.eye(num_nodes, dtype=torch.bool, device=logits.device),
                                     float("-inf"))
    if batch is not None:
        cross_graph = batch.unsqueeze(0) != batch.unsqueeze(1)
        logits = logits.masked_fill(cross_graph, float("-inf"))
    return logits


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
                batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> CandidateCells:
        """
        Z: [N, d] node embeddings
        batch: optional [N] graph-id per node (PyG batching convention). When
        several graphs are batched together, similarity is computed over all
        N nodes at once - without this mask, top-k would happily recruit
        members from an unrelated graph in the same batch. Pass it whenever
        Z may contain more than one graph.
        edge_index: unused - `topk` proposes from embeddings alone. Accepted
        so DMDModel.forward can call every Stage 2 brick with the same
        arguments (`cycle_basis` requires it).
        Returns a CandidateCells with one cell per node (num_cells == N).
        """
        num_nodes = Z.size(0)

        logits = (Z @ self.W) @ Z.t()  # [N, N]
        logits = _apply_candidate_mask(logits, num_nodes, exclude_self, batch)
        max_candidates = _max_same_graph_candidates(num_nodes, exclude_self, batch)

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


class AutoregressiveProposal(nn.Module):
    """
    Stage 2 rich brick: builds each candidate cell member-by-member instead of
    one hard top-k per anchor - member 1 conditioned on the anchor, member 2
    on {anchor, member 1}, and so on. This is the brick that makes membership
    differentiable (closes gap 1 of CLAUDE.md §7): at every step, "who gets
    picked next" is a Gumbel-softmax distribution whose SHAPE (not just the
    logit value at whichever index a hard selection lands on) is a smooth
    function of Z and this module's parameters - unlike `topk`, where nudging
    W leaves the top-k *index set* unchanged almost everywhere (a measure-zero
    exception at the swap boundary), so only the retained values' magnitude
    carries a gradient, never the membership choice itself.

    Motivation (kept here per convention): independent similarity sampling
    p(u|v) ~ exp(sim(z_v, z_u)) scores every candidate in isolation and cannot
    express long cycles - each pick is blind to the others. Conditioning
    sequentially on the running set lets the model learn "given what is
    already in this cell, who would complete a cycle", which similarity alone
    cannot express.

    Conditioning: a GRU consumes the running set's state (initialized at the
    anchor's own embedding, updated with the chosen member's embedding after
    every step) and projects it to a query that scores every remaining
    (masked) candidate via a bilinear form against Z.

    Cell size: FIXED at `max_size` members (anchor included), clamped per
    batch exactly like `top_k` is - the spec's documented stability fallback.
    A learned halting probability (variable-length cells) is left as a
    follow-on if this proves unstable; using a fixed length and logging it
    (this docstring) is the honest choice over an unlogged unstable one.

    Gradient-stop contract: `scores[cell]` accumulates, over every step, the
    Gumbel-softmax distribution's probability-weighted logit
    `sum_t (soft_t * step_logits_t).sum()`. This is fully differentiable
    w.r.t. Z and `set_gru`/`query_proj`'s parameters - the reparameterized
    Gumbel-softmax weights move continuously as those parameters move, which
    is exactly the property `topk`'s hard index selection lacks. `node_index`
    (which node the argmax of `soft_t` happened to be) is still built via a
    hard `.argmax` for downstream consumers (Stage 3/5 need concrete node
    ids) and does not itself carry gradient - the differentiable signal lives
    in `scores`, which is what Stage 4 and Stage 6 actually optimize through.
    """

    def __init__(self, latent_dim: int, max_size: int = 4, hidden_dim: Optional[int] = None,
                 tau: float = 1.0, include_anchor: bool = True):
        super().__init__()
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if tau <= 0:
            raise ValueError("tau must be > 0")
        self.max_size = max_size
        self.tau = tau
        self.include_anchor = include_anchor
        hidden_dim = hidden_dim or latent_dim
        # Conditions the next step's query on the running set's state.
        self.set_gru = nn.GRUCell(latent_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, Z: torch.Tensor, batch: Optional[torch.Tensor] = None,
                exclude_self: bool = True,
                edge_index: Optional[torch.Tensor] = None) -> CandidateCells:
        """
        edge_index: unused - membership is conditioned on embeddings only.
        Accepted for the same uniform-call-site reason as `MotifProposal`.
        """
        num_nodes = Z.size(0)
        device = Z.device
        anchor_index = torch.arange(num_nodes, device=device)

        max_candidates = _max_same_graph_candidates(num_nodes, exclude_self, batch)
        num_steps = min(self.max_size - (1 if self.include_anchor else 0), max(max_candidates, 0))
        if num_steps <= 0:
            raise ValueError(
                "max_size leaves no valid same-graph candidate: the smallest graph in this "
                "batch has too few nodes for the requested --max-size."
            )

        available = torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=device)
        if exclude_self:
            available &= ~torch.eye(num_nodes, dtype=torch.bool, device=device)
        if batch is not None:
            available &= (batch.unsqueeze(0) == batch.unsqueeze(1))

        hidden = self.query_proj.weight.new_zeros(num_nodes, self.set_gru.hidden_size)
        hidden = self.set_gru(Z, hidden)  # seed the running-set state with the anchor itself

        member_lists = [anchor_index.unsqueeze(1)] if self.include_anchor else []
        step_scores = torch.zeros(num_nodes, device=device)

        for _ in range(num_steps):
            query = self.query_proj(hidden)                      # [N, d]
            step_logits = query @ Z.t()                           # [N, N]
            step_logits = step_logits.masked_fill(~available, float("-inf"))

            if self.training:
                uniform = torch.rand_like(step_logits).clamp(min=_EPS, max=1 - _EPS)
                gumbel_noise = -torch.log(-torch.log(uniform))
                soft = F.softmax((step_logits + gumbel_noise) / self.tau, dim=-1)
            else:
                soft = F.softmax(step_logits / self.tau, dim=-1)

            chosen = torch.argmax(soft, dim=-1)                   # [N] hard pick, per anchor
            safe_logits = step_logits.masked_fill(~available, 0.0)  # avoid 0 * -inf = nan below
            step_scores = step_scores + (soft * safe_logits).sum(dim=-1)

            member_lists.append(chosen.unsqueeze(1))
            available = available & ~F.one_hot(chosen, num_classes=num_nodes).bool()
            hidden = self.set_gru(Z[chosen], hidden)

        member_index = torch.cat(member_lists, dim=1)  # [N, cell_size]
        cell_batch = anchor_index.unsqueeze(-1).expand_as(member_index).reshape(-1)
        node_index = member_index.reshape(-1)

        return CandidateCells(node_index=node_index, cell_batch=cell_batch,
                               anchor_index=anchor_index, scores=step_scores)


class CycleBasisProposal(nn.Module):
    """
    Stage 2 rich brick: the DiffLift-style contrast arm. Enumerates a cycle
    basis of each graph in the batch via `networkx.cycle_basis` - Paton's
    algorithm for undirected graphs, reused rather than hand-rolled - and uses
    each cycle as a candidate cell.

    Membership is fixed by graph topology: no gradient flows into *which*
    nodes form a cell (unlike `autoregressive`). `scores` is still a
    differentiable function of Z (the same bilinear form as `MotifProposal`,
    read at each cycle's own edges) so Stage 4 selection remains learnable -
    only the proposal step itself is non-differentiable, exactly as DiffLift's
    is.

    `num_cells != num_nodes` in general here (see `CandidateCells`'s
    docstring) - a graph contributes one cell per independent cycle, which can
    be zero (a tree/forest has none).

    This is the proposal `s5=tnn` requires (CLAUDE.md §4.5): a cycle is
    exactly a cell whose boundary is a valid 1-cycle in the 1-skeleton, which
    is what makes building a genuine cell complex on top of it well-defined -
    unlike `topk`/`autoregressive`'s arbitrary node sets.
    """

    def __init__(self, latent_dim: int, max_cycle_length: Optional[int] = None):
        super().__init__()
        self.max_cycle_length = max_cycle_length
        self.W = nn.Parameter(torch.Tensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, Z: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None,
                exclude_self: bool = True) -> CandidateCells:
        """
        edge_index: REQUIRED - cycle enumeration needs the actual graph
        topology, not just node embeddings (the one respect in which this
        brick's inputs differ from the anchor-based ones).
        """
        if edge_index is None:
            raise ValueError("cycle_basis needs edge_index (the graph to enumerate cycles of).")

        num_nodes = Z.size(0)
        device = Z.device
        num_graphs = (int(batch.max().item()) + 1 if (batch is not None and batch.numel()) else 1)

        member_lists: List[torch.Tensor] = []
        for g in range(num_graphs):
            if batch is None:
                node_ids = torch.arange(num_nodes, device=device)
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
            else:
                node_ids = torch.nonzero(batch == g, as_tuple=True)[0]
                edge_mask = batch[edge_index[0]] == g

            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(node_ids.tolist())
            sub_edges = edge_index[:, edge_mask]
            for u, v in zip(sub_edges[0].tolist(), sub_edges[1].tolist()):
                if u != v:
                    nx_graph.add_edge(u, v)

            for cycle in nx.cycle_basis(nx_graph):
                if self.max_cycle_length is not None and len(cycle) > self.max_cycle_length:
                    continue
                member_lists.append(torch.tensor(cycle, dtype=torch.long, device=device))

        if not member_lists:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            return CandidateCells(node_index=empty_long, cell_batch=empty_long,
                                   anchor_index=empty_long, scores=torch.empty(0, device=device))

        node_index = torch.cat(member_lists)
        cell_batch = torch.cat([torch.full((m.numel(),), cid, dtype=torch.long, device=device)
                                 for cid, m in enumerate(member_lists)])
        anchor_index = torch.stack([m[0] for m in member_lists])

        logits = (Z @ self.W) @ Z.t()
        scores = torch.stack([
            logits[m, torch.roll(m, shifts=-1, dims=0)].mean() for m in member_lists
        ])

        return CandidateCells(node_index=node_index, cell_batch=cell_batch,
                               anchor_index=anchor_index, scores=scores)
