# Step 5 : message passing over the rewired structure.
#
# Stage 5 simple brick (`gnn_rewired`, moved here from dmd_model.py so it has
# the same registry-brick shape as every other stage): flattens accepted
# cells into weighted star edges and runs a second GCN pass over the union
# with the original 1-skeleton. There is no genuine complex at
# message-passing time - this is gap 3 of CLAUDE.md §7.
#
# Stage 5 rich bricks (feat/rich-bricks, §4.5):
#   `tnn`            - a real cell complex (nodes / original edges / accepted
#                       cycles as 2-cells), message-passed with TopoModelX's
#                       CWNLayer (Bodnar et al. 2021). Only valid together
#                       with `s2=cycle_basis` - enforced by
#                       models/registry.py::validate_pipeline_config, not here.
#   `hypergraph_tnn` - accepted cells (from ANY proposal) become hyperedges,
#                       message-passed with TopoModelX's UniGCNLayer
#                       (Huang & Yang 2021). The valid-for-any-proposal
#                       fallback the spec calls for.
#
# Every brick keeps the contract
#     forward(Z, edge_index, candidates, alpha, num_nodes, batch=None)
#         -> (Z_rewired, rewired_edge_index, rewired_edge_weight)
# so DMDModel.forward's call site never changes across bricks, and
# `rewired_edge_index`/`rewired_edge_weight` (the rank-0 view) are always
# populated the same way `gnn_rewired` computes them - `tools/osq_proxies.py`
# / `tools/osq_metrics.py` always score that 1-skeleton restriction, never a
# brick-specific structure (§4.5's last paragraph).

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce

from models.motif_proposition import CandidateCells


def build_rewired_edges(edge_index: torch.Tensor, candidates: CandidateCells,
                         alpha: torch.Tensor, num_nodes: int,
                         include_original: bool = True):
    """
    Turns each candidate cell into a star of edges (anchor <-> each
    non-anchor member, both directions), edge weight = that cell's `alpha`
    (differentiable). Unions with the original 1-skeleton at weight 1.0 when
    `include_original=True` (set False for the "rewired 1-skeleton only"
    ablation arm). Duplicate edges (e.g. an original edge that's also
    proposed) are merged by max weight via `coalesce`, which keeps gradients
    flowing through whichever weight tensor achieved the max.

    `cycle_basis` cells have no single "anchor" node in the usual sense - the
    first member (`candidates.anchor_index`) is used as the star's center,
    which is an arbitrary-but-consistent choice for the purposes of this
    rank-0 view (every rich Stage 5 brick still needs SOME 1-skeleton
    restriction for the OSq proxies to score - see the module docstring).

    Returns (rewired_edge_index [2, E'], rewired_edge_weight [E']).
    """
    if candidates.cell_batch.numel() == 0:
        if include_original:
            orig_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            return coalesce(edge_index, orig_weight, num_nodes=num_nodes, reduce="max")
        empty = torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
        return empty, torch.empty(0, device=edge_index.device)

    anchor_per_slot = candidates.anchor_index[candidates.cell_batch]
    weight_per_slot = alpha[candidates.cell_batch]

    non_self = candidates.node_index != anchor_per_slot
    src = anchor_per_slot[non_self]
    dst = candidates.node_index[non_self]
    w = weight_per_slot[non_self]

    cand_edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    cand_edge_weight = torch.cat([w, w])

    if include_original:
        orig_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=cand_edge_weight.dtype)
        all_edge_index = torch.cat([edge_index, cand_edge_index], dim=1)
        all_edge_weight = torch.cat([orig_weight, cand_edge_weight])
    else:
        all_edge_index = cand_edge_index
        all_edge_weight = cand_edge_weight

    rewired_edge_index, rewired_edge_weight = coalesce(
        all_edge_index, all_edge_weight, num_nodes=num_nodes, reduce="max"
    )
    return rewired_edge_index, rewired_edge_weight


class GNNRewiredMP(nn.Module):
    """Stage 5 simple brick: see module docstring."""

    def __init__(self, latent_dim: int, include_original_edges: bool = True):
        super().__init__()
        self.conv = GCNConv(latent_dim, latent_dim)
        self.include_original_edges = include_original_edges

    def forward(self, Z: torch.Tensor, edge_index: torch.Tensor, candidates: CandidateCells,
                alpha: torch.Tensor, num_nodes: int, batch: Optional[torch.Tensor] = None):
        rewired_edge_index, rewired_edge_weight = build_rewired_edges(
            edge_index, candidates, alpha, num_nodes, include_original=self.include_original_edges
        )
        Z_rewired = F.relu(self.conv(Z, rewired_edge_index, edge_weight=rewired_edge_weight))
        return Z_rewired, rewired_edge_index, rewired_edge_weight


def _build_cell_complex_matrices(edge_index: torch.Tensor, candidates: CandidateCells,
                                  num_nodes: int, device: torch.device):
    """
    Builds the sparse incidence/adjacency matrices `TNNMessagePassing` needs:
    0-cells = nodes, 1-cells = the ORIGINAL graph's undirected edges, 2-cells
    = accepted cycles (`candidates`, assumed stored in cyclic member order per
    cell - true for `models.motif_proposition.CycleBasisProposal`, the only
    valid Stage 2 pairing per the validity constraint).

    Computed directly in torch rather than through `toponetx.classes.
    CellComplex`: toponetx is a real, listed dependency, but its import pulls
    in `pyarrow`, which was observed here to fail to load under a sufficiently
    long install path (`DLL load failed` - a Windows path-length artifact of
    this development environment, not a toponetx defect). The matrices built
    below are the same combinatorial objects `CellComplex.incidence_matrix` /
    `adjacency_matrix` would produce for this complex; message passing itself
    still goes through TopoModelX's real `CWNLayer`, not a hand-rolled
    convolution - only the plumbing that feeds it is inlined.

    Returns (incidence_1_t [n_1cells, n_nodes], incidence_2 [n_1cells, n_2cells],
    adjacency_0 [n_1cells, n_1cells]) as coalesced sparse COO tensors.
    """
    src, dst = edge_index[0], edge_index[1]
    undirected = src < dst
    edge_u, edge_v = src[undirected], dst[undirected]
    num_1cells = edge_u.numel()
    edge_key = edge_u.to(torch.long) * num_nodes + edge_v.to(torch.long)

    rows = torch.arange(num_1cells, device=device).repeat_interleave(2)
    cols = torch.stack([edge_u, edge_v], dim=1).reshape(-1)
    incidence_1_t = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), torch.ones(rows.numel(), device=device),
        size=(num_1cells, num_nodes)).coalesce()

    num_2cells = int(candidates.cell_batch.max().item()) + 1 if candidates.cell_batch.numel() else 0
    if num_1cells == 0 or num_2cells == 0:
        empty_idx = torch.empty(2, 0, dtype=torch.long, device=device)
        empty_val = torch.empty(0, device=device)
        incidence_2 = torch.sparse_coo_tensor(empty_idx, empty_val, size=(num_1cells, num_2cells)).coalesce()
        adjacency_0 = torch.sparse_coo_tensor(empty_idx, empty_val, size=(num_1cells, num_1cells)).coalesce()
        return incidence_1_t, incidence_2, adjacency_0

    member, cell = candidates.node_index, candidates.cell_batch
    next_member = torch.roll(member, shifts=-1, dims=0)
    next_cell = torch.roll(cell, shifts=-1, dims=0)
    within_cell = cell == next_cell  # drops the one pair that spuriously wraps across cells
    b_u, b_v, b_cell = member[within_cell], next_member[within_cell], cell[within_cell]

    cell_sizes = torch.bincount(cell, minlength=num_2cells)
    last_slot = cell_sizes.cumsum(0) - 1
    first_slot = last_slot - cell_sizes + 1
    wrap_u, wrap_v = member[last_slot], member[first_slot]
    wrap_cell = torch.arange(num_2cells, device=device)

    boundary_u = torch.cat([b_u, wrap_u])
    boundary_v = torch.cat([b_v, wrap_v])
    boundary_cell = torch.cat([b_cell, wrap_cell])

    lo = torch.minimum(boundary_u, boundary_v).to(torch.long)
    hi = torch.maximum(boundary_u, boundary_v).to(torch.long)
    boundary_key = lo * num_nodes + hi

    sorted_keys, order = torch.sort(edge_key)
    pos = torch.searchsorted(sorted_keys, boundary_key).clamp(max=max(sorted_keys.numel() - 1, 0))
    hit = (sorted_keys.numel() > 0) & (sorted_keys[pos.clamp(max=max(sorted_keys.numel() - 1, 0))] == boundary_key)
    boundary_1cell = order[pos[hit]]

    incidence_2 = torch.sparse_coo_tensor(
        torch.stack([boundary_1cell, boundary_cell[hit]]),
        torch.ones(int(hit.sum().item()), device=device),
        size=(num_1cells, num_2cells)).coalesce()

    adjacency_0 = torch.sparse.mm(incidence_2, incidence_2.t()).coalesce()
    idx, val = adjacency_0.indices(), adjacency_0.values()
    keep = idx[0] != idx[1]
    adjacency_0 = torch.sparse_coo_tensor(idx[:, keep], val[keep], size=adjacency_0.shape).coalesce()

    return incidence_1_t, incidence_2, adjacency_0


class TNNMessagePassing(nn.Module):
    """Stage 5 rich brick `tnn`: see module docstring and
    `_build_cell_complex_matrices` for the construction it relies on."""

    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None,
                 include_original_edges: bool = True):
        super().__init__()
        from topomodelx.nn.cell.cwn_layer import CWNLayer

        hidden_dim = hidden_dim or latent_dim
        self.layer = CWNLayer(in_channels_0=latent_dim, in_channels_1=latent_dim,
                               in_channels_2=latent_dim, out_channels=hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        self.include_original_edges = include_original_edges

    def forward(self, Z: torch.Tensor, edge_index: torch.Tensor, candidates: CandidateCells,
                alpha: torch.Tensor, num_nodes: int, batch: Optional[torch.Tensor] = None):
        rewired_edge_index, rewired_edge_weight = build_rewired_edges(
            edge_index, candidates, alpha, num_nodes, include_original=self.include_original_edges
        )

        device = Z.device
        incidence_1_t, incidence_2, adjacency_0 = _build_cell_complex_matrices(
            edge_index, candidates, num_nodes, device)
        num_1cells, num_2cells = incidence_1_t.shape[0], incidence_2.shape[1]

        if num_1cells == 0 or num_2cells == 0:
            # No edges, or no accepted cycle anywhere in this batch/graph:
            # nothing for a cell complex to do - identity on Z rather than
            # calling CWNLayer on an empty rank.
            return Z, rewired_edge_index, rewired_edge_weight

        x_1 = torch.sparse.mm(incidence_1_t, Z) / 2.0  # seed edge features: mean of endpoints
        cell_sizes = torch.sparse.mm(incidence_2.t(), torch.ones(num_1cells, 1, device=device)).clamp(min=1.0)
        x_2 = torch.sparse.mm(incidence_2.t(), x_1) / cell_sizes
        x_2 = x_2 * alpha.unsqueeze(-1)  # gate each 2-cell by Stage 4's differentiable accept weight

        x_1_updated = self.layer(Z, x_1, x_2, adjacency_0, incidence_2, incidence_1_t)

        node_deg = torch.sparse.mm(incidence_1_t.t(), torch.ones(num_1cells, 1, device=device)).clamp(min=1.0)
        x_0_out = torch.sparse.mm(incidence_1_t.t(), x_1_updated) / node_deg
        Z_rewired = F.relu(self.out_proj(x_0_out))
        return Z_rewired, rewired_edge_index, rewired_edge_weight


class HypergraphTNNMessagePassing(nn.Module):
    """Stage 5 rich brick `hypergraph_tnn`: see module docstring."""

    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None,
                 include_original_edges: bool = True):
        super().__init__()
        from topomodelx.nn.hypergraph.unigcn_layer import UniGCNLayer

        hidden_dim = hidden_dim or latent_dim
        self.layer = UniGCNLayer(latent_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        self.include_original_edges = include_original_edges

    def forward(self, Z: torch.Tensor, edge_index: torch.Tensor, candidates: CandidateCells,
                alpha: torch.Tensor, num_nodes: int, batch: Optional[torch.Tensor] = None):
        rewired_edge_index, rewired_edge_weight = build_rewired_edges(
            edge_index, candidates, alpha, num_nodes, include_original=self.include_original_edges
        )

        num_cells = int(candidates.cell_batch.max().item()) + 1 if candidates.cell_batch.numel() else 0
        if num_cells == 0:
            return Z, rewired_edge_index, rewired_edge_weight

        weight_per_slot = alpha[candidates.cell_batch]
        incidence_1 = torch.sparse_coo_tensor(
            torch.stack([candidates.node_index, candidates.cell_batch]),
            weight_per_slot, size=(num_nodes, num_cells)).coalesce()

        x_0_out, _ = self.layer(Z, incidence_1)
        Z_rewired = F.relu(self.out_proj(x_0_out))
        return Z_rewired, rewired_edge_index, rewired_edge_weight
