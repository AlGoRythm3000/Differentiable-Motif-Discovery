# Batched induced-subgraph extraction for Stage 3's `mini_gnn` cell encoder.
#
# "Induced subgraph of a cell" = the cell's member nodes plus whichever edges
# of the ORIGINAL graph run between two of those members. DeepSets/Set
# Transformer only ever see the cell as a bag of node embeddings; mini_gnn is
# the one encoder that needs this - it is what lets it tell a triangle from a
# 3-path on the same three nodes, which a permutation-invariant bag cannot.
#
# Implemented as one vectorized pass over all cells at once (no Python loop
# over cells), reusing the same sorted-key/`searchsorted` set-membership trick
# already used in `tools/osq_proxies.py::_original_curvature`.

from typing import Tuple

import torch
from torch_geometric.utils import to_dense_batch


def induced_cell_edges(edge_index: torch.Tensor, node_index: torch.Tensor,
                        cell_batch: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    edge_index: [2, E] the ORIGINAL (pre-rewiring) graph.
    node_index: [total_slots] global node id of each (cell, member) slot -
                same flat layout Stage 3's `forward(x, batch_index)` receives.
    cell_batch: [total_slots] which cell each slot belongs to.

    Returns (local_edge_index [2, E'], local_edge_cell_batch [E']):
    `local_edge_index` indexes into the SAME flat slot space as `node_index` /
    `cell_batch` (0 <= index < total_slots), so it can be fed straight to a
    GNN run over `x = Z[node_index]` with `cell_batch` as the graph-batch
    vector - i.e. one disjoint-union graph, one induced subgraph per cell.
    Self-loops are excluded; both directions of every induced edge are kept
    (repo-wide convention).
    """
    total_slots = node_index.numel()
    if total_slots == 0 or edge_index.numel() == 0:
        return (torch.empty(2, 0, dtype=torch.long, device=node_index.device),
                torch.empty(0, dtype=torch.long, device=node_index.device))

    device = node_index.device
    keys_orig = edge_index[0].to(torch.long) * num_nodes + edge_index[1].to(torch.long)
    sorted_keys, _ = torch.sort(keys_orig)

    slot_ids = torch.arange(total_slots, device=device)
    # [num_cells, max_cell_size]: dense view of which slot sits where in its cell.
    dense_slots, valid = to_dense_batch(slot_ids, cell_batch)
    dense_nodes = node_index[dense_slots]  # padded rows carry a garbage-but-harmless node id

    num_cells, max_size = dense_slots.shape
    src_nodes = dense_nodes.unsqueeze(2).expand(num_cells, max_size, max_size)
    dst_nodes = dense_nodes.unsqueeze(1).expand(num_cells, max_size, max_size)
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    not_self = ~torch.eye(max_size, dtype=torch.bool, device=device).unsqueeze(0)

    pair_keys = src_nodes.to(torch.long) * num_nodes + dst_nodes.to(torch.long)
    pos = torch.searchsorted(sorted_keys, pair_keys.reshape(-1)).clamp(max=sorted_keys.numel() - 1)
    is_edge = sorted_keys[pos] == pair_keys.reshape(-1)
    is_edge = is_edge.view(num_cells, max_size, max_size) & pair_valid & not_self

    cell_idx, row, col = torch.nonzero(is_edge, as_tuple=True)
    src_slot = dense_slots[cell_idx, row]
    dst_slot = dense_slots[cell_idx, col]

    local_edge_index = torch.stack([src_slot, dst_slot], dim=0)
    local_edge_cell_batch = cell_batch[src_slot]
    return local_edge_index, local_edge_cell_batch
