# runs the different steps and call the GNN / TNN for motif the prediction or the graph classification
# dmd_model.py owns model ARCHITECTURE only (stages 1-5): it assembles the
# five brick registries (models/registry.py) and orchestrates their call
# order. tasks/*.py owns the training loop, train.py/main.py own the CLI.

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.registry import (STAGE1_ENCODERS, STAGE2_PROPOSALS, STAGE3_CELL_ENCODERS,
                              STAGE4_SELECTORS, STAGE5_MP, validate_pipeline_config)


class DMDModel(nn.Module):
    """
    Orchestrates Stages 1->5 of the pipeline via the per-stage registries
    (models/registry.py): `s1..s5` select each stage's brick by
    string, `s{n}_kwargs` carries that brick's constructor kwargs beyond the
    shared dims. Stage 6 (the loss, including the OSq term) lives in
    tools/losses.py and consumes the `structure` dict this model's forward
    returns.

    `top_k` / `selector_tau` / `selector_hard` / `include_original_edges` are
    kept as flat, top-level constructor arguments (rather than folded only
    into `s2_kwargs`/`s4_kwargs`) purely for backward compatibility with every
    existing call site (train.py's CLI, checkpoints saved before this
    branch): they are used as defaults for the `topk` / `gumbel` / every
    Stage 5 brick respectively, and only when the corresponding `s*_kwargs`
    doesn't already set them explicitly.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 motif_hidden_dim: int, motif_out_dim: int, num_classes: int,
                 s1: str = "gcn", s2: str = "topk", s3: str = "deepsets",
                 s4: str = "gumbel", s5: str = "gnn_rewired",
                 top_k: int = 4, selector_tau: float = 0.5, selector_hard: bool = True,
                 include_original_edges: bool = True,
                 s1_kwargs: Optional[dict] = None, s2_kwargs: Optional[dict] = None,
                 s3_kwargs: Optional[dict] = None, s4_kwargs: Optional[dict] = None,
                 s5_kwargs: Optional[dict] = None):
        super().__init__()
        validate_pipeline_config(s1, s2, s3, s4, s5)

        s1_kwargs = dict(s1_kwargs or {})
        s2_kwargs = dict(s2_kwargs or {})
        s3_kwargs = dict(s3_kwargs or {})
        s4_kwargs = dict(s4_kwargs or {})
        s5_kwargs = dict(s5_kwargs or {})

        if s1 in ("gcn", "gin"):
            s1_kwargs.setdefault("encoder_type", s1)
        if s2 == "topk":
            s2_kwargs.setdefault("top_k", top_k)
        if s4 == "gumbel":
            s4_kwargs.setdefault("tau", selector_tau)
            s4_kwargs.setdefault("hard", selector_hard)
        s5_kwargs.setdefault("include_original_edges", include_original_edges)

        self.embedder = STAGE1_ENCODERS[s1](input_dim, hidden_dim, latent_dim, **s1_kwargs)         # Stage 1
        self.proposal = STAGE2_PROPOSALS[s2](latent_dim, **s2_kwargs)                                # Stage 2
        self.selector = STAGE4_SELECTORS[s4](**s4_kwargs)                                            # Stage 4
        self.motif_encoder = STAGE3_CELL_ENCODERS[s3](latent_dim, motif_hidden_dim,
                                                        motif_out_dim, **s3_kwargs)                   # Stage 3
        self.message_passing = STAGE5_MP[s5](latent_dim, **s5_kwargs)                                # Stage 5
        self.classifier = nn.Linear(latent_dim + motif_out_dim, num_classes)

        self.s1, self.s2, self.s3, self.s4, self.s5 = s1, s2, s3, s4, s5
        self.include_original_edges = include_original_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None,
                node_pe: Optional[torch.Tensor] = None):
        """
        batch: optional [N] graph-id per node (PyG batching convention). When
        given, (a) node representations are mean-pooled per graph before the
        classifier head, turning this into a graph-classification model (one
        logit row per graph instead of per node), and (b) Stage 2's candidate
        proposal is masked so a motif can never recruit a member from a
        different graph in the batch. None (default) preserves the original
        single-graph, per-node classification behavior.
        node_pe: optional precomputed per-node positional/structural encoding
        (e.g. a cached GPSE `pestat_GPSE`), same row order as `x`. Only
        `s1='gpse'` consumes it; every other Stage 1 brick ignores it. Pass
        `getattr(batch_or_data, "pestat_GPSE", None)` from the caller once
        `tools/gpse_cache.py::attach_gpse_cache` has populated it.
        """
        num_nodes = x.size(0)

        Z = self.embedder(x, edge_index, batch=batch, node_pe=node_pe)         # Stage 1: [N, latent_dim]
        candidates = self.proposal(Z, batch=batch, edge_index=edge_index)       # Stage 2: candidate cells
        alpha = self.selector(candidates.scores)                                # Stage 4: [num_cells] accept weight
        selector_log_prob = getattr(self.selector, "last_log_prob", None)       # REINFORCE only; else None

        # Stage 3: encode each candidate cell (a bag - or, for mini_gnn, an
        # induced subgraph - of member node embeddings), then gate its
        # contribution by that cell's accept probability.
        motif_per_cell = self.motif_encoder(Z[candidates.node_index], candidates.cell_batch,
                                             node_index=candidates.node_index, edge_index=edge_index)
        gated_per_cell = alpha.unsqueeze(-1) * motif_per_cell  # [num_cells, motif_out_dim]

        # Scattered back onto member nodes (mean over every cell a node
        # belongs to; zero for a node in none). NOT a `num_cells == num_nodes`
        # reshape: `topk`/`autoregressive` propose exactly one cell per node,
        # but a node can still be *recruited* into several other anchors'
        # cells, and `cycle_basis` breaks the one-cell-per-node count
        # entirely (CandidateCells' own docstring) - this is the one place in
        # the model that has to hold regardless of which proposal is
        # configured.
        motif_embeddings = Z.new_zeros(num_nodes, gated_per_cell.size(-1))
        if candidates.node_index.numel() > 0:
            motif_embeddings.index_add_(0, candidates.node_index, gated_per_cell[candidates.cell_batch])
            counts = Z.new_zeros(num_nodes)
            counts.index_add_(0, candidates.node_index,
                              Z.new_ones(candidates.node_index.numel()))
            motif_embeddings = motif_embeddings / counts.clamp(min=1.0).unsqueeze(-1)

        # Stage 5: a genuine second message-passing pass over the rewired
        # structure (flattened star edges, a cell complex, or a hypergraph -
        # brick-dependent; see models/message_passing.py).
        Z_rewired, rewired_edge_index, rewired_edge_weight = self.message_passing(
            Z, edge_index, candidates, alpha, num_nodes, batch=batch)

        node_repr = torch.cat([Z_rewired, motif_embeddings], dim=-1)
        if batch is not None:
            node_repr = global_mean_pool(node_repr, batch)  # [num_graphs, latent_dim + motif_out_dim]
        logits = self.classifier(node_repr)

        # `num_nodes` / `batch` / `edge_index` are carried alongside the rewired
        # structure so Stage 6 can score it without re-deriving them: the OSq
        # term is computed per graph (the batch's structure is block-diagonal,
        # and a quantity averaged over the whole batch as if it were one graph
        # would depend on the batch composition), and the curvature term needs
        # the original 1-skeleton as its paired baseline. `selector_log_prob`
        # is Stage 6's hook for the `reinforce` Stage 4 brick (see
        # tools/losses.py::DMDLoss) - always present, `None` for every other
        # selector.
        structure = {
            "alpha": alpha,
            "candidates": candidates,
            "edge_index": edge_index,
            "rewired_edge_index": rewired_edge_index,
            "rewired_edge_weight": rewired_edge_weight,
            "num_nodes": num_nodes,
            "batch": batch,
            "selector_log_prob": selector_log_prob,
        }
        return logits, structure
