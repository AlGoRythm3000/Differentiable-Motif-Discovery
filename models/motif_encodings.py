# Step 3 : Deepsets, Set Transformer, mini-GNN - cell encoders that turn a
# candidate cell (a bag of member node embeddings) into one motif embedding.
#
# Stage 3 (rich bricks, feat/rich-bricks): DeepSets is invariant to the *bag*
# of nodes - it literally cannot distinguish a triangle from a 3-path of the
# same size (see test_triangle_vs_path_only_mini_gnn_tells_them_apart below).
# set_transformer and mini_gnn exist to fix that. All three keep the interface
# `forward(x, batch_index, node_index=None, edge_index=None) -> [num_cells, d]`
# - node_index/edge_index are only meaningful for mini_gnn (it needs the
# original graph to know which members are actually connected); DeepSets and
# Set Transformer accept and ignore them so the call site (DMDModel.forward)
# can pass the same four arguments to whichever brick is configured.

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, aggr
from torch_geometric.utils import to_dense_batch

from tools.induced_subgraph import induced_cell_edges


class DeepSetsEncoder(nn.Module):
    """
    Permutation-invariant encoder based on the DeepSets architecture (from Zaheer et al., 2017).
    Computes a continuous latent representation for stochastically discovered structural motifs.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 aggregator_type: str = 'sum', dropout: float = 0.0):
        super().__init__()

        # 1. Psi network (individual transfo of each motif's element)
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh() # stabilises the representation before aggregation (empirically found to help with training stability when using 'sum' aggregator)
        )

        # LayerNorm before aggregation: keeps the per-node representation on a
        # comparable scale regardless of how much cell sizes vary, so a big
        # cell's sum doesn't dominate purely from having more terms.
        self.pre_aggregation_norm = nn.LayerNorm(hidden_dim)

        # 2. PyG aggregator (which is permutation-invariant)
        if aggregator_type == 'max':
            self.aggregator = aggr.MaxAggregation()
        elif aggregator_type == 'mean':
            self.aggregator = aggr.MeanAggregation()
        elif aggregator_type == 'sum':
            self.aggregator = aggr.SumAggregation()
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")

        # 3. Rho network (transforms the global representation of the motif)
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor,
                node_index: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward propagation of deepsets module.

        Args:
            x (torch.Tensor): nodes features [Total_Nodes_In_All_Motifs, input_dim]
            batch_index (torch.Tensor) : 1D tensor associating each node with the index of its motif, shape [Total_Nodes_In_All_Motifs]
                                  Example: [0, 0, 0, 1, 1] means the first 3 nodes form motif 0, the next 2 form motif 1.
            node_index, edge_index: unused - DeepSets only ever sees the bag of
                                  member embeddings, never the graph among them.
                                  Accepted so this stays a drop-in swap for
                                  mini_gnn under Stage 3's shared call site.

        Returns:
            torch.Tensor: Embedding of each motif, shape [Num_Motifs, output_dim]
        """
        # stp 1 : local projection of each node in the motif
        h = self.psi(x)
        h = self.pre_aggregation_norm(h)

        # stp 2 : aggreggation
        # the batch index tensor allows the aggregator to know which nodes belong to which motif, ensuring the permutation invariance property
        h_agg = self.aggregator(h, batch_index)

        # stp 3 : global projection of the aggregated motif representation
        y = self.phi(h_agg)

        return y


class _MAB(nn.Module):
    """
    Multihead Attention Block (Lee et al. 2019, "Set Transformer"):
    MAB(Q, K) = LayerNorm(Q' + Multihead(Q', K', K')), then a position-wise FF
    with its own residual + LayerNorm. `key_padding_mask` (True = ignore) lets
    K carry padding rows from `to_dense_batch` without corrupting the softmax.
    """

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int):
        super().__init__()
        self.proj_q = nn.Linear(dim_q, dim_v)
        self.proj_k = nn.Linear(dim_k, dim_v)
        self.attn = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)
        self.ln0 = nn.LayerNorm(dim_v)
        self.ff = nn.Sequential(nn.Linear(dim_v, dim_v), nn.ReLU(), nn.Linear(dim_v, dim_v))
        self.ln1 = nn.LayerNorm(dim_v)

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Qp = self.proj_q(Q)
        Kp = self.proj_k(K)
        # A fully-padded key row (an empty cell slipping through) would give
        # every query an all -inf softmax row - guard against it explicitly
        # rather than let it silently produce NaNs.
        if key_padding_mask is not None:
            fully_padded = key_padding_mask.all(dim=-1)
            if bool(fully_padded.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_padded] = False
        attn_out, _ = self.attn(Qp, Kp, Kp, key_padding_mask=key_padding_mask)
        H = self.ln0(Qp + attn_out)
        return self.ln1(H + self.ff(H))


class SetTransformerEncoder(nn.Module):
    """
    Stage 3 rich brick: Lee et al.'s Set Transformer (ISAB blocks + PMA
    pooling), hand-implemented against `nn.MultiheadAttention` +
    `torch_geometric.utils.to_dense_batch` rather than PyG's
    `aggr.SetTransformerAggregation` - the latter's constructor signature has
    moved across PyG releases, and this repo pins `pyg` generically
    (`environment.yml`), so a hand-rolled version keeps this brick correct
    regardless of which minor version resolves. The block structure is
    identical to the paper either way (this is a "wrap the standard formula",
    not "invent a new attention mechanism").

    Permutation invariant by construction (attention + softmax over the set,
    pooled by a fixed learned seed), but - unlike DeepSets - it models
    pairwise interactions between cell members through the attention weights.
    Uses `num_inducing` inducing points so cost is O(cell_size * num_inducing)
    instead of O(cell_size^2).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_inducing: int = 32, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, hidden_dim) * (hidden_dim ** -0.5))
        self.mab_induce = _MAB(hidden_dim, hidden_dim, hidden_dim, num_heads)
        self.mab_isab = _MAB(hidden_dim, hidden_dim, hidden_dim, num_heads)
        self.seed = nn.Parameter(torch.randn(1, hidden_dim) * (hidden_dim ** -0.5))
        self.rff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.mab_pool = _MAB(hidden_dim, hidden_dim, hidden_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor,
                node_index: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_cells = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_cells == 0:
            return x.new_zeros(0, self.out_proj.out_features)

        h = self.input_proj(x)
        dense_x, valid = to_dense_batch(h, batch_index, batch_size=num_cells)  # [B, S, d], [B, S]
        key_padding_mask = ~valid

        B = dense_x.size(0)
        induced = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        H = self.mab_induce(induced, dense_x, key_padding_mask=key_padding_mask)  # [B, m, d], no padding
        isab_out = self.mab_isab(dense_x, H)  # [B, S, d] - padded rows are garbage, discarded below

        seed = self.seed.unsqueeze(0).expand(B, -1, -1)  # [B, 1, d]
        pooled = self.mab_pool(seed, self.rff(isab_out), key_padding_mask=key_padding_mask)
        return self.out_proj(pooled.squeeze(1))


class MiniGNNEncoder(nn.Module):
    """
    Stage 3 rich brick: runs a small GCN over each cell's *induced* subgraph
    (cell members + whichever original-graph edges run between them), then
    mean-pools to one embedding per cell.

    This is the encoder that genuinely sees internal topology: a triangle and
    a 3-path on the same three nodes induce different edge sets, so their
    embeddings differ - DeepSets and Set Transformer both only ever see the
    unordered bag of member features and cannot make that distinction (Set
    Transformer sees pairwise *feature* interactions, not graph structure).

    Requires `node_index` (global node ids per slot) and `edge_index` (the
    ORIGINAL, pre-rewiring graph) to compute induced edges via
    `tools/induced_subgraph.py::induced_cell_edges` - if either is missing
    (e.g. a proposal brick that hasn't set them) this brick cannot function,
    and raises rather than silently falling back to a bag-of-nodes read.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.readout = aggr.MeanAggregation()
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor,
                node_index: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if node_index is None or edge_index is None:
            raise ValueError(
                "mini_gnn needs node_index and edge_index (the original graph) to build each "
                "cell's induced subgraph - got None. Only DMDModel's Stage 3 call site is "
                "expected to supply them."
            )
        num_cells = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        if num_cells == 0:
            return x.new_zeros(0, self.out_proj.out_features)

        num_nodes = int(node_index.max().item()) + 1 if node_index.numel() else 0
        local_edge_index, _ = induced_cell_edges(edge_index, node_index, batch_index, num_nodes)

        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, local_edge_index))

        pooled = self.readout(h, batch_index, dim_size=num_cells)
        return self.out_proj(pooled)

    # TODO: pistes d'améliorations futures (à implémenter en phase de recherche) :
    # 1. remplacer deepsets par un 'SetTransformer' (Self-Attention) pour capturer
    #    les relations internes du motif non-structural au lieu d'un simple "sac de nœuds".
    #    -> fait (SetTransformerEncoder), et un mini-GNN qui voit la topologie interne.
    # 2. ajouter une normalisation (layernorm ?) entre psi et l'agrégateur si la taille
    #    des motifs varie drastiquement d'un nœud à l'autre (évite l'explosion des valeurs de la somme).
    #    -> fait (pre_aggregation_norm).
