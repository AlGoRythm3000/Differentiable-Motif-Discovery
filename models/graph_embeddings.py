# Step 1 : graph embedding model to learn motif embeddings from the motif distribution

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv

from tools.gpse_cache import GPSE_ENCODING_DIM, compute_explicit_pe, compute_gpse_live, load_pretrained_gpse


class GraphEmbedder(nn.Module):
    """
    Stage 1 node embedder (simple brick): a stack of GCN or GIN layers producing
    the node latent matrix Z that all later stages consume.

    `encoder_type` is a config knob so this can be swapped for a rich brick
    (GPSE, GraphGPS) later without changing any call site.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 encoder_type: str = "gcn", num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.encoder_type = encoder_type
        self.dropout = dropout

        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(self._make_conv(dims[i], dims[i + 1], encoder_type))

    @staticmethod
    def _make_conv(in_dim: int, out_dim: int, encoder_type: str):
        if encoder_type == "gcn":
            return GCNConv(in_dim, out_dim)
        elif encoder_type == "gin":
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
            return GINConv(mlp)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                node_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, input_dim] node features
        edge_index: [2, E]
        batch, node_pe: unused - GCN/GIN don't need a graph-id vector or a
        precomputed positional encoding. Accepted so every Stage 1 registry
        entry shares one call site (`gpse` needs both).
        Returns Z: [N, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PSEExplicitEncoder(nn.Module):
    """
    Stage 1 rich brick `pse_explicit`: LapPE + RWSE computed with PyG
    transforms (`tools/gpse_cache.py::compute_explicit_pe`), concatenated with
    the raw node features and fed through a small GCN/GIN backbone - the
    mandatory fallback for `gpse` when its checkpoint is unavailable, also
    directly selectable on its own.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 backbone_encoder_type: str = "gcn", num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.backbone = GraphEmbedder(input_dim + GPSE_ENCODING_DIM, hidden_dim, output_dim,
                                       encoder_type=backbone_encoder_type,
                                       num_layers=num_layers, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                node_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        if node_pe is None:
            node_pe = self._compute_pe(x, edge_index, batch)
        return self.backbone(torch.cat([x, node_pe.to(x.dtype)], dim=-1), edge_index, batch=batch)

    @staticmethod
    def _compute_pe(x: torch.Tensor, edge_index: torch.Tensor,
                     batch: Optional[torch.Tensor]) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device
        if batch is None:
            return compute_explicit_pe(edge_index, num_nodes).to(device)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        pe_blocks = []
        for g in range(num_graphs):
            mask = batch == g
            node_ids = torch.nonzero(mask, as_tuple=True)[0]
            remap = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            remap[node_ids] = torch.arange(node_ids.numel(), device=device)
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            local_edges = remap[edge_index[:, edge_mask]]
            pe_blocks.append(compute_explicit_pe(local_edges.cpu(), node_ids.numel()).to(device))
        return torch.cat(pe_blocks, dim=0)


class GPSEEncoder(nn.Module):
    """
    Stage 1 rich brick `gpse`: frozen pretrained structural encoder (Canturk
    et al. 2024). Answers a chicken-and-egg problem: Stage 2's candidate
    proposal is otherwise fed by a GNN that already suffers the oversquashing
    this whole project tries to fix. GPSE is
    pretrained (MolPCBA) purely on graph STRUCTURE, from random input
    features, so it is not damaged by our graph's bottlenecks.

    Gradient-stop contract: `requires_grad=False` on every GPSE parameter and
    `eval()` mode, set once at construction (see `tools/gpse_cache.py::
    load_pretrained_gpse`) - gradients from our task never flow into it. This
    is a deliberate, documented limitation (a real one to state in the paper,
    not to hide): GPSE is preprocessing, it does not co-adapt with the
    lifting.

    Two input paths, matching the two ways a caller can have GPSE encodings
    ready:
      - `node_pe` given (the fast path): a precomputed `pestat_GPSE` tensor,
        already batched in the same row order as `x` - this is what
        `tools/gpse_cache.py::attach_gpse_cache` produces once per dataset
        and every grid run then reuses (it would otherwise be recomputed
        hundreds of times across the grid).
      - `node_pe=None` (the slow/live path): runs the frozen model on the fly
        via `tools/gpse_cache.py::compute_gpse_live` - used by isolated unit
        tests and ad-hoc single-graph calls (e.g. `infer.py`) that never went
        through the dataset-level cache.

    Fallback (mandatory, not optional): if the pretrained checkpoint
    cannot be fetched (a Kaggle session without internet, e.g.), construction
    falls back to `PSEExplicitEncoder` transparently and every `forward` call
    is delegated to it. `self.actual_encoder` ("gpse" or "pse_explicit")
    records which one is really running - store it as `s1_encoder_actual` in
    the results row. A grid that silently degrades without recording this is
    worse than a failed grid.

    `gpse_mode`:
      - "concat":  GPSE's encoding is concatenated with the raw node features
                   and fed through a small GCN/GIN backbone (same shape as
                   `PSEExplicitEncoder`).
      - "replace": GPSE's encoding alone (linearly projected) becomes Z - no
                   GCN/GIN backbone at all.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 gpse_mode: str = "concat", pretrained_name: str = "molpcba",
                 weights_root: str = "GPSE_pretrained",
                 backbone_encoder_type: str = "gcn", num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if gpse_mode not in ("concat", "replace"):
            raise ValueError(f"Unknown gpse_mode: {gpse_mode!r}. Choices: concat, replace.")
        self.gpse_mode = gpse_mode

        self.gpse_model = load_pretrained_gpse(pretrained_name, weights_root)
        if self.gpse_model is None:
            self.actual_encoder = "pse_explicit"
            self._fallback = PSEExplicitEncoder(input_dim, hidden_dim, output_dim,
                                                 backbone_encoder_type=backbone_encoder_type,
                                                 num_layers=num_layers, dropout=dropout)
            self.backbone = None
            self.out_proj = None
            return

        self.actual_encoder = "gpse"
        self._fallback = None
        # The pretrained checkpoint's actual output width depends on its
        # saved `repr_type`/`dim_inner` (e.g. `repr_type='no_post_mp'` returns
        # the 512-wide internal representation, not the 51-wide PSE
        # reconstruction head `GPSE.__init__`'s default `dim_out` would
        # suggest) - probed once here rather than assumed, so this brick
        # keeps working across whichever pretrained variant actually loads.
        self.gpse_dim = self._probe_output_dim()
        if gpse_mode == "replace":
            self.backbone = None
            self.out_proj = nn.Linear(self.gpse_dim, output_dim)
        else:
            self.backbone = GraphEmbedder(input_dim + self.gpse_dim, hidden_dim, output_dim,
                                           encoder_type=backbone_encoder_type,
                                           num_layers=num_layers, dropout=dropout)
            self.out_proj = None

    def _probe_output_dim(self) -> int:
        probe_edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long)
        with torch.no_grad():
            pe = compute_gpse_live(self.gpse_model, probe_edges, 3, None)
        return int(pe.size(-1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                node_pe: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._fallback is not None:
            return self._fallback(x, edge_index, batch=batch, node_pe=node_pe)

        if node_pe is None:
            node_pe = compute_gpse_live(self.gpse_model, edge_index, x.size(0), batch)
        pe = node_pe.to(x.dtype)

        if self.gpse_mode == "replace":
            return self.out_proj(pe)
        return self.backbone(torch.cat([x, pe], dim=-1), edge_index, batch=batch)
