# runs the different steps and call the GNN / TNN for motif the prediction or the graph classification
# dmd_model.py owns model ARCHITECTURE only (stages 1-5). tasks/*.py owns the
# training loop, train.py/main.py own the CLI.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import coalesce

from models.graph_embeddings import GraphEmbedder
from models.motif_proposition import MotifProposal, CandidateCells
from models.weight_assignment import GumbelSigmoidSelector
from models.motif_encodings import DeepSetsEncoder


def build_rewired_edges(edge_index: torch.Tensor, candidates: CandidateCells,
                         alpha: torch.Tensor, num_nodes: int,
                         include_original: bool = True):
    """
    Stage 5 input: turns each candidate cell into a star of edges (anchor <->
    each non-anchor member, both directions), edge weight = that cell's
    `alpha` (differentiable). Unions with the original 1-skeleton at weight
    1.0 when `include_original=True` (set False for the "rewired 1-skeleton
    only" ablation arm from CLAUDE.md §7). Duplicate edges (e.g. an original
    edge that's also proposed) are merged by max weight via `coalesce`, which
    keeps gradients flowing through whichever weight tensor achieved the max.

    Returns (rewired_edge_index [2, E'], rewired_edge_weight [E']).
    """
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


class DMDModel(nn.Module):
    """
    Orchestrates Stages 1->5 of the pipeline (CLAUDE.md §4). Stage 6 (the loss,
    including the future OSq term) lives in tools/losses.py and consumes the
    `structure` dict this model's forward returns.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 motif_hidden_dim: int, motif_out_dim: int, num_classes: int,
                 encoder_type: str = "gcn", top_k: int = 4,
                 selector_tau: float = 0.5, selector_hard: bool = True,
                 include_original_edges: bool = True):
        super().__init__()
        self.embedder = GraphEmbedder(input_dim, hidden_dim, latent_dim, encoder_type=encoder_type)  # Stage 1
        self.proposal = MotifProposal(latent_dim, top_k=top_k)                                        # Stage 2
        self.selector = GumbelSigmoidSelector(tau=selector_tau, hard=selector_hard)                    # Stage 4
        self.motif_encoder = DeepSetsEncoder(latent_dim, motif_hidden_dim, motif_out_dim)              # Stage 3
        self.rewired_conv = GCNConv(latent_dim, latent_dim)                                            # Stage 5
        self.classifier = nn.Linear(latent_dim + motif_out_dim, num_classes)
        self.include_original_edges = include_original_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        num_nodes = x.size(0)

        Z = self.embedder(x, edge_index)                       # Stage 1: [N, latent_dim]
        candidates = self.proposal(Z)                          # Stage 2: one candidate cell per node
        alpha = self.selector(candidates.scores)                # Stage 4: [num_cells] accept weight

        # Stage 3: encode each candidate cell as a bag of member node embeddings,
        # then gate its contribution by that cell's accept probability.
        motif_per_cell = self.motif_encoder(Z[candidates.node_index], candidates.cell_batch)
        motif_embeddings = alpha.unsqueeze(-1) * motif_per_cell  # [num_cells, motif_out_dim]

        # Stage 5: a genuine second message-passing pass over the rewired
        # 1-skeleton (original edges union accepted candidate cells).
        rewired_edge_index, rewired_edge_weight = build_rewired_edges(
            edge_index, candidates, alpha, num_nodes, include_original=self.include_original_edges
        )
        Z_rewired = F.relu(self.rewired_conv(Z, rewired_edge_index, edge_weight=rewired_edge_weight))

        logits = self.classifier(torch.cat([Z_rewired, motif_embeddings], dim=-1))

        structure = {
            "alpha": alpha,
            "candidates": candidates,
            "rewired_edge_index": rewired_edge_index,
            "rewired_edge_weight": rewired_edge_weight,
        }
        return logits, structure
