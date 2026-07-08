# Step 1 : graph embedding model to learn motif embeddings from the motif distribution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, input_dim] node features
        edge_index: [2, E]
        Returns Z: [N, output_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
