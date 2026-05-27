# Step 3 : Deepsets, etc. to learn a graph embedding from the motif distribution

import torch
import torch.nn as nn
from torch_geometric.nn import aggr

class DeepSetsEncoder(nn.Module):
    """
    Permutation-invariant encoder based on the DeepSets architecture (from Zaheer et al., 2017).
    Computes a continuous latent representation for stochastically discovered structural motifs.
    
    Mathematical formulation:
        $$h_S = \rho ( \bigoplus_{u \in \mathcal{S}} \psi(z_u) )$$
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

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """
        forward propagation of deepsets module.
        
        Args:
            x (torch.Tensor): nodes features [Total_Nodes_In_All_Motifs, input_dim]
            batch_index (torch.Tensor) : 1D tensor associating each node with the index of its motif, shape [Total_Nodes_In_All_Motifs]
                                  Example: [0, 0, 0, 1, 1] means the first 3 nodes form motif 0, the next 2 form motif 1.
                                  
        Returns:
            torch.Tensor: Embedding of each motif, shape [Num_Motifs, output_dim]
        """
        # stp 1 : local projection of each node in the motif
        h = self.psi(x)
        
        # stp 2 : aggreggation
        # the batch index tensor allows the aggregator to know which nodes belong to which motif, ensuring the permutation invariance property
        h_agg = self.aggregator(h, batch_index)
        
        # stp 3 : global projection of the aggregated motif representation
        y = self.phi(h_agg)
        
        return y

    # TODO: pistes d'améliorations futures (à implémenter en phase de recherche) :
    # 1. remplacer deepsets par un 'SetTransformer' (Self-Attention) pour capturer 
    #    les relations internes du motif non-structural au lieu d'un simple "sac de nœuds".
    # 2. ajouter une normalisation (layernorm ?) entre psi et l'agrégateur si la taille 
    #    des motifs varie drastiquement d'un nœud à l'autre (évite l'explosion des valeurs de la somme).