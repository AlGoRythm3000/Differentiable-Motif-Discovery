import torch
import pytest
from models.motif_encodings import DeepSetsEncoder

def test_deepsets_permutation_invariance():
    """
    permutation invariance check.
    """
    # seed for reproductibility
    torch.manual_seed(42)
    
    # hyperparameters for the test
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    
    # initialisation of deepsets encoder
    encoder = DeepSetsEncoder(input_dim, hidden_dim, output_dim, aggregator_type='sum')
    encoder.eval() # Mode évaluation pour figer le dropout
    
    # input data
    x_original = torch.randn(5, input_dim)
    batch_index_original = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    
    with torch.no_grad():
        output_original = encoder(x_original, batch_index_original)
        
    # permuted version of the input data
    # the order in x change but the composition of the motifs remains semantically identical
    # motif 0 : index node (0, 1, 2) becomes (2, 0, 1)
    # motif 1 : index node (3, 4) becomes (4, 3)
    permutation = [2, 0, 1, 4, 3]
    
    x_permuted = x_original[permutation]
    batch_index_permuted = batch_index_original[permutation]
    
    # encoding with permutation
    with torch.no_grad():
        output_permuted = encoder(x_permuted, batch_index_permuted)
        
    # the outputs should be the same regardless of the order of the nodes in the motifs
    assert torch.allclose(output_original, output_permuted, atol=1e-6), \
        "Erreur : L'encodeur DeepSets n'est pas invariant par permutation !"

if __name__ == "__main__":
    test_deepsets_permutation_invariance()