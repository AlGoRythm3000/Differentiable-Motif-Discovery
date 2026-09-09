import torch
import pytest
from models.motif_encodings import DeepSetsEncoder, MiniGNNEncoder, SetTransformerEncoder

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

def test_deepsets_layernorm_preserves_permutation_invariance():
    """The pre-aggregation LayerNorm (TODO closed on feat/rich-bricks) must not
    break the invariance the previous test already covers for the base path."""
    torch.manual_seed(1)
    encoder = DeepSetsEncoder(10, 12, 6, aggregator_type='mean')
    encoder.eval()
    x = torch.randn(6, 10)
    batch_index = torch.tensor([0, 0, 1, 1, 1, 1])
    permutation = [1, 0, 5, 2, 4, 3]
    with torch.no_grad():
        out_a = encoder(x, batch_index)
        out_b = encoder(x[permutation], batch_index[permutation])
    assert torch.allclose(out_a, out_b, atol=1e-6)


# --------------------------------------------------------------------------
# SetTransformerEncoder (feat/rich-bricks)
# --------------------------------------------------------------------------

def test_set_transformer_output_shape():
    torch.manual_seed(0)
    encoder = SetTransformerEncoder(16, 32, 8, num_inducing=4, num_heads=2)
    x = torch.randn(7, 16)
    batch_index = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    out = encoder(x, batch_index)
    assert out.shape == (3, 8)


def test_set_transformer_permutation_invariance():
    torch.manual_seed(0)
    encoder = SetTransformerEncoder(16, 32, 8, num_inducing=4, num_heads=2)
    encoder.eval()
    x = torch.randn(5, 16)
    batch_index = torch.tensor([0, 0, 0, 1, 1])
    permutation = [2, 0, 1, 4, 3]
    with torch.no_grad():
        out_a = encoder(x, batch_index)
        out_b = encoder(x[permutation], batch_index[permutation])
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_set_transformer_gradient_flows():
    torch.manual_seed(0)
    encoder = SetTransformerEncoder(8, 16, 4, num_inducing=3, num_heads=2)
    x = torch.randn(5, 8, requires_grad=True)
    batch_index = torch.tensor([0, 0, 1, 1, 1])
    out = encoder(x, batch_index)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


# --------------------------------------------------------------------------
# MiniGNNEncoder (feat/rich-bricks)
# --------------------------------------------------------------------------

def _triangle_and_path():
    """Same 3 nodes per cell; cell 0's induced edges form a triangle, cell 1's
    form a 3-path - the case DeepSets/Set Transformer cannot distinguish."""
    x = torch.randn(6, 6)
    node_index = torch.tensor([0, 1, 2, 3, 4, 5])
    cell_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    triangle_edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    path_edges = torch.tensor([[3, 4, 4, 5], [4, 3, 5, 4]])
    edge_index = torch.cat([triangle_edges, path_edges], dim=1)
    return x, node_index, cell_batch, edge_index


def test_mini_gnn_output_shape():
    torch.manual_seed(0)
    x, node_index, cell_batch, edge_index = _triangle_and_path()
    encoder = MiniGNNEncoder(6, 8, 4)
    out = encoder(x, cell_batch, node_index=node_index, edge_index=edge_index)
    assert out.shape == (2, 4)


def test_mini_gnn_permutation_invariance_within_a_cell():
    torch.manual_seed(0)
    encoder = MiniGNNEncoder(6, 8, 4)
    encoder.eval()
    x = torch.randn(3, 6)
    node_index = torch.tensor([0, 1, 2])
    cell_batch = torch.tensor([0, 0, 0])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])

    permutation = [2, 0, 1]
    with torch.no_grad():
        out_a = encoder(x, cell_batch, node_index=node_index, edge_index=edge_index)
        out_b = encoder(x[permutation], cell_batch, node_index=node_index[permutation], edge_index=edge_index)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_mini_gnn_requires_node_index_and_edge_index():
    encoder = MiniGNNEncoder(6, 8, 4)
    x = torch.randn(3, 6)
    cell_batch = torch.tensor([0, 0, 0])
    try:
        encoder(x, cell_batch)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_triangle_vs_path_only_mini_gnn_tells_them_apart():
    """
    The test that proves what mini_gnn buys over DeepSets: a triangle
    and a 3-path on the SAME node set (same features, same cell size) must
    give DIFFERENT embeddings for mini_gnn (it sees the induced edges) and
    IDENTICAL embeddings for DeepSets (it only ever sees the bag of nodes).
    """
    torch.manual_seed(0)
    x = torch.randn(3, 6)
    node_index = torch.tensor([0, 1, 2])
    cell_batch = torch.tensor([0, 0, 0])
    triangle_edges = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    path_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    mini_gnn = MiniGNNEncoder(6, 8, 4)
    mini_gnn.eval()
    with torch.no_grad():
        out_triangle = mini_gnn(x, cell_batch, node_index=node_index, edge_index=triangle_edges)
        out_path = mini_gnn(x, cell_batch, node_index=node_index, edge_index=path_edges)
    assert not torch.allclose(out_triangle, out_path)

    deepsets = DeepSetsEncoder(6, 8, 4)
    deepsets.eval()
    with torch.no_grad():
        ds_triangle = deepsets(x, cell_batch, node_index=node_index, edge_index=triangle_edges)
        ds_path = deepsets(x, cell_batch, node_index=node_index, edge_index=path_edges)
    assert torch.allclose(ds_triangle, ds_path)


if __name__ == "__main__":
    test_deepsets_permutation_invariance()