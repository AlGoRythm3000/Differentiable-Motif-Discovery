import torch

from models.message_passing import GNNRewiredMP, HypergraphTNNMessagePassing, TNNMessagePassing
from models.motif_proposition import CycleBasisProposal, MotifProposal
from models.weight_assignment import GumbelSigmoidSelector


def _two_triangles():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
                                [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]], dtype=torch.long)
    return edge_index, 6


def test_gnn_rewired_output_shape_and_gradient():
    torch.manual_seed(0)
    edge_index, num_nodes = _two_triangles()
    Z = torch.randn(num_nodes, 8, requires_grad=True)
    proposal = MotifProposal(8, top_k=2)
    candidates = proposal(Z)
    alpha = GumbelSigmoidSelector(tau=0.5, hard=False)(candidates.scores)

    mp = GNNRewiredMP(8)
    Z_rewired, rei, rew = mp(Z, edge_index, candidates, alpha, num_nodes)
    assert Z_rewired.shape == (num_nodes, 8)
    assert rei.shape[0] == 2
    assert rew.shape[0] == rei.shape[1]

    Z_rewired.sum().backward()
    assert Z.grad is not None and torch.any(Z.grad != 0)


def test_tnn_message_passing_output_shape_and_gradient():
    torch.manual_seed(0)
    edge_index, num_nodes = _two_triangles()
    Z = torch.randn(num_nodes, 8, requires_grad=True)
    proposal = CycleBasisProposal(8)
    candidates = proposal(Z, edge_index=edge_index)
    alpha = GumbelSigmoidSelector(tau=0.5, hard=False)(candidates.scores)

    mp = TNNMessagePassing(8, hidden_dim=6)
    Z_rewired, rei, rew = mp(Z, edge_index, candidates, alpha, num_nodes)
    assert Z_rewired.shape == (num_nodes, 8)

    Z_rewired.sum().backward()
    assert Z.grad is not None and torch.any(Z.grad != 0)


def test_tnn_message_passing_no_cycles_is_identity():
    """A tree has no 2-cells: TNN must not crash, and Z passes through unchanged."""
    torch.manual_seed(0)
    tree_edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    num_nodes = 4
    Z = torch.randn(num_nodes, 6)
    proposal = CycleBasisProposal(6)
    candidates = proposal(Z, edge_index=tree_edges)
    alpha = torch.ones(0)

    mp = TNNMessagePassing(6)
    Z_rewired, rei, rew = mp(Z, tree_edges, candidates, alpha, num_nodes)
    assert torch.equal(Z_rewired, Z)


def test_hypergraph_tnn_output_shape_and_gradient():
    torch.manual_seed(0)
    edge_index, num_nodes = _two_triangles()
    Z = torch.randn(num_nodes, 8, requires_grad=True)
    proposal = MotifProposal(8, top_k=2)
    candidates = proposal(Z)
    alpha = GumbelSigmoidSelector(tau=0.5, hard=False)(candidates.scores)

    mp = HypergraphTNNMessagePassing(8, hidden_dim=6)
    Z_rewired, rei, rew = mp(Z, edge_index, candidates, alpha, num_nodes)
    assert Z_rewired.shape == (num_nodes, 8)

    Z_rewired.sum().backward()
    assert Z.grad is not None and torch.any(Z.grad != 0)


def test_hypergraph_tnn_accepts_arbitrary_topk_cells():
    """Unlike `tnn`, hypergraph_tnn has no validity requirement on the proposal."""
    torch.manual_seed(0)
    edge_index, num_nodes = _two_triangles()
    Z = torch.randn(num_nodes, 5)
    proposal = MotifProposal(5, top_k=3)
    candidates = proposal(Z)
    alpha = torch.rand(candidates.cell_batch.max().item() + 1)

    mp = HypergraphTNNMessagePassing(5)
    Z_rewired, _, _ = mp(Z, edge_index, candidates, alpha, num_nodes)
    assert Z_rewired.shape == (num_nodes, 5)
