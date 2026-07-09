import torch

from models.motif_proposition import MotifProposal


def test_candidate_cells_shapes():
    torch.manual_seed(0)
    N, d, k = 10, 6, 3
    Z = torch.randn(N, d)
    proposal = MotifProposal(latent_dim=d, top_k=k, include_anchor=True)

    candidates = proposal(Z)

    assert candidates.scores.shape == (N,)
    assert candidates.anchor_index.shape == (N,)
    # one cell per node, each with (k + 1) members (anchor + k neighbors)
    assert candidates.node_index.shape == (N * (k + 1),)
    assert candidates.cell_batch.shape == (N * (k + 1),)
    assert candidates.cell_batch.max().item() == N - 1


def test_scores_not_normalized_like_a_distribution():
    """Stage 2 must not softmax/normalize - that boundary belongs to Stage 4."""
    torch.manual_seed(0)
    Z = torch.randn(8, 4)
    proposal = MotifProposal(latent_dim=4, top_k=2)
    candidates = proposal(Z)
    assert not torch.allclose(candidates.scores.sum(), torch.tensor(1.0))


def test_each_cell_contains_its_anchor():
    torch.manual_seed(0)
    N, k = 6, 2
    Z = torch.randn(N, 4)
    proposal = MotifProposal(latent_dim=4, top_k=k, include_anchor=True)
    candidates = proposal(Z)

    for cell_id in range(N):
        members = candidates.node_index[candidates.cell_batch == cell_id]
        anchor = candidates.anchor_index[cell_id]
        assert anchor in members


def test_gradient_flows_to_Z_and_W():
    torch.manual_seed(0)
    Z = torch.randn(8, 4, requires_grad=True)
    proposal = MotifProposal(latent_dim=4, top_k=3)

    candidates = proposal(Z)
    candidates.scores.sum().backward()

    assert Z.grad is not None
    assert proposal.W.grad is not None


def test_batch_prevents_cross_graph_candidates():
    """Regression test: a candidate cell for a node in graph 0 must never
    recruit a member from graph 1 - without the `batch` mask, top-k over the
    full batched similarity matrix would happily do exactly that."""
    torch.manual_seed(0)
    N, d, k = 12, 4, 3
    Z = torch.randn(N, d)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    proposal = MotifProposal(latent_dim=d, top_k=k, include_anchor=True)

    candidates = proposal(Z, batch=batch)

    member_graph = batch[candidates.node_index]
    anchor_graph = batch[candidates.anchor_index[candidates.cell_batch]]
    assert torch.equal(member_graph, anchor_graph)


def test_batch_clamps_top_k_to_smallest_graph():
    """A graph with only 2 nodes can supply at most 1 same-graph neighbor,
    even if --top-k asks for more - k must be clamped, not silently allowed
    to leak into another graph to make up the count."""
    torch.manual_seed(0)
    Z = torch.randn(7, 4)
    batch = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.long)  # graph 0 has 2 nodes
    proposal = MotifProposal(latent_dim=4, top_k=4, include_anchor=True)

    candidates = proposal(Z, batch=batch)
    # graph 0's cells: anchor + 1 same-graph neighbor = 2 members each, not top_k + 1 = 5
    for anchor in (0, 1):
        members = candidates.node_index[candidates.cell_batch == anchor]
        assert members.shape[0] == 2


def test_top_k_larger_than_smallest_graph_raises_when_no_self_slack():
    """A singleton graph (1 node) has zero valid same-graph neighbors -
    this must raise, not silently fall back to a cross-graph candidate."""
    torch.manual_seed(0)
    Z = torch.randn(4, 4)
    batch = torch.tensor([0, 1, 1, 1], dtype=torch.long)  # graph 0 has 1 node
    proposal = MotifProposal(latent_dim=4, top_k=2, include_anchor=True)

    try:
        proposal(Z, batch=batch)
        assert False, "expected ValueError"
    except ValueError:
        pass
