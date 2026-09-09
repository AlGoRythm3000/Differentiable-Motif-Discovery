import torch

from models.motif_proposition import AutoregressiveProposal, CycleBasisProposal, MotifProposal


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


# --------------------------------------------------------------------------
# AutoregressiveProposal (feat/rich-bricks)
# --------------------------------------------------------------------------

def test_autoregressive_cell_shapes():
    torch.manual_seed(0)
    N, d = 10, 6
    Z = torch.randn(N, d)
    proposal = AutoregressiveProposal(d, max_size=4)
    candidates = proposal(Z)
    assert candidates.scores.shape == (N,)
    assert candidates.anchor_index.shape == (N,)
    assert candidates.node_index.shape == (N * 4,)  # max_size members per cell


def test_autoregressive_each_cell_contains_its_anchor():
    torch.manual_seed(0)
    N, d = 8, 5
    Z = torch.randn(N, d)
    proposal = AutoregressiveProposal(d, max_size=3)
    candidates = proposal(Z)
    for cell_id in range(N):
        members = candidates.node_index[candidates.cell_batch == cell_id]
        assert candidates.anchor_index[cell_id] in members


def test_autoregressive_never_repeats_a_member_within_a_cell():
    torch.manual_seed(0)
    N, d = 12, 5
    Z = torch.randn(N, d)
    proposal = AutoregressiveProposal(d, max_size=5)
    candidates = proposal(Z)
    for cell_id in range(N):
        members = candidates.node_index[candidates.cell_batch == cell_id]
        assert members.numel() == torch.unique(members).numel()


def test_autoregressive_gradient_reaches_membership_parameters():
    """
    The key contrast test: with s2=autoregressive, gradient
    reaches the parameters that decide MEMBERSHIP (the GRU/query scorer
    conditioning each step's Gumbel-softmax) - a class of parameter `topk`
    has no equivalent of, since its only parameter (`W`) only ever sees
    gradient through the retained top-k VALUES, never through which indices
    got selected.
    """
    torch.manual_seed(0)
    Z = torch.randn(10, 6, requires_grad=True)
    proposal = AutoregressiveProposal(6, max_size=3)
    proposal.train()

    candidates = proposal(Z)
    candidates.scores.sum().backward()

    membership_params = list(proposal.set_gru.parameters()) + list(proposal.query_proj.parameters())
    assert len(membership_params) > 0
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in membership_params)


def test_autoregressive_batch_prevents_cross_graph_candidates():
    torch.manual_seed(0)
    N, d = 12, 4
    Z = torch.randn(N, d)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    proposal = AutoregressiveProposal(d, max_size=3)

    candidates = proposal(Z, batch=batch)
    member_graph = batch[candidates.node_index]
    anchor_graph = batch[candidates.anchor_index[candidates.cell_batch]]
    assert torch.equal(member_graph, anchor_graph)


def test_autoregressive_too_small_graph_raises():
    Z = torch.randn(4, 4)
    batch = torch.tensor([0, 1, 1, 1], dtype=torch.long)  # graph 0 has 1 node
    proposal = AutoregressiveProposal(4, max_size=2)
    try:
        proposal(Z, batch=batch)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# CycleBasisProposal (feat/rich-bricks)
# --------------------------------------------------------------------------

def _two_triangles_edge_index():
    # triangle (0,1,2) and triangle (3,4,5), disconnected
    return torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
                          [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]], dtype=torch.long)


def test_cycle_basis_finds_one_cycle_per_triangle():
    torch.manual_seed(0)
    Z = torch.randn(6, 5)
    proposal = CycleBasisProposal(5)
    candidates = proposal(Z, edge_index=_two_triangles_edge_index())
    num_cells = int(candidates.cell_batch.max().item()) + 1 if candidates.cell_batch.numel() else 0
    assert num_cells == 2
    assert candidates.node_index.numel() == 6  # 3 members per triangle


def test_cycle_basis_num_cells_need_not_equal_num_nodes():
    """A tree has zero cycles - num_cells can be 0 even with many nodes."""
    torch.manual_seed(0)
    Z = torch.randn(5, 4)
    tree_edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    proposal = CycleBasisProposal(4)
    candidates = proposal(Z, edge_index=tree_edges)
    assert candidates.node_index.numel() == 0
    assert candidates.scores.numel() == 0


def test_cycle_basis_requires_edge_index():
    Z = torch.randn(6, 5)
    proposal = CycleBasisProposal(5)
    try:
        proposal(Z)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cycle_basis_scores_differentiable_wrt_Z():
    torch.manual_seed(0)
    Z = torch.randn(6, 5, requires_grad=True)
    proposal = CycleBasisProposal(5)
    candidates = proposal(Z, edge_index=_two_triangles_edge_index())
    candidates.scores.sum().backward()
    assert Z.grad is not None
    assert torch.any(Z.grad != 0)
