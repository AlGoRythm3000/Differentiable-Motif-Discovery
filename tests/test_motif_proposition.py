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
