import torch
import torch.nn as nn

from models.dmd_model import DMDModel


def _toy_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                                [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    x = torch.randn(5, 6)
    return x, edge_index


def _make_model(**kwargs):
    defaults = dict(input_dim=6, hidden_dim=8, latent_dim=5, motif_hidden_dim=6,
                     motif_out_dim=4, num_classes=3, top_k=2)
    defaults.update(kwargs)
    return DMDModel(**defaults)


def test_forward_shapes():
    torch.manual_seed(0)
    x, edge_index = _toy_graph()
    model = _make_model()
    logits, structure = model(x, edge_index)

    assert logits.shape == (5, 3)
    assert structure["alpha"].shape == (5,)  # one candidate cell per node
    assert structure["rewired_edge_index"].shape[0] == 2
    assert structure["rewired_edge_weight"].shape[0] == structure["rewired_edge_index"].shape[1]


def test_end_to_end_gradients_reach_every_submodule():
    torch.manual_seed(0)
    x, edge_index = _toy_graph()
    model = _make_model()
    logits, structure = model(x, edge_index)

    loss = logits.sum() + structure["alpha"].sum()
    loss.backward()

    for name in ("embedder", "proposal", "motif_encoder", "rewired_conv", "classifier"):
        submodule = getattr(model, name)
        grads = [p.grad for p in submodule.parameters() if p.requires_grad]
        assert len(grads) > 0
        assert any(g is not None and torch.any(g != 0) for g in grads), f"{name} got no gradient"


def test_selection_actually_gates_stage5_output():
    """
    Forcing alpha to all-zero vs. all-one must change the logits - proves
    Stage 5 genuinely depends on Stage 4's output rather than being dead code.
    """
    torch.manual_seed(0)
    x, edge_index = _toy_graph()
    model = _make_model()
    model.eval()

    class _ConstantSelector(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, scores):
            return torch.full_like(scores, self.value)

    model.selector = _ConstantSelector(0.0)
    with torch.no_grad():
        logits_zero, _ = model(x, edge_index)

    model.selector = _ConstantSelector(1.0)
    with torch.no_grad():
        logits_one, _ = model(x, edge_index)

    assert not torch.allclose(logits_zero, logits_one)
