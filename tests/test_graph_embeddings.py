import torch

from models.graph_embeddings import GPSEEncoder, GraphEmbedder, PSEExplicitEncoder


def _toy_graph():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 8)
    return x, edge_index


def test_output_shape_gcn():
    x, edge_index = _toy_graph()
    model = GraphEmbedder(input_dim=8, hidden_dim=16, output_dim=5, encoder_type="gcn")
    out = model(x, edge_index)
    assert out.shape == (4, 5)


def test_output_shape_gin():
    x, edge_index = _toy_graph()
    model = GraphEmbedder(input_dim=8, hidden_dim=16, output_dim=5, encoder_type="gin")
    out = model(x, edge_index)
    assert out.shape == (4, 5)


def test_gradients_flow_to_parameters():
    x, edge_index = _toy_graph()
    model = GraphEmbedder(input_dim=8, hidden_dim=16, output_dim=5, encoder_type="gcn")
    out = model(x, edge_index)
    out.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"


def test_unknown_encoder_type_raises():
    try:
        GraphEmbedder(input_dim=8, hidden_dim=16, output_dim=5, encoder_type="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# PSEExplicitEncoder (feat/rich-bricks) - the mandatory GPSE fallback, also
# directly selectable via s1='pse_explicit'.
# --------------------------------------------------------------------------

def test_pse_explicit_output_shape():
    x, edge_index = _toy_graph()
    model = PSEExplicitEncoder(input_dim=8, hidden_dim=16, output_dim=5)
    out = model(x, edge_index)
    assert out.shape == (4, 5)


def test_pse_explicit_gradient_flows_to_backbone():
    x, edge_index = _toy_graph()
    model = PSEExplicitEncoder(input_dim=8, hidden_dim=16, output_dim=5)
    out = model(x, edge_index)
    out.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())


# --------------------------------------------------------------------------
# GPSEEncoder (feat/rich-bricks). These tests exercise whichever path
# construction actually resolves to (real pretrained GPSE if the checkpoint
# fetches, pse_explicit otherwise) - both are asserted to behave correctly,
# and `actual_encoder` says which one ran (it must never silently degrade
# without recording which encoder was used).
# --------------------------------------------------------------------------

def test_gpse_encoder_frozen_and_shape_concat_mode():
    x, edge_index = _toy_graph()
    model = GPSEEncoder(input_dim=8, hidden_dim=16, output_dim=5, gpse_mode="concat")
    out = model(x, edge_index)
    assert out.shape == (4, 5)

    if model.gpse_model is not None:
        assert all(not p.requires_grad for p in model.gpse_model.parameters())
    else:
        assert model.actual_encoder == "pse_explicit"


def test_gpse_encoder_replace_mode_ignores_raw_features():
    """
    In 'replace' mode, Z depends only on the positional encoding - only
    meaningful when the real GPSE checkpoint loaded (the pse_explicit
    fallback's own PSEExplicitEncoder always concatenates raw x).
    """
    x, edge_index = _toy_graph()
    model = GPSEEncoder(input_dim=8, hidden_dim=16, output_dim=5, gpse_mode="replace")
    out = model(x, edge_index)
    assert out.shape == (4, 5)
    if model.gpse_model is not None:
        assert model.backbone is None and model.out_proj is not None


def test_gpse_encoder_unknown_mode_raises():
    try:
        GPSEEncoder(input_dim=8, hidden_dim=16, output_dim=5, gpse_mode="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gpse_encoder_records_actual_encoder():
    model = GPSEEncoder(input_dim=8, hidden_dim=16, output_dim=5)
    assert model.actual_encoder in ("gpse", "pse_explicit")
