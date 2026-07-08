import torch

from models.graph_embeddings import GraphEmbedder


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
