# Registry-level tests: every brick is a drop-in swap at its
# stage's call site, and the config validator rejects the one known-invalid
# combination (`tnn` without `cycle_basis`) with a clear error.

import pytest
import torch

from models.dmd_model import DMDModel
from models.registry import (STAGE1_ENCODERS, STAGE2_PROPOSALS, STAGE3_CELL_ENCODERS,
                              STAGE4_SELECTORS, STAGE5_MP, validate_pipeline_config)


def _toy_batch():
    """Two disjoint graphs, batched PyG-style; graph 1 has triangles so a
    cycle_basis/tnn brick has real 2-cells to work with, not just its
    no-cycle identity fallback."""
    x1 = torch.randn(5, 6)
    edge_index1 = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    x2 = torch.randn(6, 6)
    edge_index2 = torch.tensor([[0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
                                 [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5]])
    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2 + 5], dim=1)
    batch = torch.tensor([0] * 5 + [1] * 6)
    return x, edge_index, batch


def _make_model(**overrides):
    defaults = dict(input_dim=6, hidden_dim=8, latent_dim=6, motif_hidden_dim=6,
                     motif_out_dim=4, num_classes=3, top_k=2)
    defaults.update(overrides)
    return DMDModel(**defaults)


@pytest.mark.parametrize("s1", sorted(STAGE1_ENCODERS))
def test_stage1_drop_in_equivalence(s1):
    x, edge_index, batch = _toy_batch()
    model = _make_model(s1=s1)
    logits, structure = model(x, edge_index, batch=batch)
    assert logits.shape == (2, 3)
    assert structure["alpha"].numel() > 0 or structure["alpha"].numel() == 0  # never crashes to get here


@pytest.mark.parametrize("s2", sorted(STAGE2_PROPOSALS))
def test_stage2_drop_in_equivalence(s2):
    x, edge_index, batch = _toy_batch()
    s2_kwargs = {"max_size": 3} if s2 == "autoregressive" else None
    model = _make_model(s2=s2, s2_kwargs=s2_kwargs)
    logits, structure = model(x, edge_index, batch=batch)
    assert logits.shape == (2, 3)


@pytest.mark.parametrize("s3", sorted(STAGE3_CELL_ENCODERS))
def test_stage3_drop_in_equivalence(s3):
    x, edge_index, batch = _toy_batch()
    # motif_hidden_dim=6 in _make_model's defaults; set_transformer's
    # num_heads must divide it (its own default, 4, doesn't).
    s3_kwargs = {"num_heads": 2} if s3 == "set_transformer" else None
    model = _make_model(s3=s3, s3_kwargs=s3_kwargs)
    logits, structure = model(x, edge_index, batch=batch)
    assert logits.shape == (2, 3)


@pytest.mark.parametrize("s4", sorted(STAGE4_SELECTORS))
def test_stage4_drop_in_equivalence(s4):
    x, edge_index, batch = _toy_batch()
    s4_kwargs = {"k": 2} if s4 == "ksubset" else None
    model = _make_model(s4=s4, s4_kwargs=s4_kwargs)
    logits, structure = model(x, edge_index, batch=batch)
    assert logits.shape == (2, 3)


@pytest.mark.parametrize("s5", sorted(STAGE5_MP))
def test_stage5_drop_in_equivalence(s5):
    x, edge_index, batch = _toy_batch()
    s2 = "cycle_basis" if s5 == "tnn" else "topk"
    model = _make_model(s5=s5, s2=s2)
    logits, structure = model(x, edge_index, batch=batch)
    assert logits.shape == (2, 3)


# --------------------------------------------------------------------------
# config validator
# --------------------------------------------------------------------------

def test_validator_rejects_tnn_with_topk():
    with pytest.raises(ValueError):
        validate_pipeline_config("gcn", "topk", "deepsets", "gumbel", "tnn")


def test_validator_rejects_tnn_with_autoregressive():
    with pytest.raises(ValueError):
        validate_pipeline_config("gcn", "autoregressive", "deepsets", "gumbel", "tnn")


def test_validator_accepts_tnn_with_cycle_basis():
    validate_pipeline_config("gcn", "cycle_basis", "deepsets", "gumbel", "tnn")  # must not raise


def test_validator_accepts_hypergraph_tnn_with_any_proposal():
    for s2 in STAGE2_PROPOSALS:
        validate_pipeline_config("gcn", s2, "deepsets", "gumbel", "hypergraph_tnn")  # must not raise


def test_validator_rejects_unknown_brick():
    with pytest.raises(ValueError):
        validate_pipeline_config("nope", "topk", "deepsets", "gumbel", "gnn_rewired")


def test_dmd_model_construction_rejects_tnn_with_topk():
    """The validator is also wired into DMDModel.__init__ itself, not just
    exposed as a free function."""
    with pytest.raises(ValueError):
        _make_model(s2="topk", s5="tnn")
