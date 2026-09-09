import os

import pytest
import torch

from tools.gpse_cache import GPSE_ENCODING_DIM, attach_gpse_cache, compute_explicit_pe
from tools.synthetic import bottleneck_dataset

CHECKPOINT = "GPSE_pretrained/gpse_model_molpcba_1.0.pt"
needs_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason=f"{CHECKPOINT} not present (fetch it with ./download_gpse.sh)")


def _tiny_synthetic(n=4):
    return bottleneck_dataset("tree_neighbors_match", num_graphs=n, num_classes=4,
                              depth=2, seed=0)


def test_the_synthetic_arm_is_not_an_inmemory_dataset():
    # The premise of the whole fix below. tools/synthetic.py says so in its
    # docstring, and PyG's `precompute_GPSE` reads `dataset.data`/`.slices`
    # directly, so it cannot consume this dataset.
    dataset = _tiny_synthetic()
    assert not hasattr(dataset, "data")
    assert not hasattr(dataset, "slices")


@needs_checkpoint
def test_real_gpse_runs_on_a_non_inmemory_dataset():
    # Regression: `attach_gpse_cache` used to call PyG's `precompute_GPSE`
    # unconditionally and died with
    #   AttributeError: 'SyntheticGraphDataset' object has no attribute 'data'
    # on the synthetic bottleneck arm. It had never surfaced because that arm
    # was missing from the grid that would have exercised it - it is the
    # falsification core, and s1='gpse' on it had simply never run.
    dataset = _tiny_synthetic()
    source = attach_gpse_cache(dataset, cache_path=None, weights_root="GPSE_pretrained")

    assert source == "gpse", "fell back to pse_explicit - the checkpoint was not used"
    for data in dataset:
        pe = data.pestat_GPSE
        # One encoding row per node, virtual node already dropped.
        assert pe.shape[0] == data.num_nodes
        assert torch.isfinite(pe).all()


@needs_checkpoint
def test_both_gpse_paths_agree_on_width():
    # `precompute_GPSE` (the InMemoryDataset path) and `compute_gpse_live` (the
    # path this brick uses for everything else) must produce interchangeable
    # encodings, or the GPSE arm would mean two different things depending on
    # which dataset it ran on.
    from torch_geometric.datasets import TUDataset

    from tools.gpse_cache import load_pretrained_gpse

    model = load_pretrained_gpse("molpcba", "GPSE_pretrained")
    assert model is not None

    inmemory = TUDataset(root="datasets", name="MUTAG")[:4]
    assert hasattr(inmemory, "data") and hasattr(inmemory, "slices")
    from torch_geometric.nn.models.gpse import precompute_GPSE
    precompute_GPSE(model, inmemory)

    synthetic = _tiny_synthetic()
    attach_gpse_cache(synthetic, cache_path=None, weights_root="GPSE_pretrained")

    assert inmemory[0].pestat_GPSE.shape[-1] == synthetic[0].pestat_GPSE.shape[-1]


@needs_checkpoint
def test_the_cache_round_trips_and_records_its_source(tmp_path):
    cache = tmp_path / "synthetic.gpse_cache.pt"
    first = _tiny_synthetic()
    source = attach_gpse_cache(first, cache_path=str(cache), weights_root="GPSE_pretrained")
    assert cache.exists()

    second = _tiny_synthetic()
    reloaded = attach_gpse_cache(second, cache_path=str(cache), weights_root="GPSE_pretrained")

    assert reloaded == source
    for a, b in zip(first, second):
        assert torch.equal(a.pestat_GPSE, b.pestat_GPSE)


def test_the_fallback_reports_itself_and_uses_the_explicit_width(tmp_path):
    # A missing checkpoint must degrade to pse_explicit and SAY so - a grid that
    # silently swaps Stage 1 without recording it is worse than a failed grid.
    # The explicit fallback is 51-wide (GPSE's 6 reconstructed PSEs), which is
    # NOT the width real GPSE returns; GPSEEncoder probes the real width at
    # construction rather than assuming, so the two never get confused.
    dataset = _tiny_synthetic()
    source = attach_gpse_cache(dataset, cache_path=None,
                               weights_root=str(tmp_path / "no_such_dir"))

    assert source == "pse_explicit"
    for data in dataset:
        assert data.pestat_GPSE.shape == (data.num_nodes, GPSE_ENCODING_DIM)


def test_explicit_pe_is_padded_to_the_declared_width():
    pe = compute_explicit_pe(torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
    assert pe.shape == (2, GPSE_ENCODING_DIM)
    assert torch.isfinite(pe).all()
