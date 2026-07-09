import math

import pytest

import train
from tasks import graph_classification


class _Args:
    dataset = "MUTAG"
    data_root = "datasets"
    batch_size = 16
    train_frac = 0.6
    val_frac = 0.2


def _mutag_available():
    """MUTAG auto-downloads on first use; skip if that fails (offline CI)."""
    try:
        graph_classification.load_dataset(_Args())
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _mutag_available(), reason="MUTAG (TUDataset) unavailable/offline")


def test_load_dataset_rejects_unknown_name():
    class _BadArgs(_Args):
        dataset = "NOT_A_DATASET"
    with pytest.raises(ValueError):
        graph_classification.load_dataset(_BadArgs())


def test_load_dataset_shapes():
    data, input_dim, num_classes = graph_classification.load_dataset(_Args())
    assert input_dim == 7  # MUTAG's fixed one-hot atom-type node features
    assert num_classes == 2
    assert len(data.train_loader.dataset) > 0
    assert len(data.val_mask.dataset) > 0
    assert len(data.test_mask.dataset) > 0


def test_train_main_smoke():
    """Smoke-run the CLI entrypoint end-to-end on a real dataset with tiny dims."""
    result = train.main([
        "--task", "graph_classification",
        "--dataset", "MUTAG",
        "--batch-size", "16",
        "--epochs", "2",
        "--hidden-dim", "8",
        "--latent-dim", "6",
        "--motif-hidden-dim", "6",
        "--motif-out-dim", "4",
        "--top-k", "2",
        "--log-every", "100",
    ])
    assert not math.isnan(result["best_val_acc"])
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert 0.0 <= result["test_acc_at_best_val"] <= 1.0
