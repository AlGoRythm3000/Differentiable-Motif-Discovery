import math

import torch

import utils
import train
from models.dmd_model import DMDModel
from tools.losses import DMDLoss
from tasks import node_classification


class _Args:
    num_cliques = 3
    clique_size = 5
    feature_dim = 8
    train_frac = 0.6
    val_frac = 0.2
    seed = 0


def test_direct_training_loop_reduces_loss():
    """Literal Phase 0 exit criterion: end-to-end trainable on a toy synthetic
    graph. Calibrated to avoid a flaky accuracy threshold - only checks the
    loss is finite and does not get worse over a short run."""
    utils.set_seed(0)
    args = _Args()
    data, input_dim, num_classes = node_classification.load_dataset(args)

    model = DMDModel(input_dim=input_dim, hidden_dim=8, latent_dim=6,
                      motif_hidden_dim=6, motif_out_dim=4, num_classes=num_classes, top_k=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = DMDLoss(sparsity_weight=0.05)

    losses = []
    for _ in range(15):
        metrics = node_classification.train_step(model, data, optimizer, criterion)
        assert not math.isnan(metrics["loss"])
        losses.append(metrics["loss"])

    assert losses[-1] <= losses[0]


def test_train_main_smoke():
    """Smoke-run the CLI entrypoint end-to-end with tiny dims."""
    result = train.main([
        "--epochs", "3",
        "--num-cliques", "3",
        "--clique-size", "5",
        "--feature-dim", "8",
        "--hidden-dim", "8",
        "--latent-dim", "6",
        "--motif-hidden-dim", "6",
        "--motif-out-dim", "4",
        "--top-k", "2",
        "--log-every", "100",
    ])
    assert 0.0 <= result["best_val_acc"] <= 1.0
    assert 0.0 <= result["test_acc_at_best_val"] <= 1.0
