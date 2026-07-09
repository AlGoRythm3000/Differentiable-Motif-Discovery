import torch

import utils
from tools.metrics import accuracy


def load_dataset(args):
    """
    args: argparse.Namespace with num_cliques, clique_size, feature_dim,
          train_frac, val_frac, seed.
    Returns (data, input_dim, num_classes).
    """
    data = utils.path_of_cliques_dataset(
        num_cliques=args.num_cliques, clique_size=args.clique_size,
        feature_dim=args.feature_dim, train_frac=args.train_frac,
        val_frac=args.val_frac, seed=args.seed,
    )
    input_dim = args.feature_dim
    num_classes = args.num_cliques
    return data, input_dim, num_classes


def train_step(model, data, optimizer, criterion) -> dict:
    model.train()
    optimizer.zero_grad()
    logits, structure = model(data.x, data.edge_index)
    loss_out = criterion(logits, data.y, data.train_mask, structure)
    loss_out.total.backward()
    optimizer.step()
    return {
        "loss": loss_out.total.item(),
        "task": loss_out.task.item(),
        "sparsity": loss_out.sparsity.item(),
        "osq": loss_out.osq.item(),
        "accuracy": accuracy(logits, data.y, data.train_mask),
    }


@torch.no_grad()
def eval_step(model, data, criterion, mask) -> dict:
    model.eval()
    logits, structure = model(data.x, data.edge_index)
    loss_out = criterion(logits, data.y, mask, structure)
    return {
        "loss": loss_out.total.item(),
        "task": loss_out.task.item(),
        "accuracy": accuracy(logits, data.y, mask),
    }


@torch.no_grad()
def collect_graph_samples(model, data, num_samples: int = 1) -> list:
    """
    Only one graph exists for this task (the toy path-of-cliques), so
    `num_samples` is ignored - returns that single graph's original vs.
    rewired edges for results/analyze_results.py's adjacency heatmaps.
    """
    model.eval()
    _, structure = model(data.x, data.edge_index)
    return [{
        "graph_id": 0,
        "num_nodes": data.x.size(0),
        "label": None,
        "edge_index": data.edge_index.tolist(),
        "rewired_edge_index": structure["rewired_edge_index"].tolist(),
        "rewired_edge_weight": structure["rewired_edge_weight"].tolist(),
    }]
