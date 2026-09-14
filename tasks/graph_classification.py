# Graph-level classification task glue: whole-graph label prediction on
# TUDataset benchmarks (NCI1, MUTAG, PROTEINS). DMDModel is run with graph
# batching (Stage 5's node representations are mean-pooled per graph before
# the classifier - see `batch=` in DMDModel.forward).

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from tools.metrics import accuracy

TU_DATASETS = ("NCI1", "MUTAG", "PROTEINS")


class GraphSplits:
    """
    Holds the train/val/test DataLoaders for a graph-classification dataset.

    `val_mask`/`test_mask` are named to match the generic 4th argument
    train.py passes to `eval_step` for every task (a boolean node mask for
    node_classification, a DataLoader here) - train.py stays task-agnostic
    and just forwards whatever `load_dataset` returns.
    """

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.train_loader = train_loader
        self.val_mask = val_loader
        self.test_mask = test_loader


def load_dataset(args):
    """
    args: argparse.Namespace with dataset, data_root, batch_size,
          train_frac, val_frac, seed.
    Returns (GraphSplits, input_dim, num_classes).
    """
    if args.dataset not in TU_DATASETS:
        raise ValueError(
            f"Unknown graph_classification dataset '{args.dataset}'. "
            f"Choices: {', '.join(TU_DATASETS)}"
        )

    dataset = TUDataset(root=args.data_root, name=args.dataset)
    dataset = dataset.shuffle()

    n = len(dataset)
    n_train = max(1, int(round(args.train_frac * n)))
    n_val = max(1, int(round(args.val_frac * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_set = dataset[:n_train]
    val_set = dataset[n_train:n_train + n_val]
    test_set = dataset[n_train + n_val:]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    return (GraphSplits(train_loader, val_loader, test_loader),
            dataset.num_node_features, dataset.num_classes)


def _run_loader(model, loader, criterion, optimizer=None, grad_clip=None) -> dict:
    """
    Shared batch loop: trains if `optimizer` is given, else just evaluates.

    `grad_clip` (max global gradient norm, None = off) exists for Stage 4's
    `reinforce` brick above all: the score-function estimator is unbiased but
    its variance is unbounded, and an unclipped step is what drove `scores` to
    NaN in the feat/rich-bricks grid - which `torch.bernoulli` then turned into
    a CUDA device-side assert (see models/weight_assignment.py). Applied to
    every brick rather than only that one so a clipped and an unclipped arm are
    never compared to each other.
    """
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "task": 0.0, "sparsity": 0.0, "osq": 0.0, "correct": 0.0}
    num_graphs = 0
    # Follow the model rather than taking a device argument: the caller has
    # already decided where the model lives, and a batch on the wrong device is
    # the classic silent failure when a CPU-written loop is reused on a GPU.
    device = next(model.parameters()).device

    for batch in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            node_pe = getattr(batch, "pestat_GPSE", None)
            logits, structure = model(batch.x, batch.edge_index, batch=batch.batch, node_pe=node_pe)
            mask = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
            loss_out = criterion(logits, batch.y, mask, structure)

        if is_train:
            loss_out.total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch.num_graphs
        totals["loss"] += loss_out.total.item() * bs
        totals["task"] += loss_out.task.item() * bs
        totals["sparsity"] += loss_out.sparsity.item() * bs
        totals["osq"] += loss_out.osq.item() * bs
        totals["correct"] += accuracy(logits, batch.y, mask) * bs
        num_graphs += bs

    return {
        "loss": totals["loss"] / num_graphs,
        "task": totals["task"] / num_graphs,
        "sparsity": totals["sparsity"] / num_graphs,
        "osq": totals["osq"] / num_graphs,
        "accuracy": totals["correct"] / num_graphs,
    }


def train_step(model, data, optimizer, criterion, grad_clip=None) -> dict:
    return _run_loader(model, data.train_loader, criterion, optimizer=optimizer,
                       grad_clip=grad_clip)


@torch.no_grad()
def eval_step(model, data, criterion, loader) -> dict:
    metrics = _run_loader(model, loader, criterion, optimizer=None)
    return {"loss": metrics["loss"], "task": metrics["task"], "accuracy": metrics["accuracy"]}


@torch.no_grad()
def collect_graph_samples(model, data, num_samples: int = 3) -> list:
    """
    Runs the model on the first `num_samples` test graphs (as one small
    batch) and unbatches each graph's original vs. rewired edges for
    results/analyze_results.py's adjacency heatmaps.
    """
    model.eval()
    test_dataset = data.test_mask.dataset
    n = min(num_samples, len(test_dataset))
    if n == 0:
        return []

    loader = DataLoader(test_dataset[:n], batch_size=n)
    batch = next(iter(loader)).to(next(model.parameters()).device)
    _, structure = model(batch.x, batch.edge_index, batch=batch.batch,
                          node_pe=getattr(batch, "pestat_GPSE", None))
    rewired_edge_index = structure["rewired_edge_index"]
    rewired_edge_weight = structure["rewired_edge_weight"]
    # Cell membership, so the analysis can rebuild A^col (a clique per cell) and
    # not only the star the model actually message-passes over. Without these
    # three arrays r_bar_after can only ever be re-measured on the structure the
    # proxy already minimised, which is the circularity §6.7 of the paper flags.
    candidates = structure.get("candidates")
    alpha = structure.get("alpha")

    samples = []
    ptr = batch.ptr.tolist()
    for i in range(n):
        lo, hi = ptr[i], ptr[i + 1]

        orig_mask = (batch.edge_index[0] >= lo) & (batch.edge_index[0] < hi)
        orig_edges = (batch.edge_index[:, orig_mask] - lo).tolist()

        rew_mask = (rewired_edge_index[0] >= lo) & (rewired_edge_index[0] < hi)
        rew_edges = (rewired_edge_index[:, rew_mask] - lo).tolist()
        rew_weights = rewired_edge_weight[rew_mask].tolist()

        sample = {
            "graph_id": i,
            "num_nodes": hi - lo,
            "label": int(batch.y[i].item()),
            "edge_index": orig_edges,
            "rewired_edge_index": rew_edges,
            "rewired_edge_weight": rew_weights,
        }

        if candidates is not None and alpha is not None and candidates.cell_batch.numel():
            # Keep only cells lying wholly inside this graph. A cell that
            # straddled two graphs of the batch would make A^col join entities
            # that were never related; the proposals mask by batch, so this is a
            # guard rather than an expected case.
            slot_in = (candidates.node_index >= lo) & (candidates.node_index < hi)
            cells_here = torch.unique(candidates.cell_batch[slot_in])
            keep_cell = torch.zeros(int(candidates.cell_batch.max().item()) + 1,
                                    dtype=torch.bool, device=candidates.cell_batch.device)
            keep_cell[cells_here] = True
            whole = keep_cell[candidates.cell_batch] & slot_in
            for c in cells_here.tolist():
                if not bool(slot_in[candidates.cell_batch == c].all()):
                    whole &= candidates.cell_batch != c
            local_cells = torch.unique(candidates.cell_batch[whole])
            remap = torch.full((keep_cell.numel(),), -1, dtype=torch.long,
                               device=candidates.cell_batch.device)
            remap[local_cells] = torch.arange(local_cells.numel(),
                                              device=candidates.cell_batch.device)
            sample["cell_node_index"] = (candidates.node_index[whole] - lo).tolist()
            sample["cell_batch"] = remap[candidates.cell_batch[whole]].tolist()
            sample["cell_alpha"] = alpha[local_cells].detach().cpu().tolist()

        samples.append(sample)
    return samples
