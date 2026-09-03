# The experiment grid: datasets x proxies x OSq weights x seeds.
#
# This lives in the repo rather than inside the Kaggle notebook on purpose. The
# notebook clones a branch and calls `run_grid`, so the code that produced a
# results directory is the code at a commit SHA recorded in that same directory,
# and it can be tested locally like any other module. The notebook keeps only
# what genuinely belongs to a notebook: configuration, environment capture, and
# plots.
#
# Failure policy: one run that raises must never kill the grid. Every run is
# wrapped, the traceback is stored in the run's row and raw file, and the loop
# continues - a grid that dies at run 200 of 234 because one dataset had an
# awkward graph is worse than useless.

import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

import utils
from models.dmd_model import DMDModel
from tasks import graph_classification
from tools.losses import DMDLoss
from tools.osq_metrics import before_after_report
from tools.osq_proxies import (DEFAULT_CG_MAXITER, DEFAULT_CG_TOL, DEFAULT_EPS,
                                DEFAULT_HUTCH_K, get_proxy)
from tools.results_store import ResultsStore, make_run_id
from tools.synthetic import bottleneck_dataset

SYNTHETIC_ARM = "synthetic_bottleneck"

# Order the runtime guard cuts in. Datasets are listed cheapest-and-most-
# informative first; the synthetic arm is first because it is the only place
# where an OSq-guided lifting is *supposed* to win, so it is the one arm that
# must never be dropped.
DATASET_PRIORITY = (SYNTHETIC_ARM, "MUTAG", "PROTEINS", "IMDB-BINARY", "ENZYMES", "NCI1")
DROPPABLE_DATASETS = ("NCI1", "ENZYMES")


@dataclass
class GridConfig:
    """Every knob of the grid, in one place."""
    datasets: Sequence[str] = ("MUTAG", "PROTEINS", "ENZYMES", "NCI1", "IMDB-BINARY",
                               SYNTHETIC_ARM)
    proxies: Sequence[str] = ("none", "r_bar", "lambda2", "efc", "cf_bc_efc")
    gammas: Sequence[float] = (0.0, 0.01, 0.1, 1.0)
    seeds: Sequence[int] = (0, 1, 2)

    epochs: int = 200
    patience: int = 30
    lr: float = 0.005
    weight_decay: float = 5e-4
    batch_size: int = 32
    hidden_dim: int = 64
    top_k: int = 4
    sparsity_weight: float = 0.05
    encoder: str = "gcn"

    hutch_k: int = DEFAULT_HUTCH_K
    cg_tol: float = DEFAULT_CG_TOL
    cg_maxiter: int = DEFAULT_CG_MAXITER
    osq_eps: float = DEFAULT_EPS

    train_frac: float = 0.8
    val_frac: float = 0.1
    data_root: str = "datasets"
    device: str = "cpu"

    # Number of test graphs the before/after OSq measurement is averaged over.
    # Exact (dense eigendecomposition), so it is a per-run cost, not a per-epoch
    # one, and a handful of graphs is enough for a mean +- std across seeds.
    osq_sample_graphs: int = 8
    # gamma = 0 provably ignores the proxy (DMDLoss never calls it), so the
    # (proxy, gamma=0) cells are all the same run. Keeping one of them saves a
    # quarter of the grid; set to False to run the full cartesian product.
    skip_redundant_zero_gamma: bool = True
    time_budget_s: Optional[float] = None

    # Synthetic arm shape.
    synthetic_family: str = "tree_neighbors_match"
    synthetic_graphs: int = 600
    synthetic_classes: int = 4
    synthetic_depth: int = 3

    commit_sha: str = ""


@dataclass
class RunSpec:
    dataset: str
    proxy: str
    gamma: float
    seed: int

    @property
    def run_id(self) -> str:
        return make_run_id(self.dataset, self.proxy, self.gamma, self.seed)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def build_plan(config: GridConfig) -> List[RunSpec]:
    """
    Runs ordered so that truncating the tail degrades the grid gracefully: the
    gamma = 0 baselines first (they are what every other arm is compared to),
    then one full pass per gamma, datasets inside a pass ordered by priority.
    """
    datasets = sorted(config.datasets,
                      key=lambda d: DATASET_PRIORITY.index(d) if d in DATASET_PRIORITY else 99)
    plan: List[RunSpec] = []

    zero_gammas = [g for g in config.gammas if g == 0.0]
    for gamma in zero_gammas:
        for dataset in datasets:
            proxies = ["none"] if config.skip_redundant_zero_gamma else list(config.proxies)
            for proxy in proxies:
                for seed in config.seeds:
                    plan.append(RunSpec(dataset, proxy, gamma, seed))

    for gamma in [g for g in config.gammas if g != 0.0]:
        for dataset in datasets:
            for proxy in config.proxies:
                if proxy == "none" and config.skip_redundant_zero_gamma:
                    continue  # the `none` arm is gamma-independent by construction
                for seed in config.seeds:
                    plan.append(RunSpec(dataset, proxy, gamma, seed))
    return plan


def trim_plan(remaining: Sequence[RunSpec], projected_seconds_per_run: float,
              seconds_left: float) -> List[RunSpec]:
    """
    Drops runs, in the agreed order, until the plan fits in the time left:
    largest OSq weight first, then the two expensive real datasets. Never the
    synthetic arm, never a seed - three seeds is the minimum for a mean +- std,
    and a single-seed number is not reportable.

    Returns the kept runs; the caller reports what was dropped.
    """
    kept = list(remaining)
    if projected_seconds_per_run <= 0:
        return kept

    def fits():
        return len(kept) * projected_seconds_per_run <= seconds_left

    while not fits() and kept:
        gammas = sorted({spec.gamma for spec in kept if spec.gamma > 0}, reverse=True)
        if gammas:
            victim_gamma = gammas[0]
            reduced = [s for s in kept if s.gamma != victim_gamma]
            if reduced and any(s.gamma > 0 for s in reduced):
                kept = reduced
                continue

        droppable = [s for s in kept if s.dataset in DROPPABLE_DATASETS]
        if droppable:
            # DROPPABLE_DATASETS is ordered most-expensive-first, so this cuts
            # NCI1 before ENZYMES.
            victim_dataset = min((s.dataset for s in droppable),
                                 key=lambda d: DROPPABLE_DATASETS.index(d))
            kept = [s for s in kept if s.dataset != victim_dataset]
            continue
        break  # nothing left that we are allowed to cut
    return kept


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def ensure_node_features(dataset):
    """
    Some TU benchmarks (IMDB-BINARY) ship without node features. A one-hot
    degree encoding is the usual substitute: it is purely structural, so it does
    not smuggle in information the graph does not have, and it keeps the input
    dimension finite. Falls back to a constant feature if degrees are extreme.
    """
    if dataset.num_node_features > 0:
        return dataset

    import torch_geometric.transforms as T
    from torch_geometric.utils import degree

    max_degree = 0
    for data in dataset:
        if data.edge_index.numel():
            max_degree = max(max_degree, int(degree(data.edge_index[0],
                                                    data.num_nodes).max().item()))
    transform = T.OneHotDegree(max_degree) if max_degree <= 256 else T.Constant(value=1.0)
    dataset.transform = transform
    return dataset


def load_arm(name: str, config: GridConfig):
    """Returns (dataset, num_features, num_classes) for one arm of the grid."""
    if name == SYNTHETIC_ARM:
        dataset = bottleneck_dataset(config.synthetic_family,
                                     num_graphs=config.synthetic_graphs,
                                     num_classes=config.synthetic_classes,
                                     depth=config.synthetic_depth, seed=0)
        return dataset, dataset.num_node_features, dataset.num_classes

    from torch_geometric.datasets import TUDataset

    dataset = ensure_node_features(TUDataset(root=config.data_root, name=name))
    return dataset, dataset.num_node_features, dataset.num_classes


def _labels_of(dataset) -> torch.Tensor:
    y = dataset.y
    return torch.as_tensor(y).view(-1)


def make_splits(dataset, config: GridConfig, seed: int):
    """
    Stratified 80/10/10 split, seeded per run, returned together with the index
    lists so the exact split can be stored in the run's raw file.
    """
    from torch_geometric.loader import DataLoader

    train_idx, val_idx, test_idx = utils.stratified_split(
        _labels_of(dataset), config.train_frac, config.val_frac, seed=seed)

    splits = graph_classification.GraphSplits(
        DataLoader(dataset[train_idx], batch_size=config.batch_size, shuffle=True),
        DataLoader(dataset[val_idx], batch_size=config.batch_size),
        DataLoader(dataset[test_idx], batch_size=config.batch_size),
    )
    return splits, {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

@torch.no_grad()
def alpha_statistics(model, splits) -> dict:
    """Mean/std of the acceptance weights and the number of candidate cells."""
    model.eval()
    batch = next(iter(splits.test_mask)).to(next(model.parameters()).device)
    _, structure = model(batch.x, batch.edge_index, batch=batch.batch)
    alpha = structure["alpha"]
    return {"alpha_mean": float(alpha.mean().item()),
            "alpha_std": float(alpha.std(unbiased=False).item()),
            "num_cells": int(alpha.numel())}


def run_single(spec: RunSpec, config: GridConfig, dataset, num_features: int,
               num_classes: int) -> dict:
    """
    Trains one configuration and returns (row, history, extras). Raises on
    failure - the caller decides what to do with the traceback.
    """
    utils.set_seed(spec.seed)
    splits, split_indices = make_splits(dataset, config, spec.seed)

    # One width knob: latent and motif dimensions follow `hidden_dim`. The grid
    # varies the objective, not the architecture - a run that differs in both is
    # not an ablation of either.
    model = DMDModel(input_dim=num_features, hidden_dim=config.hidden_dim,
                     latent_dim=config.hidden_dim, motif_hidden_dim=config.hidden_dim,
                     motif_out_dim=config.hidden_dim, num_classes=num_classes,
                     encoder_type=config.encoder, top_k=config.top_k).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = DMDLoss(sparsity_weight=config.sparsity_weight, osq_weight=spec.gamma,
                        osq_fn=get_proxy(spec.proxy, hutch_k=config.hutch_k,
                                         cg_tol=config.cg_tol, cg_maxiter=config.cg_maxiter,
                                         eps=config.osq_eps))

    history = []
    best_val, best_epoch, best_state, best_test = -1.0, 0, None, 0.0
    epochs_ran = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = graph_classification.train_step(model, splits, optimizer, criterion)
        val_metrics = graph_classification.eval_step(model, splits, criterion, splits.val_mask)
        test_metrics = graph_classification.eval_step(model, splits, criterion, splits.test_mask)
        epochs_ran = epoch

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_task": train_metrics["task"],
            "train_sparsity": train_metrics["sparsity"], "train_osq": train_metrics["osq"],
            "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"], "test_acc": test_metrics["accuracy"],
        })

        if val_metrics["accuracy"] > best_val:
            best_val, best_epoch = val_metrics["accuracy"], epoch
            best_test = test_metrics["accuracy"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break  # early stopping on validation accuracy

    if best_state is not None:
        model.load_state_dict(best_state)

    best_record = history[best_epoch - 1] if history else {}
    samples = graph_classification.collect_graph_samples(model, splits,
                                                         config.osq_sample_graphs)
    # The whole evidential point of the phase: the same graphs measured before
    # and after the lifting. Without this pair there is no claim to make.
    osq = before_after_report(samples) if samples else {}

    row = {
        "run_id": spec.run_id, "commit_sha": config.commit_sha, "dataset": spec.dataset,
        "proxy": spec.proxy, "gamma": spec.gamma, "sparsity_weight": config.sparsity_weight,
        "seed": spec.seed, "epochs_ran": epochs_ran, "best_epoch": best_epoch,
        "train_acc": best_record.get("train_acc", ""), "val_acc": best_val,
        "test_acc": best_test, "train_loss": best_record.get("train_loss", ""),
        "task_loss": best_record.get("train_task", ""),
        "sparsity_loss": best_record.get("train_sparsity", ""),
        "osq_loss": best_record.get("train_osq", ""),
        "r_bar_before": osq.get("r_bar_before", ""), "r_bar_after": osq.get("r_bar_after", ""),
        "lambda2_before": osq.get("lambda2_before", ""),
        "lambda2_after": osq.get("lambda2_after", ""),
        "wc": osq.get("wc", ""), "nwc": osq.get("nwc", ""),
        "status": "ok", "error": "",
    }
    row.update(alpha_statistics(model, splits))
    return {"row": row, "history": history, "split_indices": split_indices, "osq": osq}


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def run_grid(config: GridConfig, store: ResultsStore, plan: Optional[List[RunSpec]] = None,
             verbose: bool = True) -> dict:
    """
    Runs the plan, storing each run as it finishes so a session that dies still
    leaves everything it had completed. Returns a small summary of what
    happened, including the runs the time budget forced out.
    """
    plan = list(plan if plan is not None else build_plan(config))
    if not config.commit_sha:
        # Every row must be attributable to a commit; if the caller forgot, dig
        # it out rather than storing results that cannot be reproduced.
        config.commit_sha = environment_info().get("commit_sha", "unknown")
    started = time.time()
    completed, failed, skipped, dropped = 0, 0, 0, []
    cache = {}

    while plan:
        spec = plan.pop(0)
        if store.has_run(spec.run_id):
            skipped += 1
            if verbose:
                print(f"[skip] {spec.run_id} (already stored)")
            continue

        run_started = time.time()
        try:
            if spec.dataset not in cache:
                cache[spec.dataset] = load_arm(spec.dataset, config)
            dataset, num_features, num_classes = cache[spec.dataset]
            result = run_single(spec, config, dataset, num_features, num_classes)
            row, history, extras = result["row"], result["history"], result

        except Exception:  # noqa: BLE001 - one bad run must not kill the grid
            error = traceback.format_exc()
            row = {"run_id": spec.run_id, "commit_sha": config.commit_sha,
                   "dataset": spec.dataset, "proxy": spec.proxy, "gamma": spec.gamma,
                   "seed": spec.seed, "sparsity_weight": config.sparsity_weight,
                   "status": "failed", "error": error.strip().splitlines()[-1][:300]}
            history, extras = [], {"error": error}
            failed += 1
        else:
            completed += 1

        row["runtime_s"] = round(time.time() - run_started, 2)
        store.append_run(row)
        store.append_epochs(spec.run_id, history)
        store.write_raw(spec.run_id, {
            "run_id": spec.run_id, "config": vars(config), "spec": vars(spec),
            "row": row, "history": history,
            "split_indices": extras.get("split_indices"),
            "osq": extras.get("osq"), "error": extras.get("error"),
        })

        if verbose:
            print(f"[{completed + failed:>4}] {spec.run_id:<44} "
                  f"{row['status']:<7} test_acc={row.get('test_acc', '')} "
                  f"({row['runtime_s']}s)")

        if config.time_budget_s is not None and plan:
            elapsed = time.time() - started
            mean_runtime = elapsed / max(completed + failed, 1)
            kept = trim_plan(plan, mean_runtime, config.time_budget_s - elapsed)
            if len(kept) != len(plan):
                dropped += [s.run_id for s in plan if s not in kept]
                if verbose:
                    print(f"  ! time budget: dropping {len(plan) - len(kept)} runs "
                          f"(mean {mean_runtime:.1f}s/run)")
                plan = kept

    return {"completed": completed, "failed": failed, "skipped": skipped,
            "dropped": dropped, "elapsed_s": round(time.time() - started, 2)}


def environment_info(repo_dir: str = ".") -> dict:
    """Versions, hardware and commit SHA - stored next to every results table."""
    import platform
    import subprocess
    import sys

    try:
        commit = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:  # noqa: BLE001 - a missing git checkout must not stop a run
        commit = "unknown"

    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit_sha": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:  # noqa: BLE001
        info["torch_geometric"] = "unknown"
    return info
