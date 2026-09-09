# The experiment grid: Tier A/B/C pipeline configs (loaded from
# configs/ via tools/config_loader.py) x {gamma=0, gamma=best} x datasets x
# seeds.
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

import math
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

import utils
from models.dmd_model import DMDModel
from tasks import graph_classification
from tools.config_loader import load_all_configs, pipeline_kwargs
from tools.gpse_cache import attach_gpse_cache
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
DATASET_PRIORITY = (SYNTHETIC_ARM, "MUTAG", "ENZYMES", "IMDB-BINARY", "PROTEINS",
                     "DHFR", "NCI1")
DROPPABLE_DATASETS = ("NCI1", "ENZYMES")

# Priority order: Tier A complete on all datasets; then Tier B on
# MUTAG/PROTEINS/synthetic_bottleneck only; then Tier C. Encoded as a trim
# order (last tier listed is dropped first) and as Tier B's dataset subset.
TIER_TRIM_ORDER = ("C", "B")
TIER_B_DATASETS = (SYNTHETIC_ARM, "MUTAG", "PROTEINS")


@dataclass
class GridConfig:
    """Every knob of the grid, in one place."""
    datasets: Sequence[str] = ("MUTAG", "PROTEINS", "ENZYMES", "NCI1", "IMDB-BINARY",
                               SYNTHETIC_ARM)
    configs_dir: str = "configs"
    # The "gamma=<best>" arm: the winning proxy/weight from feat/osq-proxy's
    # grid. gamma=0 is always crossed alongside it and never calls osq_fn at all
    # (DMDLoss's own short-circuit) - see test_gamma_zero_equivalence.
    best_proxy: str = "r_bar"
    best_gamma: float = 0.1
    # The gamma arm(s) to cross with every config. `None` keeps the historical
    # {0, best_gamma} pair; a list runs a genuine sweep.
    #
    # Why a sweep is not optional any more: feat/osq-proxy only ever tested
    # gamma in {0, 0.01}, and feat/rich-bricks then ran its whole grid at
    # gamma=0.1 - a value no experiment had evaluated - on the strength of a
    # "best gamma" that was never selected from anything. The proxy choice is
    # no better: r_bar won at p=0.011 against cf_bc_efc at p=0.031 over 18
    # pairs with no correction for having tested four proxies, while cf_bc_efc
    # was ahead on NCI1 and on the synthetic arm. Neither knob has been
    # selected on evidence yet; `gammas` is what makes that possible.
    gammas: Optional[Sequence[float]] = None
    # Proxy arm(s). `None` = `best_proxy` alone (crossed with gamma>0 only,
    # since gamma=0 never calls the proxy).
    proxies: Optional[Sequence[str]] = None
    seeds: Sequence[int] = (0, 1, 2)

    # Cross-validation. `None` = the historical single stratified train/val/test
    # split per seed. An int k = stratified k-fold, `seed` then controlling only
    # initialisation. See utils.stratified_kfold for why this is the default the
    # protocol should have had: a single 80/10/10 split gives MUTAG a 20-graph
    # test set, i.e. 5 accuracy points of resolution.
    cv_folds: Optional[int] = 10
    # Seed of the fold layout itself. Fixed across the grid on purpose: every
    # arm must be evaluated on the same folds or the paired deltas mean nothing.
    cv_seed: int = 0

    epochs: int = 200
    patience: int = 30
    lr: float = 0.005
    weight_decay: float = 5e-4
    batch_size: int = 32
    hidden_dim: int = 64
    sparsity_weight: float = 0.05
    # Max global gradient norm. Not decoration: an unclipped REINFORCE step is
    # what drove Stage 4's scores to NaN and cost the previous grid 36 runs
    # (models/weight_assignment.py). Applied to every arm so that a clipped and
    # an unclipped arm are never compared with each other.
    grad_clip: Optional[float] = 1.0

    hutch_k: int = DEFAULT_HUTCH_K
    cg_tol: float = DEFAULT_CG_TOL
    cg_maxiter: int = DEFAULT_CG_MAXITER
    osq_eps: float = DEFAULT_EPS

    train_frac: float = 0.8
    val_frac: float = 0.1
    data_root: str = "datasets"
    gpse_cache_dir: str = "datasets/gpse_cache"
    device: str = "cpu"

    # Number of test graphs the before/after OSq measurement is averaged over.
    # Exact (dense eigendecomposition), so it is a per-run cost, not a per-epoch
    # one, and a handful of graphs is enough for a mean +- std across seeds.
    osq_sample_graphs: int = 8
    time_budget_s: Optional[float] = None

    # Synthetic arm shape.
    synthetic_family: str = "tree_neighbors_match"
    synthetic_graphs: int = 600
    synthetic_classes: int = 4
    synthetic_depth: int = 3

    commit_sha: str = ""


@dataclass
class RunSpec:
    tier: str
    config_id: str
    dataset: str
    gamma: float
    seed: int
    fold: Optional[int] = None
    proxy: Optional[str] = None

    @property
    def run_id(self) -> str:
        return make_run_id(self.tier, self.config_id, self.dataset, self.gamma, self.seed,
                           fold=self.fold, proxy=self.proxy)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _sorted_datasets(datasets: Sequence[str]) -> List[str]:
    return sorted(datasets, key=lambda d: DATASET_PRIORITY.index(d) if d in DATASET_PRIORITY else 99)


def build_plan(config: GridConfig) -> List[RunSpec]:
    """
    Every Tier A/B/C config (loaded from `config.configs_dir`, never
    hand-copied - see tools/config_loader.py) crossed with gamma in
    {0, best_gamma}, dataset, and seed. `config_id="R"` (the standalone
    reference file) is excluded - `A0` already IS `R`, run as part of Tier A.

    Ordered Tier A (all datasets) -> Tier B (3 datasets) -> Tier C (all
    datasets), so a plan truncated by the runtime guard degrades in exactly
    the intended priority order.
    """
    all_configs = load_all_configs(config.configs_dir)
    by_tier = {"A": [], "B": [], "C": []}
    for config_id, pconfig in all_configs.items():
        if pconfig["tier"] in by_tier:
            by_tier[pconfig["tier"]].append(pconfig)
    for tier in by_tier:
        by_tier[tier].sort(key=lambda c: c["config_id"])

    datasets_all = _sorted_datasets(config.datasets)
    datasets_b = _sorted_datasets([d for d in TIER_B_DATASETS if d in config.datasets])

    if config.gammas is not None:
        gammas = tuple(config.gammas)
    else:
        gammas = (0.0, config.best_gamma) if config.best_gamma != 0.0 else (0.0,)
    # `None` rather than `(config.best_proxy,)`: leaving RunSpec.proxy unset
    # keeps run_ids byte-identical to the single-proxy design (run_single falls
    # back to `config.best_proxy`), so a results tree from before the sweep
    # still resumes and every stored figure still resolves its rows.
    proxies = tuple(config.proxies) if config.proxies is not None else (None,)
    folds = range(config.cv_folds) if config.cv_folds else (None,)

    plan: List[RunSpec] = []
    for tier, datasets in (("A", datasets_all), ("B", datasets_b), ("C", datasets_all)):
        for pconfig in by_tier[tier]:
            for dataset in datasets:
                for gamma in gammas:
                    # gamma=0 short-circuits osq_fn entirely (DMDLoss), so the
                    # proxies are indistinguishable there - running one arm per
                    # proxy at gamma=0 would be the same run stored under
                    # several names.
                    for proxy in ((None,) if gamma == 0.0 else proxies):
                        for seed in config.seeds:
                            for fold in folds:
                                plan.append(RunSpec(tier, pconfig["config_id"], dataset,
                                                    gamma, seed, fold=fold, proxy=proxy))
    return plan


def trim_plan(remaining: Sequence[RunSpec], projected_seconds_per_run: float,
              seconds_left: float) -> List[RunSpec]:
    """
    Drops runs, in the agreed order, until the plan fits in the time left:
    whole tiers first (Tier C, then Tier B - Tier A is never dropped as a
    tier), then within what's left, largest gamma first, then the two
    expensive real datasets. Never the synthetic arm, never a seed - three
    seeds is the minimum for a mean +- std, and a single-seed number is not
    reportable. Never a gamma=0 twin without also dropping its gamma>0 pair.

    Returns the kept runs; the caller reports what was dropped.
    """
    kept = list(remaining)
    if projected_seconds_per_run <= 0:
        return kept

    def fits():
        return len(kept) * projected_seconds_per_run <= seconds_left

    for tier in TIER_TRIM_ORDER:
        if fits():
            return kept
        if any(s.tier == tier for s in kept):
            kept = [s for s in kept if s.tier != tier]

    while not fits() and kept:
        gammas = sorted({s.gamma for s in kept if s.gamma > 0}, reverse=True)
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


def load_arm(name: str, config: GridConfig, needs_gpse: bool = False):
    """
    Returns (dataset, num_features, num_classes, s1_encoder_actual) for one
    arm of the grid. `needs_gpse` triggers the one-per-dataset GPSE
    precomputation (`tools/gpse_cache.py::attach_gpse_cache`) - only paid by
    datasets that actually get run through an `s1='gpse'` config, and only
    once (the cache is keyed by dataset name under `config.gpse_cache_dir`).
    """
    if name == SYNTHETIC_ARM:
        dataset = bottleneck_dataset(config.synthetic_family,
                                     num_graphs=config.synthetic_graphs,
                                     num_classes=config.synthetic_classes,
                                     depth=config.synthetic_depth, seed=0)
    else:
        from torch_geometric.datasets import TUDataset

        dataset = ensure_node_features(TUDataset(root=config.data_root, name=name))

    s1_encoder_actual = None
    if needs_gpse:
        cache_path = f"{config.gpse_cache_dir}/{name}.gpse_cache.pt"
        s1_encoder_actual = attach_gpse_cache(dataset, cache_path=cache_path)

    return dataset, dataset.num_node_features, dataset.num_classes, s1_encoder_actual


def _labels_of(dataset) -> torch.Tensor:
    y = dataset.y
    return torch.as_tensor(y).view(-1)


def make_splits(dataset, config: GridConfig, seed: int, fold: Optional[int] = None):
    """
    The run's train/val/test loaders plus the index lists, so the exact split is
    stored in the run's raw file.

    Two regimes. With `config.cv_folds` set (the default) and a `fold` given,
    this is stratified k-fold: the fold picks the split, `seed` only picks the
    initialisation, and the fold layout is shared across every arm so paired
    comparisons are on literally the same graphs. Without them it falls back to
    the historical single stratified `train_frac`/`val_frac` split seeded by
    `seed` - kept so an old plan reproduces exactly.

    The k-fold layout deliberately does NOT depend on `seed`: if it did, two
    arms differing only in seed would be measured on different test sets and
    the pairing that every delta in the analysis relies on would be gone.
    """
    from torch_geometric.loader import DataLoader

    if config.cv_folds and fold is not None:
        folds = utils.stratified_kfold(_labels_of(dataset), n_splits=config.cv_folds,
                                       seed=config.cv_seed)
        train_idx, val_idx, test_idx = folds[fold]
    else:
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
    """
    Acceptance-weight statistics over the WHOLE test split, plus the collapse
    diagnostic.

    Collapse is the failure mode this project has to be able to see. When every
    alpha is 0 the lifting adds only zero-weight edges, so the rewired structure
    is the original graph, `motif_embeddings` is identically zero, and the model
    silently degenerates to a plain GCN. It still trains, still reports an
    accuracy, and still gets averaged into a brick's mean - it just is not the
    model whose name is on the row. In the feat/rich-bricks grid this happened in
    63 of 154 usable runs (41%), including all 15 of the GPSE arm's gamma=0 runs,
    and nothing in the schema recorded it: it had to be reverse-engineered from
    `r_bar_before == r_bar_after`. Hence `collapsed` and `alpha_frac_active` as
    first-class columns.

    Measured over every test batch rather than only the first one, which is what
    the previous version did - a per-batch verdict on a 32-graph sample is not a
    verdict on the run.
    """
    model.eval()
    device = next(model.parameters()).device
    total, sum_alpha, sum_sq, active, slots = 0, 0.0, 0.0, 0, 0

    for batch in splits.test_mask:
        batch = batch.to(device)
        _, structure = model(batch.x, batch.edge_index, batch=batch.batch,
                              node_pe=getattr(batch, "pestat_GPSE", None))
        alpha = structure["alpha"]
        if alpha.numel() == 0:
            continue
        total += int(alpha.numel())
        sum_alpha += float(alpha.sum().item())
        sum_sq += float((alpha ** 2).sum().item())
        active += int((alpha > 0).sum().item())
        slots += int(structure["candidates"].node_index.numel())

    if total == 0:
        return {"alpha_mean": 0.0, "alpha_std": 0.0, "num_cells": 0, "mean_cell_size": 0.0,
                "alpha_frac_active": 0.0, "collapsed": True}

    mean = sum_alpha / total
    variance = max(sum_sq / total - mean ** 2, 0.0)
    frac_active = active / total
    return {"alpha_mean": mean, "alpha_std": variance ** 0.5,
            "num_cells": total, "mean_cell_size": slots / total,
            "alpha_frac_active": frac_active,
            # Not `mean == 0`: a run that accepts a handful of cells out of
            # thousands is a collapse in every way that matters to the claim,
            # and a soft selector makes exact zeros rare. 1% is the threshold.
            "collapsed": frac_active < 0.01}


def run_single(spec: RunSpec, config: GridConfig, pipeline_config: dict, dataset, num_features: int,
               num_classes: int) -> dict:
    """
    Trains one configuration and returns (row, history, extras). Raises on
    failure - the caller decides what to do with the traceback.
    """
    utils.set_seed(spec.seed)
    splits, split_indices = make_splits(dataset, config, spec.seed, fold=spec.fold)

    proxy = (spec.proxy or config.best_proxy) if spec.gamma > 0 else "none"

    # One width knob: latent and motif dimensions follow `hidden_dim`. The grid
    # varies the objective and the pipeline stages, not the base width - a run
    # that differs in both is not an ablation of either.
    model = DMDModel(input_dim=num_features, hidden_dim=config.hidden_dim,
                     latent_dim=config.hidden_dim, motif_hidden_dim=config.hidden_dim,
                     motif_out_dim=config.hidden_dim, num_classes=num_classes,
                     **pipeline_kwargs(pipeline_config)).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    reinforce_module = model.selector if pipeline_config["s4"] == "reinforce" else None
    criterion = DMDLoss(sparsity_weight=config.sparsity_weight, osq_weight=spec.gamma,
                        osq_fn=get_proxy(proxy, hutch_k=config.hutch_k,
                                         cg_tol=config.cg_tol, cg_maxiter=config.cg_maxiter,
                                         eps=config.osq_eps),
                        reinforce_module=reinforce_module)

    history = []
    best_val, best_epoch, best_state, best_test = -1.0, 0, None, 0.0
    epochs_ran = 0

    # Stage 4's k-subset brick documents a temperature annealed towards
    # `tau_min` over training and states that "the grid runner calls this once
    # per epoch". It did not: `set_temperature` had no caller anywhere in the
    # repo, so A6 ran its entire arm at the fixed initial tau. Honoured here,
    # with the standard exponential Gumbel schedule (Jang et al. 2017),
    # `set_temperature` applying the tau_min floor itself.
    #
    # Note for whoever sets `s4_kwargs`: at the shipped default anneal_rate=1e-4
    # this schedule is nearly flat over 200 epochs (tau 1.0 -> 0.98), so A6's
    # temperature is still effectively fixed. That is a choice for the config to
    # make deliberately, not something to bury in a formula here - raise
    # `anneal_rate` if the arm is meant to actually anneal.
    anneal = getattr(model.selector, "set_temperature", None)
    tau_init = getattr(model.selector, "tau", None)
    anneal_rate = getattr(model.selector, "anneal_rate", None)
    can_anneal = anneal is not None and tau_init is not None and anneal_rate is not None

    for epoch in range(1, config.epochs + 1):
        if can_anneal:
            anneal(tau_init * math.exp(-anneal_rate * epoch))
        train_metrics = graph_classification.train_step(model, splits, optimizer, criterion,
                                                        grad_clip=config.grad_clip)
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
        "run_id": spec.run_id, "commit_sha": config.commit_sha,
        "tier": spec.tier, "config_id": spec.config_id, "dataset": spec.dataset,
        "seed": spec.seed, "fold": "" if spec.fold is None else spec.fold,
        "n_train": len(split_indices["train"]), "n_val": len(split_indices["val"]),
        "n_test": len(split_indices["test"]),
        "grad_clip": "" if config.grad_clip is None else config.grad_clip,
        "s1_encoder": pipeline_config["s1"],
        "s1_encoder_actual": getattr(model.embedder, "actual_encoder", pipeline_config["s1"]),
        "s2_proposal": pipeline_config["s2"], "s3_cell_encoder": pipeline_config["s3"],
        "s4_selector": pipeline_config["s4"], "s5_mp": pipeline_config["s5"],
        "osq_proxy": proxy, "gamma": spec.gamma, "sparsity_weight": config.sparsity_weight,
        "epochs_ran": epochs_ran, "best_epoch": best_epoch,
        "train_acc": best_record.get("train_acc", ""), "val_acc": best_val,
        "test_acc": best_test, "train_loss": best_record.get("train_loss", ""),
        "task_loss": best_record.get("train_task", ""),
        "sparsity_loss": best_record.get("train_sparsity", ""),
        "osq_loss": best_record.get("train_osq", ""),
        "r_bar_before": osq.get("r_bar_before", ""), "r_bar_after": osq.get("r_bar_after", ""),
        "lambda2_before": osq.get("lambda2_before", ""),
        "lambda2_after": osq.get("lambda2_after", ""),
        "wc": osq.get("wc", ""), "nwc": osq.get("nwc", ""),
        "params_count": sum(p.numel() for p in model.parameters()),
        "peak_mem_mb": _peak_memory_mb(config.device),
        "status": "ok", "error": "",
    }
    row.update(alpha_statistics(model, splits))
    return {"row": row, "history": history, "split_indices": split_indices, "osq": osq}


def _peak_memory_mb(device: str) -> float:
    """Peak memory since the last reset - CUDA's own counter on GPU, RSS on CPU
    (a coarser, process-wide proxy, but the only cheap option without CUDA)."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # KB -> MB on Linux
    except ImportError:  # noqa: BLE001 - no `resource` module on Windows
        return 0.0


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
    all_configs = load_all_configs(config.configs_dir)
    if not config.commit_sha:
        # Every row must be attributable to a commit; if the caller forgot, dig
        # it out rather than storing results that cannot be reproduced.
        config.commit_sha = environment_info().get("commit_sha", "unknown")
    started = time.time()
    if SYNTHETIC_ARM not in config.datasets and verbose:
        # The last grid's `datasets` list silently lacked this arm - the notebook
        # in the repo listed it, the config actually executed did not, and the
        # omission was only discoverable months later by reading the config
        # embedded in a raw result file. It is the only arm where an
        # OSq-guided lifting is *supposed* to win, so a grid without it cannot
        # falsify anything, only fail to reject.
        print(f"WARNING: {SYNTHETIC_ARM!r} is not in this grid's datasets. It is the "
              "falsification core - without it the grid can only show that the lifting "
              "costs nothing, never that its mechanism works.")
    if config.time_budget_s is not None and verbose:
        # The guard that silently deleted two thirds of the last grid. It
        # projects with a single global mean seconds-per-run, so once a few NCI1
        # runs (416 s) had landed among MUTAG runs (15 s) it concluded the plan
        # would not fit, dropped Tier C, Tier B, NCI1 and ENZYMES, and never
        # revised that downwards - the session then finished in 5.1 h of an 8 h
        # budget. It is left in place for a sequential session that genuinely has
        # a hard cutoff, but it is opt-in and it announces itself, because a plan
        # trimmed in silence is indistinguishable from a plan that was never
        # written. The Modal runner never sets it: one container per run means
        # there is no shared wall-clock to protect.
        print(f"WARNING: time_budget_s={config.time_budget_s}s is set. Runs WILL be dropped "
              "from this plan if the projection says they do not fit, and the projection is "
              "a single global mean over datasets whose costs differ by 30x. "
              "Leave it None unless this session has a hard cutoff.")
    completed, failed, skipped, dropped = 0, 0, 0, []
    cache = {}
    gpse_config_ids = {cid for cid, c in all_configs.items() if c["s1"] == "gpse"}

    while plan:
        spec = plan.pop(0)
        if store.has_run(spec.run_id):
            skipped += 1
            if verbose:
                print(f"[skip] {spec.run_id} (already stored)")
            continue

        run_started = time.time()
        try:
            pipeline_config = all_configs[spec.config_id]
            cache_key = (spec.dataset, spec.config_id in gpse_config_ids)
            if cache_key not in cache:
                cache[cache_key] = load_arm(spec.dataset, config, needs_gpse=cache_key[1])
            dataset, num_features, num_classes, _ = cache[cache_key]
            result = run_single(spec, config, pipeline_config, dataset, num_features, num_classes)
            row, history, extras = result["row"], result["history"], result

        except Exception:  # noqa: BLE001 - one bad run must not kill the grid
            error = traceback.format_exc()
            row = {"run_id": spec.run_id, "commit_sha": config.commit_sha,
                   "tier": spec.tier, "config_id": spec.config_id, "dataset": spec.dataset,
                   "gamma": spec.gamma, "seed": spec.seed,
                   "fold": "" if spec.fold is None else spec.fold,
                   "osq_proxy": (spec.proxy or config.best_proxy) if spec.gamma > 0 else "none",
                   "sparsity_weight": config.sparsity_weight,
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
