# Analyzes the feat/osq-proxy experiment grid saved by tools/results_store.py
# (results/runs.csv, results/epochs.csv) - one row per run and one row per
# (run_id, epoch) respectively. Adapted from results/analyze_grid.py (the
# feat/rich-bricks analysis) to this branch's simpler, DIFFERENT schema:
# no tier/config_id, no per-stage brick columns, no peak_mem_mb/params_count
# - the comparison axis here is `proxy` (none/r_bar/lambda2/efc/cf_bc_efc),
# and each proxy runs at exactly one gamma (none@0.0, everything else@0.01;
# see CLAUDE.md Sec9), so there is no per-proxy "OSq on/off" split to plot -
# that split lives entirely between "none" and every other proxy.
#
# Produces, under --out-dir (default results/osq_proxy/figures):
#   status_report.csv               ok/failed run counts per proxy, + a sample error
#   accuracy_by_proxy.csv/.png       mean +/- std test accuracy per proxy, pooled
#                                    over every dataset x seed
#   accuracy_on_synthetic_bottleneck.csv/.png
#       same, restricted to synthetic_bottleneck - CLAUDE.md Sec9 calls this
#       dataset "the scientific core": it's the one built so an OSq-guided
#       lifting *should* win, so it gets its own figure rather than being
#       averaged away with the 5 real benchmark datasets.
#   runtime_by_dataset.csv/.png      mean +/- std wall-clock runtime per dataset
#   curves/<run_id>_loss.png, <run_id>_accuracy.png
#       train/val/test loss and accuracy vs. epoch, for one representative
#       (highest val_acc) run per proxy, or for --run-id if given.
#
# Every plot is meant to stand on its own if printed out of context: proxy
# names are expanded with a one-line description (PROXY_LABELS), sample
# sizes (n=...) and any proxy with zero successful runs are annotated ON the
# figure, not left to a caption.
#
# Usage:
#   python results/analyze_osq_proxy.py [--results-dir results/osq_proxy/results] [--out-dir results/osq_proxy/figures]

import argparse
import csv
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fixed categorical colors as analyze_grid.py so figures read the same
# way across both branches' analyses.
COLORS = {
    "total": "#4C72B0",
    "sparsity": "#55A868",
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
}

# CLAUDE.md Sec5's proxy registry table, condensed to one phrase each - turns
# an opaque key like "cf_bc_efc" into something a reader can parse without
# opening the registry.
PROXY_LABELS = {
    "none": "no OSq term, baseline",
    "r_bar": "mean effective resistance, primary",
    "lambda2": "spectral gap, guard-rail",
    "efc": "edge Forman curvature",
    "cf_bc_efc": "current-flow-weighted EFC",
}

NUMERIC_RUN_COLUMNS = [
    "seed", "gamma", "sparsity_weight", "epochs_ran", "best_epoch",
    "train_acc", "val_acc", "test_acc", "train_loss", "task_loss",
    "sparsity_loss", "osq_loss", "alpha_mean", "alpha_std", "num_cells",
    "r_bar_before", "r_bar_after", "lambda2_before", "lambda2_after",
    "wc", "nwc", "runtime_s",
]

NUMERIC_EPOCH_COLUMNS = [
    "epoch", "train_loss", "train_task", "train_sparsity", "train_osq",
    "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
]

SYNTHETIC_DATASET = "synthetic_bottleneck"


def _coerce(row: dict, numeric_columns) -> dict:
    """csv.DictReader gives back strings; turn the numeric columns into
    floats and blank cells into None, so aggregation never has to special
    case '' again."""
    out = dict(row)
    for col in numeric_columns:
        raw = out.get(col, "")
        out[col] = float(raw) if raw not in ("", None) else None
    return out


def load_runs(results_dir) -> list:
    path = Path(results_dir) / "runs.csv"
    with open(path, newline="") as f:
        return [_coerce(row, NUMERIC_RUN_COLUMNS) for row in csv.DictReader(f)]


def load_epochs(results_dir) -> list:
    path = Path(results_dir) / "epochs.csv"
    with open(path, newline="") as f:
        return [_coerce(row, NUMERIC_EPOCH_COLUMNS) for row in csv.DictReader(f)]


def _mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def status_report(runs) -> list:
    """One row per proxy: how many runs succeeded/failed, and the first
    error seen (truncated). This grid ran 100% ok as of writing, but the
    check is cheap insurance against a future re-run that doesn't."""
    by_proxy = defaultdict(lambda: {"ok": 0, "failed": 0, "sample_error": ""})
    for row in runs:
        bucket = by_proxy[row["proxy"]]
        if row["status"] == "ok":
            bucket["ok"] += 1
        else:
            bucket["failed"] += 1
            if not bucket["sample_error"] and row.get("error"):
                bucket["sample_error"] = row["error"].splitlines()[0][:200]
    return [{"proxy": proxy, **stats} for proxy, stats in sorted(by_proxy.items())]


def summarize_by_proxy(runs, metric: str = "test_acc", dataset: "str | None" = None) -> list:
    """Mean +/- std of `metric` per proxy, over successful runs only.
    Pass `dataset` to restrict to one dataset (used for the
    synthetic_bottleneck-only figure); leave it None to pool every dataset."""
    by_proxy = defaultdict(list)
    for row in runs:
        if row["status"] == "ok" and (dataset is None or row["dataset"] == dataset):
            by_proxy[row["proxy"]].append(row[metric])
    out = []
    for proxy, values in sorted(by_proxy.items()):
        mean, std = _mean_std(values)
        out.append({"proxy": proxy, "n": len(values), "mean": mean, "std": std})
    return out


def runtime_by_dataset(runs) -> list:
    """Mean +/- std wall-clock runtime per dataset, over successful runs."""
    runtime = defaultdict(list)
    for row in runs:
        if row["status"] == "ok":
            runtime[row["dataset"]].append(row["runtime_s"])
    out = []
    for dataset in sorted(runtime):
        rt_mean, rt_std = _mean_std(runtime[dataset])
        out.append({"dataset": dataset, "n": len(runtime[dataset]),
                     "runtime_s_mean": rt_mean, "runtime_s_std": rt_std})
    return out


def best_run_per_proxy(runs) -> dict:
    """The highest-val_acc successful run for each proxy - used as the one
    representative run whose train/val/test curves get plotted."""
    best = {}
    for row in runs:
        if row["status"] != "ok":
            continue
        proxy = row["proxy"]
        if proxy not in best or (row["val_acc"] or -1.0) > (best[proxy]["val_acc"] or -1.0):
            best[proxy] = row
    return best


def _xtick_label(proxy: str) -> str:
    label = PROXY_LABELS.get(proxy)
    if not label:
        return proxy
    wrapped = "\n".join(textwrap.wrap(label, width=16))
    return f"{proxy}\n({wrapped})"


def _missing_proxies_note(all_proxies, present_proxies) -> "str | None":
    """A one-line footnote for proxies that exist in the grid but have zero
    successful runs, so their absence from a plot reads as documented
    rather than as a rendering bug."""
    if not all_proxies:
        return None
    missing = [p for p in sorted(all_proxies) if p not in present_proxies]
    if not missing:
        return None
    return f"Not shown: {', '.join(missing)} (0 successful runs - see status_report.csv)"


def write_csv(rows, path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _annotate_n(ax, x_positions, tops, ns) -> None:
    for x, top, n in zip(x_positions, tops, ns):
        if top is None or n is None:
            continue
        ax.annotate(f"n={n}", (x, top), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, color="dimgray")


def plot_accuracy_by_proxy(summary_rows, out_path, metric_label: str = "Test accuracy",
                            all_proxies=None, title_suffix: str = "pooled over every dataset x seed") -> None:
    x = list(range(len(summary_rows)))
    means = [r["mean"] for r in summary_rows]
    stds = [r["std"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x, [m if m is not None else 0.0 for m in means],
           yerr=[s if s is not None else 0.0 for s in stds], capsize=4, color=COLORS["total"])
    _annotate_n(ax, x, [(m or 0) + (s or 0) for m, s in zip(means, stds)], [r["n"] for r in summary_rows])

    ax.set_xticks(x)
    ax.set_xticklabels([_xtick_label(r["proxy"]) for r in summary_rows], fontsize=8)
    ax.set_xlabel("OSq proxy used as the loss term")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.08)

    title = f"{metric_label} by OSq proxy\n({title_suffix}; n = successful runs per bar)"
    note = _missing_proxies_note(all_proxies, {r["proxy"] for r in summary_rows})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime_by_dataset(rows, out_path) -> None:
    x = list(range(len(rows)))
    means = [r["runtime_s_mean"] for r in rows]
    stds = [r["runtime_s_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, [m if m is not None else 0.0 for m in means],
           yerr=[s if s is not None else 0.0 for s in stds], capsize=4, color=COLORS["sparsity"])
    _annotate_n(ax, x, [(m or 0) + (s or 0) for m, s in zip(means, stds)], [r["n"] for r in rows])

    labels = [f"{r['dataset']}\n(diagnostic)" if r["dataset"] == SYNTHETIC_DATASET else r["dataset"]
              for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Wall-clock training runtime (s)")
    ax.set_title("Mean training runtime by dataset\n"
                  "(mean +/- std across proxy x seed; n = successful runs)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_title(run_row: dict) -> str:
    seed = run_row["seed"]
    proxy = run_row["proxy"]
    line1 = (f"{run_row['dataset']} | proxy={proxy} | gamma={run_row['gamma']} | "
             f"seed={int(seed) if seed is not None else seed}")
    description = PROXY_LABELS.get(proxy)
    return f"{line1}\n({description})" if description else line1


def plot_loss_curves(history, run_row, out_path) -> None:
    """Train/val/test *task* (cross-entropy) loss vs. epoch for one run -
    the standard verification plot: val/test loss diverging upward away
    from train loss is overfitting; all three staying high is
    underfitting. Task loss only - excludes the sparsity and OSq loss
    terms, which is why it can differ from the logged total train_loss."""
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, [r["train_task"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_loss"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_loss"] for r in history], label="Test", color=COLORS["test"], linewidth=2)
    ax.set_title(f"Task (cross-entropy) loss: train vs. validation vs. test\n{_run_title(run_row)}", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss (task term only)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_curves(history, run_row, out_path) -> None:
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, [r["train_acc"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_acc"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_acc"] for r in history], label="Test", color=COLORS["test"], linewidth=2)

    best_epoch = run_row.get("best_epoch")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"Best val @ epoch {int(best_epoch)}")

    ax.set_title(f"Accuracy over training\n{_run_title(run_row)}", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze the feat/osq-proxy experiment grid.")
    parser.add_argument("--results-dir", type=str, default="results/osq_proxy/results",
                         help="Directory holding runs.csv/epochs.csv (default: results/osq_proxy/results)")
    parser.add_argument("--out-dir", type=str, default="results/osq_proxy/figures",
                         help="Where to write summary tables and plots (default: results/osq_proxy/figures)")
    parser.add_argument("--run-id", action="append", default=None,
                         help="Plot train/val/test curves for this run_id (repeatable). "
                              "Defaults to the best (highest val_acc) run of each proxy.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    all_proxies = {row["proxy"] for row in runs}

    status = status_report(runs)
    write_csv(status, out_dir / "status_report.csv")

    accuracy = summarize_by_proxy(runs, "test_acc")
    write_csv(accuracy, out_dir / "accuracy_by_proxy.csv")
    plot_accuracy_by_proxy(accuracy, out_dir / "accuracy_by_proxy.png", all_proxies=all_proxies)

    synthetic_accuracy = summarize_by_proxy(runs, "test_acc", dataset=SYNTHETIC_DATASET)
    write_csv(synthetic_accuracy, out_dir / "accuracy_on_synthetic_bottleneck.csv")
    plot_accuracy_by_proxy(
        synthetic_accuracy, out_dir / "accuracy_on_synthetic_bottleneck.png",
        all_proxies=all_proxies,
        title_suffix=f"{SYNTHETIC_DATASET} only - the diagnostic arm OSq-lifting should win on (CLAUDE.md Sec9)",
    )

    runtime = runtime_by_dataset(runs)
    write_csv(runtime, out_dir / "runtime_by_dataset.csv")
    plot_runtime_by_dataset(runtime, out_dir / "runtime_by_dataset.png")

    epochs = load_epochs(args.results_dir)
    epochs_by_run = defaultdict(list)
    for row in epochs:
        epochs_by_run[row["run_id"]].append(row)

    runs_by_id = {row["run_id"]: row for row in runs}
    if args.run_id:
        run_ids = args.run_id
    else:
        run_ids = [row["run_id"] for row in best_run_per_proxy(runs).values()]

    curves_dir = out_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    for run_id in run_ids:
        if run_id not in epochs_by_run:
            print(f"skip {run_id}: no epoch history found (failed run?)")
            continue
        history = sorted(epochs_by_run[run_id], key=lambda r: r["epoch"])
        run_row = runs_by_id[run_id]
        plot_loss_curves(history, run_row, curves_dir / f"{run_id}_loss.png")
        plot_accuracy_curves(history, run_row, curves_dir / f"{run_id}_accuracy.png")

    print(f"Wrote summary tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
