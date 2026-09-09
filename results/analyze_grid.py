# Analyzes the Tier A/B/C experiment grid saved by tools/results_store.py
# (results/runs.csv, results/epochs.csv) - one row per run and one row per
# (run_id, epoch) respectively. This is distinct from analyze_results.py,
# which analyzes a single training run's saved history.jsonl/config.json.
#
# Produces, under --out-dir (default results/rich_bricks/figures):
#   status_report.csv           ok/failed run counts per config_id, + a sample error
#   accuracy_by_config.csv/.png mean +/- std test accuracy per config_id
#   runtime_by_dataset.csv/.png mean +/- std wall-clock runtime and peak memory per dataset
#   osq_effect_on_accuracy.csv/.png  mean test accuracy with the OSq term off (gamma=0)
#                                    vs on (gamma>0), per config_id
#   curves/<run_id>_loss.png, <run_id>_accuracy.png
#       train/val/test loss and accuracy vs. epoch, for one representative
#       (highest val_acc) run per config_id, or for --run-id if given.
#
# Every plot is meant to stand on its own if printed out of context: the
# config_id -> brick-changed mapping, sample sizes (n=...) and any config
# with zero successful runs are all annotated ON the figure, not left to a
# caption. See brick_labels()/_xtick_label()/_missing_configs_note().
#
# Usage:
#   python results/analyze_grid.py [--results-dir results/rich_bricks/results] [--out-dir results/rich_bricks/figures]

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same fixed categorical colors as analyze_results.py so curves read the same
# way across both scripts.
COLORS = {
    "total": "#4C72B0",
    "task": "#DD8452",
    "sparsity": "#55A868",
    "osq": "#C44E52",
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
}

NUMERIC_RUN_COLUMNS = [
    "seed", "gamma", "sparsity_weight", "epochs_ran", "best_epoch",
    "train_acc", "val_acc", "test_acc", "train_loss", "task_loss",
    "sparsity_loss", "osq_loss", "alpha_mean", "alpha_std", "num_cells",
    "mean_cell_size", "r_bar_before", "r_bar_after", "lambda2_before",
    "lambda2_after", "wc", "nwc", "params_count", "runtime_s", "peak_mem_mb",
    # Appended by fix/experiment-protocol. `fold` stays a STRING on purpose:
    # it is a pairing key, and an empty fold (pre-cross-validation rows) must
    # compare equal to itself rather than becoming None.
    "n_train", "n_val", "n_test", "alpha_frac_active",
]

NUMERIC_EPOCH_COLUMNS = [
    "epoch", "train_loss", "train_task", "train_sparsity", "train_osq",
    "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
]

# (runs.csv column, short stage name) - the five brick slots every config
# fills in, used to say *what* differs from the baseline instead of just
# printing an opaque config_id like "A4" on an axis.
STAGE_COLUMNS = [
    ("s1_encoder", "s1"), ("s2_proposal", "s2"), ("s3_cell_encoder", "s3"),
    ("s4_selector", "s4"), ("s5_mp", "s5"),
]


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
    """One row per config_id: how many runs succeeded/failed, and the first
    error seen (truncated) - so a whole-config failure (e.g. a CUDA assert
    that kills every seed) is visible at a glance instead of buried in raw/."""
    by_config = defaultdict(lambda: {"ok": 0, "failed": 0, "sample_error": ""})
    for row in runs:
        bucket = by_config[row["config_id"]]
        if row["status"] == "ok":
            bucket["ok"] += 1
        else:
            bucket["failed"] += 1
            if not bucket["sample_error"] and row.get("error"):
                bucket["sample_error"] = row["error"].splitlines()[0][:200]
    return [{"config_id": cfg, **stats} for cfg, stats in sorted(by_config.items())]


def summarize_by_config(runs, metric: str = "test_acc") -> list:
    """Mean +/- std of `metric` per config_id, over successful runs only."""
    by_config = defaultdict(list)
    for row in runs:
        if row["status"] == "ok":
            by_config[row["config_id"]].append(row[metric])
    out = []
    for cfg, values in sorted(by_config.items()):
        mean, std = _mean_std(values)
        out.append({"config_id": cfg, "n": len(values), "mean": mean, "std": std})
    return out


def runtime_by_dataset(runs) -> list:
    """Mean +/- std wall-clock runtime and peak memory per dataset, over
    successful runs - the "average compute time per dataset" view."""
    runtime = defaultdict(list)
    memory = defaultdict(list)
    for row in runs:
        if row["status"] == "ok":
            runtime[row["dataset"]].append(row["runtime_s"])
            memory[row["dataset"]].append(row["peak_mem_mb"])
    out = []
    for dataset in sorted(runtime):
        rt_mean, rt_std = _mean_std(runtime[dataset])
        mem_mean, mem_std = _mean_std(memory[dataset])
        out.append({
            "dataset": dataset, "n": len(runtime[dataset]),
            "runtime_s_mean": rt_mean, "runtime_s_std": rt_std,
            "peak_mem_mb_mean": mem_mean, "peak_mem_mb_std": mem_std,
        })
    return out


def osq_effect(runs, metric: str = "test_acc") -> list:
    """
    Mean `metric` with the OSq term off (gamma=0.0) vs on (the largest
    gamma>0 present in the grid), per config_id - the main question this
    branch exists to answer: does the lifting change the metric, at all.
    Sample sizes are tracked on each side separately since
    a failed seed can make them unequal.
    """
    by_key = defaultdict(list)
    for row in runs:
        if row["status"] == "ok" and row["gamma"] is not None:
            by_key[(row["config_id"], row["gamma"])].append(row[metric])

    gammas_on = sorted({gamma for (_, gamma) in by_key if gamma > 0})
    gamma_on = gammas_on[-1] if gammas_on else None

    out = []
    for cfg in sorted({config_id for (config_id, _) in by_key}):
        off_values = by_key.get((cfg, 0.0), [])
        on_values = by_key.get((cfg, gamma_on), []) if gamma_on is not None else []
        off_mean, off_std = _mean_std(off_values)
        on_mean, on_std = _mean_std(on_values)
        delta = (on_mean - off_mean) if (on_mean is not None and off_mean is not None) else None
        out.append({
            "config_id": cfg,
            "gamma_off_mean": off_mean, "gamma_off_std": off_std, "gamma_off_n": len(off_values),
            "gamma_on": gamma_on, "gamma_on_mean": on_mean, "gamma_on_std": on_std, "gamma_on_n": len(on_values),
            "delta": delta,
        })
    return out


def best_run_per_config(runs) -> dict:
    """The highest-val_acc successful run for each config_id - used as the
    one representative run whose train/val/test curves get plotted."""
    best = {}
    for row in runs:
        if row["status"] != "ok":
            continue
        cfg = row["config_id"]
        if cfg not in best or (row["val_acc"] or -1.0) > (best[cfg]["val_acc"] or -1.0):
            best[cfg] = row
    return best


def brick_labels(runs, baseline_config_id: str = "A0") -> dict:
    """
    {config_id: "set_transformer"} naming the one pipeline stage that
    differs from `baseline_config_id`, read straight off any one successful
    run of that config (s1_encoder/s2_proposal/.../s5_mp) - so a plot never
    has to rely on the reader knowing that "A4" means stage 3 got a set
    transformer. A config with zero successful runs (e.g. a totally failed
    one) gets no entry here; callers fall back to the bare config_id.
    """
    representative = {}
    for row in runs:
        if row["status"] == "ok" and row["config_id"] not in representative:
            representative[row["config_id"]] = row

    baseline = representative.get(baseline_config_id)
    labels = {}
    for config_id, row in representative.items():
        if config_id == baseline_config_id:
            labels[config_id] = "baseline"
            continue
        if baseline is None:
            continue
        changed = [row[column] for column, _short in STAGE_COLUMNS if row[column] != baseline[column]]
        labels[config_id] = "+".join(changed) if changed else "identical to baseline"
    return labels


def _xtick_label(config_id: str, briefs) -> str:
    brief = (briefs or {}).get(config_id)
    return f"{config_id}\n({brief})" if brief else config_id


def _missing_configs_note(all_config_ids, present_config_ids) -> "str | None":
    """A one-line footnote for configs that exist in the grid but have zero
    successful runs, so their absence from a per-config plot reads as
    documented rather than as a rendering bug."""
    if not all_config_ids:
        return None
    missing = [cfg for cfg in sorted(all_config_ids) if cfg not in present_config_ids]
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


def plot_accuracy_by_config(summary_rows, out_path, metric_label: str = "Test accuracy",
                             config_briefs=None, all_config_ids=None) -> None:
    x = list(range(len(summary_rows)))
    means = [r["mean"] for r in summary_rows]
    stds = [r["std"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, [m if m is not None else 0.0 for m in means],
           yerr=[s if s is not None else 0.0 for s in stds], capsize=4, color=COLORS["total"])
    _annotate_n(ax, x, [(m or 0) + (s or 0) for m, s in zip(means, stds)], [r["n"] for r in summary_rows])

    ax.set_xticks(x)
    ax.set_xticklabels([_xtick_label(r["config_id"], config_briefs) for r in summary_rows], fontsize=9)
    ax.set_xlabel("Pipeline configuration (brick changed vs. baseline A0)")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.08)

    title = f"{metric_label} by pipeline configuration\n(mean +/- std across dataset x seed; n = successful runs)"
    note = _missing_configs_note(all_config_ids, {r["config_id"] for r in summary_rows})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_runtime_by_dataset(rows, out_path) -> None:
    x = list(range(len(rows)))
    means = [r["runtime_s_mean"] for r in rows]
    stds = [r["runtime_s_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, [m if m is not None else 0.0 for m in means],
           yerr=[s if s is not None else 0.0 for s in stds], capsize=4, color=COLORS["sparsity"])
    _annotate_n(ax, x, [(m or 0) + (s or 0) for m, s in zip(means, stds)], [r["n"] for r in rows])

    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"] for r in rows])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Wall-clock training runtime (s)")
    ax.set_title("Mean training runtime by dataset\n"
                  "(mean +/- std across configuration x seed; n = successful runs)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_osq_effect(rows, out_path, metric_label: str = "Test accuracy",
                     config_briefs=None, all_config_ids=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(rows))
    width = 0.35
    off_x = [i - width / 2 for i in x]
    on_x = [i + width / 2 for i in x]
    off_means = [r["gamma_off_mean"] for r in rows]
    on_means = [r["gamma_on_mean"] for r in rows]
    off_stds = [r["gamma_off_std"] for r in rows]
    on_stds = [r["gamma_on_std"] for r in rows]

    ax.bar(off_x, [m if m is not None else 0.0 for m in off_means], width,
           yerr=[s if s is not None else 0.0 for s in off_stds], capsize=3,
           label="OSq off (gamma=0)", color=COLORS["train"])
    ax.bar(on_x, [m if m is not None else 0.0 for m in on_means], width,
           yerr=[s if s is not None else 0.0 for s in on_stds], capsize=3,
           label="OSq on (gamma>0)", color=COLORS["osq"])
    _annotate_n(ax, off_x, [(m or 0) + (s or 0) for m, s in zip(off_means, off_stds)],
                [r.get("gamma_off_n") for r in rows])
    _annotate_n(ax, on_x, [(m or 0) + (s or 0) for m, s in zip(on_means, on_stds)],
                [r.get("gamma_on_n") for r in rows])

    ax.set_xticks(list(x))
    ax.set_xticklabels([_xtick_label(r["config_id"], config_briefs) for r in rows], fontsize=9)
    ax.set_xlabel("Pipeline configuration (brick changed vs. baseline A0)")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.08)

    gamma_on_value = next((r["gamma_on"] for r in rows if r.get("gamma_on") is not None), None)
    title = (f"Effect of the OSq term on {metric_label.lower()}\n"
             f"(OSq off = gamma 0.0 vs. OSq on = gamma {gamma_on_value}; n = successful runs per bar)")
    note = _missing_configs_note(all_config_ids, {r["config_id"] for r in rows})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_title(run_row: dict, config_briefs=None) -> str:
    seed = run_row["seed"]
    config_label = _xtick_label(run_row["config_id"], config_briefs).replace("\n", " ")
    return (f"{run_row['run_id']} | dataset={run_row['dataset']} | "
            f"config={config_label} | gamma={run_row['gamma']} | "
            f"seed={int(seed) if seed is not None else seed}")


def plot_loss_curves(history, run_row, out_path, config_briefs=None) -> None:
    """Train/val/test *task* (cross-entropy) loss vs. epoch for one run -
    the standard verification plot: val/test loss diverging upward away
    from train loss is overfitting; all three staying high is
    underfitting. This is the task loss only - it excludes the sparsity and
    OSq loss terms, which is why it can differ from the logged total
    train_loss."""
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_task"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_loss"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_loss"] for r in history], label="Test", color=COLORS["test"], linewidth=2)
    ax.set_title(f"Task (cross-entropy) loss: train vs. validation vs. test\n{_run_title(run_row, config_briefs)}",
                 fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss (task term only)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_accuracy_curves(history, run_row, out_path, config_briefs=None) -> None:
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_acc"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_acc"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_acc"] for r in history], label="Test", color=COLORS["test"], linewidth=2)

    best_epoch = run_row.get("best_epoch")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"Best val @ epoch {int(best_epoch)}")

    ax.set_title(f"Accuracy over training\n{_run_title(run_row, config_briefs)}", fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze the Tier A/B/C experiment grid.")
    parser.add_argument("--results-dir", type=str, default="results/rich_bricks/results",
                         help="Directory holding runs.csv/epochs.csv (default: results/rich_bricks/results)")
    parser.add_argument("--out-dir", type=str, default="results/rich_bricks/figures",
                         help="Where to write summary tables and plots (default: results/rich_bricks/figures)")
    parser.add_argument("--run-id", action="append", default=None,
                         help="Plot train/val/test curves for this run_id (repeatable). "
                              "Defaults to the best (highest val_acc) run of each config_id.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    all_config_ids = {row["config_id"] for row in runs}
    briefs = brick_labels(runs)

    status = status_report(runs)
    write_csv(status, out_dir / "status_report.csv")

    accuracy = summarize_by_config(runs, "test_acc")
    write_csv(accuracy, out_dir / "accuracy_by_config.csv")
    plot_accuracy_by_config(accuracy, out_dir / "accuracy_by_config.png",
                             config_briefs=briefs, all_config_ids=all_config_ids)

    runtime = runtime_by_dataset(runs)
    write_csv(runtime, out_dir / "runtime_by_dataset.csv")
    plot_runtime_by_dataset(runtime, out_dir / "runtime_by_dataset.png")

    osq = osq_effect(runs, "test_acc")
    write_csv(osq, out_dir / "osq_effect_on_accuracy.csv")
    plot_osq_effect(osq, out_dir / "osq_effect_on_accuracy.png",
                     config_briefs=briefs, all_config_ids=all_config_ids)

    epochs = load_epochs(args.results_dir)
    epochs_by_run = defaultdict(list)
    for row in epochs:
        epochs_by_run[row["run_id"]].append(row)

    runs_by_id = {row["run_id"]: row for row in runs}
    if args.run_id:
        run_ids = args.run_id
    else:
        run_ids = [row["run_id"] for row in best_run_per_config(runs).values()]

    curves_dir = out_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    for run_id in run_ids:
        if run_id not in epochs_by_run:
            print(f"skip {run_id}: no epoch history found (failed run?)")
            continue
        history = sorted(epochs_by_run[run_id], key=lambda r: r["epoch"])
        run_row = runs_by_id[run_id]
        plot_loss_curves(history, run_row, curves_dir / f"{run_id}_loss.png", config_briefs=briefs)
        plot_accuracy_curves(history, run_row, curves_dir / f"{run_id}_accuracy.png", config_briefs=briefs)

    print(f"Wrote summary tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
