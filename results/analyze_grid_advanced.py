# "Complex" analyses on top of the Tier A/B/C grid (results/runs.csv),
# built on the loaders in results/analyze_grid.py. Two questions, chosen
# because they get most directly at what feat/rich-bricks is for:
#
#   1. Brick ablation - which single-stage swap (A1..A8) actually moves test
#      accuracy relative to the all-simple-bricks baseline (A0), holding
#      gamma fixed at 0 so the comparison isn't entangled with question 2.
#   2. OSq significance - is the accuracy delta between gamma=0 (OSq loss
#      off) and gamma>0 (on) big relative to seed noise (paired t-test /
#      Wilcoxon across seeds), and does the *extra* OSq reduction gamma buys
#      (r_bar_after / lambda2_after getting smaller) actually correlate with
#      that accuracy delta - the CLAUDE.md Sec10 non-negotiable, quantified
#      instead of eyeballed.
#
# Every plot carries its config_id -> brick-changed labels, per-bar/point
# sample sizes and a note for any config with zero data, the same self-
# contained-figure convention as results/analyze_grid.py.
#
# Usage (run as a module, from the repo root, so the `results.analyze_grid`
# import below resolves - `python results/analyze_grid_advanced.py` directly
# will NOT find it):
#   python -m results.analyze_grid_advanced [--results-dir results/rich_bricks/results] [--out-dir results/rich_bricks/figures/advanced]

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from results.analyze_grid import _mean_std, _missing_configs_note, _xtick_label, brick_labels, load_runs, write_csv

_CATEGORY_COLORS = plt.get_cmap("tab10").colors


def _try_stat(fn, *args):
    """Runs a scipy paired/correlation test, turning NaN results (e.g. from
    a zero-variance sample scipy warns about rather than raising on) and any
    exception into (None, None) instead of propagating a warning or crash."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = fn(*args)
        stat, p = float(stat), float(p)
        if stat != stat or p != p:  # NaN check without importing math
            return None, None
        return stat, p
    except Exception:
        return None, None


# -- 1. Brick ablation --------------------------------------------------------

def brick_ablation(runs, metric: str = "test_acc", baseline_config: str = "A0",
                    gamma: float = 0.0) -> list:
    """
    For every (config_id, dataset) at the given gamma, delta = mean(metric)
    for that config minus mean(metric) for `baseline_config` on the SAME
    dataset. A dataset only some configs were run on (e.g. NCI1, only run
    for A0/A1) never produces a row, since there's no baseline to compare
    against on it - this is what keeps the comparison apples-to-apples.
    """
    means = defaultdict(list)
    for row in runs:
        if row["status"] == "ok" and row["gamma"] == gamma:
            means[(row["config_id"], row["dataset"])].append(row[metric])

    baseline_means = {
        dataset: _mean_std(values)[0]
        for (cfg, dataset), values in means.items() if cfg == baseline_config
    }

    out = []
    configs = sorted({cfg for (cfg, _) in means if cfg != baseline_config})
    for cfg in configs:
        datasets = sorted({dataset for (c, dataset) in means if c == cfg})
        for dataset in datasets:
            if dataset not in baseline_means:
                continue
            cfg_mean, cfg_std = _mean_std(means[(cfg, dataset)])
            baseline_mean = baseline_means[dataset]
            out.append({
                "config_id": cfg, "dataset": dataset, "gamma": gamma,
                "n": len(means[(cfg, dataset)]),
                "baseline_mean": baseline_mean, "config_mean": cfg_mean,
                "delta": cfg_mean - baseline_mean,
            })
    return out


def brick_ablation_summary(ablation_rows: list) -> list:
    """Unweighted mean of the per-dataset deltas, per config_id - the usual
    'average improvement over the baseline' ablation-table number."""
    by_config = defaultdict(list)
    for row in ablation_rows:
        by_config[row["config_id"]].append(row["delta"])
    out = []
    for cfg, deltas in sorted(by_config.items()):
        mean, std = _mean_std(deltas)
        out.append({"config_id": cfg, "n_datasets": len(deltas), "mean_delta": mean, "std_delta": std})
    return out


def plot_brick_ablation(summary_rows: list, out_path, metric_label: str = "Test accuracy",
                         config_briefs=None, all_config_ids=None, baseline_config: str = "A0") -> None:
    rows = sorted(summary_rows, key=lambda r: r["mean_delta"] if r["mean_delta"] is not None else 0.0)
    y = list(range(len(rows)))
    deltas = [r["mean_delta"] if r["mean_delta"] is not None else 0.0 for r in rows]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(rows) + 1.5)))
    ax.barh(y, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=1)

    for i, row in enumerate(rows):
        delta = deltas[i]
        ha = "left" if delta >= 0 else "right"
        offset = 6 if delta >= 0 else -6
        ax.annotate(f"{delta:+.3f} (n={row['n_datasets']})", (delta, i),
                    textcoords="offset points", xytext=(offset, 0), va="center", ha=ha, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([_xtick_label(r["config_id"], config_briefs) for r in rows])
    ax.margins(x=0.5)
    ax.set_xlabel(f"Delta {metric_label.lower()} vs. baseline ({baseline_config}) "
                  "- unweighted mean over datasets shared with the baseline")
    title = ("Which brick swap helps or hurts?\n"
             f"(gamma=0, OSq term off; positive = better than the all-simple-bricks baseline {baseline_config})")
    note = _missing_configs_note(all_config_ids, {r["config_id"] for r in rows} | {baseline_config})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- 2. OSq significance -------------------------------------------------------

def _matched_gamma_pairs(runs) -> dict:
    """
    (config_id, dataset, seed) -> (gamma_off_row, gamma_on_row) for every
    run that has both a successful gamma=0 sibling and a successful
    gamma>0 sibling (the largest gamma present, matching osq_effect() in
    analyze_grid.py).
    """
    off, on = {}, {}
    for row in runs:
        if row["status"] != "ok" or row["gamma"] is None:
            continue
        key = (row["config_id"], row["dataset"], row["seed"])
        if row["gamma"] == 0.0:
            off[key] = row
        elif row["gamma"] > 0 and (key not in on or row["gamma"] > on[key]["gamma"]):
            on[key] = row
    return {key: (off[key], on[key]) for key in sorted(set(off) & set(on))}


def paired_osq_significance(runs, metric: str = "test_acc") -> list:
    """
    Per config_id: a paired t-test and Wilcoxon signed-rank test (across
    every matched (dataset, seed) pair) of `metric` with the OSq term on
    vs. off. Both are reported because the grid has few seeds per cell
    (often <10 pairs) - the t-test assumes normality that few samples can't
    establish, Wilcoxon doesn't but has less power; read both, trust neither
    alone.
    """
    pairs_by_config = defaultdict(list)
    for (cfg, _dataset, _seed), (off_row, on_row) in _matched_gamma_pairs(runs).items():
        if off_row[metric] is None or on_row[metric] is None:
            continue
        pairs_by_config[cfg].append((off_row[metric], on_row[metric]))

    out = []
    for cfg, pairs in sorted(pairs_by_config.items()):
        offs = [p[0] for p in pairs]
        ons = [p[1] for p in pairs]
        mean_delta, _ = _mean_std([on_val - off_val for off_val, on_val in pairs])
        t_stat, t_p = _try_stat(stats.ttest_rel, ons, offs) if len(pairs) >= 2 else (None, None)
        w_stat, w_p = _try_stat(stats.wilcoxon, ons, offs) if len(pairs) >= 2 else (None, None)
        out.append({
            "config_id": cfg, "n_pairs": len(pairs), "mean_delta": mean_delta,
            "t_stat": t_stat, "t_pvalue": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
        })
    return out


def plot_osq_significance(rows: list, out_path, metric_label: str = "Test accuracy",
                           config_briefs=None, all_config_ids=None) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = list(range(len(rows)))
    deltas = [r["mean_delta"] if r["mean_delta"] is not None else 0.0 for r in rows]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    ax.bar(x, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)

    for i, row in enumerate(rows):
        p_value = row["wilcoxon_pvalue"] if row["wilcoxon_pvalue"] is not None else row["t_pvalue"]
        p_label = f"p={p_value:.3f}" if p_value is not None else "p=n/a"
        ax.annotate(f"{p_label}\nn={row['n_pairs']}", (i, deltas[i]), textcoords="offset points",
                    xytext=(0, 4 if deltas[i] >= 0 else -26), ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([_xtick_label(r["config_id"], config_briefs) for r in rows], fontsize=9)
    ax.margins(y=0.3)
    ax.set_xlabel("Pipeline configuration (brick changed vs. baseline A0)")
    ax.set_ylabel(f"Delta {metric_label.lower()} (gamma on - off)")
    title = ("Is the OSq effect bigger than seed noise?\n"
             "(paired across dataset x seed pairs; p = Wilcoxon signed-rank,\n"
             "or paired t-test as fallback; n = matched pairs per bar)")
    note = _missing_configs_note(all_config_ids, {r["config_id"] for r in rows})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def osq_reduction_vs_accuracy(runs, osq_metric: str = "r_bar_after",
                               acc_metric: str = "test_acc") -> list:
    """
    For every matched (config, dataset, seed) pair: how much extra OSq
    reduction turning gamma on bought (delta_osq, more negative = more
    reduction since r_bar/lambda2 are minimized) and the accompanying
    accuracy change (delta_acc) - the raw material for asking whether
    reducing OSq further actually buys accuracy.
    """
    pairs = []
    for (cfg, dataset, seed), (off_row, on_row) in _matched_gamma_pairs(runs).items():
        if off_row[osq_metric] is None or on_row[osq_metric] is None:
            continue
        if off_row[acc_metric] is None or on_row[acc_metric] is None:
            continue
        pairs.append({
            "config_id": cfg, "dataset": dataset, "seed": seed,
            "delta_osq": on_row[osq_metric] - off_row[osq_metric],
            "delta_acc": on_row[acc_metric] - off_row[acc_metric],
        })
    return pairs


def correlate_osq_reduction_with_accuracy(pairs: list) -> dict:
    xs = [p["delta_osq"] for p in pairs]
    ys = [p["delta_acc"] for p in pairs]
    result = {"n": len(xs), "pearson_r": None, "pearson_pvalue": None,
              "spearman_r": None, "spearman_pvalue": None}
    if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
        result["pearson_r"], result["pearson_pvalue"] = _try_stat(stats.pearsonr, xs, ys)
        result["spearman_r"], result["spearman_pvalue"] = _try_stat(stats.spearmanr, xs, ys)
    return result


def plot_osq_reduction_vs_accuracy(pairs: list, correlation: dict, out_path,
                                    osq_label: str = "Delta r_bar_after", config_briefs=None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))

    configs = sorted({p["config_id"] for p in pairs})
    color_by_config = {cfg: _CATEGORY_COLORS[i % len(_CATEGORY_COLORS)] for i, cfg in enumerate(configs)}
    for cfg in configs:
        cfg_pairs = [p for p in pairs if p["config_id"] == cfg]
        xs = [p["delta_osq"] for p in cfg_pairs]
        ys = [p["delta_acc"] for p in cfg_pairs]
        label = _xtick_label(cfg, config_briefs).replace("\n", " ")
        ax.scatter(xs, ys, color=color_by_config[cfg], alpha=0.85, label=label,
                   edgecolor="white", linewidth=0.5, s=40)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"{osq_label} (gamma on - off; more negative = more OSq reduction)")
    ax.set_ylabel("Delta test accuracy (gamma on - off)")

    r, p = correlation.get("pearson_r"), correlation.get("pearson_pvalue")
    subtitle = (f"Pearson r={r:.2f}, p={p:.3f}, n={correlation['n']} points" if r is not None
                else f"n={correlation['n']} points (not enough spread to correlate)")
    ax.set_title(f"Does extra OSq reduction buy accuracy?\n"
                 f"Each point = one (config, dataset, seed) pair | {subtitle}", fontsize=10)
    if configs:
        ax.legend(frameon=False, fontsize=8, title="Configuration", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced analyses: per-brick ablation and OSq-effect significance.")
    parser.add_argument("--results-dir", type=str, default="results/rich_bricks/results")
    parser.add_argument("--out-dir", type=str, default="results/rich_bricks/figures/advanced")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    all_config_ids = {row["config_id"] for row in runs}
    briefs = brick_labels(runs)

    ablation = brick_ablation(runs, gamma=0.0)
    write_csv(ablation, out_dir / "brick_ablation_by_dataset.csv")
    ablation_summary = brick_ablation_summary(ablation)
    write_csv(ablation_summary, out_dir / "brick_ablation_summary.csv")
    if ablation_summary:
        plot_brick_ablation(ablation_summary, out_dir / "brick_ablation.png",
                             config_briefs=briefs, all_config_ids=all_config_ids)

    significance = paired_osq_significance(runs, "test_acc")
    write_csv(significance, out_dir / "osq_significance.csv")
    if significance:
        plot_osq_significance(significance, out_dir / "osq_significance.png",
                               config_briefs=briefs, all_config_ids=all_config_ids)

    for osq_metric, label in [("r_bar_after", "Delta r_bar_after"), ("lambda2_after", "Delta lambda2_after")]:
        pairs = osq_reduction_vs_accuracy(runs, osq_metric=osq_metric)
        write_csv(pairs, out_dir / f"osq_reduction_vs_accuracy_{osq_metric}.csv")
        correlation = correlate_osq_reduction_with_accuracy(pairs)
        write_csv([correlation], out_dir / f"osq_reduction_vs_accuracy_{osq_metric}_correlation.csv")
        if pairs:
            plot_osq_reduction_vs_accuracy(
                pairs, correlation,
                out_dir / f"osq_reduction_vs_accuracy_{osq_metric}.png", osq_label=label, config_briefs=briefs)

    print(f"Wrote advanced analysis tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
