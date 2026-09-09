# "Complex" analyses on top of the feat/osq-proxy grid (results/runs.csv),
# built on the loaders in results/analyze_osq_proxy.py. Adapted from
# results/analyze_grid_advanced.py (feat/rich-bricks): same two questions,
# but the comparison axis is `proxy` (vs. `config_id`) and the baseline is
# the `none` proxy (vs. `A0`) - there is no per-proxy gamma sweep to hold
# fixed here (see analyze_osq_proxy.py's module docstring), so every
# comparison is simply "this proxy" vs. "none" on the same (dataset, seed).
#
#   1. Proxy ablation - which OSq proxy actually moves test accuracy
#      relative to the no-OSq-term baseline (`none`), per dataset.
#   2. OSq significance - is the accuracy delta (proxy vs. none) big
#      relative to seed noise (paired t-test / Wilcoxon across matched
#      (dataset, seed) pairs), and does the *measured* OSq reduction a
#      proxy buys (r_bar_after / lambda2_after getting smaller than
#      `none`'s) actually correlate with that accuracy delta - the
#      "optimize X, measure X, check X helped" rule, quantified instead
#      of eyeballed.
#
# Every plot carries proxy descriptions, per-bar/point sample sizes, and a
# note for any proxy with zero data - the same self-contained-figure
# convention as results/analyze_osq_proxy.py.
#
# Usage (run as a module, from the repo root, so the
# `results.analyze_osq_proxy` import below resolves):
#   python -m results.analyze_osq_proxy_advanced [--results-dir results/osq_proxy/results] [--out-dir results/osq_proxy/figures/advanced]

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from results.analyze_osq_proxy import _mean_std, _missing_proxies_note, _xtick_label, load_runs, write_csv

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


# -- 1. Proxy ablation --------------------------------------------------------

def proxy_ablation(runs, metric: str = "test_acc", baseline_proxy: str = "none") -> list:
    """
    For every (proxy, dataset), delta = mean(metric) for that proxy minus
    mean(metric) for `baseline_proxy` on the SAME dataset. Every proxy in
    this grid runs on every dataset (unlike feat/rich-bricks' partial
    coverage), but the per-dataset lookup is kept anyway so this still
    degrades gracefully if a future re-run doesn't cover everything.
    """
    means = defaultdict(list)
    for row in runs:
        if row["status"] == "ok":
            means[(row["proxy"], row["dataset"])].append(row[metric])

    baseline_means = {
        dataset: _mean_std(values)[0]
        for (proxy, dataset), values in means.items() if proxy == baseline_proxy
    }

    out = []
    proxies = sorted({proxy for (proxy, _) in means if proxy != baseline_proxy})
    for proxy in proxies:
        datasets = sorted({dataset for (p, dataset) in means if p == proxy})
        for dataset in datasets:
            if dataset not in baseline_means:
                continue
            proxy_mean, proxy_std = _mean_std(means[(proxy, dataset)])
            baseline_mean = baseline_means[dataset]
            out.append({
                "proxy": proxy, "dataset": dataset,
                "n": len(means[(proxy, dataset)]),
                "baseline_mean": baseline_mean, "proxy_mean": proxy_mean,
                "delta": proxy_mean - baseline_mean,
            })
    return out


def proxy_ablation_summary(ablation_rows: list) -> list:
    """Unweighted mean of the per-dataset deltas, per proxy - the usual
    'average improvement over the baseline' ablation-table number."""
    by_proxy = defaultdict(list)
    for row in ablation_rows:
        by_proxy[row["proxy"]].append(row["delta"])
    out = []
    for proxy, deltas in sorted(by_proxy.items()):
        mean, std = _mean_std(deltas)
        out.append({"proxy": proxy, "n_datasets": len(deltas), "mean_delta": mean, "std_delta": std})
    return out


def plot_proxy_ablation(summary_rows: list, out_path, metric_label: str = "Test accuracy",
                         all_proxies=None, baseline_proxy: str = "none") -> None:
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
    ax.set_yticklabels([_xtick_label(r["proxy"]) for r in rows])
    ax.margins(x=0.5)
    ax.set_xlabel(f"Delta {metric_label.lower()} vs. baseline ({baseline_proxy}) "
                  "- unweighted mean over datasets shared with the baseline")
    title = ("Which OSq proxy helps or hurts?\n"
             f"(positive = better than the no-OSq-term baseline '{baseline_proxy}')")
    note = _missing_proxies_note(all_proxies, {r["proxy"] for r in rows} | {baseline_proxy})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- 2. OSq significance -------------------------------------------------------

def _matched_proxy_pairs(runs, baseline_proxy: str = "none") -> dict:
    """
    (proxy, dataset, seed, fold) -> (baseline_row, proxy_row) for every
    successful run of a non-baseline proxy that has a successful
    `baseline_proxy` sibling on the same (dataset, seed, fold).

    The fold belongs in the key: without it, a proxy run on one fold gets paired
    against a baseline on another, so the "paired" delta absorbs the difference
    between two test sets - and under k-fold all but one run per cell would be
    dropped silently. Pre-cross-validation rows carry no `fold` and degrade to
    the old key exactly.
    """
    baseline = {}
    others = defaultdict(dict)
    for row in runs:
        if row["status"] != "ok":
            continue
        key = (row["dataset"], row["seed"], row.get("fold", ""))
        if row["proxy"] == baseline_proxy:
            baseline[key] = row
        else:
            others[row["proxy"]][key] = row

    pairs = {}
    for proxy, rows_by_key in others.items():
        for key, row in rows_by_key.items():
            if key in baseline:
                pairs[(proxy,) + key] = (baseline[key], row)
    return dict(sorted(pairs.items()))


def paired_proxy_significance(runs, metric: str = "test_acc", baseline_proxy: str = "none") -> list:
    """
    Per proxy: a paired t-test and Wilcoxon signed-rank test (across every
    matched (dataset, seed) pair) of `metric` for this proxy vs. the
    baseline. Both are reported because the grid has few seeds per cell
    (3, x6 datasets = 18 pairs at most) - the t-test assumes normality that
    few samples can't establish, Wilcoxon doesn't but has less power; read
    both, trust neither alone.
    """
    pairs_by_proxy = defaultdict(list)
    for (proxy, _dataset, _seed, _fold), (baseline_row, proxy_row) in _matched_proxy_pairs(runs, baseline_proxy).items():
        if baseline_row[metric] is None or proxy_row[metric] is None:
            continue
        pairs_by_proxy[proxy].append((baseline_row[metric], proxy_row[metric]))

    out = []
    for proxy, pairs in sorted(pairs_by_proxy.items()):
        baselines = [p[0] for p in pairs]
        proxies = [p[1] for p in pairs]
        mean_delta, _ = _mean_std([proxy_val - base_val for base_val, proxy_val in pairs])
        t_stat, t_p = _try_stat(stats.ttest_rel, proxies, baselines) if len(pairs) >= 2 else (None, None)
        w_stat, w_p = _try_stat(stats.wilcoxon, proxies, baselines) if len(pairs) >= 2 else (None, None)
        out.append({
            "proxy": proxy, "n_pairs": len(pairs), "mean_delta": mean_delta,
            "t_stat": t_stat, "t_pvalue": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
        })
    return out


def plot_proxy_significance(rows: list, out_path, metric_label: str = "Test accuracy",
                             all_proxies=None, baseline_proxy: str = "none") -> None:
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
    ax.set_xticklabels([_xtick_label(r["proxy"]) for r in rows], fontsize=9)
    ax.margins(y=0.3)
    ax.set_xlabel("OSq proxy")
    ax.set_ylabel(f"Delta {metric_label.lower()} (proxy - none)")
    title = ("Is a proxy's accuracy effect bigger than seed noise?\n"
             "(paired across dataset x seed pairs; p = Wilcoxon signed-rank,\n"
             "or paired t-test as fallback; n = matched pairs per bar)")
    note = _missing_proxies_note(all_proxies, {r["proxy"] for r in rows} | {baseline_proxy})
    ax.set_title(title if note is None else f"{title}\n{note}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def osq_reduction_vs_accuracy(runs, osq_metric: str = "r_bar_after", acc_metric: str = "test_acc",
                               baseline_proxy: str = "none") -> list:
    """
    For every matched (proxy, dataset, seed) pair: how much extra OSq
    reduction this proxy bought over the `none` baseline (delta_osq, more
    negative = more reduction since r_bar/lambda2 are minimized) and the
    accompanying accuracy change (delta_acc) - the raw material for asking
    whether reducing measured OSq further actually buys accuracy.
    """
    pairs = []
    for (proxy, dataset, seed, fold), (baseline_row, proxy_row) in _matched_proxy_pairs(runs, baseline_proxy).items():
        if baseline_row[osq_metric] is None or proxy_row[osq_metric] is None:
            continue
        if baseline_row[acc_metric] is None or proxy_row[acc_metric] is None:
            continue
        pairs.append({
            "proxy": proxy, "dataset": dataset, "seed": seed, "fold": fold,
            "delta_osq": proxy_row[osq_metric] - baseline_row[osq_metric],
            "delta_acc": proxy_row[acc_metric] - baseline_row[acc_metric],
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
                                    osq_label: str = "Delta r_bar_after", higher_is_better: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))

    proxies = sorted({p["proxy"] for p in pairs})
    color_by_proxy = {proxy: _CATEGORY_COLORS[i % len(_CATEGORY_COLORS)] for i, proxy in enumerate(proxies)}
    for proxy in proxies:
        proxy_pairs = [p for p in pairs if p["proxy"] == proxy]
        xs = [p["delta_osq"] for p in proxy_pairs]
        ys = [p["delta_acc"] for p in proxy_pairs]
        label = _xtick_label(proxy).replace("\n", " ")
        ax.scatter(xs, ys, color=color_by_proxy[proxy], alpha=0.85, label=label,
                   edgecolor="white", linewidth=0.5, s=40)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.5)
    direction_hint = ("more positive = larger, healthier spectral gap" if higher_is_better
                      else "more negative = more OSq reduction")
    ax.set_xlabel(f"{osq_label} (proxy - none; {direction_hint})")
    ax.set_ylabel("Delta test accuracy (proxy - none)")

    r, p = correlation.get("pearson_r"), correlation.get("pearson_pvalue")
    subtitle = (f"Pearson r={r:.2f}, p={p:.3f}, n={correlation['n']} points" if r is not None
                else f"n={correlation['n']} points (not enough spread to correlate)")
    ax.set_title(f"Does extra OSq reduction buy accuracy?\n"
                 f"Each point = one (proxy, dataset, seed) pair | {subtitle}", fontsize=10)
    if proxies:
        ax.legend(frameon=False, fontsize=8, title="OSq proxy", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced analyses: per-proxy ablation and OSq-effect significance.")
    parser.add_argument("--results-dir", type=str, default="results/osq_proxy/results")
    parser.add_argument("--out-dir", type=str, default="results/osq_proxy/figures/advanced")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    all_proxies = {row["proxy"] for row in runs}

    ablation = proxy_ablation(runs)
    write_csv(ablation, out_dir / "proxy_ablation_by_dataset.csv")
    ablation_summary = proxy_ablation_summary(ablation)
    write_csv(ablation_summary, out_dir / "proxy_ablation_summary.csv")
    if ablation_summary:
        plot_proxy_ablation(ablation_summary, out_dir / "proxy_ablation.png", all_proxies=all_proxies)

    significance = paired_proxy_significance(runs, "test_acc")
    write_csv(significance, out_dir / "proxy_significance.csv")
    if significance:
        plot_proxy_significance(significance, out_dir / "proxy_significance.png", all_proxies=all_proxies)

    # r_bar/effective-resistance-style proxies want this SMALLER (more negative
    # delta = more reduction, the desired direction); lambda2 is the opposite -
    # the loss maximizes it (it enters the objective as "-lambda_2"), so a bigger
    # (more positive) delta is the healthy direction there.
    for osq_metric, label, higher_is_better in [
        ("r_bar_after", "Delta r_bar_after", False),
        ("lambda2_after", "Delta lambda2_after", True),
    ]:
        pairs = osq_reduction_vs_accuracy(runs, osq_metric=osq_metric)
        write_csv(pairs, out_dir / f"osq_reduction_vs_accuracy_{osq_metric}.csv")
        correlation = correlate_osq_reduction_with_accuracy(pairs)
        write_csv([correlation], out_dir / f"osq_reduction_vs_accuracy_{osq_metric}_correlation.csv")
        if pairs:
            plot_osq_reduction_vs_accuracy(
                pairs, correlation, out_dir / f"osq_reduction_vs_accuracy_{osq_metric}.png",
                osq_label=label, higher_is_better=higher_is_better)

    print(f"Wrote advanced analysis tables and plots to {out_dir}")


if __name__ == "__main__":
    main()
