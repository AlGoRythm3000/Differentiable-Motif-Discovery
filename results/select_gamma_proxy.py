# Selects gamma and the OSq proxy from the `--stage calibrate` results tree.
#
# This step is the one the project skipped. feat/osq-proxy compared four
# proxies at the single value gamma=0.01; feat/rich-bricks then ran its entire
# grid at gamma=0.1 - a value no experiment had ever evaluated - describing it
# in the config as "the winning OSq weight from feat/osq-proxy's grid". And
# r_bar won that comparison at p=0.011 against cf_bc_efc at p=0.031, over 18
# pairs, with no correction for having tested four proxies, while cf_bc_efc was
# ahead on NCI1 and on the synthetic arm. Neither knob was selected from
# anything; this script is what selecting them looks like.
#
# Three rules it follows and the previous selection did not:
#
#   1. SELECT ON VALIDATION, REPORT ON TEST. Choosing a hyperparameter by test
#      accuracy and then reporting that same test accuracy is how a grid talks
#      itself into an effect. `--metric val_acc` is the default and the test
#      column is printed alongside purely as the confirmatory readout.
#   2. PAIR ON THE FOLD. Every arm is compared against its own gamma=0 twin on
#      the same (dataset, seed, fold) - literally the same graphs, same
#      initialisation - so a delta is not contaminated by split luck.
#   3. CORRECT FOR THE FAMILY. Every (gamma, proxy) arm tested is one more
#      chance to find a spurious winner; p-values are Holm-corrected across the
#      whole family actually examined.
#
# It also reports the mechanism (does the arm lower measured r_bar?) and the
# collapse rate separately from the task metric, because those are three
# different claims and the previous grids reported them as one.
#
# Usage (from the repo root):
#   python -m results.select_gamma_proxy --results-dir results/calibrate

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

BASELINE_PROXY = "none"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_runs(results_dir):
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        raise SystemExit(f"no runs.csv under {results_dir}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pair_key(row):
    """
    What makes two rows comparable. The fold is part of the identity: without
    it, an arm and its baseline can land on different test sets and the
    'paired' delta is measuring the split.
    """
    return (row["dataset"], row["seed"], row.get("fold", ""))


def matched_arms(runs, metric="val_acc"):
    """
    (gamma, proxy) -> list of (baseline_value, arm_value, baseline_row, arm_row)
    for every successful run that has a successful gamma=0 twin on the same
    (dataset, seed, fold).
    """
    baseline, arms = {}, defaultdict(list)
    for row in runs:
        if row.get("status") != "ok":
            continue
        gamma = _float(row.get("gamma"))
        if gamma is None:
            continue
        if gamma == 0.0:
            baseline[_pair_key(row)] = row

    for row in runs:
        if row.get("status") != "ok":
            continue
        gamma = _float(row.get("gamma"))
        if not gamma:
            continue
        twin = baseline.get(_pair_key(row))
        if twin is None:
            continue
        arm_value, base_value = _float(row.get(metric)), _float(twin.get(metric))
        if arm_value is None or base_value is None:
            continue
        proxy = row.get("osq_proxy") or row.get("proxy") or "?"
        arms[(gamma, proxy)].append((base_value, arm_value, twin, row))
    return dict(sorted(arms.items()))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _paired_tests(baselines, values):
    """Paired t-test and Wilcoxon, via scipy when it is available."""
    if len(values) < 3:
        return {}
    try:
        from scipy import stats
    except ImportError:
        return {}
    out = {}
    try:
        t_stat, t_p = stats.ttest_rel(values, baselines)
        out["t_stat"], out["t_pvalue"] = float(t_stat), float(t_p)
    except Exception:  # noqa: BLE001 - a degenerate (zero-variance) arm is not an error
        pass
    try:
        w_stat, w_p = stats.wilcoxon(values, baselines)
        out["wilcoxon_stat"], out["wilcoxon_pvalue"] = float(w_stat), float(w_p)
    except Exception:  # noqa: BLE001 - scipy raises when every delta is exactly 0
        pass
    return out


def holm(pvalues):
    """
    Holm-Bonferroni step-down correction. Uniformly more powerful than plain
    Bonferroni at the same family-wise error rate, and there is no reason to
    use the weaker one. `None` entries (arms with no test) pass through.
    """
    indexed = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    m = len(indexed)
    out = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        out[i] = adjusted
    return out


def summarize(runs, metric="val_acc"):
    """One row per (gamma, proxy) arm: task effect, mechanism effect, collapse."""
    arms = matched_arms(runs, metric=metric)
    rows = []
    for (gamma, proxy), pairs in arms.items():
        baselines = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        deltas = [v - b for b, v in zip(baselines, values)]
        n = len(deltas)
        mean = sum(deltas) / n
        std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0

        # Mechanism, kept separate from the task metric on purpose: "the term
        # lowers effective resistance" and "the term raises accuracy" are two
        # claims and the last two grids reported them as one.
        r_deltas = []
        for _, _, twin, row in pairs:
            before, after = _float(twin.get("r_bar_after")), _float(row.get("r_bar_after"))
            if before is not None and after is not None:
                r_deltas.append(after - before)

        # Confirmatory only - never the selection criterion.
        test_deltas = []
        for _, _, twin, row in pairs:
            base, arm = _float(twin.get("test_acc")), _float(row.get("test_acc"))
            if base is not None and arm is not None:
                test_deltas.append(arm - base)

        collapsed = sum(1 for _, _, _, row in pairs if str(row.get("collapsed")) == "True")

        row = {"gamma": gamma, "proxy": proxy, "n_pairs": n,
               "mean_delta": mean, "std_delta": std,
               "n_improved": sum(1 for d in deltas if d > 0),
               "n_worse": sum(1 for d in deltas if d < 0),
               "mean_delta_test": (sum(test_deltas) / len(test_deltas)) if test_deltas else None,
               "mean_delta_r_bar": (sum(r_deltas) / len(r_deltas)) if r_deltas else None,
               "r_bar_improved": sum(1 for d in r_deltas if d < 0),
               "collapse_rate": collapsed / n if n else None}
        row.update(_paired_tests(baselines, values))
        rows.append(row)

    corrected = holm([r.get("t_pvalue") for r in rows])
    for row, p in zip(rows, corrected):
        row["t_pvalue_holm"] = p
    corrected_w = holm([r.get("wilcoxon_pvalue") for r in rows])
    for row, p in zip(rows, corrected_w):
        row["wilcoxon_pvalue_holm"] = p
    return rows


def recommend(rows, alpha=0.05):
    """
    The arm to run the grid at, and the honest answer when there isn't one.

    Deliberately conservative: an arm is only recommended if it survives the
    multiplicity correction. If nothing does, that is the finding - the previous
    selection's whole problem was that it named a winner regardless.
    """
    scored = [r for r in rows if r["n_pairs"] >= 3]
    if not scored:
        return None, "not enough matched pairs to select anything"

    significant = [r for r in scored
                   if r.get("t_pvalue_holm") is not None
                   and r["t_pvalue_holm"] < alpha and r["mean_delta"] > 0]
    if significant:
        best = max(significant, key=lambda r: r["mean_delta"])
        return best, (f"gamma={best['gamma']} proxy={best['proxy']} survives Holm correction "
                      f"(p={best['t_pvalue_holm']:.4f})")

    best = max(scored, key=lambda r: r["mean_delta"])
    return best, ("no arm survives multiple-comparison correction on the task metric. "
                  f"Largest raw effect is gamma={best['gamma']} proxy={best['proxy']} "
                  f"({best['mean_delta'] * 100:+.2f} pp), which is a direction, not a result. "
                  "Report the mechanism column instead, and pick gamma on the collapse rate "
                  "and the r_bar effect rather than on accuracy.")


# ---------------------------------------------------------------------------

def write_csv(rows, path):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/calibrate")
    parser.add_argument("--out", default=None, help="where to write the summary CSV")
    parser.add_argument("--metric", default="val_acc",
                        help="selection metric. Keep val_acc: selecting on test_acc and "
                             "then reporting it is not a measurement.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    rows = summarize(runs, metric=args.metric)
    if not rows:
        raise SystemExit("no matched (arm, gamma=0 twin) pairs found - is this a calibrate tree?")

    print(f"selection metric : {args.metric}  (test_acc shown for confirmation only)")
    print(f"arms             : {len(rows)}  (Holm-corrected across all of them)\n")
    header = (f"{'gamma':>7}{'proxy':>11}{'n':>5}{'d_val pp':>10}{'std':>8}"
              f"{'+/-':>9}{'p_holm':>9}{'d_test pp':>11}{'d_r_bar':>10}{'collapse':>10}")
    print(header)
    print("-" * len(header))
    # Formatted without nested f-strings: environment.yml pins Python 3.10 and
    # the Modal image is 3.11, neither of which accepts PEP 701 nesting.
    for row in sorted(rows, key=lambda r: (-r["mean_delta"], r["gamma"])):
        p = row.get("t_pvalue_holm")
        won_lost = "{}/{}".format(row["n_improved"], row["n_worse"])
        p_text = "-" if p is None else "{:.3f}".format(p)
        test_text = ("-" if row["mean_delta_test"] is None
                     else "{:.2f}".format(row["mean_delta_test"] * 100))
        r_text = ("-" if row["mean_delta_r_bar"] is None
                  else "{:.3f}".format(row["mean_delta_r_bar"]))
        collapse_text = ("-" if row["collapse_rate"] is None
                         else "{:.0%}".format(row["collapse_rate"]))
        print("{:>7}{:>11}{:>5}{:>10.2f}{:>8.2f}{:>9}{:>9}{:>11}{:>10}{:>10}".format(
            row["gamma"], row["proxy"], row["n_pairs"],
            row["mean_delta"] * 100, row["std_delta"] * 100,
            won_lost, p_text, test_text, r_text, collapse_text))

    best, note = recommend(rows, alpha=args.alpha)
    print(f"\nrecommendation: {note}")

    out = args.out or str(Path(args.results_dir) / "gamma_proxy_selection.csv")
    write_csv(rows, out)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
