# Analysis for the three questions the previous grids could not answer.
#
#   Q-mu    Is "gamma prevents the lifting collapsing" a property of the
#           objective, or an artefact of a sparsity weight that was simply too
#           high? Every earlier grid ran at the single value mu=0.05, where the
#           selector accepts no cell in ~2/3 of runs - which leaves "L1 is
#           ineffective" and "L1 is unobservable" indistinguishable, and makes
#           the collapse finding impossible to attribute. Sweeping mu is the
#           only thing that separates them.
#
#   Q-acol  Does gamma lower R-bar on the structure the objective is DEFINED
#           on (A^col, a clique per accepted cell), and not merely on the star
#           the proxy happens to score? The old answer to "does gamma lower
#           measured R-bar" was close to circular: measurement and objective
#           were the same functional on the same structure. `r_bar_acol_after`
#           is measured at analysis time only and never optimised, so a delta
#           there is a claim about the quantity the method defines.
#
#   Q-tnn   Does message passing over a genuine cell complex (A8: cycle_basis +
#           CWN) beat the flattened rank-0 view (A3: the same proposal, star
#           rewiring)? A3 is the control that isolates the message passing from
#           the proposal - comparing A8 against A0 alone would confound the two.
#           No previous grid answered this because every A8 run died in a CUDA
#           cascade started by an unrelated arm.
#
# Usage:  python -m results.analyze_musweep_acol --musweep results/musweep --tnn results/tnn

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from results.select_gamma_proxy import _paired_tests, holm


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(results_dir):
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        raise SystemExit(f"no runs.csv under {results_dir}")
    rows = [r for r in csv.DictReader(open(path)) if r.get("status") == "ok"]
    if not rows:
        raise SystemExit(f"{path}: no successful run")
    return rows


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def _paired(rows, split_col, metric, hold):
    """Deltas between rows differing ONLY in `split_col`, keyed on `hold`."""
    keyed = defaultdict(dict)
    for r in rows:
        keyed[tuple(r.get(c, "") for c in hold)][r.get(split_col, "")] = r
    out = defaultdict(list)
    for variants in keyed.values():
        if len(variants) < 2:
            continue
        base = min(variants, key=lambda k: float(k) if _f(k) is not None else 0.0)
        for other, row in variants.items():
            if other == base:
                continue
            a, b = _f(variants[base].get(metric)), _f(row.get(metric))
            if a is not None and b is not None:
                out[(base, other)].append((a, b))
    return out


# --------------------------------------------------------------------- Q-mu
def q_mu(rows):
    print("\n" + "=" * 78)
    print("Q-mu : le collapse est-il un effet de gamma, ou de mu mal regle ?")
    print("=" * 78)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["sparsity_weight"], r["gamma"])].append(r)
    print(f"{'mu':>8}{'gamma':>8}{'n':>5}{'collapse%':>11}{'alpha_mean':>12}"
          f"{'frac_active':>13}{'test_acc':>18}")
    for (mu, g), rs in sorted(cells.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        coll = sum(1 for r in rs if str(r.get("collapsed")) == "True")
        am, _, _ = _mean_sd([_f(r.get("alpha_mean")) for r in rs])
        fa, _, _ = _mean_sd([_f(r.get("alpha_frac_active")) for r in rs])
        ta, ts, n = _mean_sd([_f(r.get("test_acc")) for r in rs])
        print(f"{mu:>8}{g:>8}{len(rs):>5}{100*coll/len(rs):>10.1f}%{am:>12.4f}"
              f"{fa:>13.3f}{ta:>12.4f} +/-{ts:.3f}")

    print("\n  -> effet de gamma SUR LE COLLAPSE, a mu fixe (paires dans (dataset, fold, mu)) :")
    for mu in sorted({r["sparsity_weight"] for r in rows}, key=float):
        sub = [r for r in rows if r["sparsity_weight"] == mu]
        pairs = _paired(sub, "gamma", "alpha_frac_active", ["dataset", "fold", "sparsity_weight"])
        for (base, other), vals in sorted(pairs.items()):
            d = [b - a for a, b in vals]
            m, sd, n = _mean_sd(d)
            t = _paired_tests([a for a, _ in vals], [b for _, b in vals])
            p = t.get("t_pvalue")
            print(f"     mu={mu:<6} gamma {base}->{other}: d_frac_active={m:+.3f} +/- {sd:.3f} "
                  f"(n={n}, p={'-' if p is None else f'{p:.4f}'})")


# ------------------------------------------------------------------- Q-acol
def q_acol(rows):
    print("\n" + "=" * 78)
    print("Q-acol : gamma baisse-t-il R-bar sur A^col (la quantite DEFINIE),")
    print("         et pas seulement sur l'etoile que le proxy minimise ?")
    print("=" * 78)
    have = [r for r in rows if _f(r.get("r_bar_acol_after")) is not None]
    if not have:
        print("  aucune colonne r_bar_acol_after : grille anterieure a cette mesure.")
        return
    print(f"  {len(have)}/{len(rows)} runs portent la mesure A^col\n")

    hold = ["dataset", "fold", "sparsity_weight"]
    pvals, labels, lines = [], [], []
    for metric, label in (("r_bar_after", "etoile (= objectif du proxy)"),
                          ("r_bar_acol_after", "A^col  (= quantite definie)")):
        pairs = _paired(have, "gamma", metric, hold)
        for (base, other), vals in sorted(pairs.items()):
            d = [b - a for a, b in vals]
            m, sd, n = _mean_sd(d)
            t = _paired_tests([a for a, _ in vals], [b for _, b in vals])
            pvals.append(t.get("t_pvalue")); labels.append((label, base, other))
            lines.append((label, base, other, m, sd, n, sum(1 for x in d if x < 0)))
    corrected = holm(pvals)
    print(f"  {'mesure sur':<32}{'gamma':>12}{'dR-bar':>12}{'+/-':>9}{'n':>5}"
          f"{'ameliores':>11}{'p_holm':>9}")
    for (label, base, other, m, sd, n, better), p in zip(lines, corrected):
        print(f"  {label:<32}{base}->{other:<6}{m:>12.4f}{sd:>9.4f}{n:>5}"
              f"{better:>7}/{n:<3}{'-' if p is None else f'{p:.4f}':>9}")
    print("\n  Lecture : si la ligne A^col est plate alors que l'etoile bouge, le mecanisme")
    print("  mesure jusqu'ici n'etait que l'optimiseur minimisant son propre objectif.")


# -------------------------------------------------------------------- Q-tnn
def q_tnn(rows):
    print("\n" + "=" * 78)
    print("Q-tnn : le vrai complexe cellulaire (A8) bat-il la vue rank-0 aplatie ?")
    print("=" * 78)
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config_id"]].append(r)
    print(f"  {'config':<6}{'s2':>14}{'s5':>16}{'n':>5}{'test_acc':>18}{'collapse%':>11}")
    for cid, rs in sorted(by_cfg.items()):
        m, sd, n = _mean_sd([_f(r.get("test_acc")) for r in rs])
        coll = sum(1 for r in rs if str(r.get("collapsed")) == "True")
        print(f"  {cid:<6}{rs[0].get('s2_proposal',''):>14}{rs[0].get('s5_mp',''):>16}"
              f"{n:>5}{m:>12.4f} +/-{sd:.3f}{100*coll/len(rs):>10.1f}%")

    print("\n  Paires within (dataset, fold, gamma) - A3 est le controle qui isole")
    print("  le message passing de la proposition cycle_basis :")
    pairs = _paired(rows, "config_id", "test_acc", ["dataset", "fold", "gamma"])
    pvals, lines = [], []
    for (base, other), vals in sorted(pairs.items()):
        d = [b - a for a, b in vals]
        m, sd, n = _mean_sd(d)
        t = _paired_tests([a for a, _ in vals], [b for _, b in vals])
        pvals.append(t.get("t_pvalue")); lines.append((base, other, m, sd, n,
                                                       sum(1 for x in d if x > 0)))
    for (base, other, m, sd, n, better), p in zip(lines, holm(pvals)):
        print(f"     {other} - {base}: {100*m:+.2f} pp +/- {100*sd:.2f} "
              f"(n={n}, {better} gagnants, p_holm={'-' if p is None else f'{p:.4f}'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musweep", default="results/musweep")
    ap.add_argument("--tnn", default="results/tnn")
    args = ap.parse_args()

    mu_rows = load(args.musweep)
    print(f"musweep : {len(mu_rows)} runs ok")
    q_mu(mu_rows)
    q_acol(mu_rows)

    tnn_rows = load(args.tnn)
    print(f"\ntnn : {len(tnn_rows)} runs ok")
    q_tnn(tnn_rows)
    q_acol(tnn_rows)


if __name__ == "__main__":
    main()
