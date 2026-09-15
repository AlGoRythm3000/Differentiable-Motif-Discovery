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


def _variant_order(key):
    """Total order on the values of a split column, numbers before strings.

    `min(..., key=float-or-0.0)` sent every non-numeric variant to the same
    sort key, so with config_ids ("A0"/"A3"/"A8") the base was whichever row
    the CSV happened to list first - the direction of every delta then
    depended on file order.
    """
    value = _f(key)
    return (0, value, "") if value is not None else (1, 0.0, str(key))


def _paired(rows, split_col, metric, hold):
    """Deltas for EVERY ordered pair of `split_col` values, keyed on `hold`.

    All pairs, not base-against-the-rest: with three arms the comparison that
    isolates one brick is often the one that does not involve the base at all
    (A8 vs A3 isolates the message passing from the cycle_basis proposal), and
    a single-base scheme never emits it.
    """
    keyed = defaultdict(dict)
    for r in rows:
        keyed[tuple(r.get(c, "") for c in hold)][r.get(split_col, "")] = r
    out = defaultdict(list)
    for variants in keyed.values():
        keys = sorted(variants, key=_variant_order)
        for i, base in enumerate(keys):
            for other in keys[i + 1:]:
                a = _f(variants[base].get(metric))
                b = _f(variants[other].get(metric))
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

    # Whether the complex is ALIVE decides what a delta on a TNN arm means. If
    # the selector accepts no 2-cell, every rank-2 feature the CWN layer reads
    # is gated to zero and the arm is a rank<=1 architecture wearing a TNN's
    # name - so an A8-A3 delta then compares CWN's edge pathway against a
    # second GCNConv, and says nothing about higher-order message passing.
    print("\n  Vivacite du complexe (une cellule 'acceptee' = alpha_frac_active > 0) :")
    print(f"  {'config':<6}{'n':>5}{'num_cells':>12}{'alpha_mean':>12}{'runs vivants':>14}"
          f"{'datasets':>10}")
    for cid, rs in sorted(by_cfg.items()):
        nc, _, _ = _mean_sd([_f(r.get("num_cells")) for r in rs])
        am, _, _ = _mean_sd([_f(r.get("alpha_mean")) for r in rs])
        live = sum(1 for r in rs if (_f(r.get("alpha_frac_active")) or 0.0) > 0.0)
        nds = len({r["dataset"] for r in rs})
        print(f"  {cid:<6}{len(rs):>5}{nc:>12.1f}{am:>12.4f}{live:>9}/{len(rs):<4}{nds:>10}")

    # Coverage first, because an unbalanced marginal is a Simpson trap: an arm
    # that only ran on the hard datasets looks worse than the arms that ran on
    # all of them, for a reason that has nothing to do with the arm. Only the
    # paired numbers below survive that.
    ds_all = {r["dataset"] for r in rows}
    ragged = [cid for cid, rs in by_cfg.items() if {r["dataset"] for r in rs} != ds_all]
    if ragged:
        print(f"\n  ATTENTION couverture inegale ({', '.join(sorted(ragged))} n'ont pas les "
              f"{len(ds_all)} datasets) :")
        print("  les moyennes marginales ci-dessus NE SONT PAS comparables entre bras.")

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
