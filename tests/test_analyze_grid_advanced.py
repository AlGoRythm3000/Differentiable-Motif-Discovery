from tools.results_store import ResultsStore, make_run_id
from results.analyze_grid import load_runs
from results.analyze_grid_advanced import (
    brick_ablation,
    brick_ablation_summary,
    correlate_osq_reduction_with_accuracy,
    osq_reduction_vs_accuracy,
    paired_osq_significance,
    plot_brick_ablation,
    plot_osq_reduction_vs_accuracy,
    plot_osq_significance,
)


def _run(config_id, dataset, gamma, seed, test_acc, r_bar_after=None, lambda2_after=None,
         status="ok", error=""):
    return {
        "run_id": make_run_id("A", config_id, dataset, gamma, seed),
        "tier": "A", "config_id": config_id, "dataset": dataset,
        "gamma": gamma, "seed": seed, "status": status, "error": error,
        "test_acc": test_acc,
        "r_bar_after": r_bar_after, "lambda2_after": lambda2_after,
    }


def _store_with_runs(tmp_path, rows):
    store = ResultsStore(tmp_path / "results")
    for row in rows:
        store.append_run(row)
    return load_runs(store.out_dir)


# -- brick_ablation / brick_ablation_summary ---------------------------------

def test_brick_ablation_computes_delta_against_baseline_per_dataset(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.7),
        _run("A0", "PROTEINS", 0.0, 0, test_acc=0.5),
        _run("A1", "MUTAG", 0.0, 0, test_acc=0.8),
        _run("A1", "MUTAG", 0.0, 1, test_acc=0.9),
        _run("A1", "PROTEINS", 0.0, 0, test_acc=0.4),
    ]
    runs = _store_with_runs(tmp_path, rows)
    ablation = {r["dataset"]: r for r in brick_ablation(runs, gamma=0.0) if r["config_id"] == "A1"}

    assert abs(ablation["MUTAG"]["baseline_mean"] - 0.65) < 1e-9
    assert abs(ablation["MUTAG"]["config_mean"] - 0.85) < 1e-9
    assert abs(ablation["MUTAG"]["delta"] - 0.2) < 1e-9
    assert abs(ablation["PROTEINS"]["delta"] - (-0.1)) < 1e-9


def test_brick_ablation_skips_datasets_the_baseline_never_ran(tmp_path):
    # A1 has an NCI1 run but A0 (the baseline) never ran NCI1 - no fair
    # comparison exists, so no row should be produced for it.
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("A1", "MUTAG", 0.0, 0, test_acc=0.7),
        _run("A1", "NCI1", 0.0, 0, test_acc=0.9),
    ]
    runs = _store_with_runs(tmp_path, rows)
    datasets = {r["dataset"] for r in brick_ablation(runs, gamma=0.0) if r["config_id"] == "A1"}
    assert datasets == {"MUTAG"}


def test_brick_ablation_only_uses_the_requested_gamma(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.9),  # must be ignored when gamma=0.0 is requested
        _run("A1", "MUTAG", 0.0, 0, test_acc=0.6),
    ]
    runs = _store_with_runs(tmp_path, rows)
    [row] = brick_ablation(runs, gamma=0.0)
    assert row["baseline_mean"] == 0.5
    assert abs(row["delta"] - 0.1) < 1e-9


def test_brick_ablation_summary_averages_deltas_unweighted_across_datasets(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5),
        _run("A0", "PROTEINS", 0.0, 0, test_acc=0.5),
        _run("A1", "MUTAG", 0.0, 0, test_acc=0.7),   # delta +0.2
        _run("A1", "PROTEINS", 0.0, 0, test_acc=0.4),  # delta -0.1
    ]
    runs = _store_with_runs(tmp_path, rows)
    [summary] = brick_ablation_summary(brick_ablation(runs, gamma=0.0))
    assert summary["config_id"] == "A1"
    assert summary["n_datasets"] == 2
    assert abs(summary["mean_delta"] - 0.05) < 1e-9


# -- paired_osq_significance --------------------------------------------------

def test_paired_osq_significance_computes_mean_delta_and_stats_over_matched_pairs(tmp_path):
    rows = []
    for seed in range(6):
        rows.append(_run("A0", "MUTAG", 0.0, seed, test_acc=0.5))
        rows.append(_run("A0", "MUTAG", 0.1, seed, test_acc=0.7))
    runs = _store_with_runs(tmp_path, rows)
    [row] = paired_osq_significance(runs, "test_acc")

    assert row["config_id"] == "A0"
    assert row["n_pairs"] == 6
    assert abs(row["mean_delta"] - 0.2) < 1e-9
    # A constant +0.2 shift with zero within-pair variance is about as
    # significant a paired difference as this test can produce.
    assert row["t_pvalue"] is not None
    assert row["t_pvalue"] < 0.05


def test_paired_osq_significance_ignores_unmatched_runs(tmp_path):
    # Only gamma=0 exists for this config - there is nothing to pair.
    rows = [_run("A0", "MUTAG", 0.0, 0, test_acc=0.5)]
    runs = _store_with_runs(tmp_path, rows)
    assert paired_osq_significance(runs, "test_acc") == []


def test_paired_osq_significance_handles_too_few_pairs_for_a_stable_test(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.5),  # zero variance in the single difference
    ]
    runs = _store_with_runs(tmp_path, rows)
    [row] = paired_osq_significance(runs, "test_acc")
    assert row["n_pairs"] == 1
    assert row["mean_delta"] == 0.0
    # scipy can't produce a meaningful stat from one zero-variance pair - the
    # wrapper must turn that into None instead of raising or returning NaN.
    assert row["t_stat"] is None or row["t_stat"] == row["t_stat"]  # never NaN if not None


# -- osq_reduction_vs_accuracy / correlate_osq_reduction_with_accuracy --------

def test_osq_reduction_vs_accuracy_pairs_matching_runs(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5, r_bar_after=2.0),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.8, r_bar_after=0.5),
    ]
    runs = _store_with_runs(tmp_path, rows)
    [pair] = osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after")
    assert pair["config_id"] == "A0" and pair["dataset"] == "MUTAG" and pair["seed"] == 0
    assert abs(pair["delta_osq"] - (-1.5)) < 1e-9
    assert abs(pair["delta_acc"] - 0.3) < 1e-9


def test_osq_reduction_vs_accuracy_skips_pairs_missing_the_osq_metric(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5, r_bar_after=None),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.8, r_bar_after=0.5),
    ]
    runs = _store_with_runs(tmp_path, rows)
    assert osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after") == []


def test_correlate_osq_reduction_with_accuracy_detects_a_perfect_negative_correlation():
    # More negative delta_osq (more reduction) paired with more positive
    # delta_acc - a textbook negative correlation.
    pairs = [
        {"delta_osq": -2.0, "delta_acc": 0.3},
        {"delta_osq": -1.0, "delta_acc": 0.2},
        {"delta_osq": 0.0, "delta_acc": 0.1},
        {"delta_osq": 1.0, "delta_acc": 0.0},
    ]
    result = correlate_osq_reduction_with_accuracy(pairs)
    assert result["n"] == 4
    assert result["pearson_r"] is not None
    assert result["pearson_r"] < -0.99


def test_correlate_osq_reduction_with_accuracy_handles_no_spread(tmp_path):
    pairs = [{"delta_osq": -1.0, "delta_acc": 0.1}]  # a single point: nothing to correlate
    result = correlate_osq_reduction_with_accuracy(pairs)
    assert result == {"n": 1, "pearson_r": None, "pearson_pvalue": None,
                       "spearman_r": None, "spearman_pvalue": None}


def test_correlate_osq_reduction_with_accuracy_handles_empty_input():
    assert correlate_osq_reduction_with_accuracy([]) == {
        "n": 0, "pearson_r": None, "pearson_pvalue": None,
        "spearman_r": None, "spearman_pvalue": None,
    }


# -- plotting: smoke tests -----------------------------------------------------

def test_plot_brick_ablation_writes_a_file(tmp_path):
    out_path = tmp_path / "ablation.png"
    plot_brick_ablation([{"config_id": "A1", "n_datasets": 3, "mean_delta": 0.05, "std_delta": 0.02}],
                         out_path, config_briefs={"A1": "gpse"}, all_config_ids={"A0", "A1", "A7"})
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_osq_significance_writes_a_file(tmp_path):
    out_path = tmp_path / "significance.png"
    plot_osq_significance(
        [{"config_id": "A0", "n_pairs": 6, "mean_delta": 0.2,
          "t_stat": 5.0, "t_pvalue": 0.01, "wilcoxon_stat": 0.0, "wilcoxon_pvalue": 0.02}],
        out_path, config_briefs={"A0": "baseline"}, all_config_ids={"A0", "A7"},
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_osq_reduction_vs_accuracy_writes_a_file(tmp_path):
    out_path = tmp_path / "scatter.png"
    pairs = [{"config_id": "A0", "dataset": "MUTAG", "seed": 0, "delta_osq": -1.0, "delta_acc": 0.2},
             {"config_id": "A1", "dataset": "MUTAG", "seed": 0, "delta_osq": 0.5, "delta_acc": -0.1}]
    correlation = correlate_osq_reduction_with_accuracy(pairs)
    plot_osq_reduction_vs_accuracy(pairs, correlation, out_path, config_briefs={"A1": "gpse"})
    assert out_path.exists() and out_path.stat().st_size > 0


# -- end-to-end sanity --------------------------------------------------------

def test_end_to_end_over_a_small_synthetic_grid(tmp_path):
    """A0 is the baseline; A1 adds a fixed +0.1 accuracy bump on every
    dataset regardless of gamma; turning gamma on adds another fixed +0.2 and
    roughly halves r_bar_after - checks the whole pipeline agrees with those
    built-in, hand-computable effects."""
    store = ResultsStore(tmp_path / "results")
    for config_id, base_bump in (("A0", 0.0), ("A1", 0.1)):
        for dataset in ("MUTAG", "PROTEINS"):
            for seed in range(4):
                for gamma, gamma_bump, r_bar in ((0.0, 0.0, 2.0), (0.1, 0.2, 1.0)):
                    store.append_run(_run(config_id, dataset, gamma, seed,
                                           test_acc=0.5 + base_bump + gamma_bump,
                                           r_bar_after=r_bar))
    runs = load_runs(store.out_dir)

    ablation_summary = {r["config_id"]: r for r in brick_ablation_summary(brick_ablation(runs, gamma=0.0))}
    assert abs(ablation_summary["A1"]["mean_delta"] - 0.1) < 1e-9

    significance = {r["config_id"]: r for r in paired_osq_significance(runs, "test_acc")}
    assert significance["A0"]["n_pairs"] == 8  # 2 datasets x 4 seeds
    assert abs(significance["A0"]["mean_delta"] - 0.2) < 1e-9
    assert abs(significance["A1"]["mean_delta"] - 0.2) < 1e-9

    pairs = osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after")
    assert len(pairs) == 16  # 2 configs x 2 datasets x 4 seeds
    assert all(abs(p["delta_osq"] - (-1.0)) < 1e-9 for p in pairs)
    assert all(abs(p["delta_acc"] - 0.2) < 1e-9 for p in pairs)

    # Every pair has the exact same (delta_osq, delta_acc): no spread to
    # correlate, so the wrapper must report None rather than a fabricated r.
    correlation = correlate_osq_reduction_with_accuracy(pairs)
    assert correlation["n"] == 16
    assert correlation["pearson_r"] is None


# ---------------------------------------------------------------------------
# fix/experiment-protocol: the fold is part of the pairing key.
#
# Under k-fold cross-validation a (config, dataset, seed) cell holds k runs on
# k different test sets. Keying without the fold kept whichever row was read
# last and dropped the other k-1 without a word, so a 600-pair comparison
# quietly became a 60-pair one - and each surviving "pair" could straddle two
# different test sets.
# ---------------------------------------------------------------------------

def _fold_run(config_id, dataset, gamma, seed, fold, test_acc, r_bar_after=None):
    return {
        "run_id": make_run_id("A", config_id, dataset, gamma, seed, fold=fold),
        "tier": "A", "config_id": config_id, "dataset": dataset,
        "gamma": gamma, "seed": seed, "fold": fold, "status": "ok", "error": "",
        "test_acc": test_acc, "r_bar_after": r_bar_after,
    }


def test_every_fold_contributes_its_own_pair(tmp_path):
    rows = []
    for fold in range(5):
        rows.append(_fold_run("A0", "MUTAG", 0.0, 0, fold, test_acc=0.60))
        rows.append(_fold_run("A0", "MUTAG", 0.1, 0, fold, test_acc=0.65))
    runs = _store_with_runs(tmp_path, rows)

    significance = paired_osq_significance(runs, metric="test_acc")
    assert len(significance) == 1
    assert significance[0]["n_pairs"] == 5  # not 1
    assert abs(significance[0]["mean_delta"] - 0.05) < 1e-9


def test_a_pair_never_straddles_two_folds(tmp_path):
    # gamma=0 is only present on fold 0; the gamma=0.1 run on fold 1 has no twin
    # and must be dropped rather than paired against fold 0's baseline.
    rows = [
        _fold_run("A0", "MUTAG", 0.0, 0, 0, test_acc=0.60),
        _fold_run("A0", "MUTAG", 0.1, 0, 0, test_acc=0.65),
        _fold_run("A0", "MUTAG", 0.1, 0, 1, test_acc=0.95),
    ]
    runs = _store_with_runs(tmp_path, rows)

    significance = paired_osq_significance(runs, metric="test_acc")
    assert significance[0]["n_pairs"] == 1
    assert abs(significance[0]["mean_delta"] - 0.05) < 1e-9  # not 0.175


def test_pairing_still_works_on_rows_from_before_cross_validation(tmp_path):
    # The two existing results trees have no `fold` column at all.
    rows = [_run("A0", "MUTAG", 0.0, seed, test_acc=0.60) for seed in range(3)]
    rows += [_run("A0", "MUTAG", 0.1, seed, test_acc=0.65) for seed in range(3)]
    runs = _store_with_runs(tmp_path, rows)

    significance = paired_osq_significance(runs, metric="test_acc")
    assert significance[0]["n_pairs"] == 3


def test_osq_reduction_pairs_carry_their_fold(tmp_path):
    rows = []
    for fold in range(3):
        rows.append(_fold_run("A0", "MUTAG", 0.0, 0, fold, test_acc=0.60, r_bar_after=2.0))
        rows.append(_fold_run("A0", "MUTAG", 0.1, 0, fold, test_acc=0.65, r_bar_after=1.0))
    runs = _store_with_runs(tmp_path, rows)

    pairs = osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after", acc_metric="test_acc")
    assert len(pairs) == 3
    assert {pair["fold"] for pair in pairs} == {"0", "1", "2"}
