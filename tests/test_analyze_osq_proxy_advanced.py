from legacy_osq_schema import LegacyOsqStore, legacy_run_id
from results.analyze_osq_proxy import load_runs
from results.analyze_osq_proxy_advanced import (
    correlate_osq_reduction_with_accuracy,
    osq_reduction_vs_accuracy,
    paired_proxy_significance,
    plot_osq_reduction_vs_accuracy,
    plot_proxy_ablation,
    plot_proxy_significance,
    proxy_ablation,
    proxy_ablation_summary,
)


def _run(proxy, dataset, gamma, seed, test_acc, r_bar_after=None, lambda2_after=None,
         status="ok", error=""):
    return {
        "run_id": legacy_run_id(dataset, proxy, gamma, seed),
        "dataset": dataset, "proxy": proxy, "gamma": gamma, "seed": seed,
        "status": status, "error": error,
        "test_acc": test_acc, "r_bar_after": r_bar_after, "lambda2_after": lambda2_after,
    }


def _store_with_runs(tmp_path, rows):
    store = LegacyOsqStore(tmp_path / "results")
    for row in rows:
        store.append_run(row)
    return load_runs(store.out_dir)


# -- proxy_ablation / proxy_ablation_summary ---------------------------------

def test_proxy_ablation_computes_delta_against_baseline_per_dataset(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("none", "MUTAG", 0.0, 1, test_acc=0.7),
        _run("none", "PROTEINS", 0.0, 0, test_acc=0.5),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.8),
        _run("r_bar", "MUTAG", 0.01, 1, test_acc=0.9),
        _run("r_bar", "PROTEINS", 0.01, 0, test_acc=0.4),
    ]
    runs = _store_with_runs(tmp_path, rows)
    ablation = {r["dataset"]: r for r in proxy_ablation(runs) if r["proxy"] == "r_bar"}

    assert abs(ablation["MUTAG"]["baseline_mean"] - 0.65) < 1e-9
    assert abs(ablation["MUTAG"]["proxy_mean"] - 0.85) < 1e-9
    assert abs(ablation["MUTAG"]["delta"] - 0.2) < 1e-9
    assert abs(ablation["PROTEINS"]["delta"] - (-0.1)) < 1e-9


def test_proxy_ablation_skips_datasets_the_baseline_never_ran(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.7),
        _run("r_bar", "NCI1", 0.01, 0, test_acc=0.9),
    ]
    runs = _store_with_runs(tmp_path, rows)
    datasets = {r["dataset"] for r in proxy_ablation(runs) if r["proxy"] == "r_bar"}
    assert datasets == {"MUTAG"}


def test_proxy_ablation_summary_averages_deltas_unweighted_across_datasets(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.5),
        _run("none", "PROTEINS", 0.0, 0, test_acc=0.5),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.7),    # delta +0.2
        _run("r_bar", "PROTEINS", 0.01, 0, test_acc=0.4),  # delta -0.1
    ]
    runs = _store_with_runs(tmp_path, rows)
    [summary] = proxy_ablation_summary(proxy_ablation(runs))
    assert summary["proxy"] == "r_bar"
    assert summary["n_datasets"] == 2
    assert abs(summary["mean_delta"] - 0.05) < 1e-9


# -- paired_proxy_significance --------------------------------------------------

def test_paired_proxy_significance_computes_mean_delta_and_stats_over_matched_pairs(tmp_path):
    rows = []
    for seed in range(6):
        rows.append(_run("none", "MUTAG", 0.0, seed, test_acc=0.5))
        rows.append(_run("r_bar", "MUTAG", 0.01, seed, test_acc=0.7))
    runs = _store_with_runs(tmp_path, rows)
    [row] = paired_proxy_significance(runs, "test_acc")

    assert row["proxy"] == "r_bar"
    assert row["n_pairs"] == 6
    assert abs(row["mean_delta"] - 0.2) < 1e-9
    assert row["t_pvalue"] is not None
    assert row["t_pvalue"] < 0.05


def test_paired_proxy_significance_ignores_unmatched_runs(tmp_path):
    # Only "none" exists on this dataset - there is nothing to pair against.
    rows = [_run("none", "MUTAG", 0.0, 0, test_acc=0.5)]
    runs = _store_with_runs(tmp_path, rows)
    assert paired_proxy_significance(runs, "test_acc") == []


def test_paired_proxy_significance_handles_too_few_pairs_for_a_stable_test(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.5),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.5),  # zero variance in the single difference
    ]
    runs = _store_with_runs(tmp_path, rows)
    [row] = paired_proxy_significance(runs, "test_acc")
    assert row["n_pairs"] == 1
    assert row["mean_delta"] == 0.0
    assert row["t_stat"] is None or row["t_stat"] == row["t_stat"]  # never NaN if not None


# -- osq_reduction_vs_accuracy / correlate_osq_reduction_with_accuracy --------

def test_osq_reduction_vs_accuracy_pairs_matching_runs(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.5, r_bar_after=2.0),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.8, r_bar_after=0.5),
    ]
    runs = _store_with_runs(tmp_path, rows)
    [pair] = osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after")
    assert pair["proxy"] == "r_bar" and pair["dataset"] == "MUTAG" and pair["seed"] == 0
    assert abs(pair["delta_osq"] - (-1.5)) < 1e-9
    assert abs(pair["delta_acc"] - 0.3) < 1e-9


def test_osq_reduction_vs_accuracy_skips_pairs_missing_the_osq_metric(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.5, r_bar_after=None),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.8, r_bar_after=0.5),
    ]
    runs = _store_with_runs(tmp_path, rows)
    assert osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after") == []


def test_correlate_osq_reduction_with_accuracy_detects_a_perfect_negative_correlation():
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


def test_correlate_osq_reduction_with_accuracy_handles_no_spread():
    pairs = [{"delta_osq": -1.0, "delta_acc": 0.1}]
    result = correlate_osq_reduction_with_accuracy(pairs)
    assert result == {"n": 1, "pearson_r": None, "pearson_pvalue": None,
                       "spearman_r": None, "spearman_pvalue": None}


def test_correlate_osq_reduction_with_accuracy_handles_empty_input():
    assert correlate_osq_reduction_with_accuracy([]) == {
        "n": 0, "pearson_r": None, "pearson_pvalue": None,
        "spearman_r": None, "spearman_pvalue": None,
    }


# -- plotting: smoke tests -----------------------------------------------------

def test_plot_proxy_ablation_writes_a_file(tmp_path):
    out_path = tmp_path / "ablation.png"
    plot_proxy_ablation([{"proxy": "r_bar", "n_datasets": 3, "mean_delta": 0.05, "std_delta": 0.02}],
                         out_path, all_proxies={"none", "r_bar", "efc"})
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_proxy_significance_writes_a_file(tmp_path):
    out_path = tmp_path / "significance.png"
    plot_proxy_significance(
        [{"proxy": "r_bar", "n_pairs": 6, "mean_delta": 0.2,
          "t_stat": 5.0, "t_pvalue": 0.01, "wilcoxon_stat": 0.0, "wilcoxon_pvalue": 0.02}],
        out_path, all_proxies={"none", "r_bar"},
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_osq_reduction_vs_accuracy_writes_a_file(tmp_path):
    out_path = tmp_path / "scatter.png"
    pairs = [{"proxy": "r_bar", "dataset": "MUTAG", "seed": 0, "delta_osq": -1.0, "delta_acc": 0.2},
             {"proxy": "efc", "dataset": "MUTAG", "seed": 0, "delta_osq": 0.5, "delta_acc": -0.1}]
    correlation = correlate_osq_reduction_with_accuracy(pairs)
    plot_osq_reduction_vs_accuracy(pairs, correlation, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0


# -- end-to-end sanity --------------------------------------------------------

def test_end_to_end_over_a_small_synthetic_grid(tmp_path):
    """"none" is the baseline; "r_bar" adds a fixed +0.2 accuracy bump on
    every dataset and roughly halves r_bar_after - checks the whole
    pipeline agrees with those built-in, hand-computable effects."""
    store = LegacyOsqStore(tmp_path / "results")
    for dataset in ("MUTAG", "PROTEINS"):
        for seed in range(4):
            store.append_run(_run("none", dataset, 0.0, seed, test_acc=0.5, r_bar_after=2.0))
            store.append_run(_run("r_bar", dataset, 0.01, seed, test_acc=0.7, r_bar_after=1.0))
    runs = load_runs(store.out_dir)

    ablation_summary = {r["proxy"]: r for r in proxy_ablation_summary(proxy_ablation(runs))}
    assert abs(ablation_summary["r_bar"]["mean_delta"] - 0.2) < 1e-9

    significance = {r["proxy"]: r for r in paired_proxy_significance(runs, "test_acc")}
    assert significance["r_bar"]["n_pairs"] == 8  # 2 datasets x 4 seeds
    assert abs(significance["r_bar"]["mean_delta"] - 0.2) < 1e-9

    pairs = osq_reduction_vs_accuracy(runs, osq_metric="r_bar_after")
    assert len(pairs) == 8
    assert all(abs(p["delta_osq"] - (-1.0)) < 1e-9 for p in pairs)
    assert all(abs(p["delta_acc"] - 0.2) < 1e-9 for p in pairs)

    correlation = correlate_osq_reduction_with_accuracy(pairs)
    assert correlation["n"] == 8
    assert correlation["pearson_r"] is None  # no spread across identical pairs


def test_a_cross_validated_tree_pairs_per_fold_not_once_per_cell(tmp_path):
    # The legacy analyser keyed pairs on (dataset, seed). Pointed at a tree with
    # folds it would have kept one run per cell and dropped the rest in silence,
    # so a 15-pair comparison would report as 3. Rows written here carry an
    # explicit `fold`; the tests above cover the legacy no-fold rows.
    store = LegacyOsqStore(tmp_path / "results", extra_columns=("fold",))
    for fold in range(5):
        store.append_run(dict(_run("none", "MUTAG", 0.0, 0, test_acc=0.60), fold=fold))
        store.append_run(dict(_run("r_bar", "MUTAG", 0.01, 0, test_acc=0.65), fold=fold))
    runs = load_runs(store.out_dir)

    significance = {r["proxy"]: r for r in paired_proxy_significance(runs, metric="test_acc")}
    assert significance["r_bar"]["n_pairs"] == 5
    assert abs(significance["r_bar"]["mean_delta"] - 0.05) < 1e-9
