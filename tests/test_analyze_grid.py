import csv

from tools.results_store import ResultsStore, make_run_id
from results.analyze_grid import (
    _missing_configs_note,
    _xtick_label,
    best_run_per_config,
    brick_labels,
    load_epochs,
    load_runs,
    osq_effect,
    plot_accuracy_by_config,
    plot_accuracy_curves,
    plot_loss_curves,
    plot_osq_effect,
    plot_runtime_by_dataset,
    runtime_by_dataset,
    status_report,
    summarize_by_config,
    write_csv,
)


def _run(config_id, dataset, gamma, seed, test_acc, val_acc=None, runtime_s=10.0,
         peak_mem_mb=100.0, status="ok", error=""):
    return {
        "run_id": make_run_id("A", config_id, dataset, gamma, seed),
        "tier": "A", "config_id": config_id, "dataset": dataset,
        "gamma": gamma, "seed": seed, "status": status, "error": error,
        "test_acc": test_acc, "val_acc": val_acc if val_acc is not None else test_acc,
        "runtime_s": runtime_s, "peak_mem_mb": peak_mem_mb,
    }


def _store_with_runs(tmp_path, rows):
    store = ResultsStore(tmp_path / "results")
    for row in rows:
        store.append_run(row)
    return store


# -- load_runs / load_epochs: numeric coercion ------------------------------

def test_load_runs_coerces_numeric_columns_and_blanks_to_none(tmp_path):
    store = _store_with_runs(tmp_path, [_run("A0", "MUTAG", 0.0, 0, test_acc=0.7)])
    runs = load_runs(store.out_dir)
    assert len(runs) == 1
    row = runs[0]
    assert row["test_acc"] == 0.7
    assert row["gamma"] == 0.0
    assert row["seed"] == 0
    # Columns never filled in (e.g. a failed run with no r_bar_after) must
    # come back as None, never as the empty string the CSV actually stores.
    assert row["r_bar_after"] is None
    # Non-numeric columns are passed through untouched.
    assert row["dataset"] == "MUTAG"
    assert row["status"] == "ok"


def test_load_epochs_coerces_numeric_columns(tmp_path):
    store = ResultsStore(tmp_path / "results")
    store.append_epochs("run_a", [{"epoch": 1, "train_loss": 0.5, "val_acc": 0.6}])
    epochs = load_epochs(store.out_dir)
    assert epochs[0]["epoch"] == 1
    assert epochs[0]["train_loss"] == 0.5
    # Never filled in this call -> None, not "".
    assert epochs[0]["test_acc"] is None


# -- status_report -----------------------------------------------------------

def test_status_report_counts_ok_and_failed_and_captures_first_error(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.7),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.8),
        _run("A7", "MUTAG", 0.0, 0, test_acc=None, status="failed", error="CUDA assert\nmore detail"),
        _run("A7", "MUTAG", 0.1, 1, test_acc=None, status="failed", error="a different error"),
    ]
    store = _store_with_runs(tmp_path, rows)
    report = {r["config_id"]: r for r in status_report(load_runs(store.out_dir))}

    assert report["A0"] == {"config_id": "A0", "ok": 2, "failed": 0, "sample_error": ""}
    assert report["A7"]["ok"] == 0
    assert report["A7"]["failed"] == 2
    # Only the FIRST error is kept (whichever row is encountered first), truncated to one line.
    assert report["A7"]["sample_error"] == "CUDA assert"


def test_status_report_handles_a_config_with_zero_ok_runs(tmp_path):
    rows = [_run("A8", "MUTAG", 0.0, 0, test_acc=None, status="failed", error="boom")]
    store = _store_with_runs(tmp_path, rows)
    report = status_report(load_runs(store.out_dir))
    assert report == [{"config_id": "A8", "ok": 0, "failed": 1, "sample_error": "boom"}]


# -- summarize_by_config -------------------------------------------------------

def test_summarize_by_config_computes_mean_and_std_over_ok_runs_only(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.8),
        _run("A0", "PROTEINS", 0.0, 0, test_acc=1.0, status="failed", error="x"),
    ]
    store = _store_with_runs(tmp_path, rows)
    summary = {r["config_id"]: r for r in summarize_by_config(load_runs(store.out_dir), "test_acc")}

    assert summary["A0"]["n"] == 2
    assert summary["A0"]["mean"] == 0.7
    assert abs(summary["A0"]["std"] - 0.14142135623730951) < 1e-9


def test_summarize_by_config_a_single_ok_run_has_zero_std(tmp_path):
    store = _store_with_runs(tmp_path, [_run("A0", "MUTAG", 0.0, 0, test_acc=0.5)])
    summary = summarize_by_config(load_runs(store.out_dir), "test_acc")
    assert summary == [{"config_id": "A0", "n": 1, "mean": 0.5, "std": 0.0}]


def test_summarize_by_config_a_config_with_no_ok_runs_is_absent_not_crashed(tmp_path):
    store = _store_with_runs(tmp_path, [_run("A8", "MUTAG", 0.0, 0, test_acc=None, status="failed")])
    assert summarize_by_config(load_runs(store.out_dir), "test_acc") == []


# -- runtime_by_dataset ---------------------------------------------------------

def test_runtime_by_dataset_aggregates_runtime_and_memory(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.7, runtime_s=10.0, peak_mem_mb=100.0),
        _run("A1", "MUTAG", 0.0, 0, test_acc=0.7, runtime_s=20.0, peak_mem_mb=200.0),
        _run("A0", "NCI1", 0.0, 0, test_acc=0.5, runtime_s=100.0, peak_mem_mb=500.0,
             status="failed", error="oom"),
    ]
    store = _store_with_runs(tmp_path, rows)
    by_ds = {r["dataset"]: r for r in runtime_by_dataset(load_runs(store.out_dir))}

    assert by_ds["MUTAG"]["n"] == 2
    assert by_ds["MUTAG"]["runtime_s_mean"] == 15.0
    assert by_ds["MUTAG"]["peak_mem_mb_mean"] == 150.0
    # NCI1's only run failed, so it contributes no (runtime, memory) sample.
    assert "NCI1" not in by_ds


# -- osq_effect -------------------------------------------------------------

def test_osq_effect_pairs_gamma_zero_against_the_largest_positive_gamma(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.6),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.8),
        _run("A0", "MUTAG", 0.1, 1, test_acc=0.8),
    ]
    store = _store_with_runs(tmp_path, rows)
    [row] = osq_effect(load_runs(store.out_dir), "test_acc")

    assert row["config_id"] == "A0"
    assert row["gamma_off_mean"] == 0.6
    assert row["gamma_on"] == 0.1
    assert row["gamma_on_mean"] == 0.8
    assert abs(row["delta"] - 0.2) < 1e-9


def test_osq_effect_handles_a_config_missing_one_side(tmp_path):
    # Only gamma=0 runs exist for this config (e.g. the gamma>0 seeds all failed).
    store = _store_with_runs(tmp_path, [_run("A0", "MUTAG", 0.0, 0, test_acc=0.6)])
    [row] = osq_effect(load_runs(store.out_dir), "test_acc")
    assert row["gamma_off_mean"] == 0.6
    assert row["gamma_off_n"] == 1
    assert row["gamma_on_mean"] is None
    assert row["gamma_on_n"] == 0
    assert row["delta"] is None


def test_osq_effect_tracks_unequal_sample_sizes_per_side(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.6),
        _run("A0", "MUTAG", 0.1, 0, test_acc=0.8),  # only one gamma=0.1 seed survived
    ]
    store = _store_with_runs(tmp_path, rows)
    [row] = osq_effect(load_runs(store.out_dir), "test_acc")
    assert row["gamma_off_n"] == 2
    assert row["gamma_on_n"] == 1


# -- brick_labels / _xtick_label / _missing_configs_note ---------------------

def _run_with_bricks(config_id, dataset, seed, s1="gcn", s2="topk", s3="deepsets",
                      s4="gumbel", s5="gnn_rewired", status="ok"):
    row = _run(config_id, dataset, 0.0, seed, test_acc=0.6, status=status)
    row.update({"s1_encoder": s1, "s2_proposal": s2, "s3_cell_encoder": s3,
                "s4_selector": s4, "s5_mp": s5})
    return row


def test_brick_labels_names_the_one_stage_that_differs_from_baseline(tmp_path):
    rows = [
        _run_with_bricks("A0", "MUTAG", 0),
        _run_with_bricks("A4", "MUTAG", 0, s3="set_transformer"),
    ]
    store = _store_with_runs(tmp_path, rows)
    labels = brick_labels(load_runs(store.out_dir))
    assert labels["A0"] == "baseline"
    assert labels["A4"] == "set_transformer"


def test_brick_labels_omits_a_config_with_zero_successful_runs(tmp_path):
    rows = [
        _run_with_bricks("A0", "MUTAG", 0),
        _run("A7", "MUTAG", 0.0, 0, test_acc=None, status="failed", error="boom"),
    ]
    store = _store_with_runs(tmp_path, rows)
    labels = brick_labels(load_runs(store.out_dir))
    assert "A7" not in labels


def test_xtick_label_falls_back_to_the_bare_config_id_without_a_brief():
    assert _xtick_label("A7", {}) == "A7"
    assert _xtick_label("A7", None) == "A7"
    assert _xtick_label("A4", {"A4": "set_transformer"}) == "A4\n(set_transformer)"


def test_missing_configs_note_lists_zero_run_configs():
    note = _missing_configs_note({"A0", "A1", "A7"}, {"A0", "A1"})
    assert note == "Not shown: A7 (0 successful runs - see status_report.csv)"


def test_missing_configs_note_is_none_when_nothing_is_missing():
    assert _missing_configs_note({"A0", "A1"}, {"A0", "A1"}) is None
    assert _missing_configs_note(None, {"A0"}) is None


# -- best_run_per_config -------------------------------------------------------

def test_best_run_per_config_picks_highest_val_acc_and_ignores_failed(tmp_path):
    rows = [
        _run("A0", "MUTAG", 0.0, 0, test_acc=0.5, val_acc=0.5),
        _run("A0", "MUTAG", 0.0, 1, test_acc=0.9, val_acc=0.9),
        _run("A0", "MUTAG", 0.0, 2, test_acc=0.99, val_acc=0.1, status="failed", error="x"),
    ]
    store = _store_with_runs(tmp_path, rows)
    best = best_run_per_config(load_runs(store.out_dir))
    assert best["A0"]["seed"] == 1


# -- CSV round trip -----------------------------------------------------------

def test_write_csv_round_trips_through_dictreader(tmp_path):
    rows = [{"config_id": "A0", "mean": 0.7, "std": 0.1}]
    path = tmp_path / "out.csv"
    write_csv(rows, path)
    with open(path, newline="") as f:
        read_back = list(csv.DictReader(f))
    assert read_back == [{"config_id": "A0", "mean": "0.7", "std": "0.1"}]


def test_write_csv_does_nothing_for_an_empty_list(tmp_path):
    path = tmp_path / "out.csv"
    write_csv([], path)
    assert not path.exists()


# -- plotting: smoke tests (a file is produced, not pixel-perfect) -----------

def test_plot_accuracy_by_config_writes_a_file(tmp_path):
    out_path = tmp_path / "acc.png"
    plot_accuracy_by_config(
        [{"config_id": "A0", "n": 2, "mean": 0.7, "std": 0.1}], out_path,
        config_briefs={"A0": "baseline"}, all_config_ids={"A0", "A7"},
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_runtime_by_dataset_writes_a_file(tmp_path):
    out_path = tmp_path / "runtime.png"
    plot_runtime_by_dataset(
        [{"dataset": "MUTAG", "n": 2, "runtime_s_mean": 10.0, "runtime_s_std": 1.0,
          "peak_mem_mb_mean": 100.0, "peak_mem_mb_std": 5.0}],
        out_path,
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_osq_effect_writes_a_file(tmp_path):
    out_path = tmp_path / "osq.png"
    plot_osq_effect(
        [{"config_id": "A0", "gamma_off_mean": 0.6, "gamma_off_std": 0.1, "gamma_off_n": 6,
          "gamma_on": 0.1, "gamma_on_mean": 0.8, "gamma_on_std": 0.1, "gamma_on_n": 6, "delta": 0.2}],
        out_path, config_briefs={"A0": "baseline"}, all_config_ids={"A0", "A7"},
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_osq_effect_writes_a_file_without_n_counts(tmp_path):
    # Older-shaped rows (no gamma_off_n/gamma_on_n) must not crash the annotator.
    out_path = tmp_path / "osq_no_n.png"
    plot_osq_effect(
        [{"config_id": "A0", "gamma_off_mean": 0.6, "gamma_off_std": 0.1,
          "gamma_on": 0.1, "gamma_on_mean": 0.8, "gamma_on_std": 0.1, "delta": 0.2}],
        out_path,
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_loss_and_accuracy_curves_write_files(tmp_path):
    history = [
        {"epoch": 1, "train_task": 0.8, "val_loss": 0.7, "test_loss": 0.75,
         "train_acc": 0.5, "val_acc": 0.5, "test_acc": 0.5},
        {"epoch": 2, "train_task": 0.5, "val_loss": 0.6, "test_loss": 0.65,
         "train_acc": 0.7, "val_acc": 0.6, "test_acc": 0.6},
    ]
    run_row = {"run_id": "A_A0_MUTAG_g0.0_s0", "dataset": "MUTAG", "config_id": "A0",
               "gamma": 0.0, "seed": 0, "best_epoch": 2}

    loss_path = tmp_path / "loss.png"
    acc_path = tmp_path / "acc.png"
    plot_loss_curves(history, run_row, loss_path)
    plot_accuracy_curves(history, run_row, acc_path)

    assert loss_path.exists() and loss_path.stat().st_size > 0
    assert acc_path.exists() and acc_path.stat().st_size > 0


# -- end-to-end sanity on a tiny grid -----------------------------------------

def test_end_to_end_over_a_small_synthetic_grid(tmp_path):
    """Builds a miniature but realistic results/ tree (2 configs x 2 datasets
    x 2 seeds x 2 gammas, one config entirely failed) through the real
    ResultsStore, then runs every analysis function over it and checks the
    numbers are internally consistent - this is the check that the analysis
    module actually agrees with the schema results_store.py writes."""
    store = ResultsStore(tmp_path / "results")
    for dataset in ("MUTAG", "PROTEINS"):
        for seed in (0, 1):
            for gamma in (0.0, 0.1):
                acc = 0.6 if gamma == 0.0 else 0.8
                store.append_run(_run("A0", dataset, gamma, seed, test_acc=acc,
                                       runtime_s=5.0, peak_mem_mb=50.0))
                store.append_epochs(
                    make_run_id("A", "A0", dataset, gamma, seed),
                    [{"epoch": e, "train_task": 1.0 / e, "val_loss": 1.0 / e, "test_loss": 1.0 / e,
                      "train_acc": acc, "val_acc": acc, "test_acc": acc} for e in (1, 2)],
                )
                store.append_run(_run("A7", dataset, gamma, seed, test_acc=None,
                                       status="failed", error="CUDA assert"))

    runs = load_runs(store.out_dir)
    epochs = load_epochs(store.out_dir)
    assert len(runs) == 16
    assert len(epochs) == 16  # 8 successful A0 runs x 2 epochs each

    report = {r["config_id"]: r for r in status_report(runs)}
    assert report["A0"]["ok"] == 8 and report["A0"]["failed"] == 0
    assert report["A7"]["ok"] == 0 and report["A7"]["failed"] == 8

    accuracy = {r["config_id"]: r for r in summarize_by_config(runs, "test_acc")}
    assert "A7" not in accuracy  # nothing succeeded, so no accuracy to report
    assert accuracy["A0"]["n"] == 8

    runtime = {r["dataset"]: r for r in runtime_by_dataset(runs)}
    assert runtime["MUTAG"]["n"] == 4 and runtime["PROTEINS"]["n"] == 4

    [osq_row] = osq_effect(runs, "test_acc")
    assert osq_row["config_id"] == "A0"
    assert abs(osq_row["gamma_off_mean"] - 0.6) < 1e-9
    assert abs(osq_row["gamma_on_mean"] - 0.8) < 1e-9

    best = best_run_per_config(runs)
    assert set(best.keys()) == {"A0"}  # A7 has no successful run to pick
