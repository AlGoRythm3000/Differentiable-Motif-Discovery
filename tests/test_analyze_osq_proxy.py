import csv

from legacy_osq_schema import LegacyOsqStore, legacy_run_id
from results.analyze_osq_proxy import (
    _missing_proxies_note,
    _xtick_label,
    best_run_per_proxy,
    load_epochs,
    load_runs,
    plot_accuracy_by_proxy,
    plot_accuracy_curves,
    plot_loss_curves,
    plot_runtime_by_dataset,
    runtime_by_dataset,
    status_report,
    summarize_by_proxy,
    write_csv,
)


def _run(proxy, dataset, gamma, seed, test_acc, val_acc=None, runtime_s=10.0,
         status="ok", error=""):
    return {
        "run_id": legacy_run_id(dataset, proxy, gamma, seed),
        "dataset": dataset, "proxy": proxy, "gamma": gamma, "seed": seed,
        "status": status, "error": error,
        "test_acc": test_acc, "val_acc": val_acc if val_acc is not None else test_acc,
        "runtime_s": runtime_s,
    }


def _store_with_runs(tmp_path, rows):
    store = LegacyOsqStore(tmp_path / "results")
    for row in rows:
        store.append_run(row)
    return store


# -- load_runs / load_epochs: numeric coercion ------------------------------

def test_load_runs_coerces_numeric_columns_and_blanks_to_none(tmp_path):
    store = _store_with_runs(tmp_path, [_run("none", "MUTAG", 0.0, 0, test_acc=0.7)])
    runs = load_runs(store.out_dir)
    assert len(runs) == 1
    row = runs[0]
    assert row["test_acc"] == 0.7
    assert row["gamma"] == 0.0
    assert row["seed"] == 0
    # Columns never filled in (e.g. a failed run with no r_bar_after) must
    # come back as None, never as the empty string the CSV actually stores.
    assert row["r_bar_after"] is None
    assert row["dataset"] == "MUTAG"
    assert row["proxy"] == "none"
    assert row["status"] == "ok"


def test_load_epochs_coerces_numeric_columns(tmp_path):
    store = LegacyOsqStore(tmp_path / "results")
    store.append_epochs("run_a", [{"epoch": 1, "train_loss": 0.5, "val_acc": 0.6}])
    epochs = load_epochs(store.out_dir)
    assert epochs[0]["epoch"] == 1
    assert epochs[0]["train_loss"] == 0.5
    assert epochs[0]["test_acc"] is None


# -- status_report -----------------------------------------------------------

def test_status_report_counts_ok_and_failed_and_captures_first_error(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.7),
        _run("none", "MUTAG", 0.0, 1, test_acc=0.8),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=None, status="failed", error="NaN loss\nmore detail"),
        _run("r_bar", "MUTAG", 0.01, 1, test_acc=None, status="failed", error="a different error"),
    ]
    store = _store_with_runs(tmp_path, rows)
    report = {r["proxy"]: r for r in status_report(load_runs(store.out_dir))}

    assert report["none"] == {"proxy": "none", "ok": 2, "failed": 0, "sample_error": ""}
    assert report["r_bar"]["ok"] == 0
    assert report["r_bar"]["failed"] == 2
    # Only the FIRST error is kept (whichever row is encountered first), truncated to one line.
    assert report["r_bar"]["sample_error"] == "NaN loss"


def test_status_report_handles_a_proxy_with_zero_ok_runs(tmp_path):
    rows = [_run("lambda2", "MUTAG", 0.01, 0, test_acc=None, status="failed", error="boom")]
    store = _store_with_runs(tmp_path, rows)
    report = status_report(load_runs(store.out_dir))
    assert report == [{"proxy": "lambda2", "ok": 0, "failed": 1, "sample_error": "boom"}]


# -- summarize_by_proxy -------------------------------------------------------

def test_summarize_by_proxy_computes_mean_and_std_over_ok_runs_only(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.6),
        _run("none", "MUTAG", 0.0, 1, test_acc=0.8),
        _run("none", "PROTEINS", 0.0, 0, test_acc=1.0, status="failed", error="x"),
    ]
    store = _store_with_runs(tmp_path, rows)
    summary = {r["proxy"]: r for r in summarize_by_proxy(load_runs(store.out_dir), "test_acc")}

    assert summary["none"]["n"] == 2
    assert summary["none"]["mean"] == 0.7
    assert abs(summary["none"]["std"] - 0.14142135623730951) < 1e-9


def test_summarize_by_proxy_can_restrict_to_one_dataset(tmp_path):
    rows = [
        _run("r_bar", "synthetic_bottleneck", 0.01, 0, test_acc=0.9),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.5),
    ]
    store = _store_with_runs(tmp_path, rows)
    runs = load_runs(store.out_dir)
    [row] = summarize_by_proxy(runs, "test_acc", dataset="synthetic_bottleneck")
    assert row == {"proxy": "r_bar", "n": 1, "mean": 0.9, "std": 0.0}


def test_summarize_by_proxy_a_proxy_with_no_ok_runs_is_absent_not_crashed(tmp_path):
    store = _store_with_runs(tmp_path, [_run("efc", "MUTAG", 0.01, 0, test_acc=None, status="failed")])
    assert summarize_by_proxy(load_runs(store.out_dir), "test_acc") == []


# -- runtime_by_dataset ---------------------------------------------------------

def test_runtime_by_dataset_aggregates_runtime(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.7, runtime_s=10.0),
        _run("r_bar", "MUTAG", 0.01, 0, test_acc=0.7, runtime_s=20.0),
        _run("none", "NCI1", 0.0, 0, test_acc=0.5, runtime_s=100.0, status="failed", error="oom"),
    ]
    store = _store_with_runs(tmp_path, rows)
    by_ds = {r["dataset"]: r for r in runtime_by_dataset(load_runs(store.out_dir))}

    assert by_ds["MUTAG"]["n"] == 2
    assert by_ds["MUTAG"]["runtime_s_mean"] == 15.0
    # NCI1's only run failed, so it contributes no runtime sample.
    assert "NCI1" not in by_ds


# -- best_run_per_proxy -------------------------------------------------------

def test_best_run_per_proxy_picks_highest_val_acc_and_ignores_failed(tmp_path):
    rows = [
        _run("none", "MUTAG", 0.0, 0, test_acc=0.5, val_acc=0.5),
        _run("none", "MUTAG", 0.0, 1, test_acc=0.9, val_acc=0.9),
        _run("none", "MUTAG", 0.0, 2, test_acc=0.99, val_acc=0.1, status="failed", error="x"),
    ]
    store = _store_with_runs(tmp_path, rows)
    best = best_run_per_proxy(load_runs(store.out_dir))
    assert best["none"]["seed"] == 1


# -- _xtick_label / _missing_proxies_note -------------------------------------

def test_xtick_label_expands_known_proxies_and_falls_back_for_unknown():
    label = _xtick_label("r_bar")
    assert label.startswith("r_bar\n(")
    assert label.endswith("primary)")
    assert "mean effective" in label and "resistance" in label
    assert _xtick_label("totally_unknown_proxy") == "totally_unknown_proxy"


def test_missing_proxies_note_lists_zero_run_proxies():
    note = _missing_proxies_note({"none", "r_bar", "efc"}, {"none", "r_bar"})
    assert note == "Not shown: efc (0 successful runs - see status_report.csv)"


def test_missing_proxies_note_is_none_when_nothing_is_missing():
    assert _missing_proxies_note({"none", "r_bar"}, {"none", "r_bar"}) is None
    assert _missing_proxies_note(None, {"none"}) is None


# -- CSV round trip -----------------------------------------------------------

def test_write_csv_round_trips_through_dictreader(tmp_path):
    rows = [{"proxy": "r_bar", "mean": 0.7, "std": 0.1}]
    path = tmp_path / "out.csv"
    write_csv(rows, path)
    with open(path, newline="") as f:
        read_back = list(csv.DictReader(f))
    assert read_back == [{"proxy": "r_bar", "mean": "0.7", "std": "0.1"}]


def test_write_csv_does_nothing_for_an_empty_list(tmp_path):
    path = tmp_path / "out.csv"
    write_csv([], path)
    assert not path.exists()


# -- plotting: smoke tests (a file is produced, not pixel-perfect) -----------

def test_plot_accuracy_by_proxy_writes_a_file(tmp_path):
    out_path = tmp_path / "acc.png"
    plot_accuracy_by_proxy(
        [{"proxy": "none", "n": 2, "mean": 0.7, "std": 0.1}], out_path,
        all_proxies={"none", "r_bar", "efc"},
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_runtime_by_dataset_writes_a_file(tmp_path):
    out_path = tmp_path / "runtime.png"
    plot_runtime_by_dataset(
        [{"dataset": "synthetic_bottleneck", "n": 2, "runtime_s_mean": 10.0, "runtime_s_std": 1.0}],
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
    run_row = {"run_id": "MUTAG_r_bar_g0.01_s0", "dataset": "MUTAG", "proxy": "r_bar",
               "gamma": 0.01, "seed": 0, "best_epoch": 2}

    loss_path = tmp_path / "loss.png"
    acc_path = tmp_path / "acc.png"
    plot_loss_curves(history, run_row, loss_path)
    plot_accuracy_curves(history, run_row, acc_path)

    assert loss_path.exists() and loss_path.stat().st_size > 0
    assert acc_path.exists() and acc_path.stat().st_size > 0


# -- end-to-end sanity on a tiny grid -----------------------------------------

def test_end_to_end_over_a_small_synthetic_grid(tmp_path):
    """Builds a miniature but realistic results/ tree (2 proxies x 2
    datasets x 2 seeds, one proxy entirely failed) through the real
    the legacy feat/osq-proxy schema, then runs every analysis function over it
    and checks the
    numbers are internally consistent."""
    store = LegacyOsqStore(tmp_path / "results")
    for dataset in ("MUTAG", "synthetic_bottleneck"):
        for seed in (0, 1):
            store.append_run(_run("none", dataset, 0.0, seed, test_acc=0.6, runtime_s=5.0))
            store.append_epochs(
                legacy_run_id(dataset, "none", 0.0, seed),
                [{"epoch": e, "train_task": 1.0 / e, "val_loss": 1.0 / e, "test_loss": 1.0 / e,
                  "train_acc": 0.6, "val_acc": 0.6, "test_acc": 0.6} for e in (1, 2)],
            )
            store.append_run(_run("lambda2", dataset, 0.01, seed, test_acc=None,
                                   status="failed", error="NaN"))

    runs = load_runs(store.out_dir)
    epochs = load_epochs(store.out_dir)
    assert len(runs) == 8
    assert len(epochs) == 8  # 4 successful "none" runs x 2 epochs each

    report = {r["proxy"]: r for r in status_report(runs)}
    assert report["none"]["ok"] == 4 and report["none"]["failed"] == 0
    assert report["lambda2"]["ok"] == 0 and report["lambda2"]["failed"] == 4

    accuracy = {r["proxy"]: r for r in summarize_by_proxy(runs, "test_acc")}
    assert "lambda2" not in accuracy  # nothing succeeded, so no accuracy to report
    assert accuracy["none"]["n"] == 4

    synthetic_only = summarize_by_proxy(runs, "test_acc", dataset="synthetic_bottleneck")
    assert synthetic_only == [{"proxy": "none", "n": 2, "mean": 0.6, "std": 0.0}]

    runtime = {r["dataset"]: r for r in runtime_by_dataset(runs)}
    assert runtime["MUTAG"]["n"] == 2 and runtime["synthetic_bottleneck"]["n"] == 2

    best = best_run_per_proxy(runs)
    assert set(best.keys()) == {"none"}  # lambda2 has no successful run to pick
