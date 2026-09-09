import csv
import json
import zipfile

from tools.results_store import EPOCH_COLUMNS, RUN_COLUMNS, ResultsStore, make_run_id


def _row(**overrides):
    row = {"run_id": make_run_id("A", "A1", "MUTAG", 0.1, 0), "tier": "A", "config_id": "A1",
           "dataset": "MUTAG", "osq_proxy": "r_bar", "gamma": 0.1, "seed": 0, "status": "ok"}
    row.update(overrides)
    return row


def test_run_id_is_deterministic_and_readable():
    assert make_run_id("A", "A1", "MUTAG", 0.1, 2) == "A_A1_MUTAG_g0.1_s2"


def test_runs_csv_keeps_the_frozen_schema(tmp_path):
    store = ResultsStore(tmp_path / "results")
    store.append_run(_row())

    with open(store.runs_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == RUN_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["osq_proxy"] == "r_bar"
    assert rows[0]["config_id"] == "A1"
    # Columns that were not filled in stay present and empty, never missing.
    assert rows[0]["r_bar_after"] == ""
    assert rows[0]["s1_encoder_actual"] == ""


def test_unknown_columns_are_rejected(tmp_path):
    store = ResultsStore(tmp_path / "results")
    try:
        store.append_run(_row(accuracy_typo=0.9))
    except ValueError as error:
        assert "accuracy_typo" in str(error)
    else:
        raise AssertionError("an unknown column must raise")


def test_a_run_id_is_never_silently_overwritten(tmp_path):
    store = ResultsStore(tmp_path / "results")
    store.append_run(_row(test_acc=0.7))
    assert store.has_run(_row()["run_id"])

    try:
        store.append_run(_row(test_acc=0.9))
    except ValueError as error:
        assert "already stored" in str(error)
    else:
        raise AssertionError("a duplicate run_id must raise")

    store.append_run(_row(test_acc=0.9), force=True)
    assert len(store.read_runs()) == 2


def test_appending_a_second_run_keeps_the_first(tmp_path):
    store = ResultsStore(tmp_path / "results")
    store.append_run(_row())
    store.append_run(_row(run_id=make_run_id("A", "A0", "MUTAG", 0.0, 0),
                          config_id="A0", osq_proxy="none", gamma=0.0))
    assert {r["run_id"] for r in store.read_runs()} == {"A_A1_MUTAG_g0.1_s0", "A_A0_MUTAG_g0.0_s0"}


def test_epochs_csv_is_one_row_per_epoch(tmp_path):
    store = ResultsStore(tmp_path / "results")
    history = [{"epoch": e, "train_loss": 1.0 / e, "val_acc": 0.5} for e in (1, 2, 3)]
    store.append_epochs("run_a", history)
    store.append_epochs("run_b", history)

    with open(store.epochs_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EPOCH_COLUMNS
        rows = list(reader)
    assert len(rows) == 6
    assert {r["run_id"] for r in rows} == {"run_a", "run_b"}


def test_raw_and_env_are_json_and_zip_contains_everything(tmp_path):
    store = ResultsStore(tmp_path / "results")
    store.write_env({"torch": "2.0.1", "commit_sha": "abc123"})
    store.append_run(_row())
    store.append_epochs(_row()["run_id"], [{"epoch": 1, "train_loss": 0.5}])
    store.write_raw(_row()["run_id"], {"config": {"gamma": 0.1}, "history": [], "error": None})

    assert json.load(open(store.env_path))["commit_sha"] == "abc123"

    archive = store.zip(tmp_path / "results.zip")
    names = zipfile.ZipFile(archive).namelist()
    assert any(name.endswith("runs.csv") for name in names)
    assert any(name.endswith("epochs.csv") for name in names)
    assert any(name.endswith("env.json") for name in names)
    assert any("raw/" in name for name in names)


def test_run_id_keeps_its_historical_shape_without_folds_or_a_proxy_sweep():
    # Results trees produced before fix/experiment-protocol must still resume.
    assert make_run_id("A", "A1", "MUTAG", 0.1, 2) == "A_A1_MUTAG_g0.1_s2"
    assert make_run_id("A", "A1", "MUTAG", 0.1, 2, fold=None, proxy=None) == "A_A1_MUTAG_g0.1_s2"


def test_run_id_encodes_the_fold_and_the_proxy_when_they_vary():
    assert make_run_id("A", "A0", "MUTAG", 0.1, 0, fold=3) == "A_A0_MUTAG_g0.1_s0_f3"
    assert make_run_id("A", "A0", "MUTAG", 0.1, 0, proxy="efc") == "A_A0_MUTAG_pefc_g0.1_s0"
    assert (make_run_id("A", "A0", "MUTAG", 0.1, 0, fold=0, proxy="r_bar")
            == "A_A0_MUTAG_pr_bar_g0.1_s0_f0")
    assert make_run_id("A", "A0", "MUTAG", 0.1, 0, fold=0) != make_run_id(
        "A", "A0", "MUTAG", 0.1, 0, fold=1)


def test_new_columns_are_appended_never_inserted():
    # The whole point of freezing the schema: an analysis script written against
    # the old column order must keep working. Anything added goes at the end.
    historical = [
        "run_id", "commit_sha", "tier", "config_id", "dataset", "seed",
        "s1_encoder", "s1_encoder_actual", "s2_proposal", "s3_cell_encoder",
        "s4_selector", "s5_mp", "osq_proxy", "gamma", "sparsity_weight",
        "epochs_ran", "best_epoch", "train_acc", "val_acc", "test_acc",
        "train_loss", "task_loss", "sparsity_loss", "osq_loss",
        "alpha_mean", "alpha_std", "num_cells", "mean_cell_size",
        "r_bar_before", "r_bar_after", "lambda2_before", "lambda2_after", "wc", "nwc",
        "params_count", "runtime_s", "peak_mem_mb", "status", "error",
    ]
    assert RUN_COLUMNS[:len(historical)] == historical
    assert "collapsed" in RUN_COLUMNS and "fold" in RUN_COLUMNS and "n_test" in RUN_COLUMNS
