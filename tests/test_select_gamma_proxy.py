import csv

from results.select_gamma_proxy import (holm, load_runs, matched_arms, recommend,
                                         summarize, write_csv)


def _row(**overrides):
    row = {"status": "ok", "dataset": "MUTAG", "seed": "0", "fold": "0",
           "gamma": "0.0", "osq_proxy": "none", "val_acc": "0.70", "test_acc": "0.68",
           "r_bar_after": "2.0", "collapsed": "False"}
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def _arm_and_twin(dataset, fold, gamma, proxy, val, **overrides):
    return [_row(dataset=dataset, fold=fold),
            _row(dataset=dataset, fold=fold, gamma=gamma, osq_proxy=proxy, val_acc=val,
                 **overrides)]


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------

def test_arms_pair_only_against_their_own_fold():
    # The failure this guards: pairing on (dataset, seed) alone lets an arm on
    # fold 3 be compared against a baseline on fold 7, so the "paired" delta is
    # partly a difference between two test sets.
    runs = [_row(fold=0), _row(fold=1, val_acc=0.90),
            _row(fold=0, gamma="0.1", osq_proxy="r_bar", val_acc=0.75)]
    arms = matched_arms(runs)
    assert list(arms) == [(0.1, "r_bar")]
    pairs = arms[(0.1, "r_bar")]
    assert len(pairs) == 1
    assert pairs[0][0] == 0.70  # fold 0's baseline, not fold 1's 0.90


def test_an_arm_without_a_baseline_twin_is_dropped_not_compared_to_a_mean():
    runs = [_row(dataset="MUTAG", fold=0),
            _row(dataset="NCI1", fold=0, gamma="0.1", osq_proxy="r_bar", val_acc=0.9)]
    assert matched_arms(runs) == {}


def test_failed_runs_never_enter_a_pair():
    runs = [_row(fold=0),
            _row(fold=0, gamma="0.1", osq_proxy="r_bar", val_acc=0.9, status="failed")]
    assert matched_arms(runs) == {}


def test_gamma_zero_is_the_baseline_and_never_its_own_arm():
    runs = [_row(fold=0), _row(fold=0, gamma="0.1", osq_proxy="r_bar", val_acc=0.75)]
    assert all(gamma > 0 for gamma, _ in matched_arms(runs))


# ---------------------------------------------------------------------------
# selection metric
# ---------------------------------------------------------------------------

def test_selection_uses_validation_and_reports_test_separately():
    # An arm that is better on test and worse on validation must not be selected
    # by the default metric - selecting on test and then reporting it is the
    # thing this whole script exists to stop.
    runs = []
    for fold in range(5):
        runs += _arm_and_twin("MUTAG", fold, "0.1", "r_bar", 0.60, test_acc=0.95)
    rows = summarize(runs)
    assert len(rows) == 1
    assert rows[0]["mean_delta"] < 0        # validation: worse
    assert rows[0]["mean_delta_test"] > 0   # test: better, reported but not used


def test_mechanism_and_collapse_are_reported_alongside_the_task_metric():
    runs = []
    for fold in range(4):
        runs += _arm_and_twin("MUTAG", fold, "0.1", "r_bar", 0.72,
                              r_bar_after=1.0, collapsed="True")
    row = summarize(runs)[0]
    assert row["mean_delta_r_bar"] == -1.0   # the term lowered measured resistance
    assert row["r_bar_improved"] == 4
    assert row["collapse_rate"] == 1.0       # ...on a model that accepted no cell


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------

def test_holm_is_monotone_and_never_below_the_raw_p_value():
    raw = [0.01, 0.02, 0.5]
    adjusted = holm(raw)
    assert adjusted == [0.03, 0.04, 0.5]
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert adjusted == sorted(adjusted, key=lambda p: raw[adjusted.index(p)])


def test_holm_passes_missing_p_values_through():
    assert holm([0.01, None, 0.5]) == [0.02, None, 0.5]


def test_holm_of_a_single_test_changes_nothing():
    assert holm([0.03]) == [0.03]


# ---------------------------------------------------------------------------
# recommendation
# ---------------------------------------------------------------------------

def test_recommendation_refuses_to_name_a_winner_without_evidence():
    # The previous selection named `r_bar` / `gamma=0.1` regardless. When
    # nothing survives correction, saying so IS the result.
    runs = []
    for fold in range(5):
        runs += _arm_and_twin("MUTAG", fold, "0.1", "r_bar", 0.70)  # zero effect
    best, note = recommend(summarize(runs))
    assert "no arm survives" in note or "not enough" in note


def test_recommendation_needs_at_least_three_pairs():
    runs = _arm_and_twin("MUTAG", 0, "0.1", "r_bar", 0.99)
    best, note = recommend(summarize(runs))
    assert best is None and "not enough" in note


def test_summary_csv_round_trips(tmp_path):
    runs = []
    for fold in range(4):
        runs += _arm_and_twin("MUTAG", fold, "0.1", "r_bar", 0.75)
    path = tmp_path / "selection.csv"
    write_csv(summarize(runs), path)

    with open(path, newline="") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 1
    assert written[0]["proxy"] == "r_bar" and written[0]["n_pairs"] == "4"


def test_load_runs_fails_loudly_on_a_directory_with_no_grid(tmp_path):
    try:
        load_runs(tmp_path)
        assert False, "expected SystemExit"
    except SystemExit:
        pass
