import torch

from tools.experiment_grid import (SYNTHETIC_ARM, GridConfig, RunSpec, build_plan,
                                    environment_info, load_arm, make_splits, run_grid,
                                    trim_plan)
from tools.results_store import ResultsStore


def _small_config(**overrides):
    config = GridConfig(
        datasets=[SYNTHETIC_ARM], proxies=["none", "r_bar"], gammas=[0.0, 0.1],
        seeds=[0], epochs=2, patience=1, batch_size=8, hidden_dim=16, top_k=2,
        hutch_k=4, osq_sample_graphs=2, synthetic_graphs=24, synthetic_depth=2,
        synthetic_classes=3,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_plan_drops_the_redundant_zero_gamma_cells():
    # gamma = 0 short-circuits the proxy, so (r_bar, 0.0) is the same run as
    # (none, 0.0) - running it four more times would buy nothing.
    config = _small_config(proxies=["none", "r_bar", "efc"], gammas=[0.0, 0.01, 0.1])
    ids = [spec.run_id for spec in build_plan(config)]

    assert ids.count(f"{SYNTHETIC_ARM}_none_g0.0_s0") == 1
    assert f"{SYNTHETIC_ARM}_r_bar_g0.0_s0" not in ids
    assert f"{SYNTHETIC_ARM}_none_g0.01_s0" not in ids
    assert len(ids) == len(set(ids))


def test_full_cartesian_product_is_one_flag_away():
    config = _small_config(proxies=["none", "r_bar"], gammas=[0.0, 0.1],
                           skip_redundant_zero_gamma=False)
    plan = build_plan(config)
    assert len(plan) == 2 * 2 * len(config.seeds)


def test_plan_runs_the_baselines_first_and_the_synthetic_arm_early():
    config = _small_config(datasets=["NCI1", SYNTHETIC_ARM, "MUTAG"],
                           proxies=["none", "r_bar"], gammas=[0.0, 0.1])
    plan = build_plan(config)
    assert all(spec.gamma == 0.0 for spec in plan[:3])
    assert plan[0].dataset == SYNTHETIC_ARM
    first_of_each = [spec.dataset for spec in plan if spec.gamma == 0.0]
    assert first_of_each.index(SYNTHETIC_ARM) < first_of_each.index("NCI1")


def test_time_budget_cuts_gammas_before_datasets_and_never_the_synthetic_arm():
    plan = [RunSpec(dataset, "r_bar", gamma, seed)
            for dataset in (SYNTHETIC_ARM, "MUTAG", "ENZYMES", "NCI1")
            for gamma in (0.01, 0.1, 1.0)
            for seed in (0, 1, 2)]

    # Budget for three quarters of the plan: the largest gamma goes first.
    kept = trim_plan(plan, projected_seconds_per_run=1.0, seconds_left=0.8 * len(plan))
    assert not any(spec.gamma == 1.0 for spec in kept)
    assert {spec.dataset for spec in kept} == {SYNTHETIC_ARM, "MUTAG", "ENZYMES", "NCI1"}

    # A much tighter budget starts dropping the expensive real datasets, but the
    # synthetic arm and the three seeds survive whatever happens.
    kept = trim_plan(plan, projected_seconds_per_run=1.0, seconds_left=10)
    assert SYNTHETIC_ARM in {spec.dataset for spec in kept}
    assert "NCI1" not in {spec.dataset for spec in kept}
    synthetic_seeds = {spec.seed for spec in kept if spec.dataset == SYNTHETIC_ARM}
    assert synthetic_seeds == {0, 1, 2}


def test_splits_are_stratified_disjoint_and_reproducible():
    config = _small_config()
    dataset, _, _ = load_arm(SYNTHETIC_ARM, config)
    _, indices = make_splits(dataset, config, seed=0)
    _, again = make_splits(dataset, config, seed=0)

    assert indices == again
    all_idx = indices["train"] + indices["val"] + indices["test"]
    assert len(all_idx) == len(set(all_idx)) == len(dataset)
    train_labels = dataset.y[torch.tensor(indices["train"])]
    assert len(train_labels.unique()) == dataset.num_classes


def test_grid_stores_one_row_per_run_with_before_and_after_measurements(tmp_path):
    store = ResultsStore(tmp_path / "results")
    summary = run_grid(_small_config(), store, verbose=False)

    assert summary["completed"] == 2 and summary["failed"] == 0
    rows = store.read_runs()
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "ok"
        assert row["commit_sha"]
        assert float(row["r_bar_before"]) > 0
        assert float(row["r_bar_after"]) > 0
        assert row["alpha_mean"] and row["num_cells"]
    assert (store.raw_dir / f"{rows[0]['run_id']}.json").exists()


def test_a_failing_run_is_recorded_and_the_grid_continues(tmp_path):
    store = ResultsStore(tmp_path / "results")
    config = _small_config()
    plan = build_plan(config)
    plan.insert(0, RunSpec("NOT_A_DATASET", "none", 0.0, 0))

    summary = run_grid(config, store, plan=plan, verbose=False)

    assert summary["failed"] == 1
    assert summary["completed"] == 2
    failed = [row for row in store.read_runs() if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["error"]


def test_a_second_pass_skips_what_is_already_stored(tmp_path):
    store = ResultsStore(tmp_path / "results")
    config = _small_config()
    run_grid(config, store, verbose=False)
    summary = run_grid(config, store, verbose=False)

    assert summary["completed"] == 0
    assert summary["skipped"] == 2
    assert len(store.read_runs()) == 2


def test_environment_info_carries_a_commit_sha_and_versions():
    info = environment_info(".")
    assert info["commit_sha"]
    assert info["torch"] and info["python"]
    assert "cuda_available" in info
