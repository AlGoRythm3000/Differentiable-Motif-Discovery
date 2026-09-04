import torch

from tools.experiment_grid import (SYNTHETIC_ARM, GridConfig, RunSpec, TIER_B_DATASETS,
                                    build_plan, environment_info, load_arm, make_splits,
                                    run_grid, trim_plan)
from tools.results_store import ResultsStore

# A tiny, self-contained configs/ tree (never the real 16-config one - these
# tests care about the grid's PLUMBING, not about running the actual Tier
# A/B/C science, which would be far too slow for a unit test).
_TIER_A_CONFIG = """\
tier: A
config_id: TA1
s1: gcn
s2: topk
s3: deepsets
s4: gumbel
s5: gnn_rewired
"""
_TIER_B_CONFIG = """\
tier: B
config_id: TB1
s1: gcn
s2: topk
s3: deepsets
s4: gumbel
s5: gnn_rewired
"""
_TIER_C_CONFIG = """\
tier: C
config_id: TC1
s1: gcn
s2: topk
s3: deepsets
s4: gumbel
s5: gnn_rewired
"""


def _write_mini_configs(tmp_path):
    (tmp_path / "a.yaml").write_text(_TIER_A_CONFIG)
    (tmp_path / "b.yaml").write_text(_TIER_B_CONFIG)
    (tmp_path / "c.yaml").write_text(_TIER_C_CONFIG)
    return str(tmp_path)


def _small_config(configs_dir, **overrides):
    config = GridConfig(
        datasets=[SYNTHETIC_ARM], configs_dir=configs_dir,
        best_proxy="r_bar", best_gamma=0.1,
        seeds=[0], epochs=2, patience=1, batch_size=8, hidden_dim=16,
        hutch_k=4, osq_sample_graphs=2, synthetic_graphs=24, synthetic_depth=2,
        synthetic_classes=3,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_plan_crosses_every_tier_a_config_with_gamma_zero_and_best(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path))
    plan = build_plan(config)
    ids = [spec.run_id for spec in plan]

    assert f"A_TA1_{SYNTHETIC_ARM}_g0.0_s0" in ids
    assert f"A_TA1_{SYNTHETIC_ARM}_g0.1_s0" in ids
    assert len(ids) == len(set(ids))


def test_plan_runs_tier_a_before_b_before_c(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path), datasets=[SYNTHETIC_ARM, "MUTAG"])
    plan = build_plan(config)
    tiers_in_order = [spec.tier for spec in plan]
    # every 'A' entry appears before every 'B' entry, which appears before every 'C' entry
    assert tiers_in_order.index("C") > max(i for i, t in enumerate(tiers_in_order) if t == "B")
    assert tiers_in_order.index("B") > max(i for i, t in enumerate(tiers_in_order) if t == "A")


def test_plan_restricts_tier_b_to_its_dataset_subset(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path),
                           datasets=[SYNTHETIC_ARM, "MUTAG", "NCI1"])
    plan = build_plan(config)
    tier_b_datasets = {spec.dataset for spec in plan if spec.tier == "B"}
    assert tier_b_datasets == {d for d in TIER_B_DATASETS if d in config.datasets}
    assert "NCI1" not in tier_b_datasets

    tier_a_datasets = {spec.dataset for spec in plan if spec.tier == "A"}
    assert tier_a_datasets == set(config.datasets)  # Tier A: every dataset


def test_reference_config_id_is_not_run_directly():
    """'R' (configs/reference.yaml) is excluded from the grid - A0 already
    is R, run as part of Tier A."""
    plan = build_plan(GridConfig(datasets=[SYNTHETIC_ARM], seeds=[0]))
    assert all(spec.tier != "R" for spec in plan)


def test_trim_plan_drops_whole_tiers_before_touching_datasets_or_seeds():
    plan = ([RunSpec("A", "TA1", SYNTHETIC_ARM, g, s) for g in (0.0, 0.1) for s in (0, 1, 2)]
           + [RunSpec("B", "TB1", SYNTHETIC_ARM, g, s) for g in (0.0, 0.1) for s in (0, 1, 2)]
           + [RunSpec("C", "TC1", SYNTHETIC_ARM, g, s) for g in (0.0, 0.1) for s in (0, 1, 2)])

    # Budget for exactly Tier A's share: Tier B and C must be dropped whole.
    kept = trim_plan(plan, projected_seconds_per_run=1.0, seconds_left=6)
    assert {spec.tier for spec in kept} == {"A"}
    assert len(kept) == 6


def test_trim_plan_never_drops_the_synthetic_arm_or_a_seed():
    plan = [RunSpec("A", "TA1", dataset, gamma, seed)
            for dataset in (SYNTHETIC_ARM, "MUTAG", "ENZYMES", "NCI1")
            for gamma in (0.0, 0.1)
            for seed in (0, 1, 2)]

    kept = trim_plan(plan, projected_seconds_per_run=1.0, seconds_left=10)
    assert SYNTHETIC_ARM in {spec.dataset for spec in kept}
    assert "NCI1" not in {spec.dataset for spec in kept}
    synthetic_seeds = {spec.seed for spec in kept if spec.dataset == SYNTHETIC_ARM}
    assert synthetic_seeds == {0, 1, 2}


def test_splits_are_stratified_disjoint_and_reproducible(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path))
    dataset, _, _, _ = load_arm(SYNTHETIC_ARM, config)
    _, indices = make_splits(dataset, config, seed=0)
    _, again = make_splits(dataset, config, seed=0)

    assert indices == again
    all_idx = indices["train"] + indices["val"] + indices["test"]
    assert len(all_idx) == len(set(all_idx)) == len(dataset)
    train_labels = dataset.y[torch.tensor(indices["train"])]
    assert len(train_labels.unique()) == dataset.num_classes


def test_grid_stores_one_row_per_run_with_before_and_after_measurements(tmp_path):
    store = ResultsStore(tmp_path / "results")
    config = _small_config(_write_mini_configs(tmp_path))
    # gamma=0 twin + gamma=best twin for the one Tier A config = 2 runs.
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]
    summary = run_grid(config, store, plan=plan, verbose=False)

    assert summary["completed"] == 2 and summary["failed"] == 0
    rows = store.read_runs()
    assert len(rows) == 2
    for row in rows:
        assert row["status"] == "ok"
        assert row["commit_sha"]
        assert row["tier"] == "A" and row["config_id"] == "TA1"
        assert row["s1_encoder"] == "gcn" and row["s1_encoder_actual"] == "gcn"
        assert float(row["r_bar_before"]) > 0
        assert float(row["r_bar_after"]) > 0
        assert row["alpha_mean"] and row["num_cells"] and row["mean_cell_size"]
        assert row["params_count"]
    assert (store.raw_dir / f"{rows[0]['run_id']}.json").exists()


def test_a_failing_run_is_recorded_and_the_grid_continues(tmp_path):
    store = ResultsStore(tmp_path / "results")
    config = _small_config(_write_mini_configs(tmp_path))
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]
    plan.insert(0, RunSpec("A", "TA1", "NOT_A_DATASET", 0.0, 0))

    summary = run_grid(config, store, plan=plan, verbose=False)

    assert summary["failed"] == 1
    assert summary["completed"] == 2
    failed = [row for row in store.read_runs() if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["error"]


def test_a_second_pass_skips_what_is_already_stored(tmp_path):
    store = ResultsStore(tmp_path / "results")
    config = _small_config(_write_mini_configs(tmp_path))
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]

    run_grid(config, store, plan=list(plan), verbose=False)
    summary = run_grid(config, store, plan=list(plan), verbose=False)

    assert summary["completed"] == 0
    assert summary["skipped"] == 2
    assert len(store.read_runs()) == 2


def test_environment_info_carries_a_commit_sha_and_versions():
    info = environment_info(".")
    assert info["commit_sha"]
    assert info["torch"] and info["python"]
    assert "cuda_available" in info
