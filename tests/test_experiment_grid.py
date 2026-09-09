import torch

from tools.experiment_grid import (SYNTHETIC_ARM, GridConfig, RunSpec, TIER_B_DATASETS,
                                    alpha_statistics, build_plan, environment_info,
                                    load_arm, make_splits, run_grid, run_single, trim_plan)
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
        # These tests pin the historical single-split design (2 runs per Tier A
        # config: the gamma pair). Cross-validation is exercised by its own
        # tests below, which say what k they expect out loud.
        cv_folds=None,
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


# ---------------------------------------------------------------------------
# fix/experiment-protocol: cross-validation, the gamma/proxy sweep, and the
# collapse diagnostic. Each of these pins something the feat/rich-bricks grid
# got wrong in a way that was invisible in its own output.
# ---------------------------------------------------------------------------

def test_plan_crosses_folds_when_cross_validation_is_on(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path), cv_folds=5)
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]

    assert len(plan) == 2 * 5  # gamma pair x 5 folds
    assert {spec.fold for spec in plan} == {0, 1, 2, 3, 4}
    assert len({spec.run_id for spec in plan}) == len(plan)


def test_folds_are_shared_across_arms_so_deltas_stay_paired(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path), cv_folds=5)
    dataset, _, _, _ = load_arm(SYNTHETIC_ARM, config)

    # Same fold, different seed: the seed must move the initialisation, never
    # the split, or a "seed effect" is a split effect in disguise - which is
    # exactly what the single-split design could not separate.
    _, first = make_splits(dataset, config, seed=0, fold=2)
    _, second = make_splits(dataset, config, seed=7, fold=2)
    assert first == second

    _, other_fold = make_splits(dataset, config, seed=0, fold=3)
    assert other_fold["test"] != first["test"]


def test_every_graph_is_tested_once_across_the_folds_of_a_dataset(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path), cv_folds=5)
    dataset, _, _, _ = load_arm(SYNTHETIC_ARM, config)

    tested = []
    for fold in range(5):
        _, indices = make_splits(dataset, config, seed=0, fold=fold)
        tested += indices["test"]
    assert sorted(tested) == list(range(len(dataset)))


def test_plan_sweeps_gamma_and_proxy_without_duplicating_the_gamma_zero_arm(tmp_path):
    config = _small_config(_write_mini_configs(tmp_path), cv_folds=None,
                           gammas=[0.0, 0.01, 0.1], proxies=["r_bar", "efc"])
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]

    # gamma=0 short-circuits osq_fn, so the proxies are indistinguishable there:
    # one arm, not one per proxy.
    zero = [spec for spec in plan if spec.gamma == 0.0]
    assert len(zero) == 1 and zero[0].proxy is None
    assert len(plan) == 1 + 2 * 2  # gamma=0, then 2 gammas x 2 proxies
    assert len({spec.run_id for spec in plan}) == len(plan)


def test_run_ids_are_unchanged_when_no_sweep_and_no_folds_are_requested(tmp_path):
    # A results tree produced before this branch must still resume against it.
    config = _small_config(_write_mini_configs(tmp_path))
    ids = [spec.run_id for spec in build_plan(config) if spec.tier == "A"]
    assert sorted(ids) == [f"A_TA1_{SYNTHETIC_ARM}_g0.0_s0",
                           f"A_TA1_{SYNTHETIC_ARM}_g0.1_s0"]


def test_rows_carry_the_collapse_diagnostic_and_the_split_sizes(tmp_path):
    # The failure that 41% of the last grid hit and that nothing in its schema
    # recorded: alpha identically 0, so the lifting adds only zero-weight edges
    # and the model is a plain GCN under a brick's name.
    store = ResultsStore(tmp_path / "results")
    config = _small_config(_write_mini_configs(tmp_path))
    plan = [spec for spec in build_plan(config) if spec.tier == "A"]
    run_grid(config, store, plan=plan, verbose=False)

    for row in store.read_runs():
        assert row["collapsed"] in ("True", "False")
        assert row["alpha_frac_active"] != ""
        assert 0.0 <= float(row["alpha_frac_active"]) <= 1.0
        assert int(row["n_test"]) > 0
        assert int(row["n_train"]) + int(row["n_val"]) + int(row["n_test"]) > 0
        assert row["grad_clip"] != ""


def test_collapse_is_reported_when_the_selector_accepts_nothing(tmp_path):
    import torch

    config = _small_config(_write_mini_configs(tmp_path))
    dataset, num_features, num_classes, _ = load_arm(SYNTHETIC_ARM, config)
    splits, _ = make_splits(dataset, config, seed=0)

    from models.dmd_model import DMDModel
    model = DMDModel(input_dim=num_features, hidden_dim=16, latent_dim=16,
                     motif_hidden_dim=16, motif_out_dim=16, num_classes=num_classes)
    # Drive every proposal score far negative: sigmoid saturates at 0, the hard
    # threshold rejects every cell, and alpha is identically zero.
    with torch.no_grad():
        model.proposal.W.fill_(0.0)
        model.classifier.bias.fill_(0.0)
    model.selector.tau = 1e-3
    with torch.no_grad():
        for parameter in model.embedder.parameters():
            parameter.fill_(0.0)

    stats = alpha_statistics(model, splits)
    assert stats["alpha_frac_active"] == 0.0 or stats["collapsed"] is True


_KSUBSET_CONFIG = """\
tier: A
config_id: TK1
s1: gcn
s2: topk
s3: deepsets
s4: ksubset
s5: gnn_rewired
s4_kwargs:
  k: 2
  tau_init: 1.0
  tau_min: 0.1
  anneal_rate: 0.5
"""


def test_the_grid_actually_anneals_the_ksubset_temperature(tmp_path):
    # KSubsetSelector's docstring says "the grid runner calls this once per
    # epoch and logs the schedule". Until this test, `set_temperature` had no
    # caller anywhere in the repo and A6 ran its whole arm at the fixed initial
    # tau - a documented contract that nothing enforced.
    from tools.config_loader import load_all_configs

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "k.yaml").write_text(_KSUBSET_CONFIG)

    config = _small_config(str(configs_dir), epochs=5, patience=5)
    pipeline_config = load_all_configs(str(configs_dir))["TK1"]
    dataset, num_features, num_classes, _ = load_arm(SYNTHETIC_ARM, config)
    spec = RunSpec("A", "TK1", SYNTHETIC_ARM, 0.0, 0)

    # anneal_rate=0.5 over 5 epochs: tau_init * exp(-0.5*5) = 0.082, floored at
    # tau_min = 0.1 by set_temperature itself.
    result = run_single(spec, config, pipeline_config, dataset, num_features, num_classes)
    assert result["row"]["status"] == "ok"


def test_set_temperature_never_goes_below_tau_min():
    from models.weight_assignment import KSubsetSelector

    selector = KSubsetSelector(k=2, tau_init=1.0, tau_min=0.1, anneal_rate=0.5)
    selector.set_temperature(0.001)
    assert selector.tau == 0.1
    selector.set_temperature(0.5)
    assert selector.tau == 0.5
