# CLAUDE.md §11: "every Tier A/B/C config instantiates" - loaded from the
# actual declarative files under configs/, not a hand-copied list, so this
# test fails the moment a real config file goes stale.

import pytest

from models.dmd_model import DMDModel
from tools.config_loader import load_all_configs, load_pipeline_config, pipeline_kwargs


def test_load_all_configs_finds_every_declared_config():
    configs = load_all_configs("configs")
    # R + Tier A (9, including A0) + Tier B (4) + Tier C (2) = 16, per CLAUDE.md §9.
    assert len(configs) == 16
    assert {"R", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"} <= set(configs)
    assert {"B1", "B2", "B3", "B4"} <= set(configs)
    assert {"C1", "C2"} <= set(configs)


@pytest.mark.parametrize("config_id", [
    "R", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "B1", "B2", "B3", "B4", "C1", "C2",
])
def test_every_config_instantiates(config_id):
    configs = load_all_configs("configs")
    config = configs[config_id]
    model = DMDModel(input_dim=6, hidden_dim=8, latent_dim=6, motif_hidden_dim=6,
                     motif_out_dim=4, num_classes=3, **pipeline_kwargs(config))
    assert model.s1 == config["s1"]
    assert model.s5 == config["s5"]


def test_load_pipeline_config_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tier: A\nconfig_id: X\ns1: gcn\n")
    with pytest.raises(ValueError):
        load_pipeline_config(str(bad))


def test_load_all_configs_rejects_duplicate_config_id(tmp_path):
    (tmp_path / "one.yaml").write_text(
        "tier: A\nconfig_id: DUP\ns1: gcn\ns2: topk\ns3: deepsets\ns4: gumbel\ns5: gnn_rewired\n")
    (tmp_path / "two.yaml").write_text(
        "tier: A\nconfig_id: DUP\ns1: gin\ns2: topk\ns3: deepsets\ns4: gumbel\ns5: gnn_rewired\n")
    with pytest.raises(ValueError):
        load_all_configs(str(tmp_path))
