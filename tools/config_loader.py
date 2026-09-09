# Loads a pipeline config (the R / Tier A / Tier B / Tier C definitions) from
# a declarative YAML file under configs/. Kept deliberately tiny: the file
# IS the config (five brick strings + kwargs), this module only reads it and
# checks it against the same validator DMDModel itself uses, so an invalid
# Tier config fails at load time - before a Kaggle session burns time on it -
# rather than only when DMDModel.__init__ runs.

from pathlib import Path
from typing import Dict

import yaml

from models.registry import validate_pipeline_config

REQUIRED_KEYS = ("tier", "config_id", "s1", "s2", "s3", "s4", "s5")
KWARG_KEYS = ("s1_kwargs", "s2_kwargs", "s3_kwargs", "s4_kwargs", "s5_kwargs")


def load_pipeline_config(path: str) -> Dict:
    """
    Returns the parsed config dict, validated. Every `configs/**/*.yaml` file
    is expected to declare `tier`, `config_id`, `s1..s5`, and optionally
    `s1_kwargs..s5_kwargs` (brick-specific kwargs, e.g. `num_inducing`,
    `ksubset_k`, `gpse_mode`).
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")

    validate_pipeline_config(config["s1"], config["s2"], config["s3"], config["s4"], config["s5"])

    for key in KWARG_KEYS:
        config.setdefault(key, {})
    return config


def load_all_configs(configs_dir: str = "configs") -> Dict[str, Dict]:
    """Loads every `*.yaml` under `configs_dir` (recursively), keyed by `config_id`."""
    configs = {}
    for path in sorted(Path(configs_dir).rglob("*.yaml")):
        config = load_pipeline_config(str(path))
        config_id = config["config_id"]
        if config_id in configs:
            raise ValueError(f"Duplicate config_id {config_id!r}: {path} and a config already loaded.")
        configs[config_id] = config
    return configs


def pipeline_kwargs(config: Dict) -> Dict:
    """
    The subset of `config` that's actually `DMDModel(**...)`-shaped -
    `tier`/`config_id` are bookkeeping for the results row, not model kwargs.
    """
    return {
        "s1": config["s1"], "s2": config["s2"], "s3": config["s3"],
        "s4": config["s4"], "s5": config["s5"],
        "s1_kwargs": config["s1_kwargs"], "s2_kwargs": config["s2_kwargs"],
        "s3_kwargs": config["s3_kwargs"], "s4_kwargs": config["s4_kwargs"],
        "s5_kwargs": config["s5_kwargs"],
    }
