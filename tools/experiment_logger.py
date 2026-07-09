# Persists per-run config, per-epoch metrics, and rewiring snapshots under
# results/runs/<config-name>/<run-timestamp>/ so results/analyze_results.py
# can plot them later. history.jsonl is flushed every epoch (not just at the
# end) so a killed/crashed run still leaves a usable partial history.

import hashlib
import json
import time
from pathlib import Path

# Windows MAX_PATH is 260 chars by default; keep the experiment name well
# under that so results_dir/name/run_id/history.jsonl never hits it, even
# when several hyperparameters are swept away from their defaults at once.
_MAX_NAME_LENGTH = 60

# CLI args that describe *how* the run was launched/logged, not the model or
# data configuration - excluded from the experiment name (seed is handled
# separately, appended last, since sweeping it is a common comparison axis).
_ADMIN_FIELDS = {"task", "dataset", "epochs", "log_every", "results_dir",
                  "data_root", "save_graph_samples", "seed"}


def build_experiment_name(args, parser) -> str:
    """
    <task>_<dataset-or-toy>_<key=value for every non-default hyperparameter>_seed=<seed>

    Two runs only share a name if they're the same configuration (up to
    seed), so sweeping datasets/hyperparameters produces distinct,
    self-describing directories without needing to hand-name experiments.
    """
    defaults = vars(parser.parse_args([]))

    dataset_label = args.dataset if (args.task == "graph_classification" and args.dataset) else "toy"
    parts = [args.task, dataset_label]

    for key in sorted(vars(args)):
        if key in _ADMIN_FIELDS:
            continue
        value = getattr(args, key)
        if key in defaults and value != defaults[key]:
            parts.append(f"{key}={value}")

    parts.append(f"seed={args.seed}")
    name = "_".join(parts)

    if len(name) > _MAX_NAME_LENGTH:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = name[:_MAX_NAME_LENGTH] + "_" + digest

    return name


class ExperimentLogger:
    def __init__(self, results_dir: str, args, parser):
        self.exp_name = build_experiment_name(args, parser)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.exp_dir = Path(results_dir) / self.exp_name / run_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        self._history_file = open(self.exp_dir / "history.jsonl", "w")

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, test_metrics: dict) -> None:
        record = {"epoch": epoch}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        self._history_file.write(json.dumps(record) + "\n")
        self._history_file.flush()

    def save_graph_samples(self, samples: list) -> None:
        with open(self.exp_dir / "graph_samples.json", "w") as f:
            json.dump(samples, f, indent=2)

    def save_summary(self, summary: dict) -> None:
        with open(self.exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        self._history_file.close()
