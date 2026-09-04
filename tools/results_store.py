# Append-only results storage for the experiment grid.
#
# The schema is fixed here rather than in the notebook that produces it, for two
# reasons: analysis scripts written months later must not have to guess column
# names, and a grid that runs on a throwaway cloud session must be able to
# resume without ever rewriting a row it already has.
#
#   results/
#   |- runs.csv          one row per run, the main analysis table
#   |- epochs.csv        one row per (run_id, epoch), the curves
#   |- env.json          versions, GPU, commit SHA, timestamp
#   |- raw/<run_id>.json full config + per-epoch history + traceback if failed
#
# CSV and JSON only, deliberately: pickles do not survive a version bump of
# torch or PyG, and these results are meant to outlive both.

import csv
import json
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

# Column names are frozen AS OF feat/rich-bricks (CLAUDE.md §12 fixes this
# exact list/order). This is a deliberate one-time break from feat/osq-proxy's
# schema (`proxy` -> `osq_proxy`, `tier`/`config_id` inserted, per-stage brick
# columns added) - the branch also changes `run_id`'s shape, so the two
# schemas were never going to share one results/ tree. From here on, the rule
# reverts to what it always was: add new columns at the end, never rename or
# reorder, or every results directory produced by this branch stops parsing.
RUN_COLUMNS: List[str] = [
    "run_id", "commit_sha", "tier", "config_id", "dataset", "seed",
    "s1_encoder", "s1_encoder_actual", "s2_proposal", "s3_cell_encoder", "s4_selector", "s5_mp",
    "osq_proxy", "gamma", "sparsity_weight",
    "epochs_ran", "best_epoch", "train_acc", "val_acc", "test_acc",
    "train_loss", "task_loss", "sparsity_loss", "osq_loss",
    "alpha_mean", "alpha_std", "num_cells", "mean_cell_size",
    "r_bar_before", "r_bar_after", "lambda2_before", "lambda2_after", "wc", "nwc",
    "params_count", "runtime_s", "peak_mem_mb", "status", "error",
]

EPOCH_COLUMNS: List[str] = [
    "run_id", "epoch",
    "train_loss", "train_task", "train_sparsity", "train_osq", "train_acc",
    "val_loss", "val_acc", "test_loss", "test_acc",
]


def make_run_id(tier: str, config_id: str, dataset: str, gamma: float, seed: int) -> str:
    """
    Deterministic (CLAUDE.md §12), so a resumed grid recognizes what it
    already ran and an analysis script can join a row back to its raw file
    without a lookup table.
    """
    return f"{tier}_{config_id}_{dataset}_g{gamma}_s{seed}"


class ResultsStore:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.raw_dir = self.out_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.out_dir / "runs.csv"
        self.epochs_path = self.out_dir / "epochs.csv"
        self.env_path = self.out_dir / "env.json"

    # -- environment ------------------------------------------------------
    def write_env(self, env: dict) -> None:
        with open(self.env_path, "w") as f:
            json.dump(env, f, indent=2, default=str)

    # -- runs -------------------------------------------------------------
    def existing_run_ids(self) -> set:
        if not self.runs_path.exists():
            return set()
        with open(self.runs_path, newline="") as f:
            return {row["run_id"] for row in csv.DictReader(f)}

    def has_run(self, run_id: str) -> bool:
        return run_id in self.existing_run_ids()

    def append_run(self, row: dict, force: bool = False) -> None:
        """
        Appends one run row. Refuses a run_id already present unless `force`:
        silently overwriting is how a grid ends up with two different numbers
        under the same name and no way to tell which one a figure used.
        """
        unknown = set(row) - set(RUN_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown result columns: {sorted(unknown)}")
        if not force and self.has_run(row["run_id"]):
            raise ValueError(f"run_id '{row['run_id']}' already stored - pass force=True to add "
                             "it anyway (it will appear twice, on purpose).")

        is_new = not self.runs_path.exists()
        with open(self.runs_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in RUN_COLUMNS})

    def read_runs(self) -> List[dict]:
        if not self.runs_path.exists():
            return []
        with open(self.runs_path, newline="") as f:
            return list(csv.DictReader(f))

    # -- epochs -----------------------------------------------------------
    def append_epochs(self, run_id: str, history: Iterable[dict]) -> None:
        is_new = not self.epochs_path.exists()
        with open(self.epochs_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPOCH_COLUMNS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            for record in history:
                row = {column: record.get(column, "") for column in EPOCH_COLUMNS}
                row["run_id"] = run_id
                writer.writerow(row)

    # -- raw --------------------------------------------------------------
    def write_raw(self, run_id: str, payload: dict) -> Path:
        path = self.raw_dir / f"{run_id}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    # -- packaging --------------------------------------------------------
    def zip(self, archive_path: Optional[str] = None) -> Path:
        """Zips the whole results tree for one-click download at the end of a session."""
        archive = Path(archive_path) if archive_path else self.out_dir.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.out_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(self.out_dir.parent))
        return archive
