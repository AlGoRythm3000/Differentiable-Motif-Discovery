# The feat/osq-proxy results schema, for the tests of the analysis scripts that
# read a tree written in it.
#
# `results/analyze_osq_proxy.py` and `..._advanced.py` analyse the Grid A tree
# under `results/osq_proxy/`, which was written by feat/osq-proxy's own
# ResultsStore: a `proxy` column, no `tier`/`config_id`, and run ids shaped
# `<dataset>_<proxy>_g<gamma>_s<seed>`. feat/rich-bricks then replaced that
# schema on purpose - the note at the top of tools/results_store.py calls it a
# "deliberate one-time break" - which renamed `proxy` to `osq_proxy` and changed
# `make_run_id` to take `(tier, config_id, ...)`.
#
# The two test modules were never updated, so they had been importing the NEW
# store to build fixtures for the OLD analysers: `append_run` rejected the
# `proxy` column and `make_run_id` refused the four-argument call. Both files
# were red from that merge until this module existed - `git stash`-ing every
# later change reproduces the same 18 failures on main.
#
# The fixture writes the legacy schema directly rather than borrowing whichever
# store happens to be current. That is not a workaround: the analysers under
# test read a frozen historical tree, so their fixtures must produce that tree,
# and they must keep producing it when the live schema moves again.

import csv
from pathlib import Path

# Verbatim from results/osq_proxy/results/runs.csv's header.
LEGACY_RUN_COLUMNS = [
    "run_id", "commit_sha", "dataset", "proxy", "gamma", "sparsity_weight", "seed",
    "epochs_ran", "best_epoch", "train_acc", "val_acc", "test_acc",
    "train_loss", "task_loss", "sparsity_loss", "osq_loss",
    "alpha_mean", "alpha_std", "num_cells",
    "r_bar_before", "r_bar_after", "lambda2_before", "lambda2_after", "wc", "nwc",
    "runtime_s", "status", "error",
]

LEGACY_EPOCH_COLUMNS = [
    "run_id", "epoch", "train_loss", "train_task", "train_sparsity", "train_osq",
    "train_acc", "val_loss", "val_acc", "test_loss", "test_acc",
]


def legacy_run_id(dataset, proxy, gamma, seed):
    """feat/osq-proxy's run id shape, e.g. `synthetic_bottleneck_none_g0.0_s0`."""
    return f"{dataset}_{proxy}_g{gamma}_s{seed}"


class LegacyOsqStore:
    """
    Minimal stand-in for feat/osq-proxy's ResultsStore: enough to write a
    runs.csv / epochs.csv that `results/analyze_osq_proxy.py::load_runs` reads.

    Columns are the full frozen legacy list, not just the keys a fixture
    happens to set, because `load_runs` promises that an unfilled numeric cell
    comes back as `None` rather than raising - which only holds if the column
    is present and blank.

    `extra_columns` appends columns the legacy schema never had - only `fold` so
    far. It exists so one test can check that a legacy analyser fed a
    cross-validated tree still pairs correctly. The default stays the frozen
    historical list, because that is the tree these analysers are actually for.
    """

    def __init__(self, out_dir, extra_columns=()):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.out_dir / "runs.csv"
        self.epochs_path = self.out_dir / "epochs.csv"
        self.run_columns = LEGACY_RUN_COLUMNS + [c for c in extra_columns
                                                  if c not in LEGACY_RUN_COLUMNS]
        self._runs = []
        self._epochs = []

    def append_run(self, row):
        self._runs.append(row)
        _write(self.runs_path, self.run_columns, self._runs)

    def append_epochs(self, run_id, history):
        for record in history:
            self._epochs.append(dict(record, run_id=run_id))
        _write(self.epochs_path, LEGACY_EPOCH_COLUMNS, self._epochs)


def _write(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})
