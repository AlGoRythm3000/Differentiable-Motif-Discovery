"""Append one results store into another, refusing anything that would corrupt it.

A grid can end up split across stores: a run dropped by an orchestrator timeout
and re-run later, a seed added on other hardware. Merging those by hand - `tail
-n +2 new/runs.csv >> old/runs.csv` - works right up until the two headers differ
by a column, and then it writes a file whose rows are silently shifted and whose
numbers are wrong in a way no check catches. This does the same job through
`ResultsStore.append_run`, which knows the canonical column order and refuses a
run_id that is already stored.

Epoch histories and raw/ payloads come along; an existing raw file is never
overwritten.

Run:  python3 tools/merge_results.py SOURCE_DIR DEST_DIR [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.results_store import EPOCH_COLUMNS, RUN_COLUMNS, ResultsStore  # noqa: E402


def _header(path: Path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return csv.DictReader(f).fieldnames


def merge(source: Path, dest: Path, dry_run: bool = False) -> int:
    src_runs, dst_runs = source / "runs.csv", dest / "runs.csv"
    if not src_runs.exists():
        print(f"no runs.csv in {source}")
        return 1

    # Both files must speak today's schema. A source written by an older store is
    # not merged and then explained away in a caption; it is refused here.
    for path in (src_runs, dst_runs):
        header = _header(path)
        if header is not None and header != RUN_COLUMNS:
            missing = [c for c in RUN_COLUMNS if c not in header]
            extra = [c for c in header if c not in RUN_COLUMNS]
            print(f"REFUSING: {path} does not match tools/results_store.RUN_COLUMNS "
                  f"(missing={missing}, unexpected={extra})")
            return 1

    store = ResultsStore(str(dest))
    already = set(store.existing_run_ids())
    with open(src_runs, newline="") as f:
        rows = list(csv.DictReader(f))

    clashes = [r["run_id"] for r in rows if r["run_id"] in already]
    if clashes:
        print(f"REFUSING: {len(clashes)} run_id(s) already in {dest}, e.g. {clashes[:3]}.\n"
              "         Nothing merged. Remove them from the source, or merge into a "
              "fresh directory if they are genuinely different runs.")
        return 1

    no_sha = [r["run_id"] for r in rows if not (r.get("commit_sha") or "").strip()]
    if no_sha:
        print(f"REFUSING: {len(no_sha)} source row(s) carry no commit_sha, e.g. {no_sha[:3]}")
        return 1

    shas = sorted({r["commit_sha"][:12] for r in rows})
    dst_shas = sorted({r["commit_sha"][:12] for r in store.read_runs() if r.get("commit_sha")})
    print(f"{len(rows)} run(s) from {source} @ {', '.join(shas)}")
    if dst_shas and set(shas) - set(dst_shas):
        # Not an error: a re-run of a dropped batch legitimately carries a later
        # SHA. It IS something the paper has to say out loud, so it is said here.
        print(f"  note: {dest} currently holds {', '.join(dst_shas)} - the merged grid "
              f"will span more than one commit.")
    if dry_run:
        print("--dry-run: nothing written")
        return 0

    for row in rows:
        store.append_run(row)

    src_epochs = source / "epochs.csv"
    if src_epochs.exists():
        header = _header(src_epochs)
        if header != EPOCH_COLUMNS:
            print(f"  skipped epochs.csv: header does not match EPOCH_COLUMNS")
        else:
            with open(src_epochs, newline="") as f:
                by_run = {}
                for record in csv.DictReader(f):
                    by_run.setdefault(record["run_id"], []).append(record)
            for run_id, history in by_run.items():
                store.append_epochs(run_id, history)
            print(f"  merged {sum(len(v) for v in by_run.values())} epoch rows")

    copied = 0
    for path in sorted((source / "raw").glob("*.json")):
        target = dest / "raw" / path.name
        if not target.exists():
            shutil.copy(path, target)
            copied += 1
    print(f"  copied {copied} raw payload(s)")
    print(f"merged -> {dst_runs} ({len(store.existing_run_ids())} runs total)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return merge(args.source, args.dest, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
