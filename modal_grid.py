# Modal front-end for the Tier A/B/C experiment grid.
#
# This replaces notebooks/kaggle_rich_bricks.ipynb as the way the grid is run,
# and it exists because of three specific things that went wrong on Kaggle:
#
#   1. ONE PROCESS FOR EVERYTHING. A7 (s4=reinforce) diverged, fed a NaN to
#      `torch.bernoulli`, and got a CUDA device-side assert - which poisons the
#      context for the whole process. The 35 runs scheduled after it died in
#      0.00 s each with the same traceback, and A8 (s5=tnn) was recorded as
#      "failed" on 18 runs it never actually attempted. Here every run is its
#      own container, so a poisoned context can take down exactly one run.
#
#   2. A SHARED WALL CLOCK. A single sequential session had to fit in Kaggle's
#      limit, so `trim_plan` projected from a global mean seconds-per-run and
#      deleted Tier B, Tier C, NCI1 and ENZYMES to make the plan fit - then the
#      session finished in 5.1 h of its 8 h budget. Runs here are independent,
#      so there is no shared budget to protect and `time_budget_s` stays None.
#
#   3. CONFIG DRIFT. The committed notebook listed the synthetic bottleneck arm;
#      the config actually executed did not, and the difference was only
#      recoverable months later from a config blob inside a raw result file. The
#      plan here is built from `configs/` and the CLI flags below, and the exact
#      GridConfig is written into env.json before the first run starts.
#
# Usage:
#   modal run modal_grid.py --stage prepare
#   modal run modal_grid.py --stage calibrate     # select gamma and proxy
#   modal run modal_grid.py --stage full          # the actual grid
#
# Results stream back to a local results directory as each container finishes,
# so an interrupted sweep loses nothing and re-running resumes.

import os
import sys
import time

import modal

REPO_REMOTE = "/root/repo"
DATA_REMOTE = "/data"

# torch from PyPI already carries its CUDA runtime, and torch_geometric >= 2.6
# is pure Python (no torch-scatter/torch-sparse compilation step), so the image
# is a plain pip install with no CUDA base image to pin.
#
# topomodelx/toponetx are installed with --no-deps on purpose: they pull
# `pyg-nightly`, a SECOND distribution of the `torch_geometric` package. Letting
# their resolver run mixes files from both distributions in site-packages and
# produces "partially initialized module 'torch_geometric' has no attribute
# 'typing'" at import time. Both are lazy-imported, and only by
# models/message_passing.py's tnn / hypergraph_tnn bricks.
#
# `--no-deps` also makes their `numpy<2` pin moot, so numpy is left unpinned to
# match the local `dmd` environment the test suite is green on (numpy 2.4.6,
# torch 2.14, PyG 2.8.0.post1) - the tnn bricks import and train fine there. A
# cloud image that disagrees with the environment the tests pass in is a second
# configuration nobody is testing.
# torch_geometric is pinned to the version that produced the two existing
# results trees (env.json: 2.8.0.post1) so a redo is comparable to them rather
# than to a different PyG's idea of what `GCNConv` normalisation or
# `precompute_GPSE` does. torch is a recent stable rather than the Kaggle
# nightly those trees used, which is the one deliberate difference.
# `--stage prepare` is the smoke test for this image: it imports everything,
# builds every dataset and runs GPSE end to end.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "scipy",
        "networkx",
        "pyyaml",
        "requests",
    )
    .pip_install("torch_geometric==2.8.0.post1")
    .pip_install("topomodelx", "toponetx", extra_options="--no-deps")
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path=REPO_REMOTE,
        ignore=["results/*", "datasets/*", "paper/*", "notebooks/*", ".git/*",
                "GPSE_pretrained/*.pt", "**/__pycache__", "**/*.pyc"],
    )
)

# Datasets, the GPSE checkpoint and the per-dataset GPSE cache. Written once by
# `--stage prepare`, read-only for every run container afterwards - otherwise
# 1500 containers each download NCI1 and each run a 20-layer GatedGCN over it.
data_volume = modal.Volume.from_name("dmd-grid-data", create_if_missing=True)

# Where a sweep's results live WHILE IT RUNS. `orchestrate` (below) is the only
# writer; `--stage collect` is a reader that can run at any time, from any
# machine, whether or not `orchestrate` has finished - this is what makes a
# sweep survive the laptop that launched it. See `orchestrate`'s docstring.
RESULTS_REMOTE = "/results"
results_volume = modal.Volume.from_name("dmd-grid-results", create_if_missing=True)

app = modal.App("dmd-grid")

# `max_containers` on @app.function needs Modal >= 1.0 (it was `concurrency_limit`
# before). Both are env-overridable so a sweep can be widened or narrowed without
# editing this file: DMD_GPU=A10G DMD_MAX_CONTAINERS=80 modal run modal_grid.py ...
# Per-container dataset cache, populated by `run_one`. Module-level so it
# survives across the inputs one container handles.
_ARM_CACHE = {}

GPU = os.environ.get("DMD_GPU", "L4")
MAX_CONTAINERS = int(os.environ.get("DMD_MAX_CONTAINERS", "15"))

# Inputs per `.map()` call. The first calibrate attempt issued all 1260 at once;
# it processed 79 in seven minutes at a perfectly healthy ~11 runs/min, then the
# map stalled and produced nothing for four hours before the client died in
# `check_lost_inputs` with an auth error. Whatever the cause, one `.map()` over
# the whole plan makes a single stall cost the entire sweep. Batching bounds the
# blast radius: a wedged batch is abandoned, the next one starts clean, and
# `run_id`s already stored are skipped - so a retry converges instead of redoing
# work.
BATCH_SIZE = int(os.environ.get("DMD_BATCH_SIZE", "150"))

# Abandon a batch if no result arrives for this long. This is a MONEY guard, not
# a politeness setting. Twice now the map produced results for six or seven
# minutes and then went silent, only surfacing `AuthError: failed to parse auth
# token` three and a half hours later - and GPU containers stay allocated (and
# billed) through that silence. Two hangs cost ~7h40 of wall clock for 2.2
# GPU-hours of actual work. Whatever wedges the map, nothing justifies waiting
# hours for it: a run that takes longer than this to produce its NEXT result has
# stopped making progress, and the batch is worth more abandoned than waited on.
STALL_TIMEOUT_S = int(os.environ.get("DMD_STALL_TIMEOUT_S", "420"))

# How often `orchestrate` commits the results Volume mid-batch, in completed
# runs. Small on purpose: `--stage collect` is meant to be checkable "any
# time", and a coarse commit cadence makes early progress indistinguishable
# from the exact silence a wedged sweep produces.
COMMIT_EVERY = int(os.environ.get("DMD_COMMIT_EVERY", "10"))


def _setup_repo():
    """
    Make the mounted repo importable and cwd-correct inside a container, and
    point its GPSE checkpoint directory at the shared volume.

    That symlink is not tidiness. `tools/gpse_cache.py::load_pretrained_gpse`
    and `GPSEEncoder.__init__` both default to `weights_root="GPSE_pretrained"`,
    a RELATIVE path - which resolves to `/root/repo/GPSE_pretrained`, inside the
    image, not the volume. And `GPSE.from_pretrained` DOWNLOADS into that root
    when the file is missing rather than failing. So without this, every s1=gpse
    container would silently re-fetch the same 254 MiB checkpoint from Zenodo,
    and the copy `prepare` put in the volume would never be read by anything.
    """
    if REPO_REMOTE not in sys.path:
        sys.path.insert(0, REPO_REMOTE)
    os.chdir(REPO_REMOTE)

    shared = f"{DATA_REMOTE}/GPSE_pretrained"
    local = f"{REPO_REMOTE}/GPSE_pretrained"
    os.makedirs(shared, exist_ok=True)
    if not os.path.islink(local):
        # The image ships the directory (README only - the .pt is gitignored
        # and excluded from the mount), so replace it with the link.
        if os.path.isdir(local):
            import shutil

            shutil.rmtree(local)
        os.symlink(shared, local)


def _remote_paths(config):
    """
    Rewrite every path in a GridConfig for the inside of a container.

    All three live here rather than in `_build_config` because that config is
    also used LOCALLY, by `build_plan` in the local entrypoint: baking
    `/root/repo/configs` into it made the plan unbuildable on the developer's
    machine (PermissionError on /root). A path that differs between the two
    sides belongs at the boundary, which is this function.
    """
    config.configs_dir = f"{REPO_REMOTE}/configs"
    config.data_root = f"{DATA_REMOTE}/datasets"
    config.gpse_cache_dir = f"{DATA_REMOTE}/gpse_cache"
    return config


# ---------------------------------------------------------------------------
# prepare: one container, one pass, everything cached into the volume
# ---------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={DATA_REMOTE: data_volume},
              timeout=60 * 60 * 2)
def prepare(datasets: list, needs_gpse: bool = True):
    """
    Materialises every dataset and (once) the GPSE encodings into the volume.

    Runs on a GPU because the GPSE precomputation is a 20-layer GatedGCN over
    every graph of every dataset - the single most expensive non-training step
    in the project, and the reason tools/gpse_cache.py caches per dataset
    instead of per run.
    """
    _setup_repo()
    import torch

    from tools.experiment_grid import GridConfig, load_arm

    os.makedirs(f"{DATA_REMOTE}/datasets", exist_ok=True)
    os.makedirs(f"{DATA_REMOTE}/gpse_cache", exist_ok=True)
    os.makedirs(f"{DATA_REMOTE}/GPSE_pretrained", exist_ok=True)

    # The pretrained checkpoint (254 MiB, Zenodo 8145095). Fetched into the
    # volume rather than baked into the image so the image stays small and a
    # variant swap does not force a rebuild. tools/gpse_cache.py falls back to
    # pse_explicit if this is unavailable, and RECORDS that it did - a silent
    # degrade to a different Stage 1 encoder is the failure this reports on.
    checkpoint = f"{DATA_REMOTE}/GPSE_pretrained/gpse_model_molpcba_1.0.pt"
    if needs_gpse and not os.path.exists(checkpoint):
        import requests

        url = "https://zenodo.org/records/8145095/files/gpse_model_molpcba_1.0.pt"
        print(f"downloading GPSE checkpoint from {url}")
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(checkpoint, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  -> {os.path.getsize(checkpoint) / 1e6:.0f} MB")

    config = _remote_paths(GridConfig(device="cuda" if torch.cuda.is_available() else "cpu"))

    sources = {}
    for name in datasets:
        started = time.time()
        _, num_features, num_classes, source = load_arm(name, config, needs_gpse=needs_gpse)
        sources[name] = source
        print(f"{name:>22}: features={num_features} classes={num_classes} "
              f"s1_encoder_actual={source} ({time.time() - started:.0f}s)")
        data_volume.commit()

    data_volume.commit()
    return sources


# ---------------------------------------------------------------------------
# one run, one container
# ---------------------------------------------------------------------------

@app.function(image=image, gpu=GPU, volumes={DATA_REMOTE: data_volume},
              timeout=60 * 60, max_containers=MAX_CONTAINERS, retries=0)
def run_one(payload: dict) -> dict:
    """
    Trains exactly one RunSpec and returns its stored payload.

    `retries=0` deliberately: a run that fails is a RESULT, and re-rolling it
    until it passes would quietly select for lucky seeds. The traceback is
    returned in the payload, stored in the row, and analysed as a failure -
    which is how the reinforce divergence should have surfaced the first time.

    The failure is caught here rather than left to Modal so that a crash still
    produces a row. It cannot cascade: the container is torn down afterwards, so
    even an unrecoverable CUDA context dies with this one run.
    """
    _setup_repo()
    import traceback

    import torch

    from tools.config_loader import load_all_configs
    from tools.experiment_grid import GridConfig, RunSpec, load_arm, run_single

    # A container that started before `prepare` committed would otherwise see a
    # stale mount and re-download the dataset (or, worse, silently fall back to
    # pse_explicit because the GPSE checkpoint "is not there").
    data_volume.reload()

    # Reconstructed from a plain dict rather than shipped as a pickled
    # GridConfig: Modal deserializes arguments BEFORE the function body runs,
    # so at that moment `tools.experiment_grid` is not yet importable (the repo
    # goes on sys.path in `_setup_repo`, above) and a pickled instance of a
    # class from it cannot be resolved.
    config = _remote_paths(GridConfig(**payload["config"]))
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = RunSpec(**payload["spec"])

    pipeline_config = load_all_configs(config.configs_dir)[spec.config_id]
    needs_gpse = pipeline_config["s1"] == "gpse"

    # A Modal container serves many inputs in sequence, so loading the arm per
    # input would re-read (and, for gpse, re-attach) the same dataset ~30 times
    # per container. Keyed on gpse too: the cached encodings are attached to the
    # Data objects in place, so a gpse run and a non-gpse run cannot share one.
    cache_key = (spec.dataset, needs_gpse)
    started = time.time()
    try:
        if cache_key not in _ARM_CACHE:
            _ARM_CACHE.clear()  # one arm at a time: these hold whole datasets
            _ARM_CACHE[cache_key] = load_arm(spec.dataset, config, needs_gpse=needs_gpse)
        dataset, num_features, num_classes, _ = _ARM_CACHE[cache_key]
        result = run_single(spec, config, pipeline_config, dataset, num_features, num_classes)
        row, history = result["row"], result["history"]
        extras = {"split_indices": result["split_indices"], "osq": result["osq"], "error": None}
    except Exception:  # noqa: BLE001 - a failed run is a row, not a lost run
        error = traceback.format_exc()
        row = {"run_id": spec.run_id, "commit_sha": config.commit_sha,
               "tier": spec.tier, "config_id": spec.config_id, "dataset": spec.dataset,
               "gamma": spec.gamma, "seed": spec.seed,
               "fold": "" if spec.fold is None else spec.fold,
               "osq_proxy": (spec.proxy or config.best_proxy) if spec.gamma > 0 else "none",
               "sparsity_weight": config.sparsity_weight,
               "status": "failed", "error": error.strip().splitlines()[-1][:300]}
        history, extras = [], {"split_indices": None, "osq": None, "error": error}

    row["runtime_s"] = round(time.time() - started, 2)
    return {"run_id": spec.run_id, "spec": vars(spec), "row": row, "history": history, **extras}


def _drain_with_stall_timeout(stream, timeout_s):
    """
    Yield from `stream`, raising TimeoutError if it goes `timeout_s` without
    producing anything.

    A background thread does the blocking iteration and hands items over a
    queue; the consumer waits with a timeout. The thread is a daemon and is
    deliberately NOT joined on timeout - a wedged Modal iterator may never
    return, and the point of this function is to stop waiting on it.
    """
    import queue
    import threading

    items = queue.Queue(maxsize=64)
    DONE = object()

    def pump():
        try:
            for item in stream:
                items.put(item)
        except Exception as error:  # noqa: BLE001 - surfaced to the consumer below
            items.put(error)
        finally:
            items.put(DONE)

    threading.Thread(target=pump, daemon=True).start()

    while True:
        try:
            item = items.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(
                f"no result for {timeout_s}s - abandoning this batch rather than "
                "holding GPU containers open on a map that has stopped progressing")
        if item is DONE:
            return
        yield item


# ---------------------------------------------------------------------------
# the whole sweep, running server-side
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={RESULTS_REMOTE: results_volume},
              timeout=10 * 60 * 60, retries=0)
def orchestrate(payloads: list, out_name: str, env: dict) -> dict:
    """
    Runs an entire sweep inside Modal: batches `payloads` through `run_one`,
    applies the stall guard, and writes every result straight to the results
    Volume as it lands - never back to the caller.

    Why this exists: `main()` used to drive `run_one.map()` directly from the
    laptop, iterating the results as they streamed back. That died three times
    when the laptop's network went away for good (twice from sleep, once from
    what the traceback shows was a plain connection drop) - the local client IS
    what keeps a `.map()` iterator alive, so losing it stalls the sweep with no
    way to reconnect, and the last two hangs cost ~7h40 of wall clock for 2.2
    GPU-hours of actual work.

    Spawned (`orchestrate.spawn(...)`, never awaited) from `main()`, together
    with `modal run --detach`, this keeps running on Modal's infrastructure
    independent of the local machine: closing the laptop, losing wifi, or
    Ctrl-C on the local process does not touch it. `--stage collect` reads back
    whatever is on the volume at any time, whether or not this call has
    finished - that is the actual repatriation mechanism, not a live tether to
    this function call.

    Resumable by construction: reads whatever `run_id`s already exist for
    `out_name` on the volume and skips them, so re-spawning after a cancelled
    or timed-out attempt continues instead of redoing.
    """
    _setup_repo()
    import csv
    import json as _json

    from tools.experiment_grid import RunSpec
    from tools.results_store import EPOCH_COLUMNS, RUN_COLUMNS

    out_dir = f"{RESULTS_REMOTE}/{out_name}"
    raw_dir = f"{out_dir}/raw"
    os.makedirs(raw_dir, exist_ok=True)

    runs_path = f"{out_dir}/runs.csv"
    epochs_path = f"{out_dir}/epochs.csv"
    is_new_runs = not os.path.exists(runs_path)
    is_new_epochs = not os.path.exists(epochs_path)

    existing = set()
    if not is_new_runs:
        with open(runs_path, newline="") as f:
            existing = {row["run_id"] for row in csv.DictReader(f)}
    payloads = [p for p in payloads if RunSpec(**p["spec"]).run_id not in existing]

    # `env` was captured on the LAPTOP that submitted this sweep, so its `gpu`,
    # `cuda_available` and `torch` describe the client, not the machines that
    # do the work - `results/calibrate5/env.json` went out recording
    # "torch 2.14.0+cpu, gpu None" for a sweep that ran entirely on L4s. Record
    # the worker's own environment here, where we are actually standing on it,
    # and keep the client's under its own key rather than overwriting it.
    from tools.experiment_grid import environment_info

    worker = environment_info(REPO_REMOTE)
    env = dict(env, client_environment={k: env.get(k) for k in
                                        ("platform", "python", "torch", "gpu",
                                         "cuda_available", "torch_geometric")})
    env.update({k: worker[k] for k in
                ("platform", "python", "torch", "gpu", "cuda_available",
                 "torch_geometric") if k in worker})
    with open(f"{out_dir}/env.json", "w") as f:
        _json.dump(env, f, indent=2, default=str)

    runs_f = open(runs_path, "a", newline="")
    epochs_f = open(epochs_path, "a", newline="")
    runs_w = csv.DictWriter(runs_f, fieldnames=RUN_COLUMNS, extrasaction="ignore")
    epochs_w = csv.DictWriter(epochs_f, fieldnames=EPOCH_COLUMNS, extrasaction="ignore")
    if is_new_runs:
        runs_w.writeheader()
    if is_new_epochs:
        epochs_w.writeheader()
    results_volume.commit()

    completed = failed = errored = wedged = 0
    print(f"orchestrate: {len(payloads)} runs to submit for '{out_name}' "
          f"({len(existing)} already on the volume)", flush=True)

    for offset in range(0, len(payloads), BATCH_SIZE):
        batch = payloads[offset:offset + BATCH_SIZE]
        print(f"\n--- batch {offset // BATCH_SIZE + 1}"
              f"/{(len(payloads) + BATCH_SIZE - 1) // BATCH_SIZE}: {len(batch)} runs ---",
              flush=True)
        try:
            # `order_outputs=False` so a slow DHFR run does not hold back the
            # MUTAG results queued behind it; `return_exceptions=True` so an
            # infrastructure error on one container is reported, not fatal.
            stream = run_one.map(batch, order_outputs=False, return_exceptions=True)
            for result in _drain_with_stall_timeout(stream, STALL_TIMEOUT_S):
                if isinstance(result, Exception):
                    errored += 1
                    print(f"[infra] {type(result).__name__}: {result}", flush=True)
                    continue

                row = result["row"]
                runs_w.writerow({column: row.get(column, "") for column in RUN_COLUMNS})
                for record in result["history"]:
                    erow = {column: record.get(column, "") for column in EPOCH_COLUMNS}
                    erow["run_id"] = result["run_id"]
                    epochs_w.writerow(erow)
                runs_f.flush()
                epochs_f.flush()

                with open(f"{raw_dir}/{result['run_id']}.json", "w") as rf:
                    _json.dump({
                        "run_id": result["run_id"], "spec": result["spec"], "row": row,
                        "history": result["history"], "split_indices": result["split_indices"],
                        "osq": result["osq"], "error": result["error"],
                    }, rf, indent=2, default=str)

                completed += row["status"] == "ok"
                failed += row["status"] != "ok"
                print(f"[{completed + failed:>5}/{len(payloads)}] {result['run_id']:<52} "
                      f"{row['status']:<7} acc={row.get('test_acc', '')} "
                      f"collapsed={row.get('collapsed', '')} ({row['runtime_s']}s)", flush=True)

                # Committing only once per BATCH_SIZE=150 runs would leave
                # `--stage collect` showing nothing for the ~10+ minutes it takes
                # 15 parallel containers to clear a batch - indistinguishable
                # from the exact silence that made the last two hangs so hard to
                # tell from a healthy run. Every COMMIT_EVERY results instead, so
                # collect reflects genuinely fresh progress.
                if (completed + failed) % COMMIT_EVERY == 0:
                    results_volume.commit()
        except Exception as error:  # noqa: BLE001 - a wedged batch must not end the sweep
            wedged += 1
            print(f"[batch abandoned] {type(error).__name__}: {error}", flush=True)
            print("  continuing with the next batch; stored runs are skipped on a retry.",
                  flush=True)
        results_volume.commit()  # durable, and visible to `--stage collect`, after every batch

    runs_f.close()
    epochs_f.close()
    results_volume.commit()

    summary = {"completed": completed, "failed": failed, "errored": errored,
              "wedged": wedged, "remaining": len(payloads) - completed - failed}
    print("\n" + "=" * 70)
    for key, value in summary.items():
        print(f"{key:11}: {value}")
    return summary


# ---------------------------------------------------------------------------
# plan presets
# ---------------------------------------------------------------------------

# DHFR rather than NCI1 as the large molecular control. NCI1 is 4110 graphs of
# ~30 nodes and costs ~622 s/run against DHFR's ~162 s - 56-60% of the entire
# grid's compute for one dataset, which does not fit Modal's monthly credit.
#
# The swap costs nothing scientifically BECAUSE of the cross-validation change:
# NCI1's real advantage was a 411-graph test split, and under 10-fold every
# graph is a test graph exactly once, so DHFR's pooled estimate is over all 756
# of its graphs (0.13 pp of resolution). AIDS is the same price and bigger, but
# is a saturated benchmark (~99% for trivial baselines) - a control that cannot
# discriminate does not control anything.
ALL_DATASETS = ["synthetic_bottleneck", "MUTAG", "PROTEINS", "IMDB-BINARY", "ENZYMES", "DHFR"]

# Stage `calibrate`: pick gamma and the proxy on evidence, on the reference
# config only. This step has never been run. feat/osq-proxy compared four
# proxies at the single value gamma=0.01; feat/rich-bricks then ran its entire
# grid at gamma=0.1, a value nothing had ever evaluated. Sweeping gamma on one
# config first is both cheaper and the only thing that makes "best_gamma" a
# claim rather than a label.
CALIBRATE_GAMMAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]
CALIBRATE_PROXIES = ["r_bar", "lambda2", "efc", "cf_bc_efc"]


def _build_config(stage: str, seeds, folds: int, proxies=None):
    from tools.experiment_grid import GridConfig

    # `configs_dir` stays LOCAL here - the plan is built on this machine.
    # `_remote_paths` swaps it for the container path inside each run.
    common = dict(datasets=ALL_DATASETS, configs_dir="configs",
                  seeds=list(seeds), cv_folds=folds, grad_clip=1.0,
                  time_budget_s=None, device="cuda")

    if stage == "calibrate":
        return GridConfig(gammas=CALIBRATE_GAMMAS,
                          proxies=list(proxies) if proxies else CALIBRATE_PROXIES,
                          **common)
    return GridConfig(**common)


def _collect(out: str, with_raw: bool = False) -> None:
    """
    `--stage collect`: pulls whatever `orchestrate` has written so far for
    this `out` directory's name from the results Volume into the local `out`
    directory. Safe to run repeatedly, at any point during or after a sweep -
    it is a plain sync, not tied to any particular spawned call.
    """
    import io

    from tools.results_store import ResultsStore

    out_name = os.path.basename(os.path.normpath(out))

    def _pull(remote_name: str, local_path: str) -> bool:
        buf = io.BytesIO()
        try:
            results_volume.read_file_into_fileobj(f"{out_name}/{remote_name}", buf)
        except FileNotFoundError:
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(buf.getvalue())
        return True

    os.makedirs(out, exist_ok=True)
    got_env = _pull("env.json", os.path.join(out, "env.json"))
    got_runs = _pull("runs.csv", os.path.join(out, "runs.csv"))
    got_epochs = _pull("epochs.csv", os.path.join(out, "epochs.csv"))

    if not got_runs:
        print(f"nothing on the volume yet for '{out_name}' - has a sweep been spawned "
              f"(modal run --detach modal_grid.py --stage calibrate --out {out})?")
        return

    rows = ResultsStore(out).read_runs()
    ok = sum(1 for row in rows if row["status"] == "ok")
    print(f"collected: {len(rows)} runs ({ok} ok, {len(rows) - ok} failed) -> {out}/runs.csv")
    if got_epochs:
        print(f"           epochs.csv -> {out}/epochs.csv")
    if not got_env:
        print("  (no env.json yet - orchestrate may not have started writing)")

    if with_raw:
        try:
            entries = results_volume.listdir(f"{out_name}/raw", recursive=False)
        except (FileNotFoundError, modal.exception.NotFoundError):
            # `read_file_into_fileobj` raises the builtin `FileNotFoundError` for
            # a missing FILE; `listdir` raises Modal's own `NotFoundError` for a
            # missing DIRECTORY. Both mean the same thing here - nothing yet.
            entries = []
        print(f"pulling {len(entries)} raw/*.json files...")
        for i, entry in enumerate(entries, 1):
            name = os.path.basename(entry.path)
            _pull(f"raw/{name}", os.path.join(out, "raw", name))
            if i % 50 == 0 or i == len(entries):
                print(f"  {i}/{len(entries)}", flush=True)


# ---------------------------------------------------------------------------
# local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "full", out: str = "results/modal", seeds: str = "0",
         folds: int = 10, only_tier: str = "", only_config: str = "",
         proxies: str = "", gammas: str = "", dry_run: bool = False,
         gpse: bool = True, with_raw: bool = False):
    """
    stage:       prepare | calibrate | full | collect
    out:         for calibrate/full, the NAME (its basename) of the directory
                 the sweep writes to on the results Volume while it runs - not
                 a local path yet. `collect` is what copies it locally, into
                 this same `out` path, and can be run again at any time to
                 pull the latest state; it never talks to a running sweep.
    seeds:       comma-separated. With 10-fold CV the fold IS the variance
                 estimate, so one seed is usually the right answer and a second
                 seed buys far less than a sixth dataset would.
    only_tier:   restrict to "A" / "B" / "C" (empty = all)
    only_config: comma-separated config_ids to restrict to. Defaults to "A0"
                 for `calibrate`, which is the point of that stage: gamma and
                 the proxy are properties of the objective, so they are selected
                 once on the reference column rather than re-swept inside every
                 brick. A LIST is what makes a single-axis sweep possible:
                 `--only-config A0,A4,A5` is the three cell encoders and nothing
                 else, run into one directory so the comparison is one table
                 rather than three that have to be stitched together.
    gammas:      comma-separated override of the gamma arm(s). Empty keeps the
                 stage's own choice - the {0, best_gamma} pair for `full`, the
                 sweep for `calibrate`. `--gammas 0.0` isolates an axis that
                 does not touch the objective: at gamma=0 DMDLoss short-circuits
                 osq_fn entirely, so the result does not depend on WHAT the
                 rewired structure looks like to the proxy. That is what lets
                 the cell-encoder comparison be run now, while the rank-0
                 structure the proxy scores (star vs. clique) is still open.
    with_raw:    `collect` only - also pull every raw/<run_id>.json (config,
                 per-epoch history, split indices). Off by default: runs.csv is
                 enough for results/select_gamma_proxy.py and the analysis
                 scripts, and a few thousand small file transfers is the slow
                 part of a collect.

    IMPORTANT for calibrate/full: run this with
        modal run --detach modal_grid.py --stage calibrate ...
    (`--detach` goes right after `modal run`, before the script). Without it,
    the App is torn down the moment this process exits or the connection to
    Modal drops - which has already happened three times: twice from this
    machine sleeping, once from what its traceback shows was a plain network
    drop. `--detach` plus the volume-backed `orchestrate`/`collect` split below
    is what makes a sweep survive the laptop that launched it being closed.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tools.experiment_grid import build_plan, environment_info
    from tools.results_store import ResultsStore

    if stage == "prepare":
        sources = prepare.remote(ALL_DATASETS, needs_gpse=gpse)
        print("\nGPSE sources (an s1_encoder_actual of 'pse_explicit' means the real "
              "encoder was NOT used):")
        for name, source in sources.items():
            print(f"  {name:>22}: {source}")
        return

    if stage == "collect":
        _collect(out, with_raw=with_raw)
        return

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    proxy_list = [p.strip() for p in proxies.split(",") if p.strip()]
    config = _build_config(stage, seed_list, folds, proxies=proxy_list)
    gamma_list = [float(g) for g in gammas.split(",") if g.strip()]
    if gamma_list:
        config.gammas = gamma_list
    config.commit_sha = environment_info(".").get("commit_sha", "unknown")

    if not only_config and stage == "calibrate":
        only_config = "A0"
    config_ids = [c.strip() for c in only_config.split(",") if c.strip()]

    plan = build_plan(config)
    if only_tier:
        plan = [spec for spec in plan if spec.tier == only_tier]
    if config_ids:
        plan = [spec for spec in plan if spec.config_id in config_ids]

    # A misspelt --only-config filters the plan down to nothing, and an empty
    # plan is indistinguishable from a finished one at the `already >= len(plan)`
    # check below (0 >= 0), which would report "nothing to do" and exit 0. Fail
    # loudly instead - a launch that quietly does nothing is worse than an error.
    if not plan:
        raise SystemExit(
            f"empty plan: nothing matches config(s) {config_ids or ['all']} in tier "
            f"{only_tier or 'A+B+C'}. Check the ids against configs/tier_*/.")

    out_name = os.path.basename(os.path.normpath(out))

    # A run_id encodes the fold INDEX but not the fold COUNT, so `_f0` under
    # 5-fold and `_f0` under 10-fold are different splits sharing one name.
    # Resuming across a change of axes would skip a run because a same-named but
    # differently-split row exists, silently mixing two protocols in one table.
    # Read from the VOLUME's env.json, not a local one - results now live there
    # first, and a local `out` dir may be stale, empty, or from before this
    # split-per-name scheme existed.
    import io
    import json as _json

    buf = io.BytesIO()
    try:
        results_volume.read_file_into_fileobj(f"{out_name}/env.json", buf)
    except FileNotFoundError:
        previous = {}
    else:
        previous = _json.loads(buf.getvalue()).get("grid_config", {})
    for axis in ("cv_folds", "gammas", "proxies", "seeds", "datasets"):
        was = previous.get(axis)
        now = getattr(config, axis, None)
        if was is not None and was != now and list(was or []) != list(now or []):
            raise SystemExit(
                f"'{out_name}' on the results volume was built with {axis}={was!r} but this "
                f"run asks for {now!r}.\nSpawning would mix two protocols under one set of "
                "run_ids. Use a different --out name.")

    try:
        already = len(results_volume.listdir(f"{out_name}/raw", recursive=False))
    except (FileNotFoundError, modal.exception.NotFoundError):
        already = 0

    print(f"stage      : {stage}")
    print(f"plan       : {len(plan)} runs ({already} already on the volume - "
          "orchestrate skips them)")
    print(f"gammas     : {config.gammas if config.gammas is not None else (0.0, config.best_gamma)}")
    print(f"proxies    : {config.proxies if config.proxies is not None else (config.best_proxy,)}")
    print(f"seeds      : {list(config.seeds)}   folds: {config.cv_folds}")
    print(f"configs    : {only_config or 'all'}   tier: {only_tier or 'A+B+C'}")
    print(f"grad_clip  : {config.grad_clip}")
    print(f"gpu        : {GPU} x up to {MAX_CONTAINERS} containers")
    if dry_run:
        for spec in plan[:15]:
            print("  ", spec.run_id)
        print(f"   ... ({len(plan)} total)")
        return
    if already >= len(plan):
        print("nothing to do (everything already on the volume)")
        return

    env = environment_info(".")
    env.update({"stage": stage, "runner": "modal", "gpu_requested": GPU,
                "grid_config": {k: (list(v) if isinstance(v, (tuple, range)) else v)
                                for k, v in vars(config).items()}})

    config_payload = {k: (list(v) if isinstance(v, (tuple, range)) else v)
                      for k, v in vars(config).items()}
    payloads = [{"config": config_payload, "spec": vars(spec)} for spec in plan]

    try:
        import subprocess

        env["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance, never worth failing a launch over
        env["branch"] = "unknown"
    call = orchestrate.spawn(payloads, out_name, env)
    print(f"\nspawned orchestrate (function_call_id = {call.object_id!r}).")
    print("This now runs on Modal's infrastructure, independent of this terminal - it is safe")
    print("to close this laptop IF this was launched with `modal run --detach`.")
    print("\nCheck progress or pull results any time with:")
    print(f"  modal run modal_grid.py --stage collect --out {out}")
    print("\nCancel it with:")
    print(f"  python -c \"import modal; "
          f"modal.functions.FunctionCall.from_id({call.object_id!r}).cancel()\"")
