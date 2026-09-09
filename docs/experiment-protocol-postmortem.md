# Post-mortem: what the first two grids actually measured

Audit of the two results trees the paper is written from, and the record of what
`fix/experiment-protocol` changes as a result.

- **Grid A** — `feat/osq-proxy`, commit `9767be65`, Tesla T4, 5.0 h, 90 runs, all usable.
- **Grid B** — `feat/rich-bricks`, commit `e1cd6624`, Tesla T4, 5.1 h, 190 rows, 154 usable.

Both are in `results/` (`osq_proxy/`, `rich_bricks/`). Everything below is derived from
those trees, not from memory: the grid config that actually ran is embedded in every
`raw/<run_id>.json`, which is how the config drift in §1 was found at all.

---

## 1. Grid B ran about a third of its plan, and the missing third was the important one

The plan was 378 runs. 190 were attempted.

| Planned | Result |
|---|---|
| Tier A, 9 configs × 6 datasets × 2 γ × 3 seeds = 270 | 190 attempted, 154 usable |
| Tier B, 4 interaction configs = 48 | **0 runs** |
| Tier C, 2 full-stack configs = 60 | **0 runs** |
| `synthetic_bottleneck` arm | **0 runs** |
| NCI1, 108 runs | 10 |
| ENZYMES, 108 runs | 18 |

Three unrelated causes, worth separating because they need different fixes.

**a. The synthetic arm was never in the executed plan.** `notebooks/kaggle_rich_bricks.ipynb`
as committed lists `synthetic_bottleneck` first in `DATASETS`. The `config` blob inside
every raw file lists five datasets and not that one. The notebook was edited before being
run and the edit was never committed. This is the arm `tools/experiment_grid.py` calls
"the falsification core" and that the runtime guard is explicitly written never to drop —
it is where an OSq-guided lifting is *supposed* to win. Without it, Grid B can show that
the lifting costs nothing; it cannot show that its mechanism works.

*Fixed by:* `run_grid` now prints a warning when the plan lacks the synthetic arm, and
`modal_grid.py` builds `DATASETS` from a module constant rather than an editable cell.

**b. The runtime guard over-trimmed, then left 3 hours unused.** Budget was 8 h;
the session finished in 5.1 h. `trim_plan` projects with a single global mean
seconds-per-run, and once a few NCI1 runs (416 s each) landed among MUTAG runs (15 s each)
that mean said the plan would not fit. It dropped Tier C, then Tier B, then NCI1, then
ENZYMES — and never revised the projection back down as the cheap datasets caught up.

*Fixed by:* the guard is opt-in and announces itself loudly. `modal_grid.py` never sets it:
one container per run means there is no shared wall clock to protect.

**c. One diverged run cost 36.** `A_A7_MUTAG_g0.0_s0` (Stage 4 = `reinforce`) died after
3.5 s. The 35 runs scheduled after it died in **0.00 s each**, at `torch.manual_seed` —
the first CUDA call of a run, before any model is built. That is the signature of a CUDA
context poisoned by an earlier run, not of a defect in the arms carrying the error. So
**A8 (`s5=tnn`) was recorded as 18 failures on runs it never attempted**, and the TNN
brick has still never been tested once.

### The root cause of (c)

`REINFORCESelector.forward` called `torch.bernoulli(probs)`. On CUDA that kernel validates
`0 <= p <= 1` with a device-side assert. The preceding `clamp(min=eps, max=1-eps)` looks
like it guarantees that, but **`clamp` propagates NaN rather than clipping it** — so once
REINFORCE's unbounded-variance gradient drove `scores` to NaN, `sigmoid` gave NaN, `clamp`
gave NaN, and `bernoulli` asserted. A device-side assert is asynchronous and unrecoverable:
it surfaces at whatever unrelated line next synchronises (here, `int(index.max())` inside a
PyG aggregator in Stage 3) and every later CUDA call in the process fails.

*Fixed by:* three separate changes, because it was three separate failures.
1. `REINFORCESelector` rejects non-finite scores with an ordinary `RuntimeError` before
   reaching `bernoulli`, so the grid can catch it, record it and continue.
2. `GridConfig.grad_clip` (default 1.0), applied in `graph_classification._run_loader` —
   the actual remedy for the divergence, applied to every arm so a clipped and an unclipped
   arm are never compared with each other.
3. One container per run in `modal_grid.py`, so even an unrecoverable context dies alone.

Regression tests: `tests/test_weight_assignment.py::test_reinforce_raises_*`, including
`test_clamp_alone_would_not_have_saved_us`.

---

## 2. The error bars are a protocol artefact, not optimisation noise

Splits were a single stratified 80/10/10 per seed. That leaves:

| Dataset | Test graphs | 1 graph is |
|---|---|---|
| MUTAG | 20 | **5.00 pp** |
| ENZYMES | 60 | 1.67 pp |
| IMDB-BINARY | 100 | 1.00 pp |
| PROTEINS | 112 | 0.89 pp |
| NCI1 | 411 | 0.24 pp |

Every MUTAG accuracy in Grid B is a multiple of 5, and the seed-to-seed standard
deviations run to 8–13 points (`A1 MUTAG γ=0: 76.7 ± 12.58`). More seeds cannot fix this:
they all re-measure the same 20 graphs. Worse, `seed` drove **both** the split and the
initialisation, so the two variance sources are confounded and not separable after the fact.

The consequence is that the two headline questions are unanswerable at this resolution:

| Comparison | Result |
|---|---|
| mini-GNN − DeepSets (Q1) | −0.8 ± 6.7 pp, n = 18 pairs |
| Set Transformer − DeepSets (Q1) | +0.5 ± 7.4 pp, n = 18 pairs |
| γ=0.1 − γ=0 on accuracy (Q2b) | −0.4 ± 6.5 pp, n = 76 pairs |

*Fixed by:* `utils.stratified_kfold` and `GridConfig.cv_folds` (default 10). Every graph
becomes a test graph exactly once, so MUTAG's pooled estimate is over 188 graphs instead of
20 — 0.53 pp of resolution instead of 5.00. The fold layout is fixed across the grid and
independent of `seed`, so the fold picks the split, the seed picks the initialisation, and
paired deltas compare arms on literally the same graphs.

---

## 3. 41% of the usable runs are a plain GCN wearing a brick's name

63 of 154 usable Grid B runs have `alpha_mean == 0` exactly: the selector accepts no cell
at all. When that happens the lifting contributes only zero-weight edges, so the rewired
structure *is* the original graph, `motif_embeddings` is identically zero, and the model
degenerates to a plain GCN. It still trains, still reports an accuracy, and still gets
averaged into its brick's mean.

Nothing in the schema recorded this. It is only visible because `r_bar_before` and
`r_bar_after` come out **bit-identical** in exactly those runs (62 of 154 — the small
mismatch is that `alpha_mean` was measured on one test batch while the OSq report used a
different sample of 8 graphs).

Collapse rate by arm:

| Config | γ=0 | γ=0.1 |
|---|---|---|
| A0 reference | 8/15 | 2/15 |
| **A1 GPSE** | **15/15** | 3/13 |
| A2 autoregressive | 7/12 | 3/12 |
| A3 cycle basis | 5/9 | 3/9 |
| A4 set transformer | 5/9 | 3/9 |
| A5 mini-GNN | 6/9 | 3/9 |
| A6 k-subset | 0/9 | 0/9 |

Three things follow.

- **The GPSE arm's γ=0 half is entirely collapsed.** `brick_ablation_summary.csv` reports
  A1 as the largest brick effect (+8.1 pp); that number compares a collapsed model against
  non-collapsed ones and cannot be read as a property of the GPSE encoder.
- **γ > 0 sharply reduces collapse**, which is a real and reportable mechanism: the OSq
  term wants edges and counteracts the sparsity penalty that is otherwise driving α to 0.
- **A6 (k-subset) never collapses**, as expected — it selects exactly *k* cells by
  construction. That it is also the only arm with no collapse contamination makes it the
  cleanest comparison point in the grid.

α is bimodal overall (63 runs at exactly 0, 33 at exactly 1, 58 strictly between): the
"differentiable" acceptance is behaving as a hard switch, so μ = 0.05 relative to the task
loss is a knob that needs sweeping, not a constant.

*Fixed by:* `alpha_frac_active` and `collapsed` are first-class columns, computed over the
**whole** test split rather than its first batch. A collapsed run is now a fact on the row.

---

## 4. `best_proxy` and `best_gamma` were never selected from anything

Grid B ran its entire plan at `gamma = 0.1` with `proxy = r_bar`, labelled "the winning
proxy/weight from feat/osq-proxy's grid".

- Grid A only ever tested `gamma ∈ {0, 0.01}`. **`gamma = 0.1` had never been evaluated
  anywhere** before the whole grid was run at it.
- `r_bar` was picked on `p = 0.011` against `cf_bc_efc` at `p = 0.031`, over 18 pairs, with
  no correction for having tested four proxies — while `cf_bc_efc` was *ahead* on NCI1
  (67.6 vs 65.5) and on the synthetic arm (43.9 vs 42.2), the one arm that is supposed to
  discriminate.

*Fixed by:* `GridConfig.gammas` / `GridConfig.proxies` turn both into real swept axes, and
`modal run modal_grid.py --stage calibrate` sweeps `γ ∈ {0, 0.003, 0.01, 0.03, 0.1, 0.3}`
× 4 proxies on the reference column under 10-fold CV, before the grid commits to either.
`γ = 0` is emitted once rather than once per proxy, since `DMDLoss` short-circuits `osq_fn`
there and the proxies are indistinguishable.

---

## 5. Smaller things found on the way

- **`KSubsetSelector.set_temperature` had no caller.** Its docstring states that "the grid
  runner calls this once per epoch and logs the schedule"; `grep` finds the method
  definition and nothing else. A6 ran its entire arm at the fixed initial τ = 1.0.
  *Fixed:* `run_single` now anneals τ towards `tau_min` each epoch.
- **`tasks/graph_classification.py::load_dataset` declares `TU_DATASETS = ("NCI1", "MUTAG",
  "PROTEINS")`** and rejects anything else, while the grid runs ENZYMES and IMDB-BINARY
  through its own `load_arm`. Only `train.py`'s CLI path goes through the narrow list, so
  the CLI cannot reproduce a grid run on two of the five datasets. Not fixed here — flagged.
- **Parameter counts span 25 410 to 21 104 962** across arms (the frozen GPSE encoder is
  the upper end), so no comparison involving A1 is budget-matched. The paper already says
  this; worth keeping in view when the redo is designed.

---

## What survives all of this

One result does not depend on any of the above and should be kept:

> The oversquashing term does what it says to the structure. Paired within configuration,
> γ = 0.1 lowers measured mean effective resistance in **58 of 76 pairs and raises it in 1**
> (Δ = −1.04 ± 1.02). It does not move task accuracy (−0.4 ± 6.5 pp).

That is a clean mechanism-works / task-does-not-follow split, and it is the honest core of
the paper. The redo should make the second half *measurable* rather than merely unrefuted —
which, given §2, it currently is not.

---

## Cost of the redo

Extrapolated from the real per-dataset runtimes in the two trees (T4).

| Protocol | Runs | T4-hours | Wall clock, 40 × L4 |
|---|---|---|---|
| `--stage calibrate` (A0, γ×proxy sweep, 10 folds, 1 seed) | ~1 260 | ~49 | ~30 min |
| `--stage full` (A+B+C, 10 folds, 1 seed) | ~1 560 | ~57 | ~35 min |
| `--stage full`, 3 seeds | ~4 680 | ~171 | ~1 h 45 |

With 10-fold CV the fold is already the variance estimate, so one seed is usually the right
answer — a second seed buys much less than a sixth dataset would.
