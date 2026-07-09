# Bayesian hyperparameter search over the DMD training pipeline, using Optuna.
#
# Each trial samples a configuration, trains via train.main() for a fixed
# epoch budget, and reports the run's best validation accuracy (train.py
# already tracks this as best_val_acc - see train.main). optuna.samplers.
# GPSampler drives the search with a Gaussian-process surrogate + LogEI
# acquisition function; --sampler tpe switches to Optuna's default
# Tree-structured Parzen Estimator instead.
#
# Usage:
#   python tune.py --task graph_classification --dataset MUTAG --n-trials 50
#   python tune.py --task node_classification --n-trials 30 --sampler tpe
#
# Results (resumable across runs):
#   results/optuna/<study-name>.db            sqlite: every trial, params + value
#   results/optuna/<study-name>/best_params.json
#   results/optuna/<study-name>/trials.csv    all trials, one row each

import argparse
import json
from pathlib import Path

import optuna

import train

# Hyperparameters searched, with their Optuna sampling spec. Kept to the
# knobs exposed on train.py's CLI (see build_arg_parser there); architecture
# constants only set as Python defaults (num_layers, dropout, ...) aren't
# reachable here without exposing them on that CLI first.
SEARCH_SPACE = {
    "lr": ("float", 1e-4, 1e-1, {"log": True}),
    "weight_decay": ("float", 1e-6, 1e-2, {"log": True}),
    "hidden_dim": ("categorical", [16, 32, 64, 128]),
    "latent_dim": ("categorical", [16, 32, 64]),
    "motif_hidden_dim": ("categorical", [16, 32, 64]),
    "motif_out_dim": ("categorical", [16, 32, 64]),
    "top_k": ("int", 2, 8, {}),
    "selector_tau": ("float", 0.1, 2.0, {}),
    "sparsity_weight": ("float", 1e-3, 1.0, {"log": True}),
}


def suggest_params(trial: optuna.Trial) -> dict:
    params = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float":
            low, high, kwargs = spec[1], spec[2], spec[3]
            params[name] = trial.suggest_float(name, low, high, **kwargs)
        elif kind == "int":
            low, high, kwargs = spec[1], spec[2], spec[3]
            params[name] = trial.suggest_int(name, low, high, **kwargs)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def build_argv(params: dict, args: argparse.Namespace) -> list:
    argv = [
        "--task", args.task,
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--log-every", str(args.epochs + 1),  # keep per-trial console noise down
        "--save-graph-samples", "0",
    ]

    if args.task == "graph_classification":
        argv += ["--dataset", args.dataset, "--data-root", args.data_root,
                 "--batch-size", str(args.batch_size)]
    else:
        argv += ["--num-cliques", str(args.num_cliques), "--clique-size", str(args.clique_size),
                 "--feature-dim", str(args.feature_dim)]

    results_dir = f"{args.results_root}/{args.study_name}/trial_runs" if args.save_trials else ""
    argv += ["--results-dir", results_dir]

    for name, value in params.items():
        argv += [f"--{name.replace('_', '-')}", str(value)]

    return argv


def make_objective(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        result = train.main(build_argv(params, args))
        return result["best_val_acc"]
    return objective


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter search for the DMD pipeline (Optuna).")
    parser.add_argument("--task", type=str, default="node_classification", choices=list(train.TASKS.keys()))
    parser.add_argument("--dataset", type=str, default=None,
                         help="graph_classification only. Choices: NCI1, MUTAG, PROTEINS.")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-cliques", type=int, default=8)
    parser.add_argument("--clique-size", type=int, default=6)
    parser.add_argument("--feature-dim", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100,
                         help="Epoch budget per trial (lower than train.py's 200-epoch default "
                              "so more trials fit in the same time).")
    parser.add_argument("--seed", type=int, default=0,
                         help="Training seed held fixed across trials, so the surrogate is fit on "
                              "hyperparameter effects rather than seed noise.")

    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sampler", type=str, default="gp", choices=["gp", "tpe"],
                         help="gp = Gaussian-process surrogate + LogEI acquisition (optuna.samplers.GPSampler). "
                              "tpe = Optuna's default Tree-structured Parzen Estimator.")
    parser.add_argument("--study-name", type=str, default=None,
                         help="Defaults to '<task>_<dataset-or-toy>'.")
    parser.add_argument("--storage", type=str, default=None,
                         help="Optuna storage URL. Defaults to a sqlite file under --results-root, "
                              "so re-running with the same study name resumes/adds trials.")
    parser.add_argument("--results-root", type=str, default="results/optuna")
    parser.add_argument("--save-trials", action="store_true",
                         help="Also log each trial's full config/history/summary via ExperimentLogger "
                              "(like a normal train.py run) under --results-root/<study-name>/trial_runs.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.study_name is None:
        dataset_label = args.dataset if (args.task == "graph_classification" and args.dataset) else "toy"
        args.study_name = f"{args.task}_{dataset_label}"

    out_dir = Path(args.results_root) / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{Path(args.results_root) / (args.study_name + '.db')}"

    sampler = (optuna.samplers.GPSampler(seed=args.seed) if args.sampler == "gp"
               else optuna.samplers.TPESampler(seed=args.seed))

    study = optuna.create_study(
        study_name=args.study_name, storage=storage, direction="maximize",
        sampler=sampler, load_if_exists=True,
    )
    study.optimize(make_objective(args), n_trials=args.n_trials)

    print(f"\nBest val acc: {study.best_value:.4f}")
    print("Best params:")
    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    with open(out_dir / "best_params.json", "w") as f:
        json.dump({"best_val_acc": study.best_value, "best_params": study.best_params}, f, indent=2)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)

    print(f"\nStudy storage: {storage}")
    print(f"Best params + full trial table saved under {out_dir}")


if __name__ == "__main__":
    main()
