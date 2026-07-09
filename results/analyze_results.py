# Loads a single training run saved by tools/experiment_logger.py
# (config.json, history.jsonl, summary.json, graph_samples.json) and renders
# three figures: training loss components, train/val/test performance, and
# original-vs-rewired adjacency heatmaps for the graphs snapshotted at the
# end of training.
#
# Usage:
#   python results/analyze_results.py results/runs/<experiment>/<run_id>

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical colors so the same series always gets the same color
# across every figure (never re-cycled/reassigned per plot).
COLORS = {
    "total": "#4C72B0",
    "task": "#DD8452",
    "sparsity": "#55A868",
    "osq": "#C44E52",
    "train": "#4C72B0",
    "val": "#DD8452",
    "test": "#55A868",
}


def load_run(exp_dir: Path):
    with open(exp_dir / "config.json") as f:
        config = json.load(f)

    history = []
    with open(exp_dir / "history.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))

    summary = {}
    summary_path = exp_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    samples = []
    samples_path = exp_dir / "graph_samples.json"
    if samples_path.exists():
        with open(samples_path) as f:
            samples = json.load(f)

    return config, history, summary, samples


def experiment_title(config: dict) -> str:
    dataset = config.get("dataset") or "toy synthetic (path-of-cliques)"
    return (f"{config['task']} | dataset={dataset} | encoder={config['encoder']} | "
            f"hidden={config['hidden_dim']} latent={config['latent_dim']} top_k={config['top_k']} | "
            f"seed={config['seed']}")


def plot_loss_components(history, config, out_path):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_loss"] for r in history], label="Total", color=COLORS["total"], linewidth=2)
    ax.plot(epochs, [r["train_task"] for r in history], label="Task (cross-entropy)",
            color=COLORS["task"], linewidth=2)
    ax.plot(epochs, [r["train_sparsity"] for r in history], label="Sparsity (mean α)",
            color=COLORS["sparsity"], linewidth=2)
    if config.get("osq_weight", 0):  # Phase 0 default: OSq term is off - skip a flat, meaningless zero line
        ax.plot(epochs, [r["train_osq"] for r in history], label="OSq", color=COLORS["osq"], linewidth=2)

    ax.set_title(f"Training loss components\n{experiment_title(config)}", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_task_loss_splits(history, config, out_path):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_task"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_task"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_task"] for r in history], label="Test", color=COLORS["test"], linewidth=2)

    ax.set_title(f"Task loss: train vs. validation vs. test\n{experiment_title(config)}", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_performance(history, summary, config, out_path):
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_accuracy"] for r in history], label="Train", color=COLORS["train"], linewidth=2)
    ax.plot(epochs, [r["val_accuracy"] for r in history], label="Validation", color=COLORS["val"], linewidth=2)
    ax.plot(epochs, [r["test_accuracy"] for r in history], label="Test", color=COLORS["test"], linewidth=2)

    best_val = summary.get("best_val_acc")
    if best_val is not None:
        best_epoch = next((r["epoch"] for r in history if r["val_accuracy"] == best_val), None)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"Best val @ epoch {best_epoch}")

    ax.set_title(f"Accuracy over training\n{experiment_title(config)}", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _adjacency_from_edges(num_nodes, edge_index, weights=None):
    mat = np.zeros((num_nodes, num_nodes))
    if not edge_index or not edge_index[0]:
        return mat
    src, dst = edge_index
    if weights is None:
        weights = [1.0] * len(src)
    for s, d, w in zip(src, dst, weights):
        mat[s, d] = w
    return mat


def plot_adjacency_heatmaps(samples, config, out_path):
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)

    for row, sample in enumerate(samples):
        num_nodes = sample["num_nodes"]
        orig = _adjacency_from_edges(num_nodes, sample["edge_index"])
        rewired = _adjacency_from_edges(num_nodes, sample["rewired_edge_index"], sample["rewired_edge_weight"])
        label_suffix = f", label={sample['label']}" if sample.get("label") is not None else ""

        for col, (mat, name) in enumerate([(orig, "Original"), (rewired, "Rewired")]):
            ax = axes[row][col]
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"Graph {sample['graph_id']} — {name}{label_suffix}\n{num_nodes} nodes", fontsize=9)
            ax.set_xlabel("Node index")
            ax.set_ylabel("Node index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Edge weight")

    fig.suptitle(f"Original vs. rewired adjacency\n{experiment_title(config)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze a saved DMD training run.")
    parser.add_argument("exp_dir", type=str,
                         help="Path to a run directory, e.g. results/runs/<experiment>/<run_id>")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Where to save plots. Defaults to <exp_dir>/plots.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    config, history, summary, samples = load_run(exp_dir)
    if not history:
        raise SystemExit(f"No history found in {exp_dir} (expected history.jsonl)")

    plot_loss_components(history, config, out_dir / "loss_components.png")
    plot_task_loss_splits(history, config, out_dir / "loss_train_val_test.png")
    plot_performance(history, summary, config, out_dir / "performance.png")
    plot_adjacency_heatmaps(samples, config, out_dir / "adjacency_heatmaps.png")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
