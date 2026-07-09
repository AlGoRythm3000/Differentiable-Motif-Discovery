# calls graph_classification.py or node_classification.py to train the model

import argparse

import torch

import utils
from models.dmd_model import DMDModel
from tools.losses import DMDLoss
from tools.experiment_logger import ExperimentLogger
from tasks import node_classification, graph_classification

TASKS = {
    "node_classification": node_classification,
    "graph_classification": graph_classification,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the DMD pipeline (Phase 0 scaffold).")
    parser.add_argument("--task", type=str, default="node_classification", choices=list(TASKS.keys()))

    # dataset: node_classification only uses the toy synthetic path-of-cliques
    # graph below. graph_classification selects a real benchmark via --dataset
    # (NCI1, MUTAG, PROTEINS - TUDataset, auto-downloaded into --data-root).
    parser.add_argument("--dataset", type=str, default=None,
                         help="graph_classification only. Choices: NCI1, MUTAG, PROTEINS.")
    parser.add_argument("--data-root", type=str, default="datasets",
                         help="graph_classification only. Where TUDataset downloads/caches data.")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="graph_classification only. Graphs per batch.")
    parser.add_argument("--num-cliques", type=int, default=8)
    parser.add_argument("--clique-size", type=int, default=6)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)

    # model
    parser.add_argument("--encoder", type=str, default="gcn", choices=["gcn", "gin"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--motif-hidden-dim", type=int, default=32)
    parser.add_argument("--motif-out-dim", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--selector-tau", type=float, default=0.5)
    parser.add_argument("--selector-soft", action="store_true",
                         help="Use soft (non straight-through) Stage 4 selection.")
    parser.add_argument("--no-original-edges", action="store_true",
                         help="Rewired-1-skeleton-only ablation (drops the original edges).")

    # optimization
    parser.add_argument("--sparsity-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)

    # results persistence
    parser.add_argument("--results-dir", type=str, default="results/runs",
                         help="Where per-run config/metrics/graph snapshots are saved "
                              "(results/analyze_results.py reads from here). Pass \"\" to disable.")
    parser.add_argument("--save-graph-samples", type=int, default=3,
                         help="Number of sample graphs (original vs. rewired adjacency) to "
                              "snapshot at the end of training.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    utils.set_seed(args.seed)

    task_module = TASKS[args.task]
    data, input_dim, num_classes = task_module.load_dataset(args)

    logger = ExperimentLogger(args.results_dir, args, parser) if args.results_dir else None

    model = DMDModel(
        input_dim=input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        motif_hidden_dim=args.motif_hidden_dim, motif_out_dim=args.motif_out_dim,
        num_classes=num_classes, encoder_type=args.encoder, top_k=args.top_k,
        selector_tau=args.selector_tau, selector_hard=not args.selector_soft,
        include_original_edges=not args.no_original_edges,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = DMDLoss(sparsity_weight=args.sparsity_weight)

    best_val_acc = 0.0
    test_acc_at_best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = task_module.train_step(model, data, optimizer, criterion)
        val_metrics = task_module.eval_step(model, data, criterion, data.val_mask)
        test_metrics = task_module.eval_step(model, data, criterion, data.test_mask)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            test_acc_at_best_val = test_metrics["accuracy"]

        if logger is not None:
            logger.log_epoch(epoch, train_metrics, val_metrics, test_metrics)

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_metrics['loss']:.4f} "
                  f"(Task: {train_metrics['task']:.4f}, Sparse: {train_metrics['sparsity']:.4f}) | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")

    result = {"best_val_acc": best_val_acc, "test_acc_at_best_val": test_acc_at_best_val}

    if logger is not None:
        logger.save_summary(result)
        logger.save_graph_samples(task_module.collect_graph_samples(model, data, args.save_graph_samples))
        logger.close()
        print(f"\nResults saved to {logger.exp_dir}")

    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc @ Best Val: {test_acc_at_best_val:.4f}")
    return result


if __name__ == "__main__":
    main()
