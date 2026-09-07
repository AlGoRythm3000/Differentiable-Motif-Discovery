# calls graph_classification.py or node_classification.py to train the model

import argparse
import copy
import json

import torch

import utils
from models import registry
from models.dmd_model import DMDModel
from tools.losses import DMDLoss
from tools.osq_metrics import before_after_report
from tools.osq_proxies import (DEFAULT_CG_MAXITER, DEFAULT_CG_TOL, DEFAULT_EPS,
                                DEFAULT_HUTCH_K, available_proxies, get_proxy)
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

    # model - the five pipeline stages (models/registry.py), selectable by
    # string. Defaults reproduce the pre-feat/rich-bricks "simple column"
    # exactly. --encoder is kept as the Stage 1 flag name (rather than
    # --s1) since every existing invocation already uses it.
    parser.add_argument("--encoder", type=str, default="gcn",
                         choices=list(registry.STAGE1_ENCODERS), help="Stage 1 brick.")
    parser.add_argument("--proposal", type=str, default="topk",
                         choices=list(registry.STAGE2_PROPOSALS), help="Stage 2 brick.")
    parser.add_argument("--cell-encoder", type=str, default="deepsets",
                         choices=list(registry.STAGE3_CELL_ENCODERS), help="Stage 3 brick.")
    parser.add_argument("--selector-type", type=str, default="gumbel",
                         choices=list(registry.STAGE4_SELECTORS), help="Stage 4 brick.")
    parser.add_argument("--message-passing", type=str, default="gnn_rewired",
                         choices=list(registry.STAGE5_MP), help="Stage 5 brick.")
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

    # oversquashing objective (Stage 6). --osq-weight is the gamma of
    # L = L_task + mu * L_sparsity + gamma * L_osq; at gamma = 0 the proxy is
    # never called, so any --osq-proxy reproduces the proxy-free run exactly.
    parser.add_argument("--osq-proxy", type=str, default="none", choices=list(available_proxies()),
                         help="Which differentiable OSq proxy to add to the loss.")
    parser.add_argument("--osq-weight", type=float, default=0.0,
                         help="gamma, the weight of the OSq term.")
    parser.add_argument("--hutch-k", type=int, default=DEFAULT_HUTCH_K,
                         help="Number of Hutchinson probes / CG right-hand sides (r_bar, cf_bc_efc).")
    parser.add_argument("--cg-tol", type=float, default=DEFAULT_CG_TOL)
    parser.add_argument("--cg-maxiter", type=int, default=DEFAULT_CG_MAXITER)
    parser.add_argument("--osq-eps", type=float, default=DEFAULT_EPS,
                         help="Laplacian grounding. Keeps a disconnected structure finite and "
                              "caps how bad a bottleneck is allowed to score.")
    parser.add_argument("--osq-report", action="store_true",
                         help="Measure OSq before/after on the sampled graphs at the end of "
                              "training and store it in summary.json.")

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
    parser.add_argument("--hparams-json", type=str, default=None,
                         help="Path to a tune.py best_params.json. Its values are applied for any "
                              "hyperparameter still at its argparse default (explicit CLI flags win).")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.hparams_json:
        with open(args.hparams_json) as f:
            best_params = json.load(f)["best_params"]
        defaults = vars(build_arg_parser().parse_args([]))
        for key, value in best_params.items():
            if getattr(args, key) == defaults.get(key):
                setattr(args, key, value)

    utils.set_seed(args.seed)

    task_module = TASKS[args.task]
    data, input_dim, num_classes = task_module.load_dataset(args)

    logger = ExperimentLogger(args.results_dir, args, parser) if args.results_dir else None

    model_kwargs = dict(
        input_dim=input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        motif_hidden_dim=args.motif_hidden_dim, motif_out_dim=args.motif_out_dim,
        num_classes=num_classes,
        s1=args.encoder, s2=args.proposal, s3=args.cell_encoder,
        s4=args.selector_type, s5=args.message_passing,
        top_k=args.top_k, selector_tau=args.selector_tau, selector_hard=not args.selector_soft,
        include_original_edges=not args.no_original_edges,
    )
    model = DMDModel(**model_kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.osq_weight > 0 and args.sparsity_weight == 0:
        print("WARNING: an OSq term with no sparsity counterweight is minimized by the "
              "complete graph. Set --sparsity-weight > 0.")

    osq_fn = get_proxy(args.osq_proxy, hutch_k=args.hutch_k, cg_tol=args.cg_tol,
                       cg_maxiter=args.cg_maxiter, eps=args.osq_eps)
    reinforce_module = model.selector if args.selector_type == "reinforce" else None
    criterion = DMDLoss(sparsity_weight=args.sparsity_weight, osq_weight=args.osq_weight,
                        osq_fn=osq_fn, reinforce_module=reinforce_module)

    best_val_acc = 0.0
    test_acc_at_best_val = 0.0
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = task_module.train_step(model, data, optimizer, criterion)
        val_metrics = task_module.eval_step(model, data, criterion, data.val_mask)
        test_metrics = task_module.eval_step(model, data, criterion, data.test_mask)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            test_acc_at_best_val = test_metrics["accuracy"]
            best_state_dict = copy.deepcopy(model.state_dict())

        if logger is not None:
            logger.log_epoch(epoch, train_metrics, val_metrics, test_metrics)

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_metrics['loss']:.4f} "
                  f"(Task: {train_metrics['task']:.4f}, Sparse: {train_metrics['sparsity']:.4f}) | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | Test Acc: {test_metrics['accuracy']:.4f}")

    result = {"best_val_acc": best_val_acc, "test_acc_at_best_val": test_acc_at_best_val}

    if logger is not None:
        samples = task_module.collect_graph_samples(model, data, args.save_graph_samples)
        if args.osq_report and samples:
            result.update(before_after_report(samples))
        logger.save_summary(result)
        if best_state_dict is not None:
            logger.save_checkpoint(best_state_dict, model_kwargs)
        logger.save_graph_samples(samples)
        logger.close()
        print(f"\nResults saved to {logger.exp_dir}")

    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc @ Best Val: {test_acc_at_best_val:.4f}")
    return result


if __name__ == "__main__":
    main()
