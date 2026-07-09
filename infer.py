# Runs inference from a checkpoint saved by train.py: model_best.pt holds the
# weights from the epoch with the best validation accuracy (see
# best_state_dict in train.main), plus the exact DMDModel constructor kwargs
# needed to rebuild that architecture. config.json (saved alongside it)
# carries the CLI args needed to rebuild the same data split. Together these
# mean no hyperparameters need to be repeated on the command line - just
# point at the run directory.
#
# Usage:
#   python infer.py results/runs/<experiment>/<run_id>
#   python infer.py results/runs/<experiment>/<run_id> --split test --dump-predictions preds.json

import argparse
import json
from pathlib import Path

import torch

import utils
from models.dmd_model import DMDModel
from tools.losses import DMDLoss
from train import TASKS


def _split_data(data, task: str, split: str):
    if split == "train" and task == "graph_classification":
        return data.train_loader  # GraphSplits has no train_mask alias, unlike val/test
    return getattr(data, f"{split}_mask")


def load_run(run_dir: Path):
    with open(run_dir / "config.json") as f:
        run_args = argparse.Namespace(**json.load(f))

    ckpt_path = run_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} - this run predates checkpoint saving, "
            "or was launched with --results-dir \"\"."
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    utils.set_seed(run_args.seed)
    task_module = TASKS[run_args.task]
    data, _, _ = task_module.load_dataset(run_args)

    model = DMDModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, data, run_args, task_module


@torch.no_grad()
def predict(model, data, task: str, split: str):
    """Per-example (prediction, label) pairs for `split` - eval_step only returns aggregate metrics."""
    split_data = _split_data(data, task, split)

    if task == "graph_classification":
        preds, labels = [], []
        for batch in split_data:
            logits, _ = model(batch.x, batch.edge_index, batch=batch.batch)
            preds += logits.argmax(dim=-1).tolist()
            labels += batch.y.tolist()
        return preds, labels

    logits, _ = model(data.x, data.edge_index)
    preds = logits[split_data].argmax(dim=-1).tolist()
    labels = data.y[split_data].tolist()
    return preds, labels


def main():
    parser = argparse.ArgumentParser(description="Run inference with a pretrained DMD checkpoint.")
    parser.add_argument("run_dir", type=str, help="e.g. results/runs/<experiment>/<run_id>")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--dump-predictions", type=str, default=None,
                         help="Optional path to write per-example predictions/labels as JSON.")
    args = parser.parse_args()

    model, data, run_args, task_module = load_run(Path(args.run_dir))

    criterion = DMDLoss(sparsity_weight=run_args.sparsity_weight)
    split_data = _split_data(data, run_args.task, args.split)
    metrics = task_module.eval_step(model, data, criterion, split_data)
    print(f"{args.split} accuracy: {metrics['accuracy']:.4f} (loss: {metrics['loss']:.4f})")

    if args.dump_predictions:
        preds, labels = predict(model, data, run_args.task, args.split)
        with open(args.dump_predictions, "w") as f:
            json.dump({"split": args.split, "predictions": preds, "labels": labels}, f, indent=2)
        print(f"Saved {len(preds)} predictions to {args.dump_predictions}")


if __name__ == "__main__":
    main()
