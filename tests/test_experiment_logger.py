import json
import sys
from pathlib import Path

import train
from tools.experiment_logger import ExperimentLogger, build_experiment_name

RESULTS_ANALYZE_PATH = Path(__file__).resolve().parent.parent / "results"


def test_experiment_name_includes_dataset_and_only_non_default_hparams():
    parser = train.build_arg_parser()
    default_args = parser.parse_args([])
    name = build_experiment_name(default_args, parser)
    assert name == "node_classification_toy_seed=0"

    custom_args = parser.parse_args(["--encoder", "gin", "--hidden-dim", "128", "--seed", "3"])
    name = build_experiment_name(custom_args, parser)
    assert "encoder=gin" in name
    assert "hidden_dim=128" in name
    assert name.endswith("seed=3")
    assert "top_k" not in name  # left at default, so not part of the name


def test_experiment_logger_writes_incrementally(tmp_path):
    parser = train.build_arg_parser()
    args = parser.parse_args(["--results-dir", str(tmp_path)])
    logger = ExperimentLogger(str(tmp_path), args, parser)

    assert (logger.exp_dir / "config.json").exists()
    with open(logger.exp_dir / "config.json") as f:
        assert json.load(f)["task"] == "node_classification"

    logger.log_epoch(1, {"loss": 1.0, "task": 0.9, "sparsity": 0.1, "osq": 0.0, "accuracy": 0.5},
                      {"loss": 1.1, "task": 1.0, "accuracy": 0.4},
                      {"loss": 1.2, "task": 1.1, "accuracy": 0.3})

    with open(logger.exp_dir / "history.jsonl") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    assert lines[0]["epoch"] == 1
    assert lines[0]["train_loss"] == 1.0
    assert lines[0]["val_accuracy"] == 0.4
    assert lines[0]["test_accuracy"] == 0.3

    logger.save_summary({"best_val_acc": 0.4})
    logger.save_graph_samples([{"graph_id": 0, "num_nodes": 2, "label": None,
                                 "edge_index": [[0], [1]],
                                 "rewired_edge_index": [[0], [1]],
                                 "rewired_edge_weight": [1.0]}])
    logger.close()

    assert json.loads((logger.exp_dir / "summary.json").read_text())["best_val_acc"] == 0.4
    assert len(json.loads((logger.exp_dir / "graph_samples.json").read_text())) == 1


def test_train_main_saves_results_and_analyze_script_reads_them(tmp_path):
    result_root = tmp_path / "runs"
    train.main([
        "--epochs", "3",
        "--num-cliques", "3",
        "--clique-size", "5",
        "--feature-dim", "8",
        "--hidden-dim", "8",
        "--latent-dim", "6",
        "--motif-hidden-dim", "6",
        "--motif-out-dim", "4",
        "--top-k", "2",
        "--log-every", "100",
        "--results-dir", str(result_root),
        "--save-graph-samples", "1",
    ])

    exp_dirs = list(result_root.glob("*/*"))
    assert len(exp_dirs) == 1
    exp_dir = exp_dirs[0]

    for fname in ("config.json", "history.jsonl", "summary.json", "graph_samples.json"):
        assert (exp_dir / fname).exists()

    history_lines = (exp_dir / "history.jsonl").read_text().strip().splitlines()
    assert len(history_lines) == 3  # one line per epoch

    sys.path.insert(0, str(RESULTS_ANALYZE_PATH))
    try:
        import analyze_results
        config, history, summary, samples = analyze_results.load_run(exp_dir)
        out_dir = exp_dir / "plots"
        out_dir.mkdir()
        analyze_results.plot_loss_components(history, config, out_dir / "loss_components.png")
        analyze_results.plot_task_loss_splits(history, config, out_dir / "loss_train_val_test.png")
        analyze_results.plot_performance(history, summary, config, out_dir / "performance.png")
        analyze_results.plot_adjacency_heatmaps(samples, config, out_dir / "adjacency_heatmaps.png")
    finally:
        sys.path.remove(str(RESULTS_ANALYZE_PATH))

    for fname in ("loss_components.png", "loss_train_val_test.png", "performance.png", "adjacency_heatmaps.png"):
        assert (out_dir / fname).exists()
        assert (out_dir / fname).stat().st_size > 0
