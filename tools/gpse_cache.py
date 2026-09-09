# GPSE (Cantürk et al. 2024) integration for Stage 1's `gpse` rich brick.
#
# GPSE is frozen and its inference is expensive (a 20-layer, 512-hidden GatedGCN
# with a virtual node) relative to everything else in this pipeline, and it does
# not depend on the downstream task at all - so it is computed ONCE per dataset
# and cached, not recomputed every forward pass across a 26-config x 2-gamma x
# 6-dataset x 3-seed grid, where it would otherwise be recomputed hundreds of
# times.
#
# Wraps PyG's own `torch_geometric.nn.models.gpse` module
# (`GPSE.from_pretrained` + `precompute_GPSE`/`gpse_process_batch`) - the
# pretrained checkpoint and the official precomputation loop are wrapped, not
# reimplemented here.
#
# Fallback contract: if the pretrained checkpoint cannot be fetched (Kaggle
# sessions may run without internet), every function here falls back to
# `pse_explicit` (LapPE + RWSE via PyG transforms) rather than raising, and
# reports which one actually ran - a caller that silently ends up on a
# different encoder without recording it is exactly the failure mode to avoid.

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# GPSE's default dim_out: the concatenation of its 6 reconstructed PSEs
# (LapPE, eigenvalues, ElstaticPE, RWSE, HKdiagSE, CycleSE). The pse_explicit
# fallback is padded/truncated to the same width so GPSEEncoder never has to
# know which one actually produced a given `pestat_GPSE` tensor.
GPSE_ENCODING_DIM = 51


def compute_explicit_pe(edge_index: torch.Tensor, num_nodes: int,
                         target_dim: int = GPSE_ENCODING_DIM) -> torch.Tensor:
    """
    LapPE + RWSE for a single graph via PyG's own transforms - the mandatory
    fallback when the GPSE checkpoint is unavailable. Padded with zeros or
    truncated to `target_dim` so it is a
    drop-in substitute for a real GPSE encoding wherever one is expected.
    """
    from torch_geometric.data import Data
    from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

    if num_nodes == 0:
        return torch.zeros(0, target_dim)

    lap_k = max(1, min(target_dim // 2, num_nodes - 1)) if num_nodes > 1 else 1
    rw_len = max(1, target_dim - lap_k)

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    try:
        data = AddLaplacianEigenvectorPE(k=lap_k, attr_name="lap_pe", is_undirected=True)(data)
        lap_pe = data.lap_pe
    except Exception:  # noqa: BLE001 - a degenerate graph (e.g. no edges) must not kill the fallback
        logger.warning("AddLaplacianEigenvectorPE failed on a %d-node graph; zero-filling.",
                        num_nodes, exc_info=True)
        lap_pe = torch.zeros(num_nodes, lap_k)

    try:
        data = AddRandomWalkPE(walk_length=rw_len, attr_name="rw_pe")(data)
        rw_pe = data.rw_pe
    except Exception:  # noqa: BLE001
        logger.warning("AddRandomWalkPE failed on a %d-node graph; zero-filling.",
                        num_nodes, exc_info=True)
        rw_pe = torch.zeros(num_nodes, rw_len)

    pe = torch.cat([lap_pe, rw_pe], dim=-1)
    if pe.size(-1) < target_dim:
        pe = torch.nn.functional.pad(pe, (0, target_dim - pe.size(-1)))
    return pe[:, :target_dim]


def load_pretrained_gpse(pretrained_name: str = "molpcba", weights_root: str = "GPSE_pretrained"):
    """
    Returns a frozen, eval-mode GPSE model, or None if the checkpoint could
    not be fetched (no internet, or a version mismatch). Never raises - the
    caller is expected to fall back to `compute_explicit_pe` when this
    returns None, and to record that it did.
    """
    try:
        from torch_geometric.nn.models import GPSE

        model = GPSE.from_pretrained(pretrained_name, root=weights_root)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    except Exception:  # noqa: BLE001 - fetch failure must not kill the grid
        logger.warning("GPSE pretrained checkpoint '%s' unavailable; falling back to pse_explicit.",
                        pretrained_name, exc_info=True)
        return None


@torch.no_grad()
def compute_gpse_live(model, edge_index: torch.Tensor, num_nodes: int,
                       node_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Runs the frozen GPSE model on a single (possibly batched) graph given as
    plain tensors, for callers that only have raw `(edge_index, batch)` and no
    pre-cached `pestat_GPSE` (the fallback path inside `GPSEEncoder.forward`;
    the primary/fast path is `attach_gpse_cache` below, run once per dataset).

    Adds a virtual node per graph (`torch_geometric.transforms.VirtualNode`,
    matching GPSE's pretraining setup) via PyG's official
    `gpse_process_batch`, then drops the virtual-node rows before returning -
    callers never see them.
    """
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn.models.gpse import gpse_process_batch
    from torch_geometric.transforms import VirtualNode

    device = edge_index.device
    if node_batch is None:
        node_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() else 0

    graphs = []
    for g in range(num_graphs):
        mask = node_batch == g
        local_nodes = torch.nonzero(mask, as_tuple=True)[0]
        remap = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        remap[local_nodes] = torch.arange(local_nodes.numel(), device=device)
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        local_edges = remap[edge_index[:, edge_mask]]
        graphs.append(VirtualNode()(Data(edge_index=local_edges.cpu(), num_nodes=local_nodes.numel())))

    vn_batch = Batch.from_data_list(graphs)
    pe, ptr = gpse_process_batch(model, vn_batch, rand_type="NormalSE", use_vn=True)

    keep = torch.ones(pe.size(0), dtype=torch.bool)
    keep[ptr[1:] - 1] = False  # the virtual node is appended last within each graph's block
    return pe[keep].to(device)


def attach_gpse_cache(dataset, cache_path: Optional[str] = None,
                       pretrained_name: str = "molpcba",
                       weights_root: str = "GPSE_pretrained") -> str:
    """
    Attaches `data.pestat_GPSE` [num_nodes, GPSE_ENCODING_DIM] to every graph
    in `dataset`, in place - the required one-per-dataset precomputation.
    Reuses a cache on disk if `cache_path` already exists; otherwise computes
    (real GPSE if the checkpoint fetches, `pse_explicit` otherwise) and writes
    one if `cache_path` is given.

    Returns "gpse" or "pse_explicit" - whichever actually ran. Store this as
    `s1_encoder_actual` in the results row: a grid that silently degrades to a
    different encoder without recording it is worse than a failed grid.
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        for data, pe in zip(dataset, cached["pestat_GPSE"]):
            data.pestat_GPSE = pe
        return cached["source"]

    model = load_pretrained_gpse(pretrained_name, weights_root)
    if model is not None:
        # PyG's `precompute_GPSE` is written against `InMemoryDataset`: it reads
        # `dataset.data` / `dataset.slices` directly. `tools/synthetic.py`'s
        # SyntheticGraphDataset is deliberately NOT one (its docstring says so -
        # the graphs are generated in seconds and never cached, so the collation
        # machinery would only add failure modes), so that call raises
        # `AttributeError: 'SyntheticGraphDataset' object has no attribute 'data'`.
        #
        # This never surfaced before because the synthetic arm was missing from
        # the grid that would have exercised it - it is the falsification core,
        # and s1='gpse' on it had simply never run.
        #
        # For those datasets we use this module's own generic path
        # (`compute_gpse_live`), the same one `GPSEEncoder.forward` falls back to,
        # which produces the identical encoding from raw tensors. One graph at a
        # time rather than batched: this runs once per dataset and is cached to
        # disk afterwards, so the batching complexity would buy seconds once.
        if hasattr(dataset, "data") and hasattr(dataset, "slices"):
            from torch_geometric.nn.models.gpse import precompute_GPSE

            if not hasattr(dataset, "transform"):
                dataset.transform = None
            precompute_GPSE(model, dataset)
        else:
            for data in dataset:
                data.pestat_GPSE = compute_gpse_live(model, data.edge_index, data.num_nodes)
        source = "gpse"
    else:
        for data in dataset:
            data.pestat_GPSE = compute_explicit_pe(data.edge_index, data.num_nodes)
        source = "pse_explicit"

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"source": source, "pestat_GPSE": [d.pestat_GPSE for d in dataset]}, cache_path)
    return source
