# Analysis-time oversquashing (OSq) MEASUREMENT.
#
# Everything here is meant to be *reported*, not optimized: the functions are
# exact (dense eigendecompositions, shortest-path betweenness) and carry no
# differentiability requirement. The train-time, differentiable counterparts
# live in tools/osq_proxies.py. Keeping the two apart is deliberate: the whole
# claim of this phase is "we optimize a proxy and the measured quantity moves",
# which is only auditable if the measured quantity never goes through the
# optimizer's code path.
#
# Convention shared with the rest of the repo: `edge_index` is [2, E] with both
# directions present, `edge_weight` is [E] and non-negative. The Laplacian used
# throughout is the *symmetrized* combinatorial one,
#     L = D - (A + A^T) / 2,
# so a symmetric edge list gives the usual L = D - A, and an accidentally
# one-sided edge list still yields a symmetric PSD operator instead of silently
# producing complex spectra.

from typing import Iterable, Optional, Sequence

import networkx as nx
import torch

# Kernel cut-off for the symmetric eigendecompositions below, expressed as a
# multiple of `n * machine_eps * lambda_max`. A tolerance tied to the dtype (and
# not a hard-coded 1e-8) matters here: in float32 the exact zero eigenvalue of a
# Laplacian typically comes back around 1e-7, so a tighter cut-off would treat
# the kernel direction as a genuine tiny eigenvalue and make lambda_2 read ~0 on
# a perfectly connected graph - and 1/lambda blow up in the resistance sums.
_EIG_TOL_FACTOR = 4.0


def dense_adjacency(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                    num_nodes: int) -> torch.Tensor:
    """
    [num_nodes, num_nodes] symmetric weighted adjacency, zero diagonal.

    Duplicate entries are summed then symmetrized by (A + A^T) / 2, which is
    the identity for the both-directions-present edge lists this repo produces.
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.get_default_dtype(),
                                 device=edge_index.device)
    A = torch.zeros(num_nodes, num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    A.index_put_((edge_index[0], edge_index[1]), edge_weight, accumulate=True)
    A = 0.5 * (A + A.t())
    A.fill_diagonal_(0.0)
    return A


def dense_laplacian(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                    num_nodes: int) -> torch.Tensor:
    """Combinatorial Laplacian L = D - A of the symmetrized weighted graph."""
    A = dense_adjacency(edge_index, edge_weight, num_nodes)
    return torch.diag(A.sum(dim=1)) - A


def normalized_laplacian(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                         num_nodes: int) -> torch.Tensor:
    """
    L_norm = I - D^-1/2 A D^-1/2 on the support of D. Isolated nodes keep a
    zero row/column rather than producing a division by zero; they then sit at
    eigenvalue 0 and are counted as their own component, which is the correct
    reading (an isolated node is disconnected).
    """
    A = dense_adjacency(edge_index, edge_weight, num_nodes)
    deg = A.sum(dim=1)
    inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5), torch.zeros_like(deg))
    A_norm = inv_sqrt.unsqueeze(1) * A * inv_sqrt.unsqueeze(0)
    eye = torch.diag((deg > 0).to(A.dtype))
    return eye - A_norm


def _pinv_spectrum(L: torch.Tensor):
    """
    (eigenvalues, eigenvectors, kernel_mask) of a symmetric PSD matrix, with the
    numerical kernel identified relative to the largest eigenvalue.
    """
    evals, evecs = torch.linalg.eigh(L)
    scale = max(float(evals[-1].item()), 1.0)
    tol = _EIG_TOL_FACTOR * L.size(0) * torch.finfo(L.dtype).eps * scale
    kernel = evals <= tol
    return evals, evecs, kernel


def connected_components(edge_index: torch.Tensor, num_nodes: int) -> list:
    """List of node-id lists, one per connected component (isolated nodes included)."""
    graph = to_networkx(edge_index, None, num_nodes)
    return [sorted(component) for component in nx.connected_components(graph)]


def to_networkx(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                num_nodes: int) -> nx.Graph:
    """
    Undirected nx.Graph with a `weight` attribute per edge. Both directions of
    the same pair collapse onto one nx edge; the larger weight wins, matching
    the `reduce="max"` coalescing the model uses when it unions the original
    1-skeleton with the accepted cells.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    weights = ([1.0] * len(src)) if edge_weight is None else edge_weight.detach().tolist()
    for u, v, w in zip(src, dst, weights):
        if u == v:
            continue
        existing = graph.get_edge_data(u, v)
        if existing is None or w > existing["weight"]:
            graph.add_edge(u, v, weight=float(w))
    return graph


def effective_resistance(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                         num_nodes: int, pairs: Sequence[Sequence[int]]) -> torch.Tensor:
    """
    Exact R_eff(u, v) = (e_u - e_v)^T L^+ (e_u - e_v) for the requested pairs.

    Returns +inf for pairs sitting in different connected components, which is
    the honest value (no current can flow) - do not silently replace it by a
    finite number when aggregating.
    """
    L = dense_laplacian(edge_index, edge_weight, num_nodes)
    evals, evecs, kernel = _pinv_spectrum(L)
    inv = torch.where(kernel, torch.zeros_like(evals), 1.0 / evals.clamp(min=1e-30))

    component_of = torch.empty(num_nodes, dtype=torch.long)
    for cid, component in enumerate(connected_components(edge_index, num_nodes)):
        component_of[torch.tensor(component, dtype=torch.long)] = cid

    out = torch.empty(len(pairs), dtype=L.dtype)
    for idx, (u, v) in enumerate(pairs):
        if component_of[u] != component_of[v]:
            out[idx] = float("inf")
            continue
        diff = evecs[u] - evecs[v]
        out[idx] = (diff * diff * inv).sum()
    return out


def resistance_summary(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                       num_nodes: int) -> dict:
    """
    Effective-resistance summary of one graph, computed per connected component.

    Returns `r_tot` (Kirchhoff index, sum of R_eff over all intra-component
    pairs) and `r_bar` (its mean over those pairs). Only `r_bar` should ever be
    compared across graphs: a lifting changes the number of nodes carrying the
    relational structure, so a total would move for a reason that has nothing
    to do with oversquashing, and inside a batch the biggest graph would
    dominate any aggregate.
    """
    L = dense_laplacian(edge_index, edge_weight, num_nodes)
    components = connected_components(edge_index, num_nodes)

    r_tot = 0.0
    num_pairs = 0
    for component in components:
        size = len(component)
        if size < 2:
            continue
        idx = torch.tensor(component, dtype=torch.long)
        evals, _, kernel = _pinv_spectrum(L[idx][:, idx])
        inv = torch.where(kernel, torch.zeros_like(evals), 1.0 / evals.clamp(min=1e-30))
        # Kirchhoff index of a component: R_tot = n * tr(L^+).
        r_tot += float(size * inv.sum().item())
        num_pairs += size * (size - 1) // 2

    r_bar = r_tot / num_pairs if num_pairs else 0.0
    return {"r_tot": r_tot, "r_bar": r_bar,
            "num_components": len(components), "num_pairs": num_pairs}


def spectral_gap(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                 num_nodes: int) -> float:
    """
    lambda_2 of the normalized Laplacian: the smallest eigenvalue outside the
    kernel. Zero on a disconnected graph (the kernel then has dimension > 1),
    which is exactly the reading we want - a disconnected structure is the
    worst possible bottleneck.
    """
    L_norm = normalized_laplacian(edge_index, edge_weight, num_nodes)
    evals, _, kernel = _pinv_spectrum(L_norm)
    num_kernel = int(kernel.sum().item())
    if num_kernel > 1 or num_kernel >= num_nodes:
        return 0.0
    return float(evals[num_kernel].item())


def edge_forman_curvature(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                          num_nodes: int) -> torch.Tensor:
    """
    Per-edge curvature, one value per column of `edge_index`:

        EFC(i, j) = 4 - d_i - d_j + 3 * T_ij + 2 * F_ij

    with `d` the weighted degree, `T_ij = (A^2)_ij` the (weighted) triangle
    support of the edge and `F_ij` its quadrangle support, obtained from
    `(A^3)_ij` by removing the walks that fold back on the edge itself
    (i -> j -> b -> j, i -> a -> i -> j, and the i -> j -> i -> j overlap
    counted twice by those two corrections).

    Deliberate deviation from the exact combinatorial definition: 4-walks whose
    quadrangle carries a chord are NOT removed, because doing so costs a per-edge
    n^2 scan while the version above is a degree-3 polynomial in the entries of
    A - three matmuls, smooth, and autograd-able with no custom backward. The
    consequence is an over-count of F on dense neighbourhoods, i.e. a curvature
    that is optimistic (less negative) exactly where the graph is already well
    connected; it does not affect the sign at a bottleneck, which is what the
    quantity is used for.

    This function is differentiable w.r.t. `edge_weight` and is therefore also
    the numerical core of the `efc` / `cf_bc_efc` proxies - measurement and
    proxy share the *formula*, not the surrounding objective.
    """
    A = dense_adjacency(edge_index, edge_weight, num_nodes)
    deg = A.sum(dim=1)
    A2 = A @ A
    A3 = A2 @ A
    self_return = (A * A).sum(dim=1)  # sum_k A_ik^2, the i -> k -> i mass

    src, dst = edge_index[0], edge_index[1]
    a = A[src, dst]
    triangles = A2[src, dst]
    quadrangles = A3[src, dst] - a * (self_return[src] + self_return[dst]) + a.pow(3)

    return 4.0 - deg[src] - deg[dst] + 3.0 * triangles + 2.0 * quadrangles


def curvature_summary(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                      num_nodes: int) -> dict:
    """Distributional read of EFC over the edges of one graph."""
    curv = edge_forman_curvature(edge_index, edge_weight, num_nodes)
    if curv.numel() == 0:
        return {"efc_mean": 0.0, "efc_std": 0.0, "efc_min": 0.0, "efc_neg_frac": 0.0}
    return {
        "efc_mean": float(curv.mean().item()),
        "efc_std": float(curv.std(unbiased=False).item()),
        "efc_min": float(curv.min().item()),
        "efc_neg_frac": float((curv < 0).to(curv.dtype).mean().item()),
    }


def weighted_curvature(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                       num_nodes: int, use_weights_as_distance: bool = False) -> dict:
    """
    wc  = sum_e bc(e) * curv(e)
    nwc = sum_e bc(e) * curv(e) restricted to negatively curved edges

    `bc` is shortest-path edge betweenness. Weighting curvature by betweenness
    is what makes the number comparable between a graph and its lifted version:
    a lifting adds relational mass, so an unweighted curvature average silently
    changes meaning, while betweenness re-weights each edge by how much traffic
    it actually has to carry.

    Measurement only. Shortest-path betweenness is piecewise-constant in the
    edge weights, so its gradient is zero almost everywhere and undefined where
    the argmin path switches - this quantity must never be used as a loss. The
    differentiable stand-in (current-flow betweenness) lives in the proxies.

    `use_weights_as_distance=False` (default) computes hop-count betweenness,
    the usual convention; True uses 1/w as a length so that a strongly accepted
    cell counts as a short edge.
    """
    graph = to_networkx(edge_index, edge_weight, num_nodes)
    if graph.number_of_edges() == 0:
        return {"wc": 0.0, "nwc": 0.0}

    if use_weights_as_distance:
        for _, _, data in graph.edges(data=True):
            data["length"] = 1.0 / max(data["weight"], 1e-12)
        bc = nx.edge_betweenness_centrality(graph, weight="length")
    else:
        bc = nx.edge_betweenness_centrality(graph, weight=None)

    curv = edge_forman_curvature(edge_index, edge_weight, num_nodes)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()

    # One curvature value per undirected pair (both directions carry the same
    # number by construction); betweenness is only defined on the pair.
    seen = {}
    for k, (u, v) in enumerate(zip(src, dst)):
        if u == v:
            continue
        seen[(min(u, v), max(u, v))] = float(curv[k].item())

    wc = 0.0
    nwc = 0.0
    for pair, curvature in seen.items():
        centrality = bc.get(pair, bc.get((pair[1], pair[0]), 0.0))
        wc += centrality * curvature
        if curvature < 0:
            nwc += centrality * curvature
    return {"wc": wc, "nwc": nwc}


def influence_decay(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                    num_nodes: int, num_layers: int = 3) -> float:
    """
    Mean off-diagonal entry of the symmetrically normalized propagation matrix
    raised to `num_layers`.

    This is the structural half of an influence/sensitivity bound: for a
    message-passing network with bounded weights and nonlinearity,
    |d h_v^(r) / d x_u| is bounded by a constant times (A_hat^r)_{vu}. The mean
    of those entries therefore decreases exactly when information has more
    trouble crossing the graph in `num_layers` hops. It is a cheap stand-in for
    the model-dependent influence distance (which needs real forward/backward
    passes and is intentionally not computed here).
    """
    A = dense_adjacency(edge_index, edge_weight, num_nodes)
    deg = A.sum(dim=1)
    inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5), torch.zeros_like(deg))
    A_hat = inv_sqrt.unsqueeze(1) * A * inv_sqrt.unsqueeze(0)

    power = torch.eye(num_nodes, dtype=A.dtype, device=A.device)
    for _ in range(num_layers):
        power = power @ A_hat

    off_diagonal = power - torch.diag(torch.diagonal(power))
    denom = num_nodes * (num_nodes - 1)
    if denom == 0:
        return 0.0
    return float(off_diagonal.sum().item() / denom)


def osq_report(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
               num_nodes: int, batch: Optional[torch.Tensor] = None,
               with_betweenness: bool = True, influence_layers: int = 3) -> dict:
    """
    The row of OSq numbers stored for a run, on one graph or on a whole batch.

    When `batch` is given the structure is block-diagonal, so every quantity is
    computed per graph and averaged over graphs - never over the concatenated
    "graph" of the batch, which would mix disconnected components and make the
    numbers depend on the batch size.

    `with_betweenness=False` skips wc/nwc, the only expensive part
    (all-pairs shortest paths), for use inside inner loops.
    """
    if batch is None:
        graphs = [(edge_index, edge_weight, num_nodes)]
    else:
        graphs = list(_split_batch(edge_index, edge_weight, num_nodes, batch))

    keys = ["r_bar", "lambda2", "efc_mean", "efc_std", "efc_min", "efc_neg_frac",
            "influence_decay"] + (["wc", "nwc"] if with_betweenness else [])
    totals = {k: 0.0 for k in keys}
    if not graphs:
        return totals

    for sub_edge_index, sub_weight, sub_n in graphs:
        report = {}
        report.update(resistance_summary(sub_edge_index, sub_weight, sub_n))
        report["lambda2"] = spectral_gap(sub_edge_index, sub_weight, sub_n)
        report.update(curvature_summary(sub_edge_index, sub_weight, sub_n))
        report["influence_decay"] = influence_decay(sub_edge_index, sub_weight, sub_n,
                                                    num_layers=influence_layers)
        if with_betweenness:
            report.update(weighted_curvature(sub_edge_index, sub_weight, sub_n))
        for k in keys:
            totals[k] += report[k]

    return {k: v / len(graphs) for k, v in totals.items()}


def _split_batch(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor],
                 num_nodes: int, batch: torch.Tensor) -> Iterable:
    """
    Yields (edge_index, edge_weight, num_nodes) per graph, re-indexed to start
    at 0. Assumes PyG's contiguous batching convention (all nodes of a graph
    occupy one contiguous id range), which is what DataLoader produces.
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    counts = torch.bincount(batch, minlength=num_graphs)
    offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)])

    edge_graph = batch[edge_index[0]]
    for g in range(num_graphs):
        mask = edge_graph == g
        sub_edge_index = edge_index[:, mask] - int(offsets[g].item())
        sub_weight = None if edge_weight is None else edge_weight[mask]
        yield sub_edge_index, sub_weight, int(counts[g].item())


def before_after_report(samples: Iterable[dict], with_betweenness: bool = True,
                        influence_layers: int = 3) -> dict:
    """
    The `*_before` / `*_after` block of a stored run, averaged over sample graphs.

    `samples` is the snapshot format the task modules already produce
    (`collect_graph_samples`): per graph, `num_nodes`, the original
    `edge_index`, and the `rewired_edge_index` / `rewired_edge_weight` the model
    produced for it.

    Measuring both sides on the same graphs is the entire evidential burden of
    this phase: without a before and an after there is no way to claim that the
    lifting reduced oversquashing rather than that the datasets differ. wc/nwc
    are reported on the rewired structure only - betweenness re-weights the
    curvature by how much traffic each edge carries, which is exactly the
    comparison the "after" side is for.
    """
    keys_before = ["r_bar", "lambda2"]
    accumulated = {f"{k}_before": 0.0 for k in keys_before}
    accumulated.update({f"{k}_after": 0.0 for k in keys_before})
    accumulated.update({"efc_mean_after": 0.0, "efc_neg_frac_after": 0.0,
                        "influence_decay_before": 0.0, "influence_decay_after": 0.0,
                        "wc": 0.0, "nwc": 0.0})

    count = 0
    for sample in samples:
        num_nodes = int(sample["num_nodes"])
        original = torch.as_tensor(sample["edge_index"], dtype=torch.long).view(2, -1)
        rewired = torch.as_tensor(sample["rewired_edge_index"], dtype=torch.long).view(2, -1)
        weights = torch.as_tensor(sample["rewired_edge_weight"], dtype=torch.get_default_dtype())

        accumulated["r_bar_before"] += resistance_summary(original, None, num_nodes)["r_bar"]
        accumulated["r_bar_after"] += resistance_summary(rewired, weights, num_nodes)["r_bar"]
        accumulated["lambda2_before"] += spectral_gap(original, None, num_nodes)
        accumulated["lambda2_after"] += spectral_gap(rewired, weights, num_nodes)
        accumulated["influence_decay_before"] += influence_decay(original, None, num_nodes,
                                                                 num_layers=influence_layers)
        accumulated["influence_decay_after"] += influence_decay(rewired, weights, num_nodes,
                                                                num_layers=influence_layers)

        curvature = curvature_summary(rewired, weights, num_nodes)
        accumulated["efc_mean_after"] += curvature["efc_mean"]
        accumulated["efc_neg_frac_after"] += curvature["efc_neg_frac"]

        if with_betweenness:
            wc = weighted_curvature(rewired, weights, num_nodes)
            accumulated["wc"] += wc["wc"]
            accumulated["nwc"] += wc["nwc"]
        count += 1

    if count == 0:
        return accumulated
    return {k: v / count for k, v in accumulated.items()}
