# Train-time oversquashing (OSq) PROXIES: differentiable, cheap, and swappable
# by config string.
#
# Why proxies at all. The quantities that actually define oversquashing are not
# usable as objectives: effective resistance needs L^+ (O(n^3) per step),
# influence distance needs full forward+backward passes, and the structure the
# model produces is discrete (a cell is accepted or not). A proxy has to be
# (a) correlated with oversquashing, (b) cheap, (c) differentiable w.r.t. the
# parameters that generate the structure.
#
# Relaxation contract (the single most important rule in this file): a proxy
# scores the SOFT structure. It reads `structure["rewired_edge_weight"]`, i.e.
# the acceptance weights alpha in [0, 1] used as continuous edge weights, and
# never a thresholded or rounded adjacency. The weighted Laplacian L(alpha) is
# continuous in alpha, which is what makes any of this differentiable at all.
# Hard sampling happens at inference, or through the straight-through estimator
# already implemented in Stage 4 - not here.
#
# Every proxy has the signature
#     proxy(structure: dict, **cfg) -> torch.Tensor   # scalar, to MINIMIZE
# and `get_proxy(name, **cfg)` binds the config so the loss only ever sees a
# one-argument callable. The registry exists so several proxies can be compared
# under one experiment grid: which proxy wins is itself a result.
#
# Degenerate optimum, stated once and for all: every proxy here is minimized by
# the complete graph. They are only meaningful next to the sparsity term. Never
# run one with sparsity_weight = 0.

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from tools.osq_metrics import edge_forman_curvature

# Defaults for the Hutchinson + conjugate-gradient estimator. `k` around 10-30
# is plenty: the estimator noise behaves like minibatch noise and does not break
# SGD. `eps` grounds the Laplacian (see `_hutchinson_solves`).
DEFAULT_HUTCH_K = 16
DEFAULT_CG_TOL = 1e-5
DEFAULT_CG_MAXITER = 100
DEFAULT_EPS = 1e-4


# --------------------------------------------------------------------------
# structure unpacking
# --------------------------------------------------------------------------

def _fields(structure: dict) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
    """
    (edge_index, edge_weight, num_nodes, node_batch, num_graphs) from the dict
    `DMDModel.forward` returns.

    `num_nodes` and `batch` are read when present and reconstructed otherwise,
    so a proxy still works on a hand-built structure dict (tests) or on an older
    caller that only passes the four original keys.
    """
    edge_index = structure["rewired_edge_index"]
    edge_weight = structure["rewired_edge_weight"]

    num_nodes = structure.get("num_nodes")
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0

    node_batch = structure.get("batch")
    if node_batch is None:
        node_batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        num_graphs = 1
    else:
        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    return edge_index, edge_weight, int(num_nodes), node_batch, num_graphs


def _graph_sizes(node_batch: torch.Tensor, num_graphs: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.bincount(node_batch, minlength=num_graphs).to(dtype)


# --------------------------------------------------------------------------
# weighted Laplacian as an operator (never materialized)
# --------------------------------------------------------------------------

def _symmetric_degree(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
    """
    Degree of the symmetrized graph. Each directed entry contributes half its
    weight to both endpoints, so that
        L(w) = sum over directed entries e=(i,j) of  0.5 * w_e * b_e b_e^T,
    with b_e = e_i - e_j. On the both-directions edge lists this repo produces,
    the two halves of a pair add back up to the usual w * b b^T; on a one-sided
    list the operator stays symmetric and PSD instead of silently breaking CG.
    That factor 0.5 is also what makes the per-edge gradient below symmetric.
    """
    deg = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
    deg.index_add_(0, edge_index[0], 0.5 * edge_weight)
    deg.index_add_(0, edge_index[1], 0.5 * edge_weight)
    return deg


def _laplacian_matvec(V: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                      deg: torch.Tensor, eps: float) -> torch.Tensor:
    """
    (L + eps*I) V for V of shape [num_nodes, k], in O(nnz) - only scatter/gather,
    no dense matrix anywhere.
    """
    src, dst = edge_index[0], edge_index[1]
    w = (0.5 * edge_weight).unsqueeze(-1)
    AV = torch.zeros_like(V)
    AV.index_add_(0, src, w * V[dst])
    AV.index_add_(0, dst, w * V[src])
    return (deg.unsqueeze(-1) + eps) * V - AV


def _project_per_graph(V: torch.Tensor, node_batch: torch.Tensor, num_graphs: int,
                       sizes: torch.Tensor) -> torch.Tensor:
    """
    Removes the Laplacian kernel: the all-ones vector of each graph. Subtracting
    the per-graph mean makes every column orthogonal to that kernel, and L maps
    the orthogonal complement into itself, so CG started at 0 stays there.
    """
    sums = torch.zeros(num_graphs, V.size(1), dtype=V.dtype, device=V.device)
    sums.index_add_(0, node_batch, V)
    means = sums / sizes.clamp(min=1.0).unsqueeze(-1)
    return V - means[node_batch]


def _cg(B: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
        deg: torch.Tensor, eps: float, tol: float, maxiter: int) -> Tuple[torch.Tensor, float]:
    """
    Jacobi-preconditioned conjugate gradient solving (L + eps*I) X = B for all
    `k` right-hand sides at once. Returns (X, worst relative residual).

    Conditioning warning, kept here on purpose: a severe bottleneck means a tiny
    lambda_2, i.e. a huge condition number, i.e. CG struggling exactly on the
    graphs we care about. The Jacobi preconditioner is the cheap minimum, the
    iteration count is capped, and the residual is returned so callers can log
    it - a capped solve is a biased estimate but still a usable proxy.
    """
    X = torch.zeros_like(B)
    R = B.clone()
    precond = 1.0 / (deg + eps).clamp(min=1e-12)
    Zp = precond.unsqueeze(-1) * R
    P = Zp.clone()
    rz = (R * Zp).sum(dim=0)
    b_norm = B.norm(dim=0).clamp(min=1e-30)

    residual = float((R.norm(dim=0) / b_norm).max().item())
    # CG's 2-norm residual is not monotone, so "best so far" is kept explicitly
    # rather than assuming the last iterate is the good one.
    best_X, best_residual = X, residual
    rz_initial = float(rz.max().item())
    floor = torch.finfo(B.dtype).eps ** 2

    for _ in range(maxiter):
        AP = _laplacian_matvec(P, edge_index, edge_weight, deg, eps)
        pAp = (P * AP).sum(dim=0)
        # Per column, never globally. The operator is positive definite, so a
        # non-positive or non-finite curvature means that column is done - it
        # has converged, or its right-hand side was zero to begin with (a
        # Rademacher probe that came out constant projects to exactly zero, and
        # with a few dozen probes that does happen). Freezing that column is
        # correct; aborting the whole solve because of it is not.
        active = torch.isfinite(pAp) & (pAp > 0)
        if not bool(active.any()):
            break

        safe_pAp = torch.where(active, pAp, torch.ones_like(pAp))
        alpha = torch.where(active, rz / safe_pAp, torch.zeros_like(rz))
        X_next = X + alpha * P
        R_next = R - alpha * AP

        # Same idea for round-off blowing up a single column: keep that column's
        # previous iterate instead of letting one NaN spread to the whole batch.
        finite = torch.isfinite(X_next).all(dim=0) & torch.isfinite(R_next).all(dim=0)
        if not bool(finite.all()):
            X_next = torch.where(finite, X_next, X)
            R_next = torch.where(finite, R_next, R)
        X, R = X_next, R_next

        residual = float((R.norm(dim=0) / b_norm).max().item())
        if residual < best_residual:
            best_X, best_residual = X, residual
        if residual < tol:
            break

        Zp = precond.unsqueeze(-1) * R
        rz_new = (R * Zp).sum(dim=0)
        # Round-off floor. Once rz has fallen to the square of the machine
        # epsilon, both rz and pAp are pure round-off, their ratio is noise, and
        # iterating on turns a converged solve into NaN - which would then
        # silently poison the loss. Asking float32 for a 1e-12 tolerance is
        # exactly how one gets there, so stop on the precision floor rather than
        # trusting `tol` alone.
        if float(rz_new.max().item()) <= rz_initial * floor:
            break

        beta = rz_new / rz.clamp(min=1e-30)
        P = Zp + beta * P
        rz = rz_new
    return best_X, best_residual


def _hutchinson_solves(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int,
                       node_batch: torch.Tensor, num_graphs: int, sizes: torch.Tensor,
                       k: int, tol: float, maxiter: int, eps: float,
                       seed: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    The one set of linear solves every resistance-flavoured quantity in this
    file is built from. Returns (Z, X, residual) with X = (L + eps*I)^-1 Z.

    Estimator. For z with i.i.d. zero-mean unit-variance entries (Rademacher
    here), E[z^T A z] = tr(A), so tr(L^+) is estimated by the mean of z^T x with
    x = L^+ z. We never form L^+: x is the solution of L x = z, obtained by CG,
    which only ever needs matrix-vector products - O(|E|) per iteration instead
    of an O(n^3) factorization.

    Grounding. `eps` makes the system positive definite so CG cannot divide by a
    zero eigenvalue and a disconnected structure stays finite. It caps the
    penalty (a bottleneck can only be counted as bad as 1/eps), which is a
    modelling choice, hence a config knob and not a hidden constant. Combined
    with the per-graph projection of z, the estimator targets
    sum_{i >= 2} 1/(lambda_i + eps), the grounded pseudo-inverse trace.

    Physical reading: x is the electric potential when injecting currents z into
    the network, so x_i - x_j is the voltage drop across edge (i, j).

    `seed` forces a fixed probe set. Leave it None in training (fresh noise each
    step, like minibatch noise); set it when a caller needs the estimator to be
    a deterministic function of the weights - a finite-difference gradient check
    is meaningless if the probes are resampled between evaluations.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=edge_weight.device)
        generator.manual_seed(int(seed))

    bits = torch.randint(0, 2, (num_nodes, k), generator=generator,
                         device=edge_weight.device, dtype=torch.int64)
    Z = (bits.to(edge_weight.dtype) * 2.0 - 1.0)
    Z = _project_per_graph(Z, node_batch, num_graphs, sizes)

    deg = _symmetric_degree(edge_index, edge_weight, num_nodes)
    X, residual = _cg(Z, edge_index, edge_weight, deg, eps, tol, maxiter)
    # Exact in theory (L preserves the kernel's orthogonal complement); done for
    # numerical hygiene, since round-off does leak into the constant direction.
    X = _project_per_graph(X, node_batch, num_graphs, sizes)
    return Z, X, residual


# --------------------------------------------------------------------------
# primary proxy: mean effective resistance
# --------------------------------------------------------------------------

class _MeanEffectiveResistance(torch.autograd.Function):
    """
    R_bar with a hand-written backward.

    Forward: `k` CG solves under no_grad. Backward: the closed form below. We do
    NOT backpropagate through the CG iterations - that would be unstable and
    would store every iterate.

    Gradient. With L(w) = sum_e 0.5 * w_e * b_e b_e^T,
        d tr(L^+) / d w_e = -0.5 * tr(L^+ b_e b_e^T L^+) = -0.5 * ||L^+ b_e||^2,
    the pseudo-inverse correction terms vanishing because b_e is orthogonal to
    the all-ones vector inside a connected component. And with the SAME
    x = L^+ z already computed in the forward,
        E_z[(x_i - x_j)^2] = b_e^T (L^+)^2 b_e = ||L^+ b_e||^2,
    so the gradient costs nothing beyond the forward solves. It is also local:
    each edge reads two entries of x, which is what lets it chain naturally back
    to the per-cell acceptance weight through autograd.

    Reading: strengthening an edge reduces total resistance most when that edge
    carries a large voltage drop - i.e. when it is a bottleneck. (Note that
    ||L^+ b_e||^2 is the biharmonic distance, not R_eff(e) = b_e^T L^+ b_e.)
    """

    @staticmethod
    def forward(ctx, edge_weight, edge_index, num_nodes, node_batch, num_graphs,
                k, tol, maxiter, eps, seed):
        with torch.no_grad():
            sizes = _graph_sizes(node_batch, num_graphs, edge_weight.dtype)
            Z, X, residual = _hutchinson_solves(edge_index, edge_weight.detach(), num_nodes,
                                                node_batch, num_graphs, sizes,
                                                k, tol, maxiter, eps, seed)

            # tr(L^+) per graph, then R_bar = R_tot / C(n,2) = 2 tr(L^+) / (n-1).
            # Always the normalized version: inside a batch a large graph would
            # otherwise dominate the gradient, and a lifting changes n, so a
            # total would move for a reason unrelated to oversquashing.
            per_node = (Z * X).sum(dim=1) / k
            trace = torch.zeros(num_graphs, dtype=edge_weight.dtype, device=edge_weight.device)
            trace.index_add_(0, node_batch, per_node)

            scale = torch.where(sizes > 1, 2.0 / (sizes - 1).clamp(min=1.0),
                                torch.zeros_like(sizes))
            counted = (sizes > 1).to(edge_weight.dtype)
            num_counted = counted.sum().clamp(min=1.0)
            value = (scale * trace * counted).sum() / num_counted

            src, dst = edge_index[0], edge_index[1]
            dx = X[src] - X[dst]
            cf = (dx * dx).mean(dim=1)           # E[(x_i - x_j)^2] per directed entry
            edge_graph = node_batch[src]
            coeff = -0.5 * cf * scale[edge_graph] * counted[edge_graph] / num_counted

        ctx.save_for_backward(coeff)
        ctx.residual = residual
        return value

    @staticmethod
    def backward(ctx, grad_output):
        (coeff,) = ctx.saved_tensors
        return (grad_output * coeff,) + (None,) * 9


def r_bar(structure: dict, hutch_k: int = DEFAULT_HUTCH_K, cg_tol: float = DEFAULT_CG_TOL,
          cg_maxiter: int = DEFAULT_CG_MAXITER, eps: float = DEFAULT_EPS,
          seed: Optional[int] = None) -> torch.Tensor:
    """
    Mean effective resistance of the soft rewired structure - the primary proxy.

    Averaged over the graphs of the batch, each one normalized by its own pair
    count, so the term is invariant to batch composition and to how many nodes
    the lifting brought into the relational structure.
    """
    edge_index, edge_weight, num_nodes, node_batch, num_graphs = _fields(structure)
    if edge_index.numel() == 0 or num_graphs == 0:
        return torch.zeros((), dtype=edge_weight.dtype, device=edge_weight.device)
    return _MeanEffectiveResistance.apply(edge_weight, edge_index, num_nodes, node_batch,
                                          num_graphs, hutch_k, cg_tol, cg_maxiter, eps, seed)


def current_flow_betweenness(structure: dict, hutch_k: int = DEFAULT_HUTCH_K,
                             cg_tol: float = DEFAULT_CG_TOL,
                             cg_maxiter: int = DEFAULT_CG_MAXITER,
                             eps: float = DEFAULT_EPS,
                             seed: Optional[int] = None) -> torch.Tensor:
    """
    Per-edge current-flow (Newman/Brandes) betweenness, [E], detached.

    This is the differentiable-friendly replacement for the shortest-path
    betweenness used in wc/nwc at measurement time: the current through an edge
    when injecting random source-sink patterns is a smooth function of L^+, and
    it is *already computed* by the Hutchinson estimator - it is exactly
    E[(x_i - x_j)^2]. One set of solves therefore feeds the global objective
    (R_bar), its gradient, and this global edge weighting.

    Returned detached: it plays the role betweenness plays in wc, a coefficient
    that says how much traffic an edge carries, not a quantity we differentiate
    through. The gradient path through the resistance itself is the `r_bar` arm.
    """
    edge_index, edge_weight, num_nodes, node_batch, num_graphs = _fields(structure)
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=edge_weight.dtype, device=edge_weight.device)
    with torch.no_grad():
        sizes = _graph_sizes(node_batch, num_graphs, edge_weight.dtype)
        _, X, _ = _hutchinson_solves(edge_index, edge_weight.detach(), num_nodes, node_batch,
                                     num_graphs, sizes, hutch_k, cg_tol, cg_maxiter, eps, seed)
        dx = X[edge_index[0]] - X[edge_index[1]]
        return (dx * dx).mean(dim=1)


# --------------------------------------------------------------------------
# guard-rail proxy: spectral gap
# --------------------------------------------------------------------------

def _normalized_laplacian_matvec(V: torch.Tensor, edge_index: torch.Tensor,
                                 edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """L_norm V = V - D^-1/2 A D^-1/2 V, isolated nodes contributing nothing."""
    deg = _symmetric_degree(edge_index, edge_weight, num_nodes)
    support = (deg > 0).to(V.dtype).unsqueeze(-1)
    inv_sqrt = torch.where(deg > 0, deg.clamp(min=1e-12).pow(-0.5), torch.zeros_like(deg))

    U = inv_sqrt.unsqueeze(-1) * V
    src, dst = edge_index[0], edge_index[1]
    w = (0.5 * edge_weight).unsqueeze(-1)
    AU = torch.zeros_like(V)
    AU.index_add_(0, src, w * U[dst])
    AU.index_add_(0, dst, w * U[src])
    return support * V - inv_sqrt.unsqueeze(-1) * AU


def lambda2(structure: dict, num_iters: int = 30, seed: Optional[int] = None,
            **_unused) -> torch.Tensor:
    """
    Guard-rail proxy: minimize -lambda_2 of the normalized Laplacian, i.e.
    push the spectral gap up. lambda_2 controls the global oversquashing bound.

    Differentiability comes from eigenvalue perturbation: d lambda_2 / dL is
    v_2 v_2^T, so evaluating the Rayleigh quotient v^T L_norm(w) v with v held
    fixed (found by a few deflated power iterations under no_grad) gives autograd
    exactly the right gradient, with no eigendecomposition in the graph.

    Secondary on purpose: a single global scalar, blind to local bottlenecks,
    and non-differentiable at eigenvalue crossings (where lambda_2 and lambda_3
    meet, the power iteration also converges slowly and the value is only a
    ceiling on the true lambda_2). Use it as a diagnostic, not as the primary
    objective.
    """
    edge_index, edge_weight, num_nodes, node_batch, num_graphs = _fields(structure)
    if edge_index.numel() == 0 or num_graphs == 0:
        return torch.zeros((), dtype=edge_weight.dtype, device=edge_weight.device)

    sizes = _graph_sizes(node_batch, num_graphs, edge_weight.dtype)

    with torch.no_grad():
        w_det = edge_weight.detach()
        deg = _symmetric_degree(edge_index, w_det, num_nodes)
        # Kernel of the normalized Laplacian: D^1/2 * 1, per graph.
        u0 = deg.clamp(min=0).sqrt().unsqueeze(-1)
        u0 = u0 / _per_graph_norm(u0, node_batch, num_graphs)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=edge_weight.device)
            generator.manual_seed(int(seed))
        V = torch.randn(num_nodes, 1, generator=generator,
                        device=edge_weight.device, dtype=edge_weight.dtype)
        V = _deflate(V, u0, node_batch, num_graphs)
        V = V / _per_graph_norm(V, node_batch, num_graphs)

        for _ in range(num_iters):
            # Power iteration on 2I - L_norm, whose spectrum is [0, 2]: its
            # dominant direction outside the kernel is exactly lambda_2's.
            V = 2.0 * V - _normalized_laplacian_matvec(V, edge_index, w_det, num_nodes)
            V = _deflate(V, u0, node_batch, num_graphs)
            V = V / _per_graph_norm(V, node_batch, num_graphs)

    LV = _normalized_laplacian_matvec(V, edge_index, edge_weight, num_nodes)
    numerator = torch.zeros(num_graphs, dtype=edge_weight.dtype, device=edge_weight.device)
    numerator.index_add_(0, node_batch, (V * LV).sum(dim=1))
    denominator = torch.zeros_like(numerator)
    denominator.index_add_(0, node_batch, (V * V).sum(dim=1))

    counted = (sizes > 1).to(edge_weight.dtype)
    per_graph = numerator / denominator.clamp(min=1e-12)
    return -(per_graph * counted).sum() / counted.sum().clamp(min=1.0)


def _per_graph_norm(V: torch.Tensor, node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    sq = torch.zeros(num_graphs, V.size(1), dtype=V.dtype, device=V.device)
    sq.index_add_(0, node_batch, V * V)
    return sq.clamp(min=1e-24).sqrt()[node_batch]


def _deflate(V: torch.Tensor, u0: torch.Tensor, node_batch: torch.Tensor,
             num_graphs: int) -> torch.Tensor:
    """Removes the per-graph component along the (already normalized) kernel u0."""
    dots = torch.zeros(num_graphs, V.size(1), dtype=V.dtype, device=V.device)
    dots.index_add_(0, node_batch, u0 * V)
    return V - dots[node_batch] * u0


# --------------------------------------------------------------------------
# curvature proxies
# --------------------------------------------------------------------------

def _edge_curvature(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int,
                    node_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """
    Per-edge EFC of the soft structure, computed one graph at a time.

    The formula itself is a degree-3 polynomial in the entries of A (shared with
    the measurement module, so proxy and report cannot drift apart), which makes
    it smooth and autograd-able with zero custom backward. The per-graph loop is
    what keeps it affordable: the batch adjacency is block-diagonal, so the
    matmuls are n_g^3 per graph instead of (sum n_g)^3 for the batch.
    """
    if num_graphs <= 1:
        return edge_forman_curvature(edge_index, edge_weight, num_nodes)

    sizes = torch.bincount(node_batch, minlength=num_graphs)
    offsets = torch.cat([torch.zeros(1, dtype=sizes.dtype, device=sizes.device),
                         sizes.cumsum(0)])
    edge_graph = node_batch[edge_index[0]]

    out = torch.zeros(edge_index.size(1), dtype=edge_weight.dtype, device=edge_weight.device)
    for g in range(num_graphs):
        mask = edge_graph == g
        if not bool(mask.any()):
            continue
        sub_index = edge_index[:, mask] - offsets[g]
        curv = edge_forman_curvature(sub_index, edge_weight[mask], int(sizes[g].item()))
        out = out.masked_scatter(mask, curv)
    return out


def _original_curvature(structure: dict, edge_index: torch.Tensor, num_nodes: int,
                        node_batch: torch.Tensor, num_graphs: int,
                        dtype: torch.dtype) -> torch.Tensor:
    """
    Baseline curvature per rewired edge: the EFC that edge had in the original
    1-skeleton, or 0 for an edge the lifting created.

    This is the paired-EFC correction, and it is not cosmetic. On a lifted
    structure every original edge gains a "free" triangle through the cell it
    belongs to, which mechanically adds +3 to its curvature and crushes the
    fraction of negatively curved edges towards zero. Comparing an edge to
    itself before the lifting is what keeps the sign meaningful; without it the
    negative part of the penalty would almost never fire.
    """
    original = structure.get("edge_index")
    if original is None or original.numel() == 0:
        return torch.zeros(edge_index.size(1), dtype=dtype, device=edge_index.device)

    ones = torch.ones(original.size(1), dtype=dtype, device=original.device)
    base = _edge_curvature(original, ones, num_nodes, node_batch, num_graphs)

    # Align by (src, dst) key: the rewired list is a superset of the original.
    keys_orig = original[0] * num_nodes + original[1]
    keys_new = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(keys_orig)
    sorted_keys = keys_orig[order]
    pos = torch.searchsorted(sorted_keys, keys_new).clamp(max=sorted_keys.numel() - 1)
    hit = sorted_keys[pos] == keys_new
    return torch.where(hit, base[order][pos], torch.zeros_like(base[order][pos]))


def efc(structure: dict, paired: bool = True, beta: float = 1.0, **_unused) -> torch.Tensor:
    """
    Smooth penalty on negatively curved edges of the soft structure:

        L = beta * mean_e softplus(-(EFC(e) - baseline(e)) / beta)

    softplus rather than an indicator on the negative part, because the
    indicator's gradient is zero almost everywhere; `beta` sets how sharply the
    penalty switches on around the baseline.

    Two caveats this proxy cannot fix, both worth remembering when reading its
    results: EFC only sees a 2-hop neighbourhood, while oversquashing is a depth
    phenomenon; and, taken alone, the term is maximized by piling up triangles
    and quadrangles - another reason the sparsity counterweight is mandatory.
    """
    edge_index, edge_weight, num_nodes, node_batch, num_graphs = _fields(structure)
    if edge_index.numel() == 0:
        return torch.zeros((), dtype=edge_weight.dtype, device=edge_weight.device)

    curv = _edge_curvature(edge_index, edge_weight, num_nodes, node_batch, num_graphs)
    if paired:
        curv = curv - _original_curvature(structure, edge_index, num_nodes, node_batch,
                                          num_graphs, edge_weight.dtype)
    return beta * F.softplus(-curv / beta).mean()


def cf_bc_efc(structure: dict, hutch_k: int = DEFAULT_HUTCH_K, cg_tol: float = DEFAULT_CG_TOL,
              cg_maxiter: int = DEFAULT_CG_MAXITER, eps: float = DEFAULT_EPS,
              seed: Optional[int] = None, **_unused) -> torch.Tensor:
    """
    L = - sum_e cf_bc(e) * EFC(e), the local+global candidate.

    wc = sum_e bc(e) * curv(e) is an excellent *measurement* but unusable as a
    loss, because shortest-path betweenness is piecewise-constant in the weights.
    Replacing bc by current-flow betweenness fixes exactly that: it is smooth in
    L^+, and it comes for free from the solves the resistance proxy already runs.
    One set of solves then feeds the global objective, its gradient, and this
    global weighting, while EFC supplies the local geometry.

    cf_bc is normalized to sum to 1 per graph, so the term is a weighted average
    curvature (comparable across graphs of different sizes and densities) rather
    than a sum that would grow with the edge count.
    """
    edge_index, edge_weight, num_nodes, node_batch, num_graphs = _fields(structure)
    if edge_index.numel() == 0 or num_graphs == 0:
        return torch.zeros((), dtype=edge_weight.dtype, device=edge_weight.device)

    cf = current_flow_betweenness(structure, hutch_k=hutch_k, cg_tol=cg_tol,
                                  cg_maxiter=cg_maxiter, eps=eps, seed=seed)
    edge_graph = node_batch[edge_index[0]]
    totals = torch.zeros(num_graphs, dtype=cf.dtype, device=cf.device)
    totals.index_add_(0, edge_graph, cf)
    cf = cf / totals.clamp(min=1e-12)[edge_graph]

    curv = _edge_curvature(edge_index, edge_weight, num_nodes, node_batch, num_graphs)
    weighted = torch.zeros(num_graphs, dtype=edge_weight.dtype, device=edge_weight.device)
    weighted.index_add_(0, edge_graph, cf * curv)

    present = (totals > 0).to(edge_weight.dtype)
    return -(weighted * present).sum() / present.sum().clamp(min=1.0)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def none(structure: dict, **_unused) -> torch.Tensor:
    """
    Baseline arm. Returns a hard zero that is still attached to the graph, so
    the registry contract ("every proxy returns a differentiable scalar") holds
    without changing a single bit of the total loss: adding 0.0 is exact, and
    the gradient it contributes is exactly zero.
    """
    return structure["alpha"].sum() * 0.0


PROXY_REGISTRY = {
    "none": none,
    "r_bar": r_bar,
    "lambda2": lambda2,
    "efc": efc,
    "cf_bc_efc": cf_bc_efc,
}


def available_proxies() -> Tuple[str, ...]:
    return tuple(PROXY_REGISTRY)


def get_proxy(name: str, **cfg) -> Callable[[dict], torch.Tensor]:
    """
    Binds a proxy's configuration and returns the one-argument callable
    `DMDLoss(osq_fn=...)` expects, so switching proxy is a config string and
    never a change at the loss's call site.
    """
    if name not in PROXY_REGISTRY:
        raise ValueError(f"Unknown OSq proxy '{name}'. Choices: {', '.join(PROXY_REGISTRY)}")
    proxy = PROXY_REGISTRY[name]

    def bound(structure: dict) -> torch.Tensor:
        return proxy(structure, **cfg)

    bound.__name__ = f"osq_proxy_{name}"
    return bound
