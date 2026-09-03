import math

import networkx as nx
import torch

from models.dmd_model import DMDModel
from tools import osq_metrics as M
from tools import osq_proxies as P
from tools.losses import DMDLoss
from tools.synthetic import barbell, ring_of_cliques

# Tolerance for the analytic-vs-finite-difference gradient check. The forward is
# a Hutchinson estimate, so it is only a deterministic function of the weights
# when the probes are frozen (`seed=`); once they are, the comparison is exact
# maths and float64 + a tight CG leaves ~1e-8 of room. 1e-5 is therefore a very
# loose bar that still fails loudly on a wrong formula (a missing factor 0.5, a
# sign, or a forgotten normalization all move this by orders of magnitude).
GRADIENT_RTOL = 1e-5


def _structure(edge_index, edge_weight, num_nodes, batch=None, original=None):
    """The subset of DMDModel.forward's output dict that the proxies read."""
    return {
        "alpha": edge_weight,
        "edge_index": edge_index if original is None else original,
        "rewired_edge_index": edge_index,
        "rewired_edge_weight": edge_weight,
        "num_nodes": num_nodes,
        "batch": batch,
    }


def _both_directions(pairs):
    ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return torch.cat([ei, ei.flip(0)], dim=1)


def _random_connected_graph(n=10, seed=2):
    graph = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=seed)
    return _both_directions(list(graph.edges()))


# ---------------------------------------------------------------------------
# r_bar: the primary proxy
# ---------------------------------------------------------------------------

def test_r_bar_gradient_matches_finite_differences():
    torch.manual_seed(0)
    n = 10
    edge_index = _random_connected_graph(n)
    weights = (torch.rand(edge_index.size(1), dtype=torch.float64) * 0.8 + 0.2)

    def evaluate(w):
        return P.r_bar(_structure(edge_index, w, n), hutch_k=16, seed=7,
                       eps=1e-6, cg_tol=1e-13, cg_maxiter=2000)

    w = weights.clone().requires_grad_(True)
    evaluate(w).backward()
    analytic = w.grad.clone()

    h = 1e-6
    for i in range(edge_index.size(1)):
        plus, minus = weights.clone(), weights.clone()
        plus[i] += h
        minus[i] -= h
        numeric = (evaluate(plus).item() - evaluate(minus).item()) / (2 * h)
        assert math.isclose(numeric, analytic[i].item(),
                            rel_tol=GRADIENT_RTOL, abs_tol=1e-9)


def test_r_bar_sensitivity_is_largest_on_the_bottleneck():
    # The qualitative test that the proxy means what we think it means: on two
    # cliques joined by one bridge, the edge whose weight most reduces the mean
    # effective resistance must be the bridge.
    edge_index, n = barbell(clique_size=5)
    w = torch.ones(edge_index.size(1), dtype=torch.float64, requires_grad=True)

    P.r_bar(_structure(edge_index, w, n), hutch_k=256, seed=5, eps=1e-9,
            cg_tol=1e-12, cg_maxiter=500).backward()

    worst = int(w.grad.abs().argmax().item())
    assert {int(edge_index[0, worst]), int(edge_index[1, worst])} == {4, 5}


def test_r_bar_estimator_agrees_with_the_exact_measurement():
    edge_index = _random_connected_graph(10)
    w = torch.ones(edge_index.size(1), dtype=torch.float64)
    exact = M.resistance_summary(edge_index, w, 10)["r_bar"]
    estimate = P.r_bar(_structure(edge_index, w, 10), hutch_k=4000, seed=3,
                       eps=1e-9, cg_tol=1e-12, cg_maxiter=500).item()
    assert math.isclose(estimate, exact, rel_tol=0.05)


def test_r_bar_is_lower_without_a_bottleneck():
    barbell_index, n = barbell(clique_size=5)
    ring_index, ring_n = ring_of_cliques(num_cliques=2, clique_size=5)
    kwargs = dict(hutch_k=512, seed=11, eps=1e-9, cg_tol=1e-12, cg_maxiter=500)

    bottlenecked = P.r_bar(_structure(barbell_index, torch.ones(barbell_index.size(1),
                                                                dtype=torch.float64), n), **kwargs)
    unbottlenecked = P.r_bar(_structure(ring_index, torch.ones(ring_index.size(1),
                                                                dtype=torch.float64), ring_n), **kwargs)
    assert unbottlenecked.item() < bottlenecked.item()


def test_r_bar_is_finite_on_a_connected_graph_and_capped_on_a_disconnected_one():
    # Kernel handling. Connected: the per-graph projection removes the all-ones
    # direction, so the estimate is finite and matches the exact value above.
    # Disconnected: the true mean resistance is infinite; the eps grounding
    # replaces it by a finite ceiling of order 1/eps. That is a documented
    # modelling choice, and the test pins the two properties that matter - it
    # stays finite, and it still scores strictly worse than the connected case.
    edge_index = _both_directions([(0, 1), (1, 2), (3, 4), (4, 5)])
    w = torch.ones(edge_index.size(1), dtype=torch.float64)
    disconnected = P.r_bar(_structure(edge_index, w, 6), hutch_k=64, seed=1, eps=1e-3)

    connected_index = _both_directions([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    connected = P.r_bar(_structure(connected_index,
                                   torch.ones(connected_index.size(1), dtype=torch.float64), 6),
                        hutch_k=64, seed=1, eps=1e-3)

    assert torch.isfinite(disconnected).item()
    assert disconnected.item() > connected.item()


def test_r_bar_is_averaged_per_graph_not_over_the_batch():
    # Two copies of the same graph must score like one copy, otherwise the term
    # would depend on how the DataLoader happened to batch the dataset.
    edge_index = _random_connected_graph(8, seed=5)
    n = 8
    w = torch.ones(edge_index.size(1), dtype=torch.float64)
    kwargs = dict(hutch_k=2000, seed=4, eps=1e-9, cg_tol=1e-12, cg_maxiter=500)

    single = P.r_bar(_structure(edge_index, w, n), **kwargs).item()

    doubled_index = torch.cat([edge_index, edge_index + n], dim=1)
    batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    doubled = P.r_bar(_structure(doubled_index, torch.ones(doubled_index.size(1),
                                                            dtype=torch.float64),
                                 2 * n, batch=batch), **kwargs).item()
    assert math.isclose(single, doubled, rel_tol=0.1)


def test_current_flow_betweenness_peaks_on_the_bridge():
    edge_index, n = barbell(clique_size=5)
    w = torch.ones(edge_index.size(1), dtype=torch.float64)
    cf = P.current_flow_betweenness(_structure(edge_index, w, n), hutch_k=256, seed=2,
                                    eps=1e-9, cg_tol=1e-12, cg_maxiter=500)
    worst = int(cf.argmax().item())
    assert {int(edge_index[0, worst]), int(edge_index[1, worst])} == {4, 5}
    assert not cf.requires_grad


# ---------------------------------------------------------------------------
# secondary proxies
# ---------------------------------------------------------------------------

def test_lambda2_proxy_recovers_the_exact_spectral_gap():
    edge_index = _random_connected_graph(10, seed=3)
    w = torch.ones(edge_index.size(1), dtype=torch.float64)
    value = P.lambda2(_structure(edge_index, w, 10), num_iters=400, seed=1)
    exact = M.spectral_gap(edge_index, w, 10)
    # The proxy returns -lambda_2 (it is minimized), and a truncated power
    # iteration can only over-estimate lambda_2, never under-estimate it.
    assert math.isclose(-value.item(), exact, rel_tol=1e-3)


def test_lambda2_prefers_the_graph_without_a_bottleneck():
    barbell_index, n = barbell(clique_size=5)
    ring_index, ring_n = ring_of_cliques(num_cliques=2, clique_size=5)
    bottlenecked = P.lambda2(_structure(barbell_index,
                                        torch.ones(barbell_index.size(1)), n), seed=0)
    unbottlenecked = P.lambda2(_structure(ring_index,
                                          torch.ones(ring_index.size(1)), ring_n), seed=0)
    assert unbottlenecked.item() < bottlenecked.item()


def test_efc_proxy_paired_baseline_is_zero_on_an_unchanged_structure():
    # If the "rewired" structure is the original one at weight 1, every paired
    # curvature is 0, so the penalty is exactly softplus(0).
    edge_index, n = barbell(clique_size=4)
    w = torch.ones(edge_index.size(1))
    value = P.efc(_structure(edge_index, w, n), paired=True)
    assert math.isclose(value.item(), math.log(2.0), rel_tol=1e-5)


def test_efc_proxy_falls_when_the_bottleneck_is_shortcut():
    edge_index, n = barbell(clique_size=4)
    original = edge_index
    shortcut = torch.cat([edge_index, _both_directions([(0, 7), (1, 6)])], dim=1)

    base = P.efc(_structure(edge_index, torch.ones(edge_index.size(1)), n,
                            original=original), paired=False)
    improved = P.efc(_structure(shortcut, torch.ones(shortcut.size(1)), n,
                                original=original), paired=False)
    assert improved.item() < base.item()


def test_curvature_proxies_treat_a_batch_as_separate_graphs():
    # The batch adjacency is block-diagonal: two copies of a graph must give the
    # same per-edge curvature as one copy, and the same proxy value. This is the
    # test that the per-graph loop's node offsets and edge scatter line up.
    edge_index, n = barbell(clique_size=4)
    w = torch.ones(edge_index.size(1))
    single = P._edge_curvature(edge_index, w, n, torch.zeros(n, dtype=torch.long), 1)

    doubled_index = torch.cat([edge_index, edge_index + n], dim=1)
    doubled_w = torch.ones(doubled_index.size(1))
    batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    doubled = P._edge_curvature(doubled_index, doubled_w, 2 * n, batch, 2)

    assert torch.allclose(doubled, torch.cat([single, single]), atol=1e-4)

    # `efc` is deterministic, so the two values must agree to float precision.
    one = P.efc(_structure(edge_index, w, n), paired=False)
    two = P.efc(_structure(doubled_index, doubled_w, 2 * n, batch=batch), paired=False)
    assert math.isclose(one.item(), two.item(), rel_tol=1e-4)

    # `cf_bc_efc` weights curvature by a Hutchinson estimate, and the batch draws
    # a different probe set (twice as many nodes), so the agreement is only up to
    # estimator noise - hence a loose tolerance rather than an equality. The
    # deterministic half of the batching logic is already pinned exactly by the
    # two assertions above; this one only checks that nothing structural (an
    # offset, a mis-scattered edge) is wrong.
    kwargs = dict(hutch_k=2048, seed=9, eps=1e-9, cg_tol=1e-10, cg_maxiter=500)
    one = P.cf_bc_efc(_structure(edge_index, w, n), **kwargs)
    two = P.cf_bc_efc(_structure(doubled_index, doubled_w, 2 * n, batch=batch), **kwargs)
    assert math.isclose(one.item(), two.item(), rel_tol=0.1)


def test_cf_bc_efc_is_differentiable_and_rewards_curvature_at_high_current():
    edge_index, n = barbell(clique_size=5)
    w = torch.ones(edge_index.size(1), requires_grad=True)
    value = P.cf_bc_efc(_structure(edge_index, w, n), hutch_k=64, seed=6)
    value.backward()
    assert torch.isfinite(w.grad).all()
    # The bridge carries nearly all the current and is the most negatively
    # curved edge, so it must be among the edges the term pushes hardest on.
    bridge = [k for k in range(edge_index.size(1))
              if {int(edge_index[0, k]), int(edge_index[1, k])} == {4, 5}]
    assert w.grad.abs()[bridge].max() > w.grad.abs().median()


# ---------------------------------------------------------------------------
# registry and loss wiring
# ---------------------------------------------------------------------------

def test_every_registry_key_returns_a_differentiable_scalar():
    edge_index, n = barbell(clique_size=4)
    for name in P.available_proxies():
        w = torch.ones(edge_index.size(1), requires_grad=True)
        value = P.get_proxy(name, seed=0)(_structure(edge_index, w, n))
        assert value.dim() == 0, name
        assert value.requires_grad, name
        value.backward()
        assert w.grad is not None and torch.isfinite(w.grad).all(), name


def test_get_proxy_rejects_an_unknown_name():
    try:
        P.get_proxy("not_a_proxy")
    except ValueError as error:
        assert "not_a_proxy" in str(error)
    else:
        raise AssertionError("an unknown proxy name must raise")


def test_none_proxy_is_exactly_zero_and_keeps_the_total_bit_identical():
    logits = torch.randn(4, 3)
    target = torch.randint(0, 3, (4,))
    mask = torch.ones(4, dtype=torch.bool)
    alpha = torch.rand(4, requires_grad=True)
    structure = {"alpha": alpha}

    plain = DMDLoss(sparsity_weight=0.05)(logits, target, mask, structure)
    with_none = DMDLoss(sparsity_weight=0.05, osq_weight=1.0,
                        osq_fn=P.get_proxy("none"))(logits, target, mask, structure)

    assert with_none.osq.item() == 0.0
    assert with_none.total.item() == plain.total.item()


def test_gamma_zero_reproduces_the_proxy_free_loss_bit_for_bit():
    torch.manual_seed(0)
    edge_index, n = barbell(clique_size=4)
    logits = torch.randn(n, 3)
    target = torch.randint(0, 3, (n,))
    mask = torch.ones(n, dtype=torch.bool)
    alpha = torch.rand(edge_index.size(1), requires_grad=True)
    structure = _structure(edge_index, alpha, n)
    structure["alpha"] = alpha

    plain = DMDLoss(sparsity_weight=0.05)(logits, target, mask, structure)
    for name in P.available_proxies():
        gated = DMDLoss(sparsity_weight=0.05, osq_weight=0.0,
                        osq_fn=P.get_proxy(name))(logits, target, mask, structure)
        assert gated.total.item() == plain.total.item(), name
        assert gated.osq.item() == 0.0, name


def test_proxy_gradient_reaches_the_model_through_the_soft_structure():
    # End-to-end contract: the OSq term alone (no task loss) must move the
    # parameters that generate the structure, which is only possible because it
    # reads the soft acceptance weights rather than a thresholded adjacency.
    torch.manual_seed(0)
    model = DMDModel(input_dim=5, hidden_dim=8, latent_dim=6, motif_hidden_dim=6,
                     motif_out_dim=4, num_classes=2, top_k=2)
    x = torch.randn(9, 5)
    edge_index = _both_directions([(i, i + 1) for i in range(8)])

    _, structure = model(x, edge_index)
    P.get_proxy("r_bar", hutch_k=16, seed=0)(structure).backward()

    assert model.proposal.W.grad is not None
    assert model.proposal.W.grad.abs().sum() > 0
