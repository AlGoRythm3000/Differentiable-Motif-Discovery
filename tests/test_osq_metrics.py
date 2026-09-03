import itertools
import math

import torch

from tools import osq_metrics as M


def _edge_index(pairs):
    """Both directions, the convention every module here expects."""
    ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return torch.cat([ei, ei.flip(0)], dim=1)


def _complete_graph(n):
    return _edge_index([(i, j) for i, j in itertools.combinations(range(n), 2)])


def _path_graph(n):
    return _edge_index([(i, i + 1) for i in range(n - 1)])


def test_effective_resistance_of_a_path_is_the_number_of_hops():
    # Series resistors: R_eff(0, n-1) on an unweighted path is exactly n-1.
    ei = _path_graph(5)
    r = M.effective_resistance(ei, None, 5, [(0, 4), (0, 1), (1, 3)])
    assert torch.allclose(r, torch.tensor([4.0, 1.0, 2.0]), atol=1e-4)


def test_effective_resistance_of_a_complete_graph_is_two_over_n():
    ei = _complete_graph(6)
    r = M.effective_resistance(ei, None, 6, [(0, 3)])
    assert math.isclose(r.item(), 2.0 / 6, rel_tol=1e-4)


def test_resistance_across_components_is_infinite_and_summary_stays_finite():
    # Two disjoint triangles: no current can flow between them, so the pairwise
    # value is +inf, while the summary only averages within-component pairs.
    tri = _complete_graph(3)
    ei = torch.cat([tri, tri + 3], dim=1)
    assert math.isinf(M.effective_resistance(ei, None, 6, [(0, 4)]).item())

    summary = M.resistance_summary(ei, None, 6)
    assert summary["num_components"] == 2
    assert math.isfinite(summary["r_bar"])
    assert math.isclose(summary["r_bar"], 2.0 / 3, rel_tol=1e-4)


def test_spectral_gap_matches_closed_forms():
    # lambda_2 of the normalized Laplacian: n/(n-1) for K_n, 1-cos(k pi/(n-1))
    # for a path, and exactly 0 as soon as the graph is disconnected.
    assert math.isclose(M.spectral_gap(_complete_graph(4), None, 4), 4 / 3, rel_tol=1e-4)
    assert math.isclose(M.spectral_gap(_path_graph(4), None, 4), 0.5, rel_tol=1e-4)

    tri = _complete_graph(3)
    disconnected = torch.cat([tri, tri + 3], dim=1)
    assert M.spectral_gap(disconnected, None, 6) == 0.0


def test_edge_forman_curvature_reference_values():
    # Hand-checkable cases: a lone edge has no support (4 - 1 - 1 = 2), a
    # triangle edge gains one triangle (4 - 2 - 2 + 3 = 3), a 4-cycle edge gains
    # one quadrangle (4 - 2 - 2 + 2 = 2).
    assert torch.allclose(M.edge_forman_curvature(_edge_index([(0, 1)]), None, 2),
                          torch.full((2,), 2.0))
    assert torch.allclose(M.edge_forman_curvature(_complete_graph(3), None, 3),
                          torch.full((6,), 3.0))
    cycle = _edge_index([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert torch.allclose(M.edge_forman_curvature(cycle, None, 4), torch.full((8,), 2.0))


def test_bridge_of_a_barbell_is_the_most_negatively_curved_edge():
    from tools.synthetic import barbell

    ei, n = barbell(clique_size=5)
    curv = M.edge_forman_curvature(ei, None, n)
    worst = int(curv.argmin().item())
    assert {int(ei[0, worst]), int(ei[1, worst])} == {4, 5}
    assert curv[worst] < 0


def test_curvature_is_differentiable_in_the_edge_weights():
    ei = _complete_graph(4)
    w = torch.rand(ei.size(1), requires_grad=True)
    M.edge_forman_curvature(ei, w, 4).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_weighted_curvature_signs():
    # A barbell has one strongly negative edge, so nwc must be negative; a
    # complete graph has none, so nwc is exactly 0 while wc stays positive.
    from tools.synthetic import barbell

    ei, n = barbell(clique_size=4)
    barbell_wc = M.weighted_curvature(ei, None, n)
    assert barbell_wc["nwc"] < 0

    complete = M.weighted_curvature(_complete_graph(5), None, 5)
    assert complete["nwc"] == 0.0
    assert complete["wc"] > 0


def test_influence_decay_drops_when_the_graph_gets_longer():
    short = M.influence_decay(_path_graph(4), None, 4, num_layers=3)
    long = M.influence_decay(_path_graph(12), None, 12, num_layers=3)
    assert short > long > 0


def test_osq_report_averages_per_graph_and_ignores_batch_concatenation():
    # Two copies of the same graph batched together must report the numbers of
    # one copy - not those of their (disconnected) union.
    tri = _complete_graph(3)
    single = M.osq_report(tri, None, 3)

    batched_index = torch.cat([tri, tri + 3], dim=1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    batched = M.osq_report(batched_index, None, 6, batch=batch)

    for key, value in single.items():
        assert math.isclose(value, batched[key], rel_tol=1e-4, abs_tol=1e-6)


def test_before_after_report_reads_the_snapshot_format():
    ei = _path_graph(4)
    samples = [{
        "num_nodes": 4,
        "edge_index": ei.tolist(),
        "rewired_edge_index": _complete_graph(4).tolist(),
        "rewired_edge_weight": [1.0] * 12,
    }]
    report = M.before_after_report(samples)
    # Completing the graph is the extreme rewiring: resistance down, gap up.
    assert report["r_bar_after"] < report["r_bar_before"]
    assert report["lambda2_after"] > report["lambda2_before"]
