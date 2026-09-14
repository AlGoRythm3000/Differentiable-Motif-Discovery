# A^col (Taha et al. Def. 4.1) as an ANALYSIS-TIME measurement.
#
# The point of these tests is the one §6.7 of the paper raises: r_bar_after is
# measured on the very star the proxy minimises, so it cannot distinguish "the
# mechanism improved" from "the optimiser reduced its own objective". Measuring
# R-bar on A^col instead is only useful if A^col is genuinely a different
# structure from the star, and if it is the structure the paper defines.
import torch

from tools.osq_metrics import collapsed_adjacency_edges, resistance_summary, before_after_report
from models.message_passing import build_rewired_edges
from models.motif_proposition import CandidateCells


def _undirected(pairs):
    src = [u for u, v in pairs] + [v for u, v in pairs]
    dst = [v for u, v in pairs] + [u for u, v in pairs]
    return torch.tensor([src, dst], dtype=torch.long)


def test_acol_of_one_cell_is_the_clique_on_its_members():
    # Path 0-1-2-3, one cell on {0,1,2,3}: A^col must close every pair.
    edge_index = _undirected([(0, 1), (1, 2), (2, 3)])
    index, weight = collapsed_adjacency_edges([0, 1, 2, 3], [0, 0, 0, 0], [1.0],
                                              edge_index, num_nodes=4)
    got = {(min(u, v), max(u, v)) for u, v in zip(index[0].tolist(), index[1].tolist())}
    assert got == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert torch.allclose(weight, torch.ones_like(weight))


def test_acol_carries_the_cells_acceptance_weight_on_new_pairs():
    edge_index = _undirected([(0, 1), (1, 2)])
    index, weight = collapsed_adjacency_edges([0, 1, 2], [0, 0, 0], [0.25],
                                              edge_index, num_nodes=3)
    w = {(min(u, v), max(u, v)): float(x)
         for u, v, x in zip(index[0].tolist(), index[1].tolist(), weight.tolist())}
    # Original edges keep weight 1.0 (merged with reduce="max"); the pair the
    # cell newly relates carries the cell's alpha.
    assert w[(0, 1)] == 1.0 and w[(1, 2)] == 1.0
    assert w[(0, 2)] == 0.25


def test_acol_is_denser_and_lower_resistance_than_the_star_it_replaces():
    # THE test: the star is what the proxy scores, the clique is what the paper
    # defines. If these two agreed there would be no defect to report.
    edge_index = _undirected([(0, 1), (1, 2), (2, 3), (3, 4)])
    n = 5
    members, cell_batch = [0, 1, 2, 3, 4], [0] * 5
    alpha = torch.tensor([1.0])
    candidates = CandidateCells(node_index=torch.tensor(members),
                                cell_batch=torch.tensor(cell_batch),
                                anchor_index=torch.tensor([0]),
                                scores=torch.zeros(1))
    star_i, star_w = build_rewired_edges(edge_index, candidates, alpha, n)
    acol_i, acol_w = collapsed_adjacency_edges(members, cell_batch, alpha.tolist(),
                                               edge_index, n)
    assert acol_i.size(1) > star_i.size(1)
    r_star = resistance_summary(star_i, star_w, n)["r_bar"]
    r_acol = resistance_summary(acol_i, acol_w, n)["r_bar"]
    assert r_acol < r_star


def test_acol_with_no_accepted_cell_is_the_original_graph():
    edge_index = _undirected([(0, 1), (1, 2)])
    index, weight = collapsed_adjacency_edges([], [], [], edge_index, num_nodes=3)
    assert index.size(1) == edge_index.size(1)
    assert torch.allclose(weight, torch.ones_like(weight))


def test_report_omits_acol_when_a_sample_carries_no_cells():
    # A results tree produced before this measurement existed must not come back
    # with A^col silently equal to the star value.
    sample = {"num_nodes": 3, "edge_index": [[0, 1], [1, 0]],
              "rewired_edge_index": [[0, 1], [1, 0]], "rewired_edge_weight": [1.0, 1.0]}
    out = before_after_report([sample], with_betweenness=False)
    assert "r_bar_acol_after" not in out


def test_report_emits_acol_when_cells_are_present():
    sample = {"num_nodes": 4,
              "edge_index": _undirected([(0, 1), (1, 2), (2, 3)]).tolist(),
              "rewired_edge_index": _undirected([(0, 1), (1, 2), (2, 3)]).tolist(),
              "rewired_edge_weight": [1.0] * 6,
              "cell_node_index": [0, 1, 2, 3], "cell_batch": [0, 0, 0, 0],
              "cell_alpha": [1.0]}
    out = before_after_report([sample], with_betweenness=False)
    assert out["r_bar_acol_after"] < out["r_bar_acol_before"]
    # and it is NOT the star number: the star was never even built here
    assert out["r_bar_acol_after"] < out["r_bar_after"]
