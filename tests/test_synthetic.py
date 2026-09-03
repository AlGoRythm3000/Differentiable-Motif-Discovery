import networkx as nx
import torch
from torch_geometric.loader import DataLoader

from tools import osq_metrics as M
from tools import synthetic as S


def test_every_family_builds_a_usable_graph_classification_dataset():
    for name in S.BOTTLENECK_FAMILIES:
        dataset = S.bottleneck_dataset(name, num_graphs=12, num_classes=3, seed=0)
        assert len(dataset) == 12
        assert dataset.num_classes == 3
        graph = dataset[0]
        assert graph.x.size(1) == dataset.num_node_features
        assert graph.y.shape == (1,)
        assert 0 <= int(graph.y.item()) < 3


def test_graphs_are_connected_so_the_signal_has_a_path_to_travel():
    for name in S.BOTTLENECK_FAMILIES:
        graph = S.bottleneck_dataset(name, num_graphs=1, seed=0)[0]
        nx_graph = M.to_networkx(graph.edge_index, None, graph.num_nodes)
        assert nx.is_connected(nx_graph)


def test_the_label_is_the_queried_slot_not_a_marginal_of_the_features():
    # The whole point of the match task: reading the pooled features is not
    # enough, the query has to be resolved against the right answer node. Here
    # we resolve it by hand and check the stored label agrees.
    layout = S.binary_tree_layout(depth=2)
    num_slots = len(layout.answer_nodes)
    import numpy as np

    rng = np.random.RandomState(0)
    graph = S.make_match_graph(layout, num_classes=3, rng=rng)

    query_slot = int(graph.x[layout.query_node, 1:1 + num_slots].argmax().item())
    answer_node = layout.answer_nodes[query_slot]
    label = int(graph.x[answer_node, 1 + 2 * num_slots:].argmax().item())
    assert label == int(graph.y.item())


def test_generation_is_seeded():
    a = S.bottleneck_dataset("tree_neighbors_match", num_graphs=5, seed=1)
    b = S.bottleneck_dataset("tree_neighbors_match", num_graphs=5, seed=1)
    c = S.bottleneck_dataset("tree_neighbors_match", num_graphs=5, seed=2)
    assert torch.equal(a.y, b.y)
    assert not torch.equal(a.y, c.y)


def test_dataset_supports_index_lists_and_batching():
    dataset = S.bottleneck_dataset("path_of_cliques_match", num_graphs=8, seed=0)
    subset = dataset[[0, 2, 4]]
    assert len(subset) == 3
    batch = next(iter(DataLoader(subset, batch_size=3)))
    assert batch.num_graphs == 3
    assert batch.batch.max().item() == 2


def test_bottleneck_families_are_actually_bottlenecked():
    # A path of cliques must be measurably worse than the same cliques closed
    # into a ring: same local density, one has cut edges and the other does not.
    path_index, path_n = S.path_of_cliques_layout(4, 5).edge_index, 20
    ring_index, ring_n = S.ring_of_cliques(4, 5)

    assert (M.resistance_summary(path_index, None, path_n)["r_bar"]
            > M.resistance_summary(ring_index, None, ring_n)["r_bar"])
    assert M.spectral_gap(path_index, None, path_n) < M.spectral_gap(ring_index, None, ring_n)


def test_barbell_has_exactly_one_bridge():
    edge_index, n = S.barbell(clique_size=4, bridge_length=0)
    graph = M.to_networkx(edge_index, None, n)
    assert list(nx.bridges(graph)) == [(3, 4)]


def test_majority_baseline_is_reported_not_guessed():
    dataset = S.bottleneck_dataset("tree_neighbors_match", num_graphs=200,
                                    num_classes=4, seed=0)
    baseline = S.solvable_upper_bound(dataset)
    assert 0.2 < baseline < 0.45  # near chance for 4 balanced classes


def test_unknown_family_is_rejected():
    try:
        S.bottleneck_dataset("not_a_family")
    except ValueError as error:
        assert "not_a_family" in str(error)
    else:
        raise AssertionError("an unknown family must raise")
