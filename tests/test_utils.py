import torch

from utils import make_path_of_cliques, path_of_cliques_dataset


def test_graph_structure_counts():
    num_cliques, clique_size = 4, 5
    graph = make_path_of_cliques(num_cliques, clique_size)

    assert graph.number_of_nodes() == num_cliques * clique_size
    expected_intra_edges = num_cliques * (clique_size * (clique_size - 1) // 2)
    expected_bridges = num_cliques - 1
    assert graph.number_of_edges() == expected_intra_edges + expected_bridges


def test_dataset_masks_partition_and_stratify():
    num_cliques, clique_size = 4, 6
    data = path_of_cliques_dataset(num_cliques=num_cliques, clique_size=clique_size,
                                    feature_dim=8, seed=0)

    overlap = ((data.train_mask & data.val_mask)
               | (data.val_mask & data.test_mask)
               | (data.train_mask & data.test_mask))
    assert not overlap.any()
    assert (data.train_mask | data.val_mask | data.test_mask).all()

    for c in range(num_cliques):
        offset = c * clique_size
        clique_slice = slice(offset, offset + clique_size)
        assert data.train_mask[clique_slice].any()
        assert data.val_mask[clique_slice].any()
        assert data.test_mask[clique_slice].any()


def test_labels_in_range():
    num_cliques, clique_size = 5, 4
    data = path_of_cliques_dataset(num_cliques=num_cliques, clique_size=clique_size, feature_dim=8)
    assert data.y.min().item() >= 0
    assert data.y.max().item() == num_cliques - 1


def test_beacon_marker_is_unique():
    data = path_of_cliques_dataset(num_cliques=4, clique_size=5, feature_dim=8)
    beacon_channel = data.x[:, 0]
    assert beacon_channel[0].item() == 1.0
    assert torch.all(beacon_channel[1:] == 0.0)
