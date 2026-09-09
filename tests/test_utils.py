import torch

from utils import make_path_of_cliques, path_of_cliques_dataset, stratified_kfold


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


# ---------------------------------------------------------------------------
# stratified_kfold - the protocol fix. A single 80/10/10 split gave MUTAG a
# 20-graph test set, i.e. 5 accuracy points per graph, which is most of why the
# feat/rich-bricks error bars were unreadable.
# ---------------------------------------------------------------------------

def _labels(n_per_class):
    return torch.tensor([c for c, n in enumerate(n_per_class) for _ in range(n)])


def test_kfold_makes_every_graph_a_test_graph_exactly_once():
    labels = _labels([125, 63])  # MUTAG's class balance
    folds = stratified_kfold(labels, n_splits=10, seed=0)

    tested = [i for _, _, test in folds for i in test]
    assert sorted(tested) == list(range(len(labels)))
    assert len(tested) == len(set(tested))


def test_kfold_splits_are_disjoint_and_cover_the_dataset():
    labels = _labels([60, 40, 50])
    for train, val, test in stratified_kfold(labels, n_splits=5, seed=1):
        assert not set(train) & set(val)
        assert not set(train) & set(test)
        assert not set(val) & set(test)
        assert len(train) + len(val) + len(test) == len(labels)


def test_kfold_test_blocks_keep_the_class_balance():
    labels = _labels([125, 63])
    global_balance = float((labels == 1).float().mean())
    for _, _, test in stratified_kfold(labels, n_splits=10, seed=0):
        block = labels[torch.tensor(test)]
        assert abs(float((block == 1).float().mean()) - global_balance) < 0.1


def test_kfold_layout_is_reproducible_and_seed_dependent():
    labels = _labels([50, 50])
    assert stratified_kfold(labels, 5, seed=0) == stratified_kfold(labels, 5, seed=0)
    assert stratified_kfold(labels, 5, seed=0) != stratified_kfold(labels, 5, seed=1)


def test_kfold_rejects_too_few_splits():
    # Fold i tests on block i and validates on block i+1, so k < 3 would leave
    # no training data at all.
    try:
        stratified_kfold(_labels([10, 10]), n_splits=2)
        assert False, "expected ValueError"
    except ValueError:
        pass
