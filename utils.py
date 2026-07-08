import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_path_of_cliques(num_cliques: int = 8, clique_size: int = 6) -> nx.Graph:
    """
    num_cliques disjoint K_{clique_size} cliques, chained by a single bridge
    edge between consecutive cliques (last node of clique i <-> first node of
    clique i+1). Each node gets a `clique_id` attribute. This is the toy
    synthetic graph family named in CLAUDE.md §9 (path-of-cliques) as an
    eventual falsification benchmark for oversquashing.
    """
    if num_cliques < 1 or clique_size < 2:
        raise ValueError("num_cliques must be >= 1 and clique_size must be >= 2")

    graph = nx.Graph()
    for c in range(num_cliques):
        offset = c * clique_size
        nodes = range(offset, offset + clique_size)
        graph.add_nodes_from(nodes, clique_id=c)
        graph.add_edges_from((i, j) for i in nodes for j in nodes if i < j)
        if c > 0:
            prev_last = offset - 1
            graph.add_edge(prev_last, offset)
    return graph


def _stratified_counts(size: int, train_frac: float, val_frac: float):
    n_train = max(1, int(round(train_frac * size)))
    n_val = max(1, int(round(val_frac * size)))
    if n_train + n_val >= size:
        n_train = max(1, size - 2)
        n_val = 1
    n_test = size - n_train - n_val
    return n_train, n_val, n_test


def path_of_cliques_dataset(num_cliques: int = 8, clique_size: int = 6, feature_dim: int = 16,
                             train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0) -> Data:
    """
    Toy synthetic node-classification task: label = clique_id.

    Correctness note: a plain path-of-cliques with i.i.d. random node features
    and label = clique_id is UNSOLVABLE in principle - the graph has
    clique-permutation/reflection automorphisms, and exchangeable random
    features give a permutation-equivariant GNN no way to recover an arbitrary
    global clique id. We break this symmetry with a single one-hot "beacon"
    marker at node 0 (clique 0's bridge-adjacent node, reserved feature
    channel 0): recovering `clique_id` then requires propagating the beacon's
    signal across the bottleneck bridge edges, which is exactly the kind of
    long-range/oversquashing-sensitive task this project targets.

    With a shallow GNN, don't expect near-perfect accuracy for distant
    cliques - the Phase 0 bar is "loss decreases end-to-end and beats the
    1/num_cliques random-guess baseline", not full solvability (that's gated
    by depth/oversquashing, which Phase 1 formally measures).
    """
    set_seed(seed)
    graph = make_path_of_cliques(num_cliques, clique_size)
    num_nodes = graph.number_of_nodes()

    edge_list = list(graph.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    x = torch.randn(num_nodes, feature_dim)
    x[:, 0] = 0.0
    beacon_node = 0
    x[beacon_node, 0] = 1.0

    y = torch.tensor([graph.nodes[n]["clique_id"] for n in range(num_nodes)], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    rng = np.random.RandomState(seed)
    n_train, n_val, n_test = _stratified_counts(clique_size, train_frac, val_frac)
    for c in range(num_cliques):
        offset = c * clique_size
        idx = list(range(offset, offset + clique_size))
        rng.shuffle(idx)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:n_train + n_val + n_test]] = True

    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
