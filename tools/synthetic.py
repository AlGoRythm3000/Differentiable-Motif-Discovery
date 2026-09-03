# Synthetic bottleneck graph families.
#
# These are the falsification core of the oversquashing work: the real
# benchmarks only show that a lifting does not *lose* anything, whereas a graph
# built so that its label is unreachable without crossing a bottleneck is where
# an OSq-guided lifting is supposed to win. If it does not win here, the thesis
# is wrong, and that is the point of having them.
#
# All families share one task shape, the "match" task: exactly one node carries
# a query (a slot index), a set of far-away nodes each carry (slot, label), and
# the graph's label is the label of the node whose slot matches the query.
# Why this shape rather than "read a marker off a far node": the readout is a
# mean over node representations, and a mean can recover the *multiset* of node
# features, so any task whose answer is a marginal of that multiset can be
# solved with no message passing at all. Matching a query to a slot is not such
# a marginal - the pairing has to be resolved somewhere in the network, which
# forces the query and the answers to meet, which forces information across the
# bottleneck. The label is then only recoverable through the very edges the
# lifting is supposed to fix.

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

import utils


class SyntheticGraphDataset:
    """
    Minimal stand-in for a PyG dataset, enough for `DataLoader` and for the
    experiment runner: `len`, integer/slice/index-list access,
    `num_node_features`, `num_classes`, and `y` for stratified splitting.

    Deliberately not a `torch_geometric.data.InMemoryDataset` subclass: these
    graphs are generated in seconds and never cached to disk, so the collation /
    processing machinery would only add failure modes on a Kaggle runtime.
    """

    def __init__(self, graphs: List[Data], num_classes: int, name: str = "synthetic"):
        if not graphs:
            raise ValueError("SyntheticGraphDataset needs at least one graph")
        self.graphs = graphs
        self.num_classes = num_classes
        self.name = name

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx]
        if isinstance(idx, slice):
            return SyntheticGraphDataset(self.graphs[idx], self.num_classes, self.name)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        return SyntheticGraphDataset([self.graphs[int(i)] for i in idx],
                                     self.num_classes, self.name)

    def __iter__(self):
        return iter(self.graphs)

    @property
    def num_node_features(self) -> int:
        return int(self.graphs[0].x.size(1))

    @property
    def y(self) -> torch.Tensor:
        return torch.tensor([int(g.y.item()) for g in self.graphs], dtype=torch.long)


@dataclass
class MatchLayout:
    """Where the query sits, which nodes hold answers, and how big the graph is."""
    edge_index: torch.Tensor
    num_nodes: int
    query_node: int
    answer_nodes: Sequence[int]


def _edge_index_from_graph(graph: nx.Graph) -> torch.Tensor:
    """[2, 2E] with both directions, matching the repo-wide convention."""
    edges = list(graph.edges())
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([ei, ei.flip(0)], dim=1)


def binary_tree_layout(depth: int = 3) -> MatchLayout:
    """
    Complete binary tree, query at the root, answers at the leaves.

    The bottleneck here is not a single edge but the branching itself: the
    root's receptive field doubles at every hop, so `depth` hops of messages
    have to be squeezed through a representation of fixed width - the textbook
    setting where oversquashing bites.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    num_nodes = 2 ** (depth + 1) - 1
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for parent in range(2 ** depth - 1):
        graph.add_edge(parent, 2 * parent + 1)
        graph.add_edge(parent, 2 * parent + 2)
    leaves = list(range(2 ** depth - 1, num_nodes))
    return MatchLayout(_edge_index_from_graph(graph), num_nodes, 0, leaves)


def path_of_cliques_layout(num_cliques: int = 4, clique_size: int = 5) -> MatchLayout:
    """
    Cliques chained by single bridge edges, query in the first clique, answers
    in the last one.

    Here the bottleneck is explicit and localized: every bridge edge is a cut
    edge that the whole signal must cross, so it is the natural place to watch
    the effective-resistance gradient concentrate.
    """
    graph = utils.make_path_of_cliques(num_cliques=num_cliques, clique_size=clique_size)
    num_nodes = graph.number_of_nodes()
    last_offset = (num_cliques - 1) * clique_size
    answers = list(range(last_offset, last_offset + clique_size))
    return MatchLayout(_edge_index_from_graph(graph), num_nodes, 0, answers)


def _match_features(layout: MatchLayout, num_classes: int,
                    rng: np.random.RandomState) -> Tuple[torch.Tensor, int]:
    """
    Feature block layout, per node:
        [ is_query | query slot one-hot (K) | own slot one-hot (K) | label one-hot (C) ]

    Keeping "which slot am I" and "which slot is asked for" in two different
    blocks is what makes the task un-poolable: their product, not their sum, is
    what identifies the answer.
    """
    num_slots = len(layout.answer_nodes)
    dim = 1 + 2 * num_slots + num_classes
    x = torch.zeros(layout.num_nodes, dim)

    labels = rng.randint(0, num_classes, size=num_slots)
    query_slot = int(rng.randint(0, num_slots))

    x[layout.query_node, 0] = 1.0
    x[layout.query_node, 1 + query_slot] = 1.0

    for slot, node in enumerate(layout.answer_nodes):
        x[node, 1 + num_slots + slot] = 1.0
        x[node, 1 + 2 * num_slots + int(labels[slot])] = 1.0

    return x, int(labels[query_slot])


def make_match_graph(layout: MatchLayout, num_classes: int,
                     rng: np.random.RandomState) -> Data:
    x, label = _match_features(layout, num_classes, rng)
    return Data(x=x, edge_index=layout.edge_index.clone(),
                y=torch.tensor([label], dtype=torch.long))


BOTTLENECK_FAMILIES = ("tree_neighbors_match", "path_of_cliques_match")


def bottleneck_dataset(name: str = "tree_neighbors_match", num_graphs: int = 600,
                       num_classes: int = 4, seed: int = 0,
                       depth: int = 3, num_cliques: int = 4,
                       clique_size: int = 5) -> SyntheticGraphDataset:
    """
    A graph-classification dataset over one bottleneck family.

    Topology is shared by every graph in the dataset; only the query and the
    answer labels are resampled, so the difficulty is entirely the routing
    problem and not some incidental structural variation. Chance level is
    1 / num_classes, which is the number any result table must be read against.
    """
    if name not in BOTTLENECK_FAMILIES:
        raise ValueError(f"Unknown bottleneck family '{name}'. "
                         f"Choices: {', '.join(BOTTLENECK_FAMILIES)}")
    if name == "tree_neighbors_match":
        layout = binary_tree_layout(depth=depth)
    else:
        layout = path_of_cliques_layout(num_cliques=num_cliques, clique_size=clique_size)

    rng = np.random.RandomState(seed)
    graphs = [make_match_graph(layout, num_classes, rng) for _ in range(num_graphs)]
    return SyntheticGraphDataset(graphs, num_classes, name=name)


def barbell(clique_size: int = 5, bridge_length: int = 0) -> Tuple[torch.Tensor, int]:
    """
    Two cliques joined by a single bridge (or by a path of `bridge_length`
    intermediate nodes). Returns (edge_index, num_nodes).

    The canonical qualitative test for any OSq proxy: whatever the proxy is, its
    largest sensitivity must sit on the bridge, because that is the only place
    where adding capacity changes how the two halves communicate.
    """
    graph = nx.barbell_graph(clique_size, bridge_length)
    graph = nx.convert_node_labels_to_integers(graph)
    return _edge_index_from_graph(graph), graph.number_of_nodes()


def ring_of_cliques(num_cliques: int = 4, clique_size: int = 5) -> Tuple[torch.Tensor, int]:
    """
    Path-of-cliques closed into a ring: same local density, no cut edge. Useful
    as the negative control - a proxy that scores this as badly as the barbell
    is reacting to density, not to bottlenecks.
    """
    graph = utils.make_path_of_cliques(num_cliques=num_cliques, clique_size=clique_size)
    graph.add_edge(num_cliques * clique_size - 1, 0)
    return _edge_index_from_graph(graph), graph.number_of_nodes()


def solvable_upper_bound(dataset: SyntheticGraphDataset) -> float:
    """
    Accuracy a model reaches by always predicting the majority class, i.e. the
    bar a run has to clear before "it learned something" means anything. Kept
    here so no report has to hard-code a chance level by hand.
    """
    labels = dataset.y
    counts = torch.bincount(labels, minlength=dataset.num_classes)
    return float(counts.max().item() / labels.numel())
