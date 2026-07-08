# Graph-level classification task glue - out of scope for Phase 0 (node
# classification on the toy synthetic graph only). Stubbed so `train.py`'s
# TASKS registry stays structurally complete; wire this up when a graph
# classification benchmark (e.g. LRGB Peptides) enters scope.


def load_dataset(args):
    raise NotImplementedError("graph_classification task is out of scope for Phase 0")


def train_step(model, data, optimizer, criterion):
    raise NotImplementedError("graph_classification task is out of scope for Phase 0")


def eval_step(model, data, criterion, mask):
    raise NotImplementedError("graph_classification task is out of scope for Phase 0")
