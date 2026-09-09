# Per-stage brick registries: every brick selectable by a
# config string, so assembling a DMDModel is data (five strings + kwargs)
# rather than code. Every brick within a stage keeps that stage's forward
# contract (documented in its own module), so swapping the string is always a
# drop-in change at DMDModel's call sites - no code surgery.

from typing import Callable, Dict, List, Tuple

from models.graph_embeddings import GPSEEncoder, GraphEmbedder, PSEExplicitEncoder
from models.message_passing import GNNRewiredMP, HypergraphTNNMessagePassing, TNNMessagePassing
from models.motif_encodings import DeepSetsEncoder, MiniGNNEncoder, SetTransformerEncoder
from models.motif_proposition import AutoregressiveProposal, CycleBasisProposal, MotifProposal
from models.weight_assignment import GumbelSigmoidSelector, KSubsetSelector, REINFORCESelector

STAGE1_ENCODERS: Dict[str, Callable] = {
    "gcn": GraphEmbedder,
    "gin": GraphEmbedder,
    "gpse": GPSEEncoder,
    "pse_explicit": PSEExplicitEncoder,
}

STAGE2_PROPOSALS: Dict[str, Callable] = {
    "topk": MotifProposal,
    "autoregressive": AutoregressiveProposal,
    "cycle_basis": CycleBasisProposal,
}

STAGE3_CELL_ENCODERS: Dict[str, Callable] = {
    "deepsets": DeepSetsEncoder,
    "set_transformer": SetTransformerEncoder,
    "mini_gnn": MiniGNNEncoder,
}

STAGE4_SELECTORS: Dict[str, Callable] = {
    "gumbel": GumbelSigmoidSelector,
    "ksubset": KSubsetSelector,
    "reinforce": REINFORCESelector,
}

STAGE5_MP: Dict[str, Callable] = {
    "gnn_rewired": GNNRewiredMP,
    "tnn": TNNMessagePassing,
    "hypergraph_tnn": HypergraphTNNMessagePassing,
}

# Cells whose boundary is an actual cycle in the 1-skeleton are what a cell
# complex's 2-cells require - today only `cycle_basis`
# guarantees that. `hypergraph_tnn` has no such requirement (any node subset
# is a legitimate hyperedge), which is exactly why it is the fallback for
# every other proposal.
_TNN_VALID_PROPOSALS = ("cycle_basis",)

# (predicate, message) pairs, checked in order. `predicate(s1, s2, s3, s4, s5)
# -> True` means the combination is INVALID. Kept as a list (not a single
# if-chain) so more rules can be added without touching the call site below.
_VALIDATION_RULES: List[Tuple[Callable[[str, str, str, str, str], bool], str]] = [
    (lambda s1, s2, s3, s4, s5: s5 == "tnn" and s2 not in _TNN_VALID_PROPOSALS,
     "s5='tnn' requires a Stage 2 proposal whose cells are valid 1-skeleton cycles "
     f"(one of {_TNN_VALID_PROPOSALS}), got s2={{s2!r}}. A cell complex's 2-cells must have a "
     "boundary that is an actual cycle - 'topk'/'autoregressive' cells are arbitrary node sets "
     "and would silently build an invalid complex. Use s5='hypergraph_tnn' instead, which has "
     "no such requirement."),
]


def validate_pipeline_config(s1: str, s2: str, s3: str, s4: str, s5: str) -> None:
    """Raises ValueError with a clear message for a known-invalid brick combination."""
    for stage, name, registry in (("s1", s1, STAGE1_ENCODERS), ("s2", s2, STAGE2_PROPOSALS),
                                   ("s3", s3, STAGE3_CELL_ENCODERS), ("s4", s4, STAGE4_SELECTORS),
                                   ("s5", s5, STAGE5_MP)):
        if name not in registry:
            raise ValueError(f"Unknown {stage} brick {name!r}. Choices: {sorted(registry)}")

    for predicate, message in _VALIDATION_RULES:
        if predicate(s1, s2, s3, s4, s5):
            raise ValueError(message.format(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5))
