"""BEATs audio encoder module — Version 2 integration.

This module wraps the official Microsoft BEATs implementation to produce
frozen 768-dim audio embeddings for use in the Fusion Fingerprint pipeline.

SDD v4 §4.1 — BEATs is the deep block of the Fusion Fingerprint:
    Fingerprint = DSP Features (V1) ⊕ BEATs Embedding (V2, 768-dim, frozen)

The DSP pathway from Version 1 is never removed; BEATs is strictly additive.

Public API (available after implementation):
    BEATsEncoder   — loads the pretrained model and encodes waveforms
    BEATsEmbedding — dataclass holding one 768-dim embedding with metadata
"""

from .embedding import BEATsEmbedding
from .encoder import BEATsEncoder

__all__ = ["BEATsEncoder", "BEATsEmbedding"]
