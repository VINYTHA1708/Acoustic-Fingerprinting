"""Feature Fusion module — Version 2.

Combines DSP features (Version 1) with BEATs embeddings (Version 2) into a
single Fusion Fingerprint vector per recording.

SDD v4 §4.1:
    Fingerprint = DSP Features ⊕ BEATs Embedding (768-dim, frozen)
    DSP features always appear first.

Public API:
    FusionBuilder       — builds a FusedFeatureVector from DSP + BEATs inputs
    FusedFeatureVector  — dataclass holding the fused vector with metadata
    FusedVectorSerializer — save/load to JSON and NPZ
"""

from .cache import FusionCache
from .fused_vector import FusedFeatureVector
from .fusion import FusionBuilder
from .serializer import FusedVectorSerializer

__all__ = ["FusionBuilder", "FusedFeatureVector", "FusedVectorSerializer", "FusionCache"]
