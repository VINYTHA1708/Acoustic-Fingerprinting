"""Learned Fingerprint Profile module — Version 3.

Builds a healthy learned profile from a trained ProjectionHead by running
every normal recording through the full pipeline:
    Audio → Preprocessing → DSP → BEATs → Fusion → ProjectionHead → 256-dim embedding

Public API:
    LearnedFingerprintProfile  — dataclass holding embeddings + statistics
    LearnedProfileBuilder      — builds the profile from a checkpoint + DatasetLoader
    LearnedProfileSerializer   — save/load to JSON and NPZ
"""

from .learned_profile import LearnedFingerprintProfile
from .builder import LearnedProfileBuilder
from .serializer import LearnedProfileSerializer

__all__ = [
    "LearnedFingerprintProfile",
    "LearnedProfileBuilder",
    "LearnedProfileSerializer",
]
