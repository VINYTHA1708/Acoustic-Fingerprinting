"""Machine Health Pipeline module — Version 3.

End-to-end pipeline that runs a single recording through the full
FusionCache → ContrastiveInference → LearnedDrift → LearnedHealth chain
and returns a MachineHealthReport.

No preprocessing, DSP, BEATs, fusion, inference, drift, or health logic
is duplicated. All computation is delegated to the existing modules.

Public API:
    MachineHealthReport   — dataclass holding all metrics and health fields
    MachineHealthPipeline — orchestrates the full pipeline for one recording
"""

from .result import MachineHealthReport
from .pipeline import MachineHealthPipeline

__all__ = [
    "MachineHealthReport",
    "MachineHealthPipeline",
]
