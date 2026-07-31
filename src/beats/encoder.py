"""BEATsEncoder — scaffold for the Microsoft BEATs pretrained audio encoder.

SDD v4 §4.1, §5:
    BEATs is used as a frozen pretrained encoder producing a 768-dim embedding
    per audio clip. It is never fine-tuned; only the small contrastive head
    trained in Version 3 is learnable.

Integration plan (Version 2):
    1. Place the official Microsoft BEATs source files under third_party/beats/.
       Repository: https://github.com/microsoft/unilm/tree/master/beats
       Required files: BEATs.py, backbone.py, modules.py, tokenizers.py

    2. Download the pretrained checkpoint (BEATs_iter3_plus_AS2M.pt or equivalent)
       and place it under models/beats/.
       Do NOT commit the checkpoint to version control.

    3. Implement the _load_model() and encode() methods below.

    4. Update src/beats/__init__.py to export BEATsEncoder.

This file is a scaffold only. No inference code is written here yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# TODO (V2): import BEATs from third_party once the source files are in place
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "third_party" / "beats"))
# from BEATs import BEATs, BEATsConfig

_EMBEDDING_DIM = 768  # BEATs output dimensionality (fixed by the pretrained model)
_EXPECTED_SAMPLE_RATE = 16_000  # BEATs expects 16 kHz mono audio


class BEATsEncoder:
    """Frozen BEATs encoder that maps a waveform to a 768-dim embedding.

    The model weights are loaded once at construction time and never updated.
    All gradient computation is disabled for the BEATs backbone.

    Args:
        checkpoint_path: Path to the pretrained BEATs checkpoint (.pt file).
                         Expected location: models/beats/<checkpoint>.pt

    Example (once implemented)::

        encoder = BEATsEncoder("models/beats/BEATs_iter3_plus_AS2M.pt")
        embedding = encoder.encode(waveform, sample_rate=16000)
        # embedding.vector.shape == (768,)
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._model = None  # populated by _load_model() once implemented

        # TODO (V2): call self._load_model() here once the implementation is ready
        # self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, waveform: np.ndarray, sample_rate: int) -> "BEATsEmbedding":  # noqa: F821
        """Encode a waveform into a 768-dim BEATs embedding.

        Args:
            waveform: Mono audio waveform as a float32 numpy array, shape ``(T,)``.
                      Must be sampled at 16 kHz (use PreprocessingPipeline first).
            sample_rate: Sample rate of the waveform. Must equal 16 000 Hz.

        Returns:
            A :class:`BEATsEmbedding` containing the 768-dim feature vector.

        Raises:
            ValueError: If ``sample_rate`` is not 16 000 Hz.
            RuntimeError: If the model has not been loaded yet.

        TODO (V2): implement this method.
            Steps:
            1. Validate sample_rate == _EXPECTED_SAMPLE_RATE.
            2. Convert waveform to a torch.Tensor of shape (1, T).
            3. Pass through self._model with torch.no_grad().
            4. Pool the frame-level outputs to a single 768-dim vector
               (mean pooling over the time dimension is the standard approach).
            5. Convert to float32 numpy array.
            6. Return BEATsEmbedding(vector=..., embedding_dim=768).
        """
        # TODO (V2): replace this stub with the real implementation
        raise NotImplementedError(
            "BEATsEncoder.encode() is not yet implemented. "
            "Complete the TODO steps in encoder.py (Version 2)."
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the BEATs output embedding (always 768)."""
        return _EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """True if the pretrained model has been loaded from the checkpoint."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the pretrained BEATs model from the checkpoint file.

        TODO (V2): implement this method.
            Steps:
            1. Verify self._checkpoint_path exists; raise FileNotFoundError if not.
            2. Load the checkpoint dict with torch.load(..., map_location="cpu").
            3. Instantiate BEATsConfig from the checkpoint's cfg entry.
            4. Instantiate BEATs(cfg) and call .load_state_dict().
            5. Call .eval() and wrap with torch.no_grad() context where needed.
            6. Assign to self._model.
            7. Log the checkpoint path and embedding dim at INFO level.
        """
        # TODO (V2): replace this stub with the real implementation
        raise NotImplementedError(
            "BEATsEncoder._load_model() is not yet implemented. "
            "Complete the TODO steps in encoder.py (Version 2)."
        )
