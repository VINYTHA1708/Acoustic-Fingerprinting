"""BEATsEncoder — wraps the official Microsoft BEATs pretrained audio encoder.

SDD v4 §4.1, §5:
    BEATs is used as a frozen pretrained encoder producing a 768-dim embedding
    per audio clip. It is never fine-tuned.

Integration:
    - Official source: third_party/beats/
    - Checkpoint:      models/beats/BEATs_iter3_plus_AS2M.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make third_party/beats importable without modifying those files
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "beats"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from BEATs import BEATs, BEATsConfig  # noqa: E402  (third-party import)

from .embedding import BEATsEmbedding
from .utils import mean_pool_frames, resolve_checkpoint_path, validate_waveform

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768
_EXPECTED_SAMPLE_RATE = 16_000


class BEATsEncoder:
    """Frozen BEATs encoder that maps a waveform to a 768-dim embedding.

    The model weights are loaded once at construction time and never updated.
    All gradient computation is disabled for the BEATs backbone.

    Args:
        checkpoint_path: Path to the pretrained BEATs checkpoint (.pt file).
                         Expected location: models/beats/<checkpoint>.pt

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint is incompatible with the BEATs model.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        self._checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self._model: BEATs = self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, waveform: np.ndarray, sample_rate: int, filename: str = "") -> BEATsEmbedding:
        """Encode a waveform into a 768-dim BEATs embedding.

        Args:
            waveform: Mono float32 waveform, shape ``(T,)``, at 16 kHz.
                      Pass the output of PreprocessingPipeline directly.
            sample_rate: Sample rate of the waveform. Must be 16 000 Hz.
            filename: Source filename for provenance metadata.

        Returns:
            :class:`BEATsEmbedding` with the 768-dim mean-pooled vector.

        Raises:
            ValueError: If ``waveform`` is not 1-D float32 or ``sample_rate`` != 16 000.
        """
        validate_waveform(waveform, sample_rate)

        audio_tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, T)
        padding_mask = torch.zeros(1, waveform.shape[0], dtype=torch.bool)

        with torch.no_grad():
            frame_embeddings, _ = self._model.extract_features(audio_tensor, padding_mask=padding_mask)

        # frame_embeddings: (1, T_frames, 768) → (T_frames, 768)
        frames_np = frame_embeddings.squeeze(0).cpu().numpy()
        vector = mean_pool_frames(frames_np)  # (768,) float32

        return BEATsEmbedding(
            vector=vector,
            embedding_dim=_EMBEDDING_DIM,
            filename=filename,
            machine_type="",
            machine_id="",
            sample_rate=sample_rate,
        )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the BEATs output embedding (always 768)."""
        return _EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> BEATs:
        """Load the pretrained BEATs model from the checkpoint file.

        Returns:
            A BEATs model in eval mode with gradients disabled.

        Raises:
            FileNotFoundError: If the checkpoint does not exist (via resolve_checkpoint_path).
            RuntimeError: If the checkpoint keys are incompatible with the model.
        """
        try:
            checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BEATs checkpoint from {self._checkpoint_path}: {exc}"
            ) from exc

        if "cfg" not in checkpoint or "model" not in checkpoint:
            raise RuntimeError(
                f"Incompatible checkpoint at {self._checkpoint_path}: "
                "expected keys 'cfg' and 'model'."
            )

        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)

        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint state dict is incompatible with BEATs model: {exc}"
            ) from exc

        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        logger.info(
            "BEATs checkpoint loaded: %s | embedding_dim=%d",
            self._checkpoint_path.name,
            _EMBEDDING_DIM,
        )
        return model
