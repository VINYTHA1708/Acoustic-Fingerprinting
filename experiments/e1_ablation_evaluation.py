"""Experiment E1 — Ablation Study Evaluation.

Evaluates all five configurations defined in e1_ablation_definition.py using
the identical dataset, split protocol (train_ratio=0.70, profile_ratio=0.15,
seed=42), machine IDs, and metrics as Phase 7.

Each configuration modifies the pipeline as specified in Phase 8.1:

    FM_full_method    -- 921-dim FusionCache + trained ProjectionHead -> 256-dim
    A1_no_beats       -- 153-dim DSP-only + freshly trained head (153->256)
    A2_no_dsp         -- 768-dim BEATs-only + freshly trained head (768->256)
    A3_no_contrastive -- 921-dim FusionCache + random (untrained) head -> 256-dim
    A4_no_projection  -- 921-dim FusionCache, raw vector scored directly (921-dim)

All five configurations read from the existing fusion cache (NPZ disk hits).
A1 reads dsp_feature_vector; A2 reads beats_embedding; FM/A3/A4 read
fused_feature_vector.  No raw audio or BEATs inference is performed at
evaluation time.

Results are saved to:
    experiments/results/e1/ablation_study/ablation_results.csv

Usage:
    python experiments/e1_ablation_evaluation.py
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.e1_ablation_definition import ABLATIONS, PROTOCOL, get_all_ablations
from experiments.e1_baseline_evaluation import compute_auroc, compute_separation_ratio
from src.beats.encoder import BEATsEncoder
from src.contrastive_learning.loss import NTXentLoss
from src.contrastive_learning.model import ProjectionHead
from src.contrastive_learning.serializer import ContrastiveSerializer
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata
from src.dataset.split import DatasetSplitter
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.preprocessing.pipeline import PreprocessingPipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("data/raw/MIMII")
BEATS_CHECKPOINT = Path("models/beats/BEATs_iter3_plus_AS2M.pt")
CONTRASTIVE_CHECKPOINT = Path("models/contrastive/e1/best_projection_head.pt")
CACHE_ROOT = Path("data/fusion_cache")
RESULTS_PATH = Path("experiments/results/e1/ablation_study/ablation_results.csv")

CSV_COLUMNS = [
    "ablation_id",
    "ablation_name",
    "machine_id",
    "n_normal",
    "n_abnormal",
    "auroc",
    "separation_ratio",
]

# Training hyper-parameters for A1 / A2 (match e1_train.py)
_EPOCHS = 20
_BATCH_SIZE = 16
_LR = 1e-3
_TEMPERATURE = 0.07
_OUTPUT_DIM = 256
_SEED = 42


# ---------------------------------------------------------------------------
# Flexible projection head (used by A1, A2, and A3)
# ---------------------------------------------------------------------------

class _FlexProjectionHead(nn.Module):
    """Minimal projection head with configurable input dimension.

    Architecture mirrors ProjectionHead but accepts any input_dim.
    Used for A1 (input_dim=153), A2 (input_dim=768), A3 (input_dim=921).

    Args:
        input_dim: Dimensionality of the input vector.
        output_dim: Dimensionality of the L2-normalised output (default 256).
    """

    def __init__(self, input_dim: int, output_dim: int = _OUTPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared infrastructure builders
# ---------------------------------------------------------------------------

def _build_fusion_cache(encoder: BEATsEncoder) -> FusionCache:
    pipeline = PreprocessingPipeline(target_sr=16_000)
    extractor = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    return FusionCache(
        cache_root=CACHE_ROOT,
        pipeline=pipeline,
        extractor=extractor,
        vec_builder=vec_builder,
        encoder=encoder,
        fusion=FusionBuilder(),
    )


# ---------------------------------------------------------------------------
# Per-recording feature extractors
# ---------------------------------------------------------------------------

def _vec_fm_a3_a4(rec: AudioMetadata, cache: FusionCache) -> np.ndarray:
    """Return the full 921-dim fused vector (FM / A3 / A4)."""
    return cache.load_or_create(rec).fused_feature_vector


def _vec_a1_dsp(rec: AudioMetadata, cache: FusionCache) -> np.ndarray:
    """Return the 153-dim DSP-only vector (A1).

    Reads dsp_feature_vector directly from the existing fusion cache NPZ.
    The FusedFeatureVector already contains DSP and BEATs as separate fields,
    so no raw audio processing or BEATs inference is needed.
    """
    return cache.load_or_create(rec).dsp_feature_vector


def _vec_a2_beats(rec: AudioMetadata, cache: FusionCache) -> np.ndarray:
    """Return the 768-dim BEATs-only vector (A2).

    Reads beats_embedding directly from the existing fusion cache NPZ.
    """
    return cache.load_or_create(rec).beats_embedding


# ---------------------------------------------------------------------------
# Contrastive training for A1 / A2
# ---------------------------------------------------------------------------

def _train_flex_head(
    input_dim: int,
    train_vecs: list[np.ndarray],
    machine_ids: list[str],
) -> _FlexProjectionHead:
    """Train a _FlexProjectionHead on pre-computed feature vectors.

    Constructs positive pairs (same machine_id, different recording) and
    trains with NT-Xent loss, mirroring the ContrastiveTrainer logic.

    Args:
        input_dim: Input dimensionality (153 for A1, 768 for A2).
        train_vecs: Feature vectors for all train_normal recordings.
        machine_ids: Corresponding machine_id for each vector.

    Returns:
        Trained _FlexProjectionHead in eval mode.
    """
    _set_seeds(_SEED)

    head = _FlexProjectionHead(input_dim=input_dim, output_dim=_OUTPUT_DIM)
    criterion = NTXentLoss(temperature=_TEMPERATURE)
    optimizer = optim.Adam(head.parameters(), lr=_LR)

    # Group vectors by machine_id
    groups: dict[str, list[np.ndarray]] = {}
    for vec, mid in zip(train_vecs, machine_ids):
        groups.setdefault(mid, []).append(vec)

    # Build positive pairs: for each recording, pick a random same-machine partner
    rng = random.Random(_SEED)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for mid, vecs in groups.items():
        if len(vecs) < 2:
            continue
        for i, anchor in enumerate(vecs):
            pool = [v for j, v in enumerate(vecs) if j != i]
            paired = rng.choice(pool)
            pairs.append((anchor, paired))

    if len(pairs) < 2:
        head.eval()
        return head

    best_loss = math.inf
    best_state = {k: v.clone() for k, v in head.state_dict().items()}

    for epoch in range(1, _EPOCHS + 1):
        head.train()
        rng.shuffle(pairs)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(pairs) - 1, _BATCH_SIZE):
            batch = pairs[start : start + _BATCH_SIZE]
            if len(batch) < 2:
                continue

            anchors = torch.from_numpy(np.stack([p[0] for p in batch])).float()
            paired_t = torch.from_numpy(np.stack([p[1] for p in batch])).float()

            optimizer.zero_grad()
            emb_a = head(anchors)
            emb_b = head(paired_t)
            loss = criterion(emb_a, emb_b)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            mean_loss = epoch_loss / n_batches
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if epoch % 5 == 0 or epoch == _EPOCHS:
            print(f"    epoch {epoch}/{_EPOCHS}  loss={mean_loss:.4f}", flush=True)

    head.load_state_dict(best_state)
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Embedding generator
# ---------------------------------------------------------------------------

def _project(vec: np.ndarray, head: nn.Module) -> np.ndarray:
    """Pass a vector through a projection head and return a numpy array."""
    with torch.no_grad():
        t = torch.from_numpy(vec).float()
        return head(t).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Per-ablation evaluation
# ---------------------------------------------------------------------------

def evaluate_ablation_machine_id(
    ablation_id: str,
    machine_id: str,
    profile_records: list[AudioMetadata],
    test_records: list[tuple[AudioMetadata, str]],
    *,
    cache: FusionCache | None = None,
    fm_head: ProjectionHead | None = None,
    a1_head: _FlexProjectionHead | None = None,
    a2_head: _FlexProjectionHead | None = None,
    a3_head: _FlexProjectionHead | None = None,
) -> dict:
    """Evaluate one ablation configuration on one machine ID.

    All five configurations read from the fusion cache (NPZ disk hits).
    A1 reads dsp_feature_vector; A2 reads beats_embedding; FM/A3/A4 read
    fused_feature_vector.  No raw audio or BEATs inference is performed.

    Args:
        ablation_id: One of the five registered ablation IDs.
        machine_id: Machine identifier (e.g. ``"id_00"``).
        profile_records: Normal recordings used to build the profile mean.
        test_records: List of ``(AudioMetadata, label)`` tuples for scoring.
        cache: FusionCache shared across all configurations.
        fm_head: Trained ProjectionHead for FM.
        a1_head: Trained _FlexProjectionHead (153->256) for A1.
        a2_head: Trained _FlexProjectionHead (768->256) for A2.
        a3_head: Random (untrained) _FlexProjectionHead (921->256) for A3.

    Returns:
        Result dict with keys matching ``CSV_COLUMNS``.
    """
    def _get_vec(rec: AudioMetadata) -> np.ndarray:
        if ablation_id == "FM_full_method":
            return _project(_vec_fm_a3_a4(rec, cache), fm_head)
        if ablation_id == "A1_no_beats":
            return _project(_vec_a1_dsp(rec, cache), a1_head)
        if ablation_id == "A2_no_dsp":
            return _project(_vec_a2_beats(rec, cache), a2_head)
        if ablation_id == "A3_no_contrastive":
            return _project(_vec_fm_a3_a4(rec, cache), a3_head)
        # A4_no_projection -- raw fusion vector, no head
        return _vec_fm_a3_a4(rec, cache)

    profile_vecs = [_get_vec(r) for r in profile_records]
    profile_mean = np.mean(profile_vecs, axis=0).astype(np.float32)

    scores, labels = [], []
    for rec, label in test_records:
        vec = _get_vec(rec)
        scores.append(float(np.linalg.norm(vec - profile_mean)))
        labels.append(1 if label == "abnormal" else 0)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    normal_scores = scores_arr[labels_arr == 0]
    abnormal_scores = scores_arr[labels_arr == 1]

    auroc = compute_auroc(scores_arr, labels_arr)
    sep = compute_separation_ratio(normal_scores, abnormal_scores)

    return {
        "ablation_id": ablation_id,
        "ablation_name": ABLATIONS[ablation_id].name,
        "machine_id": machine_id,
        "n_normal": int((labels_arr == 0).sum()),
        "n_abnormal": int((labels_arr == 1).sum()),
        "auroc": round(auroc, 4),
        "separation_ratio": round(sep, 4),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_inputs() -> None:
    """Raise FileNotFoundError if required resources are missing."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"MIMII dataset not found: {DATASET_ROOT}")
    if not BEATS_CHECKPOINT.exists():
        raise FileNotFoundError(f"BEATs checkpoint not found: {BEATS_CHECKPOINT}")
    if not CONTRASTIVE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Contrastive checkpoint not found: {CONTRASTIVE_CHECKPOINT}\n"
            "Run experiments/e1_train.py first."
        )


def validate_results(rows: list[dict]) -> None:
    """Raise ValueError if any result row has invalid metric values."""
    for row in rows:
        auroc = row["auroc"]
        sep = row["separation_ratio"]
        if not (isinstance(auroc, float) and (0.0 <= auroc <= 1.0 or auroc != auroc)):
            raise ValueError(
                f"Invalid AUROC={auroc} for {row['ablation_id']}/{row['machine_id']}"
            )
        if not (isinstance(sep, float) and (sep >= 0.0 or sep != sep)):
            raise ValueError(
                f"Invalid separation_ratio={sep} for {row['ablation_id']}/{row['machine_id']}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    validate_inputs()
    _set_seeds(_SEED)

    proto = PROTOCOL
    loader = DatasetLoader(DATASET_ROOT)
    all_recordings = [
        r for r in loader.get_all_files()
        if r.machine_type == proto.machine_type
        and r.machine_id in proto.machine_ids
    ]

    splitter = DatasetSplitter(
        train_ratio=proto.train_ratio,
        profile_ratio=proto.profile_ratio,
        seed=proto.seed,
    )
    split = splitter.split(all_recordings)

    # ------------------------------------------------------------------
    # Build shared infrastructure (one BEATsEncoder, one FusionCache)
    # ------------------------------------------------------------------
    encoder = BEATsEncoder(BEATS_CHECKPOINT)
    cache = _build_fusion_cache(encoder)

    # FM -- load trained checkpoint into standard ProjectionHead
    fm_head = ProjectionHead()
    ckpt = ContrastiveSerializer.load_checkpoint(CONTRASTIVE_CHECKPOINT)
    fm_head.load_state_dict(ckpt["model_state_dict"])
    fm_head.eval()

    # A3 -- random (untrained) head on 921-dim input
    _set_seeds(_SEED)
    a3_head = _FlexProjectionHead(input_dim=921, output_dim=_OUTPUT_DIM)
    a3_head.eval()

    # A1 -- read dsp_feature_vector from cache, then train flex head (153->256)
    print("\n[A1] Loading DSP vectors from fusion cache for train_normal ...")
    a1_train_vecs = [_vec_a1_dsp(r, cache) for r in split.train_normal]
    a1_machine_ids = [r.machine_id for r in split.train_normal]
    print(f"[A1] Training flex head (153->256) on {len(a1_train_vecs)} vectors ...")
    a1_head = _train_flex_head(153, a1_train_vecs, a1_machine_ids)

    # A2 -- read beats_embedding from cache, then train flex head (768->256)
    print("\n[A2] Loading BEATs vectors from fusion cache for train_normal ...")
    a2_train_vecs = [_vec_a2_beats(r, cache) for r in split.train_normal]
    a2_machine_ids = [r.machine_id for r in split.train_normal]
    print(f"[A2] Training flex head (768->256) on {len(a2_train_vecs)} vectors ...")
    a2_head = _train_flex_head(768, a2_train_vecs, a2_machine_ids)

    # ------------------------------------------------------------------
    # Evaluate all five configurations
    # ------------------------------------------------------------------
    rows: list[dict] = []

    print("\n" + "=" * 65)
    print("Experiment E1 -- Ablation Study Evaluation")
    print("=" * 65)
    print(f"Machine type : {proto.machine_type}")
    print(f"Machine IDs  : {list(proto.machine_ids)}")
    print()

    for ablation in get_all_ablations():
        aid = ablation.ablation_id
        print(f"[{aid}] {ablation.name}")

        for machine_id in proto.machine_ids:
            profile_records = [
                r for r in split.profile_normal if r.machine_id == machine_id
            ]
            test_records = (
                [(r, "normal") for r in split.test_normal if r.machine_id == machine_id]
                + [(r, "abnormal") for r in split.test_abnormal if r.machine_id == machine_id]
            )

            print(
                f"  {machine_id} -- profile={len(profile_records)}  "
                f"test_normal={sum(1 for _, l in test_records if l == 'normal')}  "
                f"test_abnormal={sum(1 for _, l in test_records if l == 'abnormal')}",
                end="  ... ",
                flush=True,
            )

            row = evaluate_ablation_machine_id(
                ablation_id=aid,
                machine_id=machine_id,
                profile_records=profile_records,
                test_records=test_records,
                cache=cache,
                fm_head=fm_head,
                a1_head=a1_head,
                a2_head=a2_head,
                a3_head=a3_head,
            )
            rows.append(row)
            print(f"AUROC={row['auroc']:.4f}  sep={row['separation_ratio']:.4f}")

        print()

    validate_results(rows)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 65)
    print("Results saved to:", RESULTS_PATH)
    print("=" * 65)
    print()
    print(f"{'Ablation':<24} {'Machine':<8} {'N_norm':>7} {'N_abn':>6} {'AUROC':>7} {'Sep':>7}")
    print("-" * 65)
    for row in rows:
        print(
            f"{row['ablation_id']:<24} {row['machine_id']:<8} "
            f"{row['n_normal']:>7} {row['n_abnormal']:>6} "
            f"{row['auroc']:>7.4f} {row['separation_ratio']:>7.4f}"
        )


if __name__ == "__main__":
    main()
