"""Phase 19 — Controlled BEATs-only vs Full-Method comparison.

Evaluates two conditions on IDENTICAL test recordings across all 16 machine IDs:

  Full method  : Phase 9 checkpoint  (DSP + BEATs, 921→256)
  BEATs-only   : Phase 19 checkpoint (BEATs only,  768→256)

Both conditions use the same split (train_ratio=0.70, profile_ratio=0.15,
seed=42), the same profile_normal recordings, and the same test recordings.

Reports:
  - ROC-AUC per machine ID
  - Mean ROC-AUC per machine type
  - Overall mean ROC-AUC
  - Side-by-side comparison table

Results saved to: experiments/results/phase19/

Usage:
    python experiments/phase19_evaluate.py
    python experiments/phase19_evaluate.py --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.serializer import ContrastiveSerializer
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.fusion.cache import FusionCache
from src.fusion.fusion import FusionBuilder
from src.beats.encoder import BEATsEncoder
from src.feature_extraction.extractor import FeatureExtractor
from src.feature_extraction.feature_vector import FeatureVectorBuilder
from src.preprocessing.pipeline import PreprocessingPipeline
from src.learned_drift.metrics import LearnedDriftMetrics
from src.learned_health_index.calculator import LearnedHealthCalculator
from src.learned_profile.learned_profile import LearnedFingerprintProfile
from src.learned_profile.builder import LearnedProfileBuilder
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.dataset.metadata import AudioMetadata

# ---------------------------------------------------------------------------
# Constants — identical split parameters to phase9_train.py / phase9_evaluate.py
# ---------------------------------------------------------------------------

EXPERIMENT_ID   = "phase19"
DATASET_ROOT    = Path("data/raw/MIMII")
CACHE_ROOT      = Path("data/fusion_cache")
BEATS_CKPT      = Path("models/beats/BEATs_iter3_plus_AS2M.pt")

PHASE9_CKPT     = Path("models/contrastive/phase9/best_projection_head.pt")
PHASE19_CKPT    = Path("models/contrastive/phase19/best_projection_head_beats_only.pt")

RESULTS_DIR     = Path("experiments/results/phase19")
PROFILE_DIR     = RESULTS_DIR / "profiles"

MACHINE_TYPES   = ["fan", "pump", "slider", "valve"]
MACHINE_IDS     = ["id_00", "id_02", "id_04", "id_06"]

TRAIN_RATIO     = 0.70
PROFILE_RATIO   = 0.15
SEED            = 42

BEATS_DIM       = 768
PROJ_DIM        = 256

CSV_COLUMNS = [
    "machine_type", "machine_id", "filename", "true_label",
    "normalized_euclidean", "normalized_manhattan", "normalized_cosine",
    "health_score",
]


# ---------------------------------------------------------------------------
# BEATs-only ProjectionHead  (must match phase19_beats_only_train.py)
# ---------------------------------------------------------------------------

class _BeatsOnlyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BEATS_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, PROJ_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# BEATs-only inference helpers
# ---------------------------------------------------------------------------

class _BeatsOnlyInference:
    """Loads the Phase 19 checkpoint and projects a 768-dim BEATs vector → 256-dim."""

    def __init__(self, checkpoint_path: Path) -> None:
        ckpt = ContrastiveSerializer.load_checkpoint(checkpoint_path)
        self._head = _BeatsOnlyHead()
        self._head.load_state_dict(ckpt["model_state_dict"])
        self._head.eval()

    def embed(self, beats_vector: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(beats_vector.astype(np.float32))
        with torch.no_grad():
            out = self._head(x)
        return out.numpy().astype(np.float32)


class _BeatsOnlyProfileBuilder:
    """Builds a LearnedFingerprintProfile using only the BEATs embedding."""

    def __init__(self, inference: _BeatsOnlyInference, cache: FusionCache) -> None:
        self._inf   = inference
        self._cache = cache
        self._metrics = LearnedDriftMetrics()

    def build(
        self,
        machine_type: str,
        machine_id: str,
        recordings: list[AudioMetadata],
    ) -> LearnedFingerprintProfile:
        embeddings = []
        for rec in recordings:
            try:
                fused = self._cache.load_or_create(rec)
                emb   = self._inf.embed(fused.beats_embedding)
                embeddings.append(emb)
            except Exception as exc:
                print(f"  [WARN] skipping {rec.filename}: {exc}")

        if not embeddings:
            raise ValueError(f"No embeddings for {machine_type}/{machine_id}")

        matrix   = np.stack(embeddings).astype(np.float32)
        mean_vec = matrix.mean(axis=0)
        std_vec  = matrix.std(axis=0)
        return LearnedFingerprintProfile(
            machine_type=machine_type,
            machine_id=machine_id,
            embedding_dimension=PROJ_DIM,
            embeddings=matrix,
            mean_vector=mean_vec,
            std_vector=std_vec,
        )

    def analyze(
        self,
        rec: AudioMetadata,
        profile: LearnedFingerprintProfile,
    ) -> dict:
        fused = self._cache.load_or_create(rec)
        emb   = self._inf.embed(fused.beats_embedding)
        (
            cosine, euclid, manhat,
            _z, _abs_diff,
            norm_euclid, norm_manhat, norm_cosine, _norm_vec,
        ) = self._metrics.compute(emb, profile)

        mu_norm, sigma_norm = _profile_norm_stats(profile)
        calc = LearnedHealthCalculator()
        health_score, _, _ = calc.calculate(
            normalized_euclidean=norm_euclid,
            normalized_manhattan=norm_manhat,
            normalized_cosine=norm_cosine,
            profile_healthy_norm=mu_norm,
            profile_healthy_norm_std=sigma_norm,
        )
        return {
            "normalized_euclidean": norm_euclid,
            "normalized_manhattan": norm_manhat,
            "normalized_cosine":    norm_cosine,
            "health_score":         health_score,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STD_FLOOR = 1e-10


def _profile_norm_stats(profile: LearnedFingerprintProfile) -> tuple[float, float]:
    mean     = profile.mean_vector.astype(np.float32)
    std      = profile.std_vector.astype(np.float32)
    safe_std = np.where(std < _STD_FLOOR, 1.0, std)
    z        = np.where(
        std < _STD_FLOOR, 0.0,
        (profile.embeddings.astype(np.float32) - mean) / safe_std,
    )
    norms = np.linalg.norm(z, axis=1)
    return float(norms.mean()), float(norms.std())


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_true, scores))
        return auc if auc >= 0.5 else float(roc_auc_score(y_true, -scores))
    except Exception:
        return float("nan")


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-12 else 0.0


def _compute_metrics(rows: list[dict]) -> dict:
    """ROC-AUC and Cohen's d for normalized_euclidean (primary metric)."""
    if not rows:
        return {}
    labels = np.array([1 if r["true_label"] == "abnormal" else 0 for r in rows])
    if len(set(labels)) < 2:
        return {}
    scores = np.array([float(r["normalized_euclidean"]) for r in rows])
    auc    = _roc_auc(labels, scores)
    normal_scores   = scores[labels == 0]
    abnormal_scores = scores[labels == 1]
    d = _cohens_d(abnormal_scores, normal_scores)
    return {"roc_auc": round(auc, 6), "cohens_d": round(d, 6)}


def _validate_isolation(split) -> None:
    train_paths   = {r.absolute_path for r in split.train_normal}
    profile_paths = {r.absolute_path for r in split.profile_normal}
    test_n_paths  = {r.absolute_path for r in split.test_normal}
    for overlap, label in [
        (train_paths & profile_paths,  "train_normal ∩ profile_normal"),
        (train_paths & test_n_paths,   "train_normal ∩ test_normal"),
        (profile_paths & test_n_paths, "profile_normal ∩ test_normal"),
    ]:
        if overlap:
            raise ValueError(f"ISOLATION FAIL: {label} ({len(overlap)} files)")


# ---------------------------------------------------------------------------
# Full-method evaluation (reuses LearnedProfileBuilder + LearnedHealthAnalyzer)
# ---------------------------------------------------------------------------

def _run_full_method(
    splits: dict,
    smoke_test: bool,
) -> list[dict]:
    """Evaluate Phase 9 full method; returns flat list of result dicts."""
    print("\n" + "=" * 60)
    print("CONDITION: Full Method (DSP + BEATs, 921→256, Phase 9 checkpoint)")
    print("=" * 60)

    builder  = LearnedProfileBuilder(checkpoint_path=PHASE9_CKPT)
    analyzer = LearnedHealthAnalyzer(checkpoint_path=PHASE9_CKPT)

    profiles: dict[tuple[str, str], LearnedFingerprintProfile] = {}
    full_dir = PROFILE_DIR / "full_method"
    full_dir.mkdir(parents=True, exist_ok=True)

    # Build profiles
    for mt in MACHINE_TYPES:
        split = splits[mt]
        for mid in MACHINE_IDS:
            recs = [r for r in split.profile_normal if r.machine_id == mid]
            if not recs:
                continue
            if smoke_test:
                recs = recs[:1]
            print(f"  Building profile: {mt}/{mid}  ({len(recs)} recs)")
            profile = builder.build(mt, mid, recordings=recs)
            profiles[(mt, mid)] = profile

    # Evaluate
    rows: list[dict] = []
    for mt in MACHINE_TYPES:
        split = splits[mt]
        test_recs = (
            [(r, "normal")   for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        if smoke_test:
            normal_sample   = []
            abnormal_sample = []
            for mid in MACHINE_IDS:
                n = [r for r, _ in test_recs if r.machine_id == mid and _ == "normal"]
                a = [r for r, _ in test_recs if r.machine_id == mid and _ == "abnormal"]
                if n: normal_sample.append((n[0], "normal"))
                if a: abnormal_sample.append((a[0], "abnormal"))
            test_recs = normal_sample + abnormal_sample

        total = len(test_recs)
        for i, (rec, true_label) in enumerate(test_recs, 1):
            key = (rec.machine_type, rec.machine_id)
            if key not in profiles:
                continue
            if i % 200 == 0 or i == 1:
                print(f"  [{mt}] {i}/{total}")
            result = analyzer.analyze(rec, profiles[key])
            rows.append({
                "machine_type":        rec.machine_type,
                "machine_id":          rec.machine_id,
                "filename":            rec.filename,
                "true_label":          true_label,
                "normalized_euclidean": result.normalized_euclidean,
                "normalized_manhattan": result.normalized_manhattan,
                "normalized_cosine":    result.normalized_cosine,
                "health_score":         result.health_score,
            })

    return rows


# ---------------------------------------------------------------------------
# BEATs-only evaluation
# ---------------------------------------------------------------------------

def _run_beats_only(
    splits: dict,
    smoke_test: bool,
) -> list[dict]:
    """Evaluate Phase 19 BEATs-only method; returns flat list of result dicts."""
    print("\n" + "=" * 60)
    print("CONDITION: BEATs-only (768→256, Phase 19 checkpoint)")
    print("=" * 60)

    # Build shared FusionCache (reuses existing cache — no re-encoding)
    pipeline    = PreprocessingPipeline(target_sr=16_000)
    extractor   = FeatureExtractor(sample_rate=16_000)
    vec_builder = FeatureVectorBuilder()
    encoder     = BEATsEncoder(BEATS_CKPT)
    fusion      = FusionBuilder()
    cache       = FusionCache(
        cache_root=CACHE_ROOT,
        pipeline=pipeline,
        extractor=extractor,
        vec_builder=vec_builder,
        encoder=encoder,
        fusion=fusion,
    )

    inference      = _BeatsOnlyInference(PHASE19_CKPT)
    profile_builder = _BeatsOnlyProfileBuilder(inference, cache)

    profiles: dict[tuple[str, str], LearnedFingerprintProfile] = {}
    bo_dir = PROFILE_DIR / "beats_only"
    bo_dir.mkdir(parents=True, exist_ok=True)

    # Build profiles
    for mt in MACHINE_TYPES:
        split = splits[mt]
        for mid in MACHINE_IDS:
            recs = [r for r in split.profile_normal if r.machine_id == mid]
            if not recs:
                continue
            if smoke_test:
                recs = recs[:1]
            print(f"  Building profile: {mt}/{mid}  ({len(recs)} recs)")
            profile = profile_builder.build(mt, mid, recs)
            profiles[(mt, mid)] = profile

    # Evaluate
    rows: list[dict] = []
    for mt in MACHINE_TYPES:
        split = splits[mt]
        test_recs = (
            [(r, "normal")   for r in split.test_normal]
            + [(r, "abnormal") for r in split.test_abnormal]
        )
        if smoke_test:
            normal_sample   = []
            abnormal_sample = []
            for mid in MACHINE_IDS:
                n = [r for r, _ in test_recs if r.machine_id == mid and _ == "normal"]
                a = [r for r, _ in test_recs if r.machine_id == mid and _ == "abnormal"]
                if n: normal_sample.append((n[0], "normal"))
                if a: abnormal_sample.append((a[0], "abnormal"))
            test_recs = normal_sample + abnormal_sample

        total = len(test_recs)
        for i, (rec, true_label) in enumerate(test_recs, 1):
            key = (rec.machine_type, rec.machine_id)
            if key not in profiles:
                continue
            if i % 200 == 0 or i == 1:
                print(f"  [{mt}] {i}/{total}")
            metrics = profile_builder.analyze(rec, profiles[key])
            rows.append({
                "machine_type":        rec.machine_type,
                "machine_id":          rec.machine_id,
                "filename":            rec.filename,
                "true_label":          true_label,
                "normalized_euclidean": metrics["normalized_euclidean"],
                "normalized_manhattan": metrics["normalized_manhattan"],
                "normalized_cosine":    metrics["normalized_cosine"],
                "health_score":         metrics["health_score"],
            })

    return rows


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict]) -> dict:
    """Return per-id, per-type, and overall ROC-AUC dicts."""
    per_id:   dict[tuple[str, str], dict] = {}
    per_type: dict[str, dict]             = {}

    for mt in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            id_rows = [r for r in rows if r["machine_type"] == mt and r["machine_id"] == mid]
            if id_rows:
                per_id[(mt, mid)] = _compute_metrics(id_rows)

        type_rows = [r for r in rows if r["machine_type"] == mt]
        if type_rows:
            per_type[mt] = _compute_metrics(type_rows)

    overall = _compute_metrics(rows)
    return {"per_id": per_id, "per_type": per_type, "overall": overall}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_comparison(
    agg_full: dict,
    agg_bo:   dict,
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Phase 19 — Controlled BEATs-only vs Full-Method Comparison")
    lines.append("=" * 72)
    lines.append(f"Full method  : Phase 9 checkpoint  (DSP+BEATs, 921→256)")
    lines.append(f"BEATs-only   : Phase 19 checkpoint (BEATs only, 768→256)")
    lines.append(f"Split        : train=0.70  profile=0.15  seed=42")
    lines.append(f"Epochs       : 20  |  Batch: 16  |  LR: 0.001  |  T: 0.07")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Per-machine-ID ROC-AUC")
    lines.append("-" * 72)
    lines.append(f"  {'Machine':<18} {'Full AUC':>10} {'BEATs AUC':>10} {'Delta':>8}")
    lines.append("  " + "-" * 50)

    for mt in MACHINE_TYPES:
        for mid in MACHINE_IDS:
            key = (mt, mid)
            f_auc = agg_full["per_id"].get(key, {}).get("roc_auc", float("nan"))
            b_auc = agg_bo["per_id"].get(key,   {}).get("roc_auc", float("nan"))
            delta = f_auc - b_auc if not (np.isnan(f_auc) or np.isnan(b_auc)) else float("nan")
            f_s   = f"{f_auc:.4f}" if not np.isnan(f_auc) else "  N/A"
            b_s   = f"{b_auc:.4f}" if not np.isnan(b_auc) else "  N/A"
            d_s   = f"{delta:+.4f}" if not np.isnan(delta) else "  N/A"
            lines.append(f"  {mt}/{mid:<12} {f_s:>10} {b_s:>10} {d_s:>8}")
        lines.append("")

    lines.append("-" * 72)
    lines.append("Per-machine-type mean ROC-AUC")
    lines.append("-" * 72)
    lines.append(f"  {'Type':<18} {'Full AUC':>10} {'BEATs AUC':>10} {'Delta':>8}")
    lines.append("  " + "-" * 50)
    for mt in MACHINE_TYPES:
        f_auc = agg_full["per_type"].get(mt, {}).get("roc_auc", float("nan"))
        b_auc = agg_bo["per_type"].get(mt,   {}).get("roc_auc", float("nan"))
        delta = f_auc - b_auc if not (np.isnan(f_auc) or np.isnan(b_auc)) else float("nan")
        f_s   = f"{f_auc:.4f}" if not np.isnan(f_auc) else "  N/A"
        b_s   = f"{b_auc:.4f}" if not np.isnan(b_auc) else "  N/A"
        d_s   = f"{delta:+.4f}" if not np.isnan(delta) else "  N/A"
        lines.append(f"  {mt:<18} {f_s:>10} {b_s:>10} {d_s:>8}")
    lines.append("")

    lines.append("-" * 72)
    lines.append("Overall mean ROC-AUC (all 4 types combined)")
    lines.append("-" * 72)
    f_auc = agg_full["overall"].get("roc_auc", float("nan"))
    b_auc = agg_bo["overall"].get("roc_auc",   float("nan"))
    delta = f_auc - b_auc if not (np.isnan(f_auc) or np.isnan(b_auc)) else float("nan")
    lines.append(f"  Full method  : {f_auc:.4f}")
    lines.append(f"  BEATs-only   : {b_auc:.4f}")
    lines.append(f"  Delta (F-B)  : {delta:+.4f}")
    lines.append("")

    winner = "Full method" if delta > 0 else ("BEATs-only" if delta < 0 else "Tie")
    lines.append(f"  DSP contribution: {winner} has higher overall AUC")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print(report)
    return report


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _agg_to_serialisable(agg: dict) -> dict:
    """Convert tuple keys to strings for JSON serialisation."""
    return {
        "per_id":   {f"{k[0]}/{k[1]}": v for k, v in agg["per_id"].items()},
        "per_type": agg["per_type"],
        "overall":  agg["overall"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False) -> None:
    # Validate checkpoints
    if not PHASE9_CKPT.exists():
        raise FileNotFoundError(
            f"Phase 9 checkpoint not found: {PHASE9_CKPT}\n"
            "Run experiments/phase9_train.py first."
        )
    if not PHASE19_CKPT.exists():
        raise FileNotFoundError(
            f"Phase 19 checkpoint not found: {PHASE19_CKPT}\n"
            "Run experiments/phase19_beats_only_train.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment : {EXPERIMENT_ID}{'  [SMOKE TEST]' if smoke_test else ''}")
    print("=" * 60)

    # 1. Load recordings and reproduce the identical Phase 9 split
    loader   = DatasetLoader(DATASET_ROOT)
    all_recs = loader.get_all_files()

    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    splits   = {}
    for mt in MACHINE_TYPES:
        type_recs  = [r for r in all_recs if r.machine_type == mt]
        splits[mt] = splitter.split(type_recs)
        _validate_isolation(splits[mt])

    print(f"\n{'Type':<8} {'profile_n':>10} {'test_n':>8} {'test_ab':>8}")
    print("-" * 38)
    for mt in MACHINE_TYPES:
        s = splits[mt]
        print(f"{mt:<8} {len(s.profile_normal):>10} {len(s.test_normal):>8} "
              f"{len(s.test_abnormal):>8}")
    print()

    # 2. Run both conditions
    rows_full = _run_full_method(splits, smoke_test)
    rows_bo   = _run_beats_only(splits, smoke_test)

    # 3. Aggregate metrics
    agg_full = _aggregate(rows_full)
    agg_bo   = _aggregate(rows_bo)

    # 4. Save CSVs
    suffix = "_smoketest" if smoke_test else ""
    _save_csv(rows_full, RESULTS_DIR / f"evaluation_full_method{suffix}.csv")
    _save_csv(rows_bo,   RESULTS_DIR / f"evaluation_beats_only{suffix}.csv")

    # 5. Save summary JSON
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "smoke_test":    smoke_test,
        "split": {
            "train_ratio":   TRAIN_RATIO,
            "profile_ratio": PROFILE_RATIO,
            "seed":          SEED,
        },
        "checkpoints": {
            "full_method": str(PHASE9_CKPT),
            "beats_only":  str(PHASE19_CKPT),
        },
        "full_method":  _agg_to_serialisable(agg_full),
        "beats_only":   _agg_to_serialisable(agg_bo),
        "delta_full_minus_beats": {
            "per_id": {
                f"{k[0]}/{k[1]}": round(
                    agg_full["per_id"].get(k, {}).get("roc_auc", float("nan"))
                    - agg_bo["per_id"].get(k, {}).get("roc_auc", float("nan")),
                    6,
                )
                for k in agg_full["per_id"]
            },
            "per_type": {
                mt: round(
                    agg_full["per_type"].get(mt, {}).get("roc_auc", float("nan"))
                    - agg_bo["per_type"].get(mt, {}).get("roc_auc", float("nan")),
                    6,
                )
                for mt in MACHINE_TYPES
            },
            "overall": round(
                agg_full["overall"].get("roc_auc", float("nan"))
                - agg_bo["overall"].get("roc_auc", float("nan")),
                6,
            ),
        },
    }

    summary_path = RESULTS_DIR / f"comparison_summary{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # 6. Print and save report
    report = _print_comparison(agg_full, agg_bo)
    report_path = RESULTS_DIR / f"phase19_report{suffix}.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"\nOutputs written to {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 19 evaluation")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test (1 recording per split per machine ID).")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
