"""Phase 19 smoke test.

Verifies, without running full training:
  1. Both Phase 19 scripts import cleanly.
  2. Dataset loading, splitting, and BEATs-only pair creation work.
  3. Every pair's fused_feature_vector is exactly 768-dim after the swap.
  4. BeatsOnlyProjectionHead produces 256-dim L2-normalised embeddings.
  5. Phase 9 full-method can process one recording end-to-end.
  6. Phase 19 BEATs-only path can process one recording end-to-end
     (uses a randomly initialised head — no trained checkpoint required).

Does NOT modify any Phase 9, Phase 14, or E1 code or results.
Does NOT run the 20-epoch training loop.

Usage:
    python experiments/phase19_smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants (mirror phase19_beats_only_train.py)
# ---------------------------------------------------------------------------

DATASET_ROOT  = ROOT / "data" / "raw" / "MIMII"
CACHE_ROOT    = ROOT / "data" / "fusion_cache"
PHASE9_CKPT   = ROOT / "models" / "contrastive" / "phase9" / "best_projection_head.pt"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42

PASS = "  PASS"
FAIL = "  FAIL"


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Step 1 — import both Phase 19 modules
# ---------------------------------------------------------------------------

def step1_imports() -> bool:
    _section("Step 1: Import Phase 19 modules")
    ok = True
    for mod_path in [
        "experiments.phase19_beats_only_train",
        "experiments.phase19_evaluate",
    ]:
        try:
            import importlib
            importlib.import_module(mod_path)
            print(f"{PASS}  import {mod_path}")
        except Exception:
            print(f"{FAIL}  import {mod_path}")
            traceback.print_exc()
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Step 2 — dataset loading, splitting, BEATs-only pair creation, 768-dim check
# ---------------------------------------------------------------------------

def step2_dataset_and_pairs() -> tuple[bool, list | None]:
    _section("Step 2: Dataset loading, splitting, BEATs-only pair creation")
    try:
        from src.dataset.loader import DatasetLoader
        from src.dataset.split import DatasetSplitter
        from src.contrastive_learning.dataset import ContrastiveDataset
        from experiments.phase19_beats_only_train import _beats_only_pairs

        loader = DatasetLoader(DATASET_ROOT)
        all_recs = loader.get_all_files()
        print(f"  Total recordings loaded : {len(all_recs)}")
        assert len(all_recs) > 0, "No recordings found"
        print(f"{PASS}  DatasetLoader")

        splitter = DatasetSplitter(
            train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED
        )
        splits = {}
        for mt in MACHINE_TYPES:
            type_recs = [r for r in all_recs if r.machine_type == mt]
            splits[mt] = splitter.split(type_recs)
        print(f"{PASS}  DatasetSplitter (4 machine types)")

        # Use only pump/id_00 train_normal for speed
        pump_train = [
            r for r in splits["pump"].train_normal if r.machine_id == "id_00"
        ][:10]
        assert len(pump_train) >= 2, f"Need >=2 recordings, got {len(pump_train)}"
        print(f"  Using {len(pump_train)} pump/id_00 train_normal recordings for pair test")

        dataset = ContrastiveDataset(
            recordings=pump_train,
            cache_root=CACHE_ROOT,
            seed=SEED,
            val_split=0.20,
        )
        all_pairs = dataset.train_positive_pairs + dataset.val_positive_pairs
        assert len(all_pairs) > 0, "No pairs generated"
        print(f"  Pairs generated         : {len(all_pairs)}")
        print(f"{PASS}  ContrastiveDataset")

        # Swap to BEATs-only
        bo_pairs = _beats_only_pairs(all_pairs)
        assert len(bo_pairs) == len(all_pairs)

        # Verify every pair vector is 768-dim
        bad = [
            i for i, p in enumerate(bo_pairs)
            if p.anchor.fused_feature_vector.shape != (768,)
            or p.paired.fused_feature_vector.shape != (768,)
        ]
        if bad:
            print(f"{FAIL}  _beats_only_pairs — wrong shapes at indices {bad[:5]}")
            return False, None

        sample_dim = bo_pairs[0].anchor.fused_feature_vector.shape[0]
        print(f"  BEATs-only vector dim   : {sample_dim}")
        assert sample_dim == 768, f"Expected 768, got {sample_dim}"
        print(f"{PASS}  _beats_only_pairs — all {len(bo_pairs)} pairs are 768-dim")

        # Also verify original fused vectors are 921-dim (sanity)
        orig_dim = all_pairs[0].anchor.fused_feature_vector.shape[0]
        assert orig_dim == 921, f"Expected original 921-dim, got {orig_dim}"
        print(f"  Original fused dim      : {orig_dim}  (expected 921)")
        print(f"{PASS}  Original fused vectors unchanged")

        return True, bo_pairs

    except Exception:
        print(f"{FAIL}  step2_dataset_and_pairs")
        traceback.print_exc()
        return False, None


# ---------------------------------------------------------------------------
# Step 3 — BeatsOnlyProjectionHead: 256-dim L2-normalised output
# ---------------------------------------------------------------------------

def step3_projection_head(bo_pairs: list) -> bool:
    _section("Step 3: BeatsOnlyProjectionHead output shape and L2 norm")
    try:
        from experiments.phase19_beats_only_train import BeatsOnlyProjectionHead

        head = BeatsOnlyProjectionHead()
        head.eval()

        vec = bo_pairs[0].anchor.fused_feature_vector  # 768-dim
        x   = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)  # (1, 768)

        with torch.no_grad():
            out = head(x)

        assert out.shape == (1, 256), f"Expected (1, 256), got {out.shape}"
        print(f"  Output shape            : {tuple(out.shape)}")
        print(f"{PASS}  Output is 256-dim")

        norm = float(torch.norm(out, p=2, dim=-1).item())
        assert abs(norm - 1.0) < 1e-5, f"Expected L2 norm ≈ 1.0, got {norm:.6f}"
        print(f"  L2 norm                 : {norm:.6f}  (expected ~1.0)")
        print(f"{PASS}  Output is L2-normalised")

        return True

    except Exception:
        print(f"{FAIL}  step3_projection_head")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 4 — Phase 9 full-method: process one recording
# ---------------------------------------------------------------------------

def step4_phase9_one_recording() -> bool:
    _section("Step 4: Phase 9 full-method — one recording end-to-end")
    if not PHASE9_CKPT.exists():
        print(f"  SKIP  Phase 9 checkpoint not found: {PHASE9_CKPT}")
        return True  # not a failure — checkpoint may not be present in CI

    try:
        from src.dataset.loader import DatasetLoader
        from src.dataset.split import DatasetSplitter
        from src.learned_profile.builder import LearnedProfileBuilder
        from src.learned_health_index.analyzer import LearnedHealthAnalyzer

        loader   = DatasetLoader(DATASET_ROOT)
        all_recs = loader.get_all_files()
        splitter = DatasetSplitter(
            train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED
        )
        pump_split = splitter.split([r for r in all_recs if r.machine_type == "pump"])

        profile_recs = [r for r in pump_split.profile_normal if r.machine_id == "id_00"][:3]
        test_rec     = next(
            (r for r in pump_split.test_normal if r.machine_id == "id_00"), None
        )
        assert profile_recs, "No pump/id_00 profile recordings"
        assert test_rec,     "No pump/id_00 test recording"

        builder  = LearnedProfileBuilder(checkpoint_path=PHASE9_CKPT)
        profile  = builder.build("pump", "id_00", recordings=profile_recs)
        assert profile.embeddings.shape[1] == 256
        print(f"  Profile embeddings shape: {profile.embeddings.shape}")
        print(f"{PASS}  LearnedProfileBuilder (Phase 9)")

        analyzer = LearnedHealthAnalyzer(checkpoint_path=PHASE9_CKPT)
        result   = analyzer.analyze(test_rec, profile)
        assert 0.0 <= result.health_score <= 100.0
        print(f"  Health score            : {result.health_score:.2f}  state={result.health_state}")
        print(f"{PASS}  LearnedHealthAnalyzer (Phase 9)")

        return True

    except Exception:
        print(f"{FAIL}  step4_phase9_one_recording")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Step 5 — Phase 19 BEATs-only path: process one recording (random head)
# ---------------------------------------------------------------------------

def step5_beats_only_one_recording() -> bool:
    _section("Step 5: Phase 19 BEATs-only — one recording end-to-end (random head)")
    try:
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
        from experiments.phase19_beats_only_train import BeatsOnlyProjectionHead
        from experiments.phase19_evaluate import _BeatsOnlyInference, _BeatsOnlyProfileBuilder

        loader   = DatasetLoader(DATASET_ROOT)
        all_recs = loader.get_all_files()
        splitter = DatasetSplitter(
            train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED
        )
        pump_split = splitter.split([r for r in all_recs if r.machine_type == "pump"])

        profile_recs = [r for r in pump_split.profile_normal if r.machine_id == "id_00"][:3]
        test_rec     = next(
            (r for r in pump_split.test_normal if r.machine_id == "id_00"), None
        )
        assert profile_recs, "No pump/id_00 profile recordings"
        assert test_rec,     "No pump/id_00 test recording"

        # Build FusionCache (reuses existing cache — no re-encoding)
        beats_ckpt  = ROOT / "models" / "beats" / "BEATs_iter3_plus_AS2M.pt"
        pipeline    = PreprocessingPipeline(target_sr=16_000)
        extractor   = FeatureExtractor(sample_rate=16_000)
        vec_builder = FeatureVectorBuilder()
        encoder     = BEATsEncoder(beats_ckpt)
        fusion      = FusionBuilder()
        cache       = FusionCache(
            cache_root=CACHE_ROOT,
            pipeline=pipeline,
            extractor=extractor,
            vec_builder=vec_builder,
            encoder=encoder,
            fusion=fusion,
        )

        # Verify cache can load one fused vector and beats_embedding is 768-dim
        fused = cache.load_or_create(profile_recs[0])
        assert fused.beats_embedding.shape == (768,), (
            f"Expected beats_embedding (768,), got {fused.beats_embedding.shape}"
        )
        print(f"  beats_embedding shape   : {fused.beats_embedding.shape}")
        print(f"{PASS}  FusionCache.load_or_create — beats_embedding is 768-dim")

        # Use a randomly initialised BeatsOnlyProjectionHead (no checkpoint needed)
        head = BeatsOnlyProjectionHead()
        head.eval()

        # Manually wire up _BeatsOnlyInference without loading a checkpoint file
        class _RandomInference:
            def __init__(self, h: BeatsOnlyProjectionHead) -> None:
                self._head = h
            def embed(self, beats_vector: np.ndarray) -> np.ndarray:
                x = torch.from_numpy(beats_vector.astype(np.float32))
                with torch.no_grad():
                    out = self._head(x)
                return out.numpy().astype(np.float32)

        inference       = _RandomInference(head)
        profile_builder = _BeatsOnlyProfileBuilder(inference, cache)

        profile = profile_builder.build("pump", "id_00", profile_recs)
        assert profile.embeddings.shape == (len(profile_recs), 256), (
            f"Expected ({len(profile_recs)}, 256), got {profile.embeddings.shape}"
        )
        print(f"  Profile embeddings shape: {profile.embeddings.shape}")
        print(f"{PASS}  _BeatsOnlyProfileBuilder.build")

        metrics = profile_builder.analyze(test_rec, profile)
        assert "health_score" in metrics
        assert 0.0 <= metrics["health_score"] <= 100.0
        print(f"  Health score            : {metrics['health_score']:.2f}")
        print(f"  norm_euclidean          : {metrics['normalized_euclidean']:.4f}")
        print(f"{PASS}  _BeatsOnlyProfileBuilder.analyze")

        return True

    except Exception:
        print(f"{FAIL}  step5_beats_only_one_recording")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Phase 19 Smoke Test")
    print("=" * 60)

    results = {}
    results["imports"]            = step1_imports()
    ok2, bo_pairs                 = step2_dataset_and_pairs()
    results["dataset_and_pairs"]  = ok2
    if ok2 and bo_pairs:
        results["projection_head"] = step3_projection_head(bo_pairs)
    else:
        results["projection_head"] = False
    results["phase9_one_rec"]     = step4_phase9_one_recording()
    results["beats_only_one_rec"] = step5_beats_only_one_recording()

    _section("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All smoke-test steps passed. Safe to run full Phase 19 training.")
    else:
        print("  One or more steps FAILED. Fix errors before running full training.")
    print()


if __name__ == "__main__":
    main()
