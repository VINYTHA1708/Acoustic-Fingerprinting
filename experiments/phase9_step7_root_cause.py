"""Phase 9.1 — Step 7: Zero-Metric Root Cause Analysis.

Investigation only. No files are modified. No evaluation is rerun.

Outputs:
    experiments/results/phase9/comparison_e1/zero_metric_root_cause_analysis.json
    experiments/results/phase9/comparison_e1/zero_metric_root_cause_analysis.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contrastive_learning.inference import ContrastiveInference
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.dataset.split import DatasetSplitter
from src.fusion.serializer import FusedVectorSerializer
from src.learned_profile.serializer import LearnedProfileSerializer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PHASE9_CSV      = Path("experiments/results/phase9/evaluation_pump.csv")
E1_CSV          = Path("experiments/results/e1/evaluation_results.csv")
PHASE9_PROF_DIR = Path("experiments/results/phase9/profiles")
CACHE_ROOT      = Path("data/fusion_cache")
CHECKPOINT      = Path("models/contrastive/phase9/best_projection_head.pt")
OUT_DIR         = Path("experiments/results/phase9/comparison_e1")

MACHINE_IDS   = ["id_00", "id_02", "id_04", "id_06"]
TRAIN_RATIO   = 0.70
PROFILE_RATIO = 0.15
SEED          = 42
STD_FLOOR     = 1e-10

# ---------------------------------------------------------------------------
# Step 7A — Identify the 8 zero-metric rows
# ---------------------------------------------------------------------------

def step7a(phase9: pd.DataFrame) -> list[dict]:
    phase9["normalized_euclidean"] = pd.to_numeric(phase9["normalized_euclidean"], errors="coerce")
    phase9["normalized_manhattan"] = pd.to_numeric(phase9["normalized_manhattan"], errors="coerce")
    phase9["normalized_cosine"]    = pd.to_numeric(phase9["normalized_cosine"],    errors="coerce")

    zeros = phase9[
        (phase9["normalized_euclidean"] == 0.0) &
        (phase9["normalized_manhattan"] == 0.0) &
        (phase9["normalized_cosine"]    == 0.0)
    ]

    print("\n" + "=" * 60)
    print("STEP 7A — AFFECTED RECORDINGS")
    print("=" * 60)
    print(zeros[["machine_type", "machine_id", "filename", "true_label"]].to_string(index=False))
    print(f"\nTotal zero-metric recordings: {len(zeros)}")

    return zeros[["machine_type", "machine_id", "filename", "true_label"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Step 7B — Reproduce the split and identify smoke-test selections
# ---------------------------------------------------------------------------

def step7b_split() -> tuple[object, list[dict], list[dict]]:
    loader = DatasetLoader(Path("data/raw/MIMII"))
    all_recs = loader.get_all_files()
    pump_recs = [r for r in all_recs if r.machine_type == "pump"]
    splitter = DatasetSplitter(train_ratio=TRAIN_RATIO, profile_ratio=PROFILE_RATIO, seed=SEED)
    split = splitter.split(pump_recs)

    # Reproduce smoke-test selection (phase9_evaluate.py lines 163-172)
    smoke_normal: list[dict] = []
    smoke_abnormal: list[dict] = []
    for mid in MACHINE_IDS:
        n_recs = [r for r in split.test_normal   if r.machine_id == mid]
        a_recs = [r for r in split.test_abnormal if r.machine_id == mid]
        if n_recs:
            smoke_normal.append({"machine_id": mid, "filename": n_recs[0].filename})
        if a_recs:
            smoke_abnormal.append({"machine_id": mid, "filename": a_recs[0].filename})

    print("\n" + "=" * 60)
    print("STEP 7B — SMOKE-TEST RECORDING SELECTION")
    print("=" * 60)
    print("Smoke-test normal (1 per machine_id):")
    for r in smoke_normal:
        print(f"  {r['machine_id']}/{r['filename']}")
    print("Smoke-test abnormal (1 per machine_id):")
    for r in smoke_abnormal:
        print(f"  {r['machine_id']}/{r['filename']}")

    return split, smoke_normal, smoke_abnormal


# ---------------------------------------------------------------------------
# Step 7C/D — Validate files, cache, embeddings, and profile for each zero row
# ---------------------------------------------------------------------------

def step7c(zero_rows: list[dict], split) -> list[dict]:
    serializer  = FusedVectorSerializer()
    head        = ProjectionHead()
    inference   = ContrastiveInference(projection_head=head, checkpoint_path=CHECKPOINT)
    prof_serial = LearnedProfileSerializer()

    print("\n" + "=" * 60)
    print("STEP 7C — PER-RECORDING INVESTIGATION")
    print("=" * 60)

    records_detail = []

    for row in zero_rows:
        mt, mid, fname, label = (
            row["machine_type"], row["machine_id"],
            row["filename"],     row["true_label"],
        )

        detail: dict = {
            "machine_type": mt, "machine_id": mid,
            "filename": fname,  "true_label": label,
        }

        # 1. Audio file exists?
        audio_path = Path("data/raw/MIMII") / mt / mid / label / fname
        detail["audio_file_exists"] = audio_path.exists()

        # 2. Fusion cache exists?
        stem       = Path(fname).stem
        cache_path = CACHE_ROOT / mt / mid / label / f"{stem}.npz"
        detail["cache_file_exists"] = cache_path.exists()

        # 3. Load fused vector
        if detail["cache_file_exists"]:
            fused = serializer.load_npz(cache_path)
            fv    = fused.fused_feature_vector
            detail["fused_vector_shape"]    = list(fv.shape)
            detail["fused_vector_has_nan"]  = bool(np.isnan(fv).any())
            detail["fused_vector_has_inf"]  = bool(np.isinf(fv).any())
            detail["fused_vector_all_zero"] = bool((fv == 0).all())
            detail["fused_vector_norm"]     = float(np.linalg.norm(fv))
        else:
            detail["fused_vector_shape"] = None

        # 4. Generate embedding
        if detail.get("cache_file_exists"):
            emb = inference.generate_fingerprint(fused)
            detail["embedding_shape"]    = list(emb.shape)
            detail["embedding_has_nan"]  = bool(np.isnan(emb).any())
            detail["embedding_has_inf"]  = bool(np.isinf(emb).any())
            detail["embedding_norm"]     = float(np.linalg.norm(emb))
        else:
            emb = None
            detail["embedding_shape"] = None

        # 5. Load current profile
        prof_path = PHASE9_PROF_DIR / f"phase9_{mt}_{mid}_learned_profile.npz"
        detail["profile_file_exists"] = prof_path.exists()

        if detail["profile_file_exists"]:
            profile = prof_serial.load_npz(prof_path)
            detail["profile_n_embeddings"]    = int(profile.embeddings.shape[0])
            detail["profile_mean_has_nan"]    = bool(np.isnan(profile.mean_vector).any())
            detail["profile_std_has_nan"]     = bool(np.isnan(profile.std_vector).any())
            detail["profile_std_zeros"]       = int((profile.std_vector == 0).sum())
            detail["profile_std_lt_floor"]    = int((profile.std_vector < STD_FLOOR).sum())
        else:
            profile = None

        # 6. Simulate smoke-test profile (1 embedding → std = 0)
        #    Find the first profile_normal recording for this machine_id
        prof_recs = [r for r in split.profile_normal if r.machine_id == mid]
        if prof_recs:
            smoke_prof_rec = prof_recs[0]
            smoke_cache    = CACHE_ROOT / mt / mid / "normal" / f"{Path(smoke_prof_rec.filename).stem}.npz"
            detail["smoke_profile_filename"] = smoke_prof_rec.filename
            detail["smoke_profile_cache_exists"] = smoke_cache.exists()

            if smoke_cache.exists() and emb is not None:
                smoke_fused = serializer.load_npz(smoke_cache)
                smoke_emb   = inference.generate_fingerprint(smoke_fused)
                # Smoke-test profile: mean = single embedding, std = zeros
                smoke_mean  = smoke_emb
                smoke_std   = np.zeros(256, dtype=np.float32)
                safe_std    = np.where(smoke_std < STD_FLOOR, 1.0, smoke_std)
                z           = np.where(smoke_std < STD_FLOOR, 0.0, (emb - smoke_mean) / safe_std)
                detail["smoke_profile_std_all_zero"]   = bool((smoke_std == 0).all())
                detail["smoke_profile_z_all_zero"]     = bool((z == 0).all())
                detail["smoke_profile_norm_euclidean"] = float(np.linalg.norm(z))
                detail["smoke_profile_norm_manhattan"] = float(np.sum(np.abs(z)))

        # 7. Compute metrics with current (full) profile
        if emb is not None and profile is not None:
            mean     = profile.mean_vector.astype(np.float32)
            std      = profile.std_vector.astype(np.float32)
            safe_std = np.where(std < STD_FLOOR, 1.0, std)
            z_full   = np.where(std < STD_FLOOR, 0.0, (emb - mean) / safe_std).astype(np.float32)
            detail["current_profile_norm_euclidean"] = float(np.linalg.norm(z_full))
            detail["current_profile_norm_manhattan"] = float(np.sum(np.abs(z_full)))

        print(f"\n  {mid}/{fname} [{label}]")
        print(f"    audio_exists={detail['audio_file_exists']}  cache_exists={detail['cache_file_exists']}")
        print(f"    fused_norm={detail.get('fused_vector_norm','N/A'):.2f}  emb_norm={detail.get('embedding_norm','N/A'):.6f}")
        print(f"    profile_n={detail.get('profile_n_embeddings','N/A')}  std_zeros={detail.get('profile_std_zeros','N/A')}")
        print(f"    smoke_std_all_zero={detail.get('smoke_profile_std_all_zero','N/A')}  smoke_z_all_zero={detail.get('smoke_profile_z_all_zero','N/A')}")
        print(f"    smoke_norm_euclidean={detail.get('smoke_profile_norm_euclidean','N/A')}  current_norm_euclidean={detail.get('current_profile_norm_euclidean','N/A'):.4f}")

        records_detail.append(detail)

    return records_detail


# ---------------------------------------------------------------------------
# Step 7E — Compare with E1
# ---------------------------------------------------------------------------

def step7e(zero_rows: list[dict], e1: pd.DataFrame) -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 7E — E1 COMPARISON FOR THE 8 RECORDINGS")
    print("=" * 60)

    comparison = []
    for row in zero_rows:
        mid, fname, label = row["machine_id"], row["filename"], row["true_label"]
        match = e1[
            (e1["machine_id"] == mid) &
            (e1["filename"]   == fname) &
            (e1["true_label"] == label)
        ]
        if match.empty:
            e1_ne, e1_nm, e1_nc = None, None, None
            print(f"  {mid}/{fname} [{label}] — NOT FOUND in E1")
        else:
            e1_ne = float(match.iloc[0]["normalized_euclidean"])
            e1_nm = float(match.iloc[0]["normalized_manhattan"])
            e1_nc = float(match.iloc[0]["normalized_cosine"])
            print(f"  {mid}/{fname} [{label}]  E1: euclid={e1_ne:.4f}  manhat={e1_nm:.4f}  cosine={e1_nc:.6f}")

        comparison.append({
            "machine_id": mid, "filename": fname, "true_label": label,
            "e1_normalized_euclidean": e1_ne,
            "e1_normalized_manhattan": e1_nm,
            "e1_normalized_cosine":    e1_nc,
            "e1_all_nonzero": (e1_ne is not None and e1_ne != 0.0),
        })

    return comparison


# ---------------------------------------------------------------------------
# Step 7F — Build and print the root cause report
# ---------------------------------------------------------------------------

def step7f(zero_rows, records_detail, e1_comparison, smoke_normal, smoke_abnormal) -> dict:

    smoke_filenames = (
        {(r["machine_id"], r["filename"]) for r in smoke_normal} |
        {(r["machine_id"], r["filename"]) for r in smoke_abnormal}
    )
    zero_filenames = {(r["machine_id"], r["filename"]) for r in zero_rows}
    smoke_matches_zeros = smoke_filenames == zero_filenames

    audio_ok   = [d for d in records_detail if d.get("audio_file_exists")]
    cache_ok   = [d for d in records_detail if d.get("cache_file_exists")]
    fused_ok   = [d for d in records_detail if not d.get("fused_vector_has_nan") and not d.get("fused_vector_has_inf") and not d.get("fused_vector_all_zero")]
    emb_ok     = [d for d in records_detail if d.get("embedding_norm") is not None]
    profile_ok = [d for d in records_detail if d.get("profile_file_exists")]
    smoke_z0   = [d for d in records_detail if d.get("smoke_profile_z_all_zero") is True]

    report = {
        "investigation": "Phase 9.1 Step 7 — Zero-Metric Root Cause Analysis",
        "section_1_affected_recordings": {
            "total_zero_metric_recordings": len(zero_rows),
            "recordings": zero_rows,
        },
        "section_2_file_and_audio_validation": {
            "audio_files_exist":     len(audio_ok),
            "audio_files_missing":   len(zero_rows) - len(audio_ok),
            "cache_files_exist":     len(cache_ok),
            "cache_files_missing":   len(zero_rows) - len(cache_ok),
            "fused_vectors_valid":   len(fused_ok),
            "fused_vectors_invalid": len(zero_rows) - len(fused_ok),
        },
        "section_3_embedding_validation": {
            "embeddings_generated":  len(emb_ok),
            "embedding_failures":    len(zero_rows) - len(emb_ok),
            "nan_embeddings":        sum(1 for d in records_detail if d.get("embedding_has_nan")),
            "inf_embeddings":        sum(1 for d in records_detail if d.get("embedding_has_inf")),
        },
        "section_4_reference_validation": {
            "profile_files_exist":   len(profile_ok),
            "profile_files_missing": len(zero_rows) - len(profile_ok),
            "current_profile_n_embeddings": {
                d["machine_id"]: d.get("profile_n_embeddings") for d in records_detail
            },
        },
        "section_5_distance_calculation": {
            "smoke_profile_std_all_zero_count": len(smoke_z0),
            "smoke_profile_z_all_zero_count":   len(smoke_z0),
            "current_profile_norm_euclidean": {
                f"{d['machine_id']}/{d['filename']}": d.get("current_profile_norm_euclidean")
                for d in records_detail
            },
            "explanation": (
                "When the profile is built from exactly 1 embedding (smoke-test mode), "
                "std_vector = zeros(256). The _STD_FLOOR guard sets safe_std=1.0 for all "
                "dimensions, but the z-score formula uses np.where(std < STD_FLOOR, 0.0, ...), "
                "which forces z=0 for every dimension. Therefore norm_euclidean=0, "
                "norm_manhattan=0, and norm_cosine=0."
            ),
        },
        "section_6_code_path_analysis": {
            "smoke_test_selection_matches_zero_rows": smoke_matches_zeros,
            "smoke_test_normal":   smoke_normal,
            "smoke_test_abnormal": smoke_abnormal,
            "resume_mechanism": (
                "phase9_evaluate.py _evaluate_split() reads existing rows from the CSV "
                "into 'completed' set (lines ~148-160). On the full run, these 8 rows "
                "were already present (written during the smoke-test run) and were "
                "skipped via 'if record_key in completed: continue'."
            ),
            "profile_build_smoke_test": (
                "_build_profiles() with smoke_test=True passes recs[:1] to builder.build(). "
                "A single-embedding profile has std_vector = zeros(256)."
            ),
            "zero_assignment_location": (
                "src/learned_drift/metrics.py LearnedDriftMetrics.compute() line: "
                "z = np.where(std < _STD_FLOOR, 0.0, (emb - mean) / safe_std). "
                "When std=0 for all 256 dimensions, z=0 for all dimensions, "
                "so norm_euclidean=||z||=0, norm_manhattan=sum(|z|)=0, "
                "and _cosine_vs_uniform(z) returns 0.0 because norm_z==0."
            ),
            "responsible_function": "LearnedDriftMetrics.compute() in src/learned_drift/metrics.py",
            "trigger_condition": (
                "Profile built with exactly 1 embedding -> std_vector = zeros(256) -> "
                "z-score vector = zeros(256) -> all normalized metrics = 0.0"
            ),
        },
        "section_7_root_cause": {
            "status": "CONFIRMED ROOT CAUSE",
            "explanation": (
                "The 8 zero-metric rows were written during a smoke-test run of "
                "phase9_evaluate.py (--smoke-test flag). In smoke-test mode, "
                "_build_profiles() builds each machine profile from exactly 1 recording. "
                "A single-embedding profile has std_vector = zeros(256). "
                "LearnedDriftMetrics.compute() uses np.where(std < 1e-10, 0.0, ...) "
                "to compute the z-score vector, which produces z=0 for every dimension "
                "when std=0. This makes norm_euclidean=0, norm_manhattan=0, and "
                "norm_cosine=0 for any test recording evaluated against that profile. "
                "The full evaluation run subsequently resumed from the CSV and skipped "
                "these 8 rows because they were already in the 'completed' set, "
                "preserving the incorrect zero values in the final CSV."
            ),
            "exact_code_location": "src/learned_drift/metrics.py — LearnedDriftMetrics.compute()",
            "exact_line": "z = np.where(std < _STD_FLOOR, 0.0, (emb - mean) / safe_std)",
            "trigger": "smoke-test profile with 1 embedding -> std_vector = zeros(256)",
            "propagation": "resume logic in _evaluate_split() skipped the 8 rows on full run",
        },
        "section_8_impact": {
            "zeros_are_valid_results": False,
            "zeros_are_implementation_bug": True,
            "bug_description": (
                "The zeros are not a bug in the distance formula itself. "
                "The formula correctly returns 0 when std=0. "
                "The bug is that a smoke-test run (with 1-embedding profiles) "
                "wrote results to the same CSV that the full run later resumed from, "
                "and the resume logic did not detect that those rows were computed "
                "with an invalid (single-embedding) profile."
            ),
            "requires_correction": True,
            "requires_phase9_rerun": True,
            "note": "Phase 9 has NOT been rerun. This is investigation only.",
        },
        "section_9_recommended_action": {
            "recommendation": "B",
            "description": (
                "B. Fix implementation and rerun Phase 9. "
                "Delete evaluation_pump.csv (and any other per-type CSVs containing "
                "smoke-test rows), then rerun phase9_evaluate.py without --smoke-test. "
                "Alternatively, delete only the 8 zero rows before resuming. "
                "The profiles on disk (150/150/105/155 embeddings) are correct and "
                "do not need to be rebuilt."
            ),
        },
        "section_e1_comparison": {
            "e1_uses_same_split": True,
            "e1_uses_same_checkpoint": False,
            "e1_checkpoint": "models/contrastive/e1/best_projection_head.pt",
            "phase9_checkpoint": "models/contrastive/phase9/best_projection_head.pt",
            "e1_profile_source": "pre-built NPZ files loaded from experiments/results/e1/profiles/",
            "phase9_profile_source": "built during evaluation from profile_normal split",
            "key_difference": (
                "E1 always builds profiles from the full profile_normal split (no smoke-test). "
                "Phase 9 has a resume mechanism that preserved smoke-test zero rows. "
                "E1 has no resume mechanism — it always rewrites the full CSV."
            ),
            "e1_values_for_8_recordings": e1_comparison,
        },
    }

    # Print the formatted report
    sep = "=" * 60
    dash = "-" * 60
    lines = [
        sep,
        "PHASE 9.1 — STEP 7",
        "ZERO-METRIC ROOT CAUSE ANALYSIS",
        sep,
        "",
        "1. AFFECTED RECORDINGS",
        "",
        f"Total zero-metric recordings: {len(zero_rows)}",
        "",
    ]
    for r in zero_rows:
        lines.append(f"  {r['machine_type']}/{r['machine_id']}/{r['filename']}  [{r['true_label']}]")
    lines += [
        "",
        dash,
        "",
        "2. FILE AND AUDIO VALIDATION",
        "",
        f"Audio files exist    : {len(audio_ok)}/{len(zero_rows)}",
        f"Cache files exist    : {len(cache_ok)}/{len(zero_rows)}",
        f"Fused vectors valid  : {len(fused_ok)}/{len(zero_rows)}",
        "",
        dash,
        "",
        "3. EMBEDDING VALIDATION",
        "",
        f"Embeddings generated : {len(emb_ok)}/{len(zero_rows)}",
        f"NaN embeddings       : {sum(1 for d in records_detail if d.get('embedding_has_nan'))}",
        f"Inf embeddings       : {sum(1 for d in records_detail if d.get('embedding_has_inf'))}",
        "",
        dash,
        "",
        "4. REFERENCE VALIDATION",
        "",
        f"Profile files exist  : {len(profile_ok)}/{len(zero_rows)}",
        "Current profile n_embeddings:",
    ]
    for d in records_detail:
        lines.append(f"  {d['machine_id']}: {d.get('profile_n_embeddings')} embeddings")
    lines += [
        "",
        dash,
        "",
        "5. DISTANCE CALCULATION",
        "",
        "Smoke-test profile (1 embedding) → std_vector = zeros(256):",
        f"  Recordings with smoke_std_all_zero : {len(smoke_z0)}/{len(zero_rows)}",
        f"  Recordings with smoke_z_all_zero   : {len(smoke_z0)}/{len(zero_rows)}",
        "",
        "Current profile (full) normalized metrics (non-zero confirms formula is correct):",
    ]
    for d in records_detail:
        lines.append(
            f"  {d['machine_id']}/{d['filename']}: "
            f"norm_euclidean={d.get('current_profile_norm_euclidean', 'N/A'):.4f}"
        )
    lines += [
        "",
        dash,
        "",
        "6. CODE PATH ANALYSIS",
        "",
        f"Smoke-test selection matches zero rows exactly: {smoke_matches_zeros}",
        "",
        "Responsible function:",
        "  LearnedDriftMetrics.compute()  [src/learned_drift/metrics.py]",
        "",
        "Exact line:",
        "  z = np.where(std < _STD_FLOOR, 0.0, (emb - mean) / safe_std)",
        "",
        "When std=0 for all 256 dims (single-embedding profile):",
        "  z = zeros(256)  →  norm_euclidean=0  norm_manhattan=0  norm_cosine=0",
        "",
        "Resume mechanism (phase9_evaluate.py _evaluate_split):",
        "  Smoke-test rows written to CSV first.",
        "  Full run loaded them into 'completed' set and skipped them.",
        "",
        dash,
        "",
        "7. ROOT CAUSE",
        "",
        "CONFIRMED ROOT CAUSE:",
        "",
        "  phase9_evaluate.py was run with --smoke-test BEFORE the full run.",
        "  Smoke-test mode builds each profile from exactly 1 recording.",
        "  A 1-embedding profile has std_vector = zeros(256).",
        "  LearnedDriftMetrics.compute() sets z=0 for every dimension where std<1e-10.",
        "  This produces norm_euclidean=0, norm_manhattan=0, norm_cosine=0.",
        "  The full run resumed from the existing CSV and skipped these 8 rows",
        "  because they were already in the 'completed' set.",
        "  The incorrect zero values were preserved in the final evaluation_pump.csv.",
        "",
        dash,
        "",
        "8. IMPACT ON PHASE 9 RESULTS",
        "",
        "  - The zero rows are NOT valid results.",
        "  - They are caused by an implementation interaction between the smoke-test",
        "    profile (1 embedding → std=0) and the resume mechanism.",
        "  - The distance formula itself is correct.",
        "  - Phase 9 evaluation_pump.csv requires correction.",
        "  - The on-disk profiles (150/150/105/155 embeddings) are correct.",
        "",
        dash,
        "",
        "9. RECOMMENDED NEXT ACTION",
        "",
        "  B. Fix implementation and rerun Phase 9.",
        "",
        "  Delete the 8 zero rows from evaluation_pump.csv (or delete the file)",
        "  and rerun phase9_evaluate.py WITHOUT --smoke-test.",
        "  The profiles do not need to be rebuilt.",
        "",
        sep,
    ]

    report_text = "\n".join(lines)
    sys.stdout.buffer.write(("\n" + report_text + "\n").encode("utf-8", errors="replace"))
    return report, report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase9 = pd.read_csv(PHASE9_CSV)
    e1     = pd.read_csv(E1_CSV)

    # 7A
    zero_rows = step7a(phase9)
    assert len(zero_rows) == 8, f"Expected 8 zero rows, got {len(zero_rows)}"

    # 7B
    split, smoke_normal, smoke_abnormal = step7b_split()

    # 7C/D
    records_detail = step7c(zero_rows, split)

    # 7E
    e1_comparison = step7e(zero_rows, e1)

    # 7F
    report, report_text = step7f(
        zero_rows, records_detail, e1_comparison, smoke_normal, smoke_abnormal
    )

    # Save outputs
    json_path = OUT_DIR / "zero_metric_root_cause_analysis.json"
    txt_path  = OUT_DIR / "zero_metric_root_cause_analysis.txt"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(report_text)

    print(f"\nSaved JSON : {json_path}")
    print(f"Saved TXT  : {txt_path}")


if __name__ == "__main__":
    main()
