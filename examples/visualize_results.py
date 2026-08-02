"""Visualization example — generates five publication-quality PNG figures.

Builds a healthy learned fingerprint profile, runs InferencePipeline on up to
50 healthy and 50 abnormal recordings, then calls ResultVisualizer to produce:

    outputs/visualizations/health_scores.png
    outputs/visualizations/drift_scores.png
    outputs/visualizations/embedding_pca.png
    outputs/visualizations/confusion_matrix.png
    outputs/visualizations/roc_curve.png

Usage:
    python examples/visualize_results.py \\
        --root data/raw/MIMII \\
        --machine-type pump \\
        --machine-id id_00 \\
        --checkpoint models/contrastive/best_projection_head.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.contrastive_learning.inference import ContrastiveInference
from src.contrastive_learning.model import ProjectionHead
from src.dataset.loader import DatasetLoader
from src.fusion.cache import FusionCache
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import InferencePipeline
from src.visualization import ResultVisualizer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_MAX_RECORDINGS = 50
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "visualizations"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--machine-type", required=True, help="Machine type (e.g. pump)")
    parser.add_argument("--machine-id", required=True, help="Machine ID (e.g. id_00)")
    parser.add_argument("--checkpoint", required=True, help="Path to ProjectionHead checkpoint")
    parser.add_argument(
        "--max-recordings", type=int, default=100,
        help="Maximum number of healthy recordings used to build the profile.",
    )
    args = parser.parse_args()

    loader = DatasetLoader(args.root)
    all_records = loader.get_all_files()

    normal_records = [
        r for r in all_records
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "normal"
    ]
    abnormal_records = [
        r for r in all_records
        if r.machine_type == args.machine_type
        and r.machine_id == args.machine_id
        and r.label == "abnormal"
    ]

    if not normal_records:
        print(f"ERROR: No normal recordings found for {args.machine_type}/{args.machine_id}.")
        sys.exit(1)
    if not abnormal_records:
        print(f"ERROR: No abnormal recordings found for {args.machine_type}/{args.machine_id}.")
        sys.exit(1)

    # Hold out inference recordings so they are never in the profile.
    infer_normal = normal_records[:_MAX_RECORDINGS]
    infer_abnormal = abnormal_records[:_MAX_RECORDINGS]
    exclude = {r.filename for r in infer_normal}

    # --- Build healthy learned profile ---
    print("Building healthy learned profile...")
    builder = LearnedProfileBuilder(checkpoint_path=args.checkpoint)
    profile = builder.build(
        loader=loader,
        machine_type=args.machine_type,
        machine_id=args.machine_id,
        max_recordings=args.max_recordings,
        exclude_filenames=exclude,
    )
    print(f"Profile built — {len(profile.embeddings)} embeddings\n")

    # --- Shared pipeline and analyzers ---
    pipeline = InferencePipeline(checkpoint_path=args.checkpoint)
    drift_analyzer = LearnedDriftAnalyzer(checkpoint_path=args.checkpoint)
    health_analyzer = LearnedHealthAnalyzer(checkpoint_path=args.checkpoint)

    # Reuse ContrastiveInference from the drift analyzer to obtain embeddings.
    inference = drift_analyzer._inference

    health_scores: list[float] = []
    drift_scores: list[float] = []
    embeddings: list[np.ndarray] = []
    embed_labels: list[str] = []
    y_true: list[int] = []

    def _process(records, label_str: str, y_val: int) -> None:
        for rec in records:
            try:
                result = pipeline.analyze(rec, profile)
                health_scores.append(result.health_score)
                drift_scores.append(result.normalized_euclidean)
                y_true.append(y_val)
                embed_labels.append(label_str)

                fused = drift_analyzer._cache.load_or_create(rec)
                emb = inference.generate_fingerprint(fused)
                embeddings.append(emb)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Skipping %s — %s", rec.filename, exc)

    print(f"Running inference on up to {_MAX_RECORDINGS} healthy recordings...")
    _process(infer_normal, "healthy", 0)

    print(f"Running inference on up to {_MAX_RECORDINGS} abnormal recordings...")
    _process(infer_abnormal, "abnormal", 1)

    n_healthy = embed_labels.count("healthy")
    n_abnormal = embed_labels.count("abnormal")
    print(f"\nProcessed: {n_healthy} healthy, {n_abnormal} abnormal\n")

    # --- Derive binary predictions from health score threshold (50) ---
    # score < 50 → CRITICAL → predicted abnormal (1), else healthy (0)
    y_pred = [1 if s < 50.0 else 0 for s in health_scores]

    # Anomaly score for ROC: lower health score = higher anomaly probability
    anomaly_scores = [100.0 - s for s in health_scores]

    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)

    # --- Generate plots ---
    vis = ResultVisualizer()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vis.plot_health_scores(
        scores=health_scores,
        labels=[r.filename for r in infer_normal + infer_abnormal],
        output_path=_OUTPUT_DIR / "health_scores.png",
    )

    vis.plot_drift_scores(
        drift_scores=drift_scores,
        labels=[r.filename for r in infer_normal + infer_abnormal],
        output_path=_OUTPUT_DIR / "drift_scores.png",
    )

    vis.plot_embedding_distribution(
        embeddings=emb_matrix,
        labels=embed_labels,
        output_path=_OUTPUT_DIR / "embedding_pca.png",
    )

    vis.plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=_OUTPUT_DIR / "confusion_matrix.png",
    )

    vis.plot_roc_curve(
        y_true=y_true,
        scores=anomaly_scores,
        output_path=_OUTPUT_DIR / "roc_curve.png",
    )

    print("Visualizations saved to:")
    print(f"  {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
