"""Streamlit dashboard for Acoustic Fingerprinting of Industrial Machines.

Provides an interactive UI for:
    - Building a healthy learned fingerprint profile
    - Running end-to-end inference on an uploaded audio file
    - Displaying health score, drift metrics, and rule-based explanation
    - Rendering all five ResultVisualizer plots inline
    - Reporting per-stage benchmark timings

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project imports — all inference delegated to existing modules
# ---------------------------------------------------------------------------
from src.benchmark.benchmark import PipelineBenchmark
from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata
from src.explainability.explainer import ExplainabilityEngine
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import InferencePipeline
from src.visualization.visualizer import ResultVisualizer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EVAL_LIMIT = 50  # recordings used for trend / PCA / confusion / ROC plots
_VIZ_TMP = Path(tempfile.gettempdir()) / "af_dashboard_viz"
_VIZ_TMP.mkdir(parents=True, exist_ok=True)

_STATE_COLORS: dict[str, str] = {
    "EXCELLENT": "🟢",
    "GOOD": "🟡",
    "WARNING": "🟠",
    "CRITICAL": "🔴",
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading pipeline resources…")
def _load_pipeline(checkpoint: str) -> InferencePipeline:
    """Load and cache the InferencePipeline (BEATs + ProjectionHead).

    Args:
        checkpoint: Path to the trained ProjectionHead ``.pt`` checkpoint.

    Returns:
        A ready :class:`~pipeline.pipeline.InferencePipeline` instance.
    """
    return InferencePipeline(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading drift analyzer…")
def _load_drift_analyzer(checkpoint: str) -> LearnedDriftAnalyzer:
    """Load and cache the LearnedDriftAnalyzer.

    Args:
        checkpoint: Path to the trained ProjectionHead ``.pt`` checkpoint.

    Returns:
        A ready :class:`~learned_drift.analyzer.LearnedDriftAnalyzer` instance.
    """
    return LearnedDriftAnalyzer(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading health analyzer…")
def _load_health_analyzer(checkpoint: str) -> LearnedHealthAnalyzer:
    """Load and cache the LearnedHealthAnalyzer.

    Args:
        checkpoint: Path to the trained ProjectionHead ``.pt`` checkpoint.

    Returns:
        A ready :class:`~learned_health_index.analyzer.LearnedHealthAnalyzer` instance.
    """
    return LearnedHealthAnalyzer(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading benchmark…")
def _load_benchmark(checkpoint: str) -> PipelineBenchmark:
    """Load and cache the PipelineBenchmark.

    Args:
        checkpoint: Path to the trained ProjectionHead ``.pt`` checkpoint.

    Returns:
        A ready :class:`~benchmark.benchmark.PipelineBenchmark` instance.
    """
    return PipelineBenchmark(checkpoint_path=checkpoint)


# ---------------------------------------------------------------------------
# Helper — build AudioMetadata from an uploaded file saved to a temp path
# ---------------------------------------------------------------------------


def _make_audio_metadata(
    wav_path: Path,
    machine_type: str,
    machine_id: str,
) -> AudioMetadata:
    """Construct an :class:`~dataset.metadata.AudioMetadata` for an uploaded file.

    The uploaded file is treated as a ``normal`` recording for inference
    purposes.  The label does not affect drift or health computation.

    Args:
        wav_path: Absolute path to the saved temporary WAV file.
        machine_type: Machine type selected in the sidebar.
        machine_id: Machine ID selected in the sidebar.

    Returns:
        :class:`~dataset.metadata.AudioMetadata` pointing at the temp file.
    """
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label="normal",
        filename=wav_path.name,
        relative_path=wav_path,
        absolute_path=wav_path,
    )


# ---------------------------------------------------------------------------
# Helper — collect evaluation data for trend / PCA / confusion / ROC plots
# ---------------------------------------------------------------------------


def _collect_eval_data(
    loader: DatasetLoader,
    machine_type: str,
    machine_id: str,
    pipeline: InferencePipeline,
    drift_analyzer: LearnedDriftAnalyzer,
    profile,
) -> tuple[list[float], list[float], np.ndarray, list[str], list[int]]:
    """Run inference on up to _EVAL_LIMIT healthy and abnormal recordings.

    Collects health scores, drift scores, embeddings, embedding labels, and
    ground-truth binary labels for visualization.

    Args:
        loader: :class:`~dataset.loader.DatasetLoader` for the dataset root.
        machine_type: Machine type to evaluate.
        machine_id: Machine ID to evaluate.
        pipeline: Cached :class:`~pipeline.pipeline.InferencePipeline`.
        drift_analyzer: Cached :class:`~learned_drift.analyzer.LearnedDriftAnalyzer`.
        profile: :class:`~learned_profile.learned_profile.LearnedFingerprintProfile`.

    Returns:
        Tuple of (health_scores, drift_scores, emb_matrix, embed_labels, y_true).
    """
    all_records = loader.get_all_files()
    normal_recs = [
        r for r in all_records
        if r.machine_type == machine_type and r.machine_id == machine_id and r.label == "normal"
    ][:_EVAL_LIMIT]
    abnormal_recs = [
        r for r in all_records
        if r.machine_type == machine_type and r.machine_id == machine_id and r.label == "abnormal"
    ][:_EVAL_LIMIT]

    health_scores: list[float] = []
    drift_scores: list[float] = []
    embeddings: list[np.ndarray] = []
    embed_labels: list[str] = []
    y_true: list[int] = []

    inference = drift_analyzer._inference

    for rec, y_val, lbl in (
        [(r, 0, "healthy") for r in normal_recs]
        + [(r, 1, "abnormal") for r in abnormal_recs]
    ):
        try:
            result = pipeline.analyze(rec, profile)
            health_scores.append(result.health_score)
            drift_scores.append(result.normalized_euclidean)
            y_true.append(y_val)
            embed_labels.append(lbl)
            fused = drift_analyzer._cache.load_or_create(rec)
            embeddings.append(inference.generate_fingerprint(fused))
        except Exception:  # noqa: BLE001
            pass

    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32) if embeddings else np.empty((0, 256))
    return health_scores, drift_scores, emb_matrix, embed_labels, y_true


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_machine_info(machine_type: str, machine_id: str, filename: str) -> None:
    """Render the Machine Information section.

    Args:
        machine_type: Machine type string.
        machine_id: Machine ID string.
        filename: Source audio filename.
    """
    st.subheader("Machine Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Machine Type", machine_type)
    c2.metric("Machine ID", machine_id)
    c3.metric("Filename", filename)


def _render_health(health_score: float, health_state: str) -> None:
    """Render the Health section with a large metric and state badge.

    Args:
        health_score: Bounded health score in [0, 100].
        health_state: Qualitative state string.
    """
    st.subheader("Health")
    badge = _STATE_COLORS.get(health_state, "⚪")
    c1, c2 = st.columns(2)
    c1.metric("Health Score", f"{health_score:.1f} / 100")
    c2.metric("Health State", f"{badge} {health_state}")

    # Colour-coded progress bar
    bar_color = {"EXCELLENT": "green", "GOOD": "blue", "WARNING": "orange", "CRITICAL": "red"}.get(
        health_state, "gray"
    )
    st.progress(int(health_score), text=f"{health_score:.1f}%")
    _ = bar_color  # reserved for future custom styling


def _render_drift(
    raw_euclidean: float,
    normalized_euclidean: float,
    raw_manhattan: float,
    normalized_manhattan: float,
    cosine_similarity: float,
) -> None:
    """Render the Drift Metrics section.

    Args:
        raw_euclidean: Raw Euclidean distance.
        normalized_euclidean: Normalized Euclidean distance.
        raw_manhattan: Raw Manhattan distance.
        normalized_manhattan: Normalized Manhattan distance.
        cosine_similarity: Raw cosine similarity.
    """
    st.subheader("Drift Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Euclidean", f"{raw_euclidean:.4f}")
    c2.metric("Normalized Euclidean", f"{normalized_euclidean:.4f}")
    c3.metric("Cosine Similarity", f"{cosine_similarity:.4f}")
    c4, c5 = st.columns(2)
    c4.metric("Raw Manhattan", f"{raw_manhattan:.4f}")
    c5.metric("Normalized Manhattan", f"{normalized_manhattan:.4f}")


def _render_explanation(summary: str, possible_causes: list[str], recommendation: str) -> None:
    """Render the Explanation section.

    Args:
        summary: One-sentence condition summary.
        possible_causes: List of potential root causes.
        recommendation: Suggested operator action.
    """
    st.subheader("Explanation")
    st.info(f"**Summary:** {summary}")
    if possible_causes:
        st.warning("**Possible Causes:**\n" + "\n".join(f"- {c}" for c in possible_causes))
    else:
        st.success("No fault causes identified.")
    st.info(f"**Recommendation:** {recommendation}")


def _render_performance(
    dsp_dim: int,
    beats_dim: int,
    fusion_dim: int,
    embedding_dim: int,
    inference_time: float,
    cache_hit: bool,
) -> None:
    """Render the Performance section.

    Args:
        dsp_dim: DSP feature vector dimension.
        beats_dim: BEATs embedding dimension.
        fusion_dim: Fused vector dimension.
        embedding_dim: Learned embedding dimension.
        inference_time: Total wall-clock inference time in seconds.
        cache_hit: Whether the fused vector was loaded from disk cache.
    """
    st.subheader("Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DSP Dimension", dsp_dim)
    c2.metric("BEATs Dimension", beats_dim)
    c3.metric("Fusion Dimension", fusion_dim)
    c4.metric("Embedding Dimension", embedding_dim)
    c5, c6 = st.columns(2)
    c5.metric("Inference Time", f"{inference_time * 1000:.1f} ms")
    c6.metric("Cache Hit", "✅ Yes" if cache_hit else "❌ No")


def _render_visualizations(
    health_scores: list[float],
    drift_scores: list[float],
    emb_matrix: np.ndarray,
    embed_labels: list[str],
    y_true: list[int],
) -> None:
    """Render all five ResultVisualizer plots inline.

    Saves each figure to a temporary directory and displays it with
    ``st.image``.  Requires at least two classes in ``y_true`` for the
    confusion matrix and ROC curve.

    Args:
        health_scores: Health scores for all evaluated recordings.
        drift_scores: Normalized Euclidean drift scores.
        emb_matrix: Embedding matrix of shape ``(N, 256)``.
        embed_labels: Per-row label strings (``"healthy"`` / ``"abnormal"``).
        y_true: Ground-truth binary labels (0 = healthy, 1 = abnormal).
    """
    if not health_scores:
        st.info("No evaluation recordings available for visualization.")
        return

    vis = ResultVisualizer()
    filenames = [f"rec_{i}" for i in range(len(health_scores))]

    st.subheader("Visualizations")

    # Health trend
    p_health = _VIZ_TMP / "health_scores.png"
    vis.plot_health_scores(health_scores, filenames, p_health)
    st.image(str(p_health), caption="Health Score Trend", use_container_width=True)

    # Drift trend
    p_drift = _VIZ_TMP / "drift_scores.png"
    vis.plot_drift_scores(drift_scores, filenames, p_drift)
    st.image(str(p_drift), caption="Normalized Euclidean Drift Trend", use_container_width=True)

    # Embedding PCA — needs at least 2 samples
    if emb_matrix.shape[0] >= 2:
        p_pca = _VIZ_TMP / "embedding_pca.png"
        vis.plot_embedding_distribution(emb_matrix, embed_labels, p_pca)
        st.image(str(p_pca), caption="Embedding Distribution (PCA)", use_container_width=True)

    # Confusion matrix and ROC — need both classes present
    has_both_classes = len(set(y_true)) == 2
    if has_both_classes:
        y_pred = [1 if s < 50.0 else 0 for s in health_scores]
        anomaly_scores = [100.0 - s for s in health_scores]

        p_cm = _VIZ_TMP / "confusion_matrix.png"
        vis.plot_confusion_matrix(y_true, y_pred, p_cm)
        st.image(str(p_cm), caption="Confusion Matrix", use_container_width=True)

        p_roc = _VIZ_TMP / "roc_curve.png"
        vis.plot_roc_curve(y_true, anomaly_scores, p_roc)
        st.image(str(p_roc), caption="ROC Curve", use_container_width=True)
    else:
        st.info(
            "Confusion matrix and ROC curve require both healthy and abnormal "
            "recordings in the dataset."
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Streamlit dashboard.

    Renders the sidebar, handles the Analyze button click, and orchestrates
    all section renderers.  All heavy computation is delegated to the
    existing project modules — no inference logic is duplicated here.
    """
    st.set_page_config(
        page_title="Acoustic Fingerprinting Dashboard",
        page_icon="🔊",
        layout="wide",
    )

    st.title("🔊 Acoustic Fingerprinting — Industrial Machine Health Monitor")
    st.caption(
        "Detects deviations from a machine's healthy acoustic reference profile "
        "using BEATs embeddings and contrastive learning."
    )

    # ------------------------------------------------------------------
    # Sidebar — configuration inputs
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")

        dataset_root = st.text_input(
            "Dataset Root",
            value="data/raw/MIMII",
            help="Path to the MIMII dataset root directory.",
        )
        machine_type = st.text_input(
            "Machine Type",
            value="pump",
            help="Machine type (e.g. fan, pump, valve, slider).",
        )
        machine_id = st.text_input(
            "Machine ID",
            value="id_00",
            help="Machine identifier (e.g. id_00, id_02).",
        )
        checkpoint = st.text_input(
            "Projection Head Checkpoint",
            value="models/contrastive/best_projection_head.pt",
            help="Path to the trained ProjectionHead .pt checkpoint.",
        )
        max_recordings = st.number_input(
            "Maximum Healthy Recordings",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help="Maximum number of healthy recordings used to build the profile.",
        )
        uploaded_file = st.file_uploader(
            "Upload Audio File (.wav)",
            type=["wav"],
            help="WAV file to analyze against the healthy profile.",
        )

        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    # ------------------------------------------------------------------
    # Validate inputs before doing any work
    # ------------------------------------------------------------------
    if not analyze_clicked:
        st.info("Configure the sidebar and click **Analyze** to begin.")
        return

    if uploaded_file is None:
        st.error("Please upload a WAV file before clicking Analyze.")
        return

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        st.error(f"Checkpoint not found: `{checkpoint_path}`")
        return

    dataset_path = Path(dataset_root)
    if not dataset_path.exists():
        st.error(f"Dataset root not found: `{dataset_path}`")
        return

    # ------------------------------------------------------------------
    # Save uploaded file to a temp path so existing modules can read it
    # ------------------------------------------------------------------
    tmp_wav = _VIZ_TMP / uploaded_file.name
    tmp_wav.write_bytes(uploaded_file.read())

    audio_record = _make_audio_metadata(tmp_wav, machine_type, machine_id)

    # ------------------------------------------------------------------
    # Load cached resources (BEATs + ProjectionHead loaded once)
    # ------------------------------------------------------------------
    pipeline = _load_pipeline(checkpoint)
    drift_analyzer = _load_drift_analyzer(checkpoint)
    health_analyzer = _load_health_analyzer(checkpoint)
    benchmark = _load_benchmark(checkpoint)

    # ------------------------------------------------------------------
    # Build healthy learned profile
    # ------------------------------------------------------------------
    with st.spinner(f"Building healthy profile from up to {max_recordings} recordings…"):
        loader = DatasetLoader(dataset_root)
        builder = LearnedProfileBuilder(checkpoint_path=checkpoint)
        try:
            profile = builder.build(
                loader=loader,
                machine_type=machine_type,
                machine_id=machine_id,
                max_recordings=int(max_recordings),
                exclude_filenames={audio_record.filename},
            )
        except ValueError as exc:
            st.error(f"Profile build failed: {exc}")
            return

    # ------------------------------------------------------------------
    # Run inference pipeline + benchmark
    # ------------------------------------------------------------------
    with st.spinner("Running inference…"):
        t0 = time.perf_counter()
        try:
            pipeline_result = pipeline.analyze(audio_record, profile)
            drift_result = drift_analyzer.analyze(audio_record, profile)
            health_result = health_analyzer.analyze(audio_record, profile)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
            return
        inference_time = time.perf_counter() - t0

        cache_hit = drift_analyzer._cache.exists(audio_record)

        # Benchmark for dimension metadata
        try:
            bench = benchmark.benchmark(audio_record, profile)
            dsp_dim = bench.dsp_dimension
            beats_dim = bench.beats_dimension
            fusion_dim = bench.fusion_dimension
            embedding_dim = bench.embedding_dimension
        except Exception:  # noqa: BLE001
            dsp_dim, beats_dim, fusion_dim, embedding_dim = 153, 768, 921, 256

    # ------------------------------------------------------------------
    # Generate explanation
    # ------------------------------------------------------------------
    engine = ExplainabilityEngine()
    explanation = engine.explain(drift_result, health_result)

    st.success("Inference complete.")

    # ------------------------------------------------------------------
    # Render all sections
    # ------------------------------------------------------------------
    st.divider()
    _render_machine_info(
        machine_type=pipeline_result.machine_type,
        machine_id=pipeline_result.machine_id,
        filename=pipeline_result.filename,
    )

    st.divider()
    _render_health(
        health_score=pipeline_result.health_score,
        health_state=pipeline_result.health_state,
    )

    st.divider()
    _render_drift(
        raw_euclidean=pipeline_result.raw_euclidean,
        normalized_euclidean=pipeline_result.normalized_euclidean,
        raw_manhattan=pipeline_result.raw_manhattan,
        normalized_manhattan=pipeline_result.normalized_manhattan,
        cosine_similarity=pipeline_result.raw_cosine,
    )

    st.divider()
    _render_explanation(
        summary=explanation.summary,
        possible_causes=explanation.possible_causes,
        recommendation=explanation.recommendation,
    )

    st.divider()
    _render_performance(
        dsp_dim=dsp_dim,
        beats_dim=beats_dim,
        fusion_dim=fusion_dim,
        embedding_dim=embedding_dim,
        inference_time=inference_time,
        cache_hit=cache_hit,
    )

    # ------------------------------------------------------------------
    # Collect evaluation data and render visualizations
    # ------------------------------------------------------------------
    st.divider()
    with st.spinner("Collecting evaluation data for visualizations…"):
        health_scores, drift_scores, emb_matrix, embed_labels, y_true = _collect_eval_data(
            loader=loader,
            machine_type=machine_type,
            machine_id=machine_id,
            pipeline=pipeline,
            drift_analyzer=drift_analyzer,
            profile=profile,
        )

    _render_visualizations(
        health_scores=health_scores,
        drift_scores=drift_scores,
        emb_matrix=emb_matrix,
        embed_labels=embed_labels,
        y_true=y_true,
    )


if __name__ == "__main__":
    main()
