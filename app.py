"""Streamlit dashboard - Acoustic Fingerprinting of Industrial Machines.

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

from src.dataset.loader import DatasetLoader
from src.dataset.metadata import AudioMetadata
from src.explainability.explainer import ExplainabilityEngine
from src.learned_drift.analyzer import LearnedDriftAnalyzer
from src.learned_health_index.analyzer import LearnedHealthAnalyzer
from src.learned_profile.builder import LearnedProfileBuilder
from src.pipeline.pipeline import InferencePipeline
from src.visualization.visualizer import ResultVisualizer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

_CHECKPOINT    = "models/contrastive/best_projection_head.pt"
_DATASET_ROOT  = "data/raw/MIMII"
_VIZ_TMP       = Path(tempfile.gettempdir()) / "af_dashboard_viz"
_VIZ_TMP.mkdir(parents=True, exist_ok=True)
_EVAL_LIMIT    = 50

# ---------------------------------------------------------------------------
# CSS  — light professional theme
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Page background ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #f5f7fa;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* ── Remove default top padding ─────────────────────────────── */
.block-container { padding-top: 1.5rem !important; }

/* ── Header banner ───────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 60%, #2563eb 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
}
.app-header-icon { font-size: 2.8rem; line-height: 1; }
.app-header-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
}
.app-header-sub {
    font-size: 0.95rem;
    color: #bfdbfe;
    margin: 0.2rem 0 0 0;
}

/* ── Step card ───────────────────────────────────────────────── */
.step-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem 1.75rem 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    height: 100%;
}
.step-number {
    display: inline-block;
    background: #1d4ed8;
    color: #ffffff;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.6rem;
}
.step-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0 0 0.25rem 0;
}
.step-desc {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0 0 1rem 0;
}

/* ── Welcome info box ────────────────────────────────────────── */
.welcome-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    display: flex;
    gap: 1.25rem;
    align-items: flex-start;
}
.welcome-icon { font-size: 2rem; line-height: 1.2; }
.welcome-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1e40af;
    margin: 0 0 0.35rem 0;
}
.welcome-text {
    font-size: 0.88rem;
    color: #374151;
    margin: 0;
    line-height: 1.6;
}
.welcome-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.75rem;
}
.pill {
    background: #dbeafe;
    color: #1e40af;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
}

/* ── Result: status card ─────────────────────────────────────── */
.status-card {
    border-radius: 14px;
    padding: 2rem 1.5rem;
    text-align: center;
}
.status-healthy {
    background: #f0fdf4;
    border: 2px solid #22c55e;
}
.status-abnormal {
    background: #fff1f2;
    border: 2px solid #ef4444;
}
.status-icon { font-size: 3rem; line-height: 1; margin-bottom: 0.5rem; }
.status-label {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0.3rem 0 0.4rem;
    letter-spacing: 0.02em;
}
.status-healthy .status-label { color: #16a34a; }
.status-abnormal .status-label { color: #dc2626; }
.status-desc { font-size: 0.88rem; color: #64748b; }

/* ── Metric card ─────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #1e293b;
}
.metric-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}

/* ── Section heading ─────────────────────────────────────────── */
.section-heading {
    font-size: 0.78rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.45rem;
    margin: 1.75rem 0 0.9rem;
}

/* ── Detail table row ────────────────────────────────────────── */
.detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.87rem;
}
.detail-key { color: #64748b; }
.detail-val { color: #1e293b; font-weight: 600; }

/* ── Sidebar info ────────────────────────────────────────────── */
.sidebar-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-top: 0.5rem;
}
.sidebar-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.sidebar-key { color: #64748b; }
.sidebar-val { color: #1e293b; font-weight: 600; }
</style>
"""

# ---------------------------------------------------------------------------
# Cached resource loaders — each loaded exactly once per session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading BEATs model and ProjectionHead...")
def _load_pipeline(checkpoint: str) -> InferencePipeline:
    return InferencePipeline(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading drift analyzer...")
def _load_drift_analyzer(checkpoint: str) -> LearnedDriftAnalyzer:
    return LearnedDriftAnalyzer(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Loading health analyzer...")
def _load_health_analyzer(checkpoint: str) -> LearnedHealthAnalyzer:
    return LearnedHealthAnalyzer(checkpoint_path=checkpoint)


@st.cache_resource(show_spinner="Building healthy machine profile...")
def _get_profile(machine_type: str, machine_id: str, max_recordings: int, checkpoint: str):
    """Build and cache the healthy profile. Re-runs only when key changes."""
    loader  = DatasetLoader(_DATASET_ROOT)
    builder = LearnedProfileBuilder(checkpoint_path=checkpoint)
    return builder.build(
        loader=loader,
        machine_type=machine_type,
        machine_id=machine_id,
        max_recordings=max_recordings,
    )


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------


def _make_audio_metadata(wav_path: Path, machine_type: str, machine_id: str) -> AudioMetadata:
    return AudioMetadata(
        machine_type=machine_type,
        machine_id=machine_id,
        label="upload",
        filename=wav_path.name,
        relative_path=wav_path,
        absolute_path=wav_path,
        is_uploaded=True,
    )


def _metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{sub_html}</div>'
    )


def _detail_row(key: str, val: str) -> str:
    return (
        f'<div class="detail-row">'
        f'<span class="detail-key">{key}</span>'
        f'<span class="detail-val">{val}</span>'
        f'</div>'
    )


def _sidebar_row(key: str, val: str) -> str:
    return (
        f'<div class="sidebar-row">'
        f'<span class="sidebar-key">{key}</span>'
        f'<span class="sidebar-val">{val}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Acoustic Fingerprinting",
        page_icon="🔊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar — system info only ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### System Information")
        st.markdown(
            '<div class="sidebar-section">'
            + _sidebar_row("Model", "BEATs + DSP Fusion")
            + _sidebar_row("Embedding", "256 dimensions")
            + _sidebar_row("Metric", "Norm. Euclidean")
            + _sidebar_row("Threshold", "Score &lt; 50")
            + _sidebar_row("Runtime", "CPU")
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "<small style='color:#64748b;line-height:1.7'>"
            "Learns only from <b>healthy</b> recordings. "
            "Detects faults by measuring acoustic drift from "
            "the machine's healthy reference profile using "
            "contrastive embeddings."
            "</small>",
            unsafe_allow_html=True,
        )

    # ── Header banner ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-header">'
        '<div class="app-header-icon">🔊</div>'
        '<div>'
        '<p class="app-header-title">Acoustic Fingerprinting</p>'
        '<p class="app-header-sub">AI-Based Industrial Machine Health Monitoring</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Three-step workflow cards ──────────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<span class="step-number">STEP 1</span>', unsafe_allow_html=True)
        st.markdown('<p class="step-title">Select Machine</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="step-desc">Choose the machine type and ID to analyze.</p>',
            unsafe_allow_html=True,
        )
        machine_type = st.selectbox(
            "Machine Type",
            ["pump", "fan", "valve", "slider"],
            index=0,
            label_visibility="collapsed",
        )
        machine_id = st.selectbox(
            "Machine ID",
            ["id_00", "id_02", "id_04", "id_06"],
            index=0,
            label_visibility="collapsed",
        )
        max_recordings = st.number_input(
            "Max Healthy Recordings",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of normal recordings used to build the healthy reference profile.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<span class="step-number">STEP 2</span>', unsafe_allow_html=True)
        st.markdown('<p class="step-title">Upload Audio</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="step-desc">Upload a WAV recording from the selected machine.</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "WAV file",
            type=["wav"],
            label_visibility="collapsed",
        )
        if uploaded_file:
            st.success(f"Ready: **{uploaded_file.name}**")
        else:
            st.info("Accepted format: .wav")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<span class="step-number">STEP 3</span>', unsafe_allow_html=True)
        st.markdown('<p class="step-title">Analyze Machine Health</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="step-desc">'
            "Run the acoustic fingerprint analysis against the healthy reference profile."
            "</p>",
            unsafe_allow_html=True,
        )
        analyze_clicked = st.button(
            "Analyze Audio",
            type="primary",
            use_container_width=True,
        )
        st.markdown(
            "<small style='color:#94a3b8'>"
            "The healthy profile is built once and cached. "
            "Only the uploaded file is processed on each run."
            "</small>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Welcome / idle state ───────────────────────────────────────────────
    if not analyze_clicked:
        st.markdown(
            '<div class="welcome-box">'
            '<div class="welcome-icon">&#128268;</div>'
            '<div>'
            '<p class="welcome-title">How it works</p>'
            '<p class="welcome-text">'
            "Upload an industrial machine audio recording to detect deviations from its "
            "healthy acoustic fingerprint. The system uses a frozen BEATs encoder combined "
            "with DSP features and contrastive learning to produce a 256-dimensional "
            "fingerprint, then measures drift from the machine's healthy reference profile."
            "</p>"
            '<div class="welcome-pills">'
            '<span class="pill">BEATs Encoder</span>'
            '<span class="pill">DSP Features</span>'
            '<span class="pill">Contrastive Learning</span>'
            '<span class="pill">256-dim Fingerprint</span>'
            '<span class="pill">Drift Analysis</span>'
            "</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Input validation ───────────────────────────────────────────────────
    if uploaded_file is None:
        st.error("Please upload a WAV file in Step 2 before clicking Analyze.")
        return

    if not Path(_CHECKPOINT).exists():
        st.error(f"Checkpoint not found: `{_CHECKPOINT}`")
        return

    if not Path(_DATASET_ROOT).exists():
        st.error(f"Dataset root not found: `{_DATASET_ROOT}`")
        return

    # ── Save uploaded WAV to temp path ─────────────────────────────────────
    tmp_wav = _VIZ_TMP / Path(uploaded_file.name).name
    tmp_wav.write_bytes(uploaded_file.read())
    audio_record = _make_audio_metadata(tmp_wav, machine_type, machine_id)

    # ── Load cached model resources (loaded once per session) ──────────────
    pipeline       = _load_pipeline(_CHECKPOINT)
    drift_analyzer = _load_drift_analyzer(_CHECKPOINT)
    health_analyzer = _load_health_analyzer(_CHECKPOINT)

    # ── Load/build healthy profile (cached per machine + max_recordings) ───
    try:
        profile = _get_profile(machine_type, machine_id, int(max_recordings), _CHECKPOINT)
    except ValueError as exc:
        st.error(f"Profile build failed: {exc}")
        return

    # ── Run inference on the uploaded file only ────────────────────────────
    with st.spinner("Analyzing audio recording..."):
        t0 = time.perf_counter()
        try:
            pipeline_result = pipeline.analyze(audio_record, profile)
            drift_result    = drift_analyzer.analyze(audio_record, profile)
            health_result   = health_analyzer.analyze(audio_record, profile)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
            return
        inference_time_ms = (time.perf_counter() - t0) * 1000

    engine      = ExplainabilityEngine()
    explanation = engine.explain(drift_result, health_result)

    health_score   = pipeline_result.health_score
    health_state   = pipeline_result.health_state
    norm_euclidean = pipeline_result.normalized_euclidean
    is_healthy     = health_score >= 50.0
    threshold      = 50.0

    # ── Results section heading ────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    # ── Status card + metric cards ─────────────────────────────────────────
    res_left, res_right = st.columns([1, 2], gap="large")

    with res_left:
        if is_healthy:
            st.markdown(
                '<div class="status-card status-healthy">'
                '<div class="status-icon">&#9989;</div>'
                '<div class="status-label">HEALTHY</div>'
                '<div class="status-desc">Within normal operating range</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-card status-abnormal">'
                '<div class="status-icon">&#9888;&#65039;</div>'
                '<div class="status-label">ABNORMAL</div>'
                '<div class="status-desc">Deviation detected from healthy profile</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    with res_right:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                _metric_card("Health Score", f"{health_score:.1f}", f"out of 100  \u00b7  {health_state}"),
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                _metric_card("Drift Score", f"{norm_euclidean:.4f}", "Normalized Euclidean"),
                unsafe_allow_html=True,
            )
        r3, r4 = st.columns(2)
        with r3:
            st.markdown(
                _metric_card(
                    "Prediction",
                    "HEALTHY" if is_healthy else "ABNORMAL",
                    f"Threshold: score < {threshold:.0f}",
                ),
                unsafe_allow_html=True,
            )
        with r4:
            st.markdown(
                _metric_card("Processing Time", f"{inference_time_ms:.0f} ms", "Uploaded file only"),
                unsafe_allow_html=True,
            )

    # ── Health score progress bar ──────────────────────────────────────────
    st.progress(
        int(health_score),
        text=f"Health Score: {health_score:.1f} / 100  ({health_state})",
    )

    # ── Recommendation ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-heading">Recommendation</div>',
        unsafe_allow_html=True,
    )
    if is_healthy:
        st.success(f"**{explanation.summary}**  \n{explanation.recommendation}")
    else:
        st.error(f"**{explanation.summary}**  \n{explanation.recommendation}")
        if explanation.possible_causes:
            st.warning(
                "**Possible causes:** " + "  \u00b7  ".join(explanation.possible_causes)
            )

    # ── Technical details (expandable) ────────────────────────────────────
    with st.expander("Technical Details"):
        td1, td2 = st.columns(2)
        with td1:
            st.markdown(
                "".join([
                    _detail_row("Machine Type",       pipeline_result.machine_type),
                    _detail_row("Machine ID",         pipeline_result.machine_id),
                    _detail_row("File",               pipeline_result.filename),
                    _detail_row("Embedding Dim",      str(pipeline_result.embedding_dimension)),
                    _detail_row("Fusion Dim",         str(pipeline_result.fusion_dimension)),
                ]),
                unsafe_allow_html=True,
            )
        with td2:
            st.markdown(
                "".join([
                    _detail_row("Detection Metric",   "Normalized Euclidean"),
                    _detail_row("Anomaly Threshold",  f"Score < {threshold:.0f}"),
                    _detail_row("Raw Euclidean",      f"{pipeline_result.raw_euclidean:.4f}"),
                    _detail_row("Norm. Euclidean",    f"{norm_euclidean:.4f}"),
                    _detail_row("Cosine Similarity",  f"{pipeline_result.raw_cosine:.4f}"),
                ]),
                unsafe_allow_html=True,
            )

    # ── Visualizations (optional, evaluation dataset) ─────────────────────
    with st.expander("Visualizations  (Evaluation Dataset)"):
        with st.spinner("Running evaluation recordings for plots..."):
            loader      = DatasetLoader(_DATASET_ROOT)
            all_records = loader.get_all_files()
            normal_recs = [
                r for r in all_records
                if r.machine_type == machine_type
                and r.machine_id == machine_id
                and r.label == "normal"
            ][:_EVAL_LIMIT]
            abnormal_recs = [
                r for r in all_records
                if r.machine_type == machine_type
                and r.machine_id == machine_id
                and r.label == "abnormal"
            ][:_EVAL_LIMIT]

            health_scores: list[float] = []
            drift_scores:  list[float] = []
            embeddings:    list[np.ndarray] = []
            embed_labels:  list[str] = []
            y_true:        list[int] = []
            inference_obj  = drift_analyzer._inference

            for rec, y_val, lbl in (
                [(r, 0, "healthy")  for r in normal_recs]
                + [(r, 1, "abnormal") for r in abnormal_recs]
            ):
                try:
                    res = pipeline.analyze(rec, profile)
                    health_scores.append(res.health_score)
                    drift_scores.append(res.normalized_euclidean)
                    y_true.append(y_val)
                    embed_labels.append(lbl)
                    fused = drift_analyzer._cache.load_or_create(rec)
                    embeddings.append(inference_obj.generate_fingerprint(fused))
                except Exception:  # noqa: BLE001
                    pass

        if not health_scores:
            st.info("No evaluation recordings found for this machine.")
        else:
            vis       = ResultVisualizer()
            filenames = [f"rec_{i}" for i in range(len(health_scores))]
            emb_matrix = (
                np.stack(embeddings, axis=0).astype(np.float32)
                if embeddings else np.empty((0, 256))
            )

            v1, v2 = st.columns(2)
            p_health = _VIZ_TMP / "health_scores.png"
            vis.plot_health_scores(health_scores, filenames, p_health)
            v1.image(str(p_health), caption="Health Score Trend", use_container_width=True)

            p_drift = _VIZ_TMP / "drift_scores.png"
            vis.plot_drift_scores(drift_scores, filenames, p_drift)
            v2.image(str(p_drift), caption="Drift Score Trend", use_container_width=True)

            if emb_matrix.shape[0] >= 2:
                p_pca = _VIZ_TMP / "embedding_pca.png"
                vis.plot_embedding_distribution(emb_matrix, embed_labels, p_pca)
                st.image(str(p_pca), caption="Embedding Distribution (PCA)", use_container_width=True)

            if len(set(y_true)) == 2:
                y_pred        = [1 if s < 50.0 else 0 for s in health_scores]
                anomaly_scores = [100.0 - s for s in health_scores]
                c1, c2 = st.columns(2)
                p_cm = _VIZ_TMP / "confusion_matrix.png"
                vis.plot_confusion_matrix(y_true, y_pred, p_cm)
                c1.image(str(p_cm), caption="Confusion Matrix", use_container_width=True)
                p_roc = _VIZ_TMP / "roc_curve.png"
                vis.plot_roc_curve(y_true, anomaly_scores, p_roc)
                c2.image(str(p_roc), caption="ROC Curve", use_container_width=True)


if __name__ == "__main__":
    main()
