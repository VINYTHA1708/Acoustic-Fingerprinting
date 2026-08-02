"""ResultVisualizer — publication-quality plots for acoustic fingerprinting results.

Provides five plot methods:
    plot_health_scores         — line chart of health score trend
    plot_drift_scores          — line chart of normalized Euclidean drift trend
    plot_embedding_distribution — PCA scatter of healthy vs abnormal embeddings
    plot_confusion_matrix      — annotated confusion matrix heatmap
    plot_roc_curve             — ROC curve with AUC annotation

All methods save figures as PNG files and close the figure after saving to
avoid memory accumulation across multiple calls.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Generates and saves publication-quality PNG plots for pipeline results.

    All methods are stateless — each call creates, saves, and closes its own
    figure.  No state is stored between calls.
    """

    # ------------------------------------------------------------------
    # 1. Health Score Trend
    # ------------------------------------------------------------------

    def plot_health_scores(
        self,
        scores: list[float],
        labels: list[str],
        output_path: Path,
    ) -> None:
        """Plot health score trend as a line chart and save to PNG.

        Args:
            scores: Health scores in [0, 100], one per recording.
            labels: Recording labels (used for x-tick display when few points).
            output_path: Destination PNG file path.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(scores)), scores, marker="o", markersize=3, linewidth=1.2)
        ax.set_title("Health Score Trend")
        ax.set_xlabel("Recording Index")
        ax.set_ylabel("Health Score")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved health score trend → %s", output_path)

    # ------------------------------------------------------------------
    # 2. Drift Trend
    # ------------------------------------------------------------------

    def plot_drift_scores(
        self,
        drift_scores: list[float],
        labels: list[str],
        output_path: Path,
    ) -> None:
        """Plot normalized Euclidean drift trend as a line chart and save to PNG.

        Args:
            drift_scores: Normalized Euclidean drift values, one per recording.
            labels: Recording labels (used for x-tick display when few points).
            output_path: Destination PNG file path.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(drift_scores)), drift_scores, marker="o", markersize=3,
                linewidth=1.2, color="tab:orange")
        ax.set_title("Normalized Euclidean Drift Trend")
        ax.set_xlabel("Recording Index")
        ax.set_ylabel("Normalized Euclidean Drift")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved drift trend → %s", output_path)

    # ------------------------------------------------------------------
    # 3. Embedding Distribution (PCA)
    # ------------------------------------------------------------------

    def plot_embedding_distribution(
        self,
        embeddings: np.ndarray,
        labels: list[str],
        output_path: Path,
    ) -> None:
        """Scatter plot of 256-dim embeddings reduced to 2D via PCA.

        Args:
            embeddings: Float32 array of shape ``(N, 256)``.
            labels: Per-row label string — ``"healthy"`` or ``"abnormal"``.
            output_path: Destination PNG file path.
        """
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)  # (N, 2)

        label_arr = np.array(labels)
        fig, ax = plt.subplots(figsize=(7, 6))

        for tag, color, marker in [
            ("healthy", "tab:blue", "o"),
            ("abnormal", "tab:red", "^"),
        ]:
            mask = label_arr == tag
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, marker=marker, s=20, alpha=0.7, label=tag,
                )

        ax.set_title("Embedding Distribution (PCA)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved embedding PCA → %s", output_path)

    # ------------------------------------------------------------------
    # 4. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: list[int],
        y_pred: list[int],
        output_path: Path,
    ) -> None:
        """Plot an annotated confusion matrix and save to PNG.

        Positive class = 1 (abnormal).  Negative class = 0 (healthy).

        Args:
            y_true: Ground-truth binary labels (0 = healthy, 1 = abnormal).
            y_pred: Predicted binary labels.
            output_path: Destination PNG file path.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        tick_labels = ["Healthy (0)", "Abnormal (1)"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        cell_labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, cell_labels[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=11,
                )

        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved confusion matrix → %s", output_path)

    # ------------------------------------------------------------------
    # 5. ROC Curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: list[int],
        scores: list[float],
        output_path: Path,
    ) -> None:
        """Plot the ROC curve with AUC annotation and save to PNG.

        Args:
            y_true: Ground-truth binary labels (0 = healthy, 1 = abnormal).
            scores: Anomaly scores (higher = more likely abnormal).
            output_path: Destination PNG file path.
        """
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved ROC curve → %s", output_path)
