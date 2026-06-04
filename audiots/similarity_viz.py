"""
Visualization module for audio similarity analysis.

Companion to ``audiots.similarity`` — produces publication‑quality
matplotlib figures for every output of the similarity pipeline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# Set font to ensure no encoding issues - try multiple fonts
import matplotlib.font_manager as fm
# Try to find available fonts
available_fonts = [f.name for f in fm.fontManager.ttflist]
# Preferred fonts in order of preference
font_candidates = ['DejaVu Sans', 'Arial', 'Helvetica', 'Tahoma', 'Verdana', 'Calibri', 'Segoe UI', 'Microsoft YaHei', 'SimHei']
selected_font = 'DejaVu Sans'
for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break
plt.rcParams['font.family'] = selected_font
plt.rcParams['axes.unicode_minus'] = False
# Additional settings to prevent encoding issues
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------

def _similarity_cmap():
    """Divergent blue‑white‑red for similarity matrices."""
    colors = [(0.05, 0.05, 0.35), (0.2, 0.4, 0.8), (1.0, 1.0, 1.0),
              (0.95, 0.4, 0.2), (0.6, 0.05, 0.05)]
    return LinearSegmentedColormap.from_list("sim_div", colors, N=256)


# ---------------------------------------------------------------------------
# 1. Local similarity matrix
# ---------------------------------------------------------------------------

def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    times_1: np.ndarray,
    times_2: np.ndarray,
    annotations: Optional[List[Dict]] = None,
    aggregation: Optional[Dict] = None,
    save_path: Optional[str] = None,
    title: str = "Local Similarity Matrix",
) -> plt.Figure:
    """
    Heatmap of pairwise window similarity between two audio files.

    High‑similarity regions appear as bright spots; annotated segment
    pairs are overlaid as markers.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.pcolormesh(
        times_2, times_1, sim_matrix,
        cmap=_similarity_cmap(), shading="gouraud",
        vmin=0.0, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, label="Similarity (cosine)")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # Overlay matched pairs
    if annotations:
        ann_times_a = [a["time_a"] for a in annotations]
        ann_times_b = [a["time_b"] for a in annotations]
        ax.scatter(ann_times_b, ann_times_a, c="lime", edgecolors="black",
                   s=40, zorder=5, marker="o", alpha=0.9, label="Top matches")

    ax.set_xlabel("Audio 2 - Time (s)", fontsize=12)
    ax.set_ylabel("Audio 1 - Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14)

    if aggregation:
        gs = aggregation.get("global_similarity", 0)
        ax.text(
            0.98, 0.02,
            f"Global similarity: {gs:.1f}%",
            transform=ax.transAxes, fontsize=11,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

    if annotations:
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 2. High‑similarity segments overlaid on waveforms
# ---------------------------------------------------------------------------

def plot_similar_segments(
    y1: np.ndarray,
    sr1: int,
    y2: np.ndarray,
    sr2: int,
    annotations: List[Dict],
    window_size: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "High-Similarity Segment Pairs",
) -> plt.Figure:
    """
    Draw both waveforms and shade the top matching window pairs.
    """
    # Downsample for display
    max_display = 40000
    ds_factor_1 = max(1, len(y1) // max_display)
    ds_factor_2 = max(1, len(y2) // max_display)
    y1_ds = y1[::ds_factor_1]
    y2_ds = y2[::ds_factor_2]
    t1_ds = np.arange(len(y1_ds)) / sr1 * ds_factor_1
    t2_ds = np.arange(len(y2_ds)) / sr2 * ds_factor_2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(title, fontsize=14)

    # Waveform 1
    ax1.plot(t1_ds, y1_ds, color="#1f77b4", linewidth=0.6, alpha=0.9)
    ax1.set_ylabel("Audio 1 - Amplitude", fontsize=11)
    ax1.set_xlim(0, t1_ds[-1])

    # Waveform 2
    ax2.plot(t2_ds, y2_ds, color="#ff7f0e", linewidth=0.6, alpha=0.9)
    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Audio 2 - Amplitude", fontsize=11)
    ax2.set_xlim(0, t2_ds[-1])

    # Colour for each annotation based on similarity
    cmap = plt.cm.RdYlGn
    n_ann = min(len(annotations), 20)
    for k, ann in enumerate(annotations[:n_ann]):
        color = cmap(ann["similarity"])
        t_a = ann["time_a"]
        t_b = ann["time_b"]
        ax1.axvspan(t_a - window_size / 2, t_a + window_size / 2,
                    alpha=0.25, color=color, edgecolor="none")
        ax2.axvspan(t_b - window_size / 2, t_b + window_size / 2,
                    alpha=0.25, color=color, edgecolor="none")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.RdYlGn(1.0), alpha=0.4, label="High similarity"),
        Patch(facecolor=plt.cm.RdYlGn(0.5), alpha=0.4, label="Medium"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 3. Feature distribution comparison
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    feat_mat_1: np.ndarray,
    feat_mat_2: np.ndarray,
    per_dim_distance: np.ndarray,
    feature_names: List[str],
    top_n: int = 12,
    save_path: Optional[str] = None,
    title: str = "Feature Distribution Comparison (Top Discriminating Features)",
) -> plt.Figure:
    """
    Side‑by‑side violin/box plots for the features with largest
    distribution distance between the two audio files.
    """
    top_idx = np.argsort(per_dim_distance)[-top_n:][::-1]

    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    fig.suptitle(title, fontsize=14, y=1.01)
    axes = axes.flatten() if top_n > 1 else [axes]

    for k, idx in enumerate(top_idx):
        ax = axes[k]
        data = [feat_mat_1[:, idx], feat_mat_2[:, idx]]
        bp = ax.boxplot(
            data, labels=["A1", "A2"], patch_artist=True,
            widths=0.5, showfliers=False,
        )
        bp["boxes"][0].set_facecolor("#1f77b4")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#ff7f0e")
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(
            f"{feature_names[idx]}\n(dist={per_dim_distance[idx]:.3f})",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for k in range(top_n, len(axes)):
        axes[k].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 4. Pattern annotation (climax + repetition)
# ---------------------------------------------------------------------------

def plot_pattern_annotations(
    y: np.ndarray,
    sr: int,
    times: np.ndarray,
    patterns: Dict,
    feat_list: List[Dict],
    audio_label: str = "Audio",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Multi‑panel plot showing waveform with climax markers, RMS envelope,
    novelty curve, and onset density.
    """
    if title is None:
        title = f"{audio_label} - Hidden Pattern Annotations"

    max_display = 60000
    ds_factor = max(1, len(y) // max_display)
    y_ds = y[::ds_factor]
    t_ds = np.arange(len(y_ds)) / sr * ds_factor

    rms = np.array([f["rms_energy"] for f in feat_list])
    onset_d = np.array([f["onset_density"] for f in feat_list])
    novelty = patterns["novelty_curve"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # (a) Waveform + climax
    ax = axes[0]
    ax.plot(t_ds, y_ds, color="#1f77b4", linewidth=0.5, alpha=0.9)
    for ci in patterns["climax_indices"]:
        if ci < len(times):
            ax.axvline(times[ci], color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.legend(["Waveform", "Climax"], loc="upper right", fontsize=8)
    ax.set_xlim(0, t_ds[-1])

    # (b) RMS envelope
    ax = axes[1]
    ax.plot(times, rms, color="#ff7f0e", linewidth=1.2)
    ax.fill_between(times, 0, rms, alpha=0.15, color="#ff7f0e")
    ax.set_ylabel("RMS Energy", fontsize=10)
    ax.set_ylim(0, None)

    # (c) Novelty curve
    ax = axes[2]
    ax.plot(times, novelty, color="#2ca02c", linewidth=1.2)
    for b in patterns["segment_boundaries"]:
        if b < len(times):
            ax.axvline(times[b], color="purple", linestyle=":", alpha=0.6)
    ax.set_ylabel("Novelty", fontsize=10)
    ax.legend(["Novelty", "Boundary"], loc="upper right", fontsize=8)

    # (d) Onset density
    ax = axes[3]
    ax.plot(times, onset_d, color="#d62728", linewidth=1.2)
    ax.set_ylabel("Onset Density", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 5. Summary dashboard
# ---------------------------------------------------------------------------

def plot_similarity_summary(
    results: Dict,
    save_path: Optional[str] = None,
    title: str = "Audio Similarity Analysis - Summary Dashboard",
) -> plt.Figure:
    """
    Single‑figure dashboard combining:
      - Local similarity matrix (top‑left)
      - Matching score breakdown (top‑right bar chart)
      - Feature distribution top‑discriminators (bottom‑left)
      - Annotation summary table (bottom‑right)
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    agg = results["aggregation"]
    sim_matrix = results["local_similarity_matrix"]
    times_1 = results["window_times_1"]
    times_2 = results["window_times_2"]
    annotations = results["annotations"]
    feature_names = results["feature_names"]
    per_dim_dist = results["per_dim_distance"]
    feat_mat_1 = results["feature_matrix_1"]
    feat_mat_2 = results["feature_matrix_2"]

    # ---- Top‑left: Similarity matrix (small) ----
    ax_mat = fig.add_axes([0.05, 0.52, 0.40, 0.40])
    im = ax_mat.pcolormesh(
        times_2, times_1, sim_matrix,
        cmap=_similarity_cmap(), shading="gouraud", vmin=0, vmax=1,
    )
    ax_mat.set_xlabel("Audio 2 (s)", fontsize=9)
    ax_mat.set_ylabel("Audio 1 (s)", fontsize=9)
    ax_mat.set_title("Local similarity matrix", fontsize=11)
    ax_mat.text(
        0.98, 0.02,
        f"Global: {agg['global_similarity']:.1f}%",
        transform=ax_mat.transAxes, fontsize=10, ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.colorbar(im, ax=ax_mat, shrink=0.8, label="Similarity")

    # ---- Top‑right: Score breakdown bar chart ----
    ax_bar = fig.add_axes([0.52, 0.55, 0.44, 0.37])
    score_labels = [
        "Global\nSimilarity",
        "Top‑K\nMatching",
        "Entropy\nWeighted",
        "Distribution\nOverlap",
    ]
    score_values = [
        agg["global_similarity"],
        agg["top_k_match_score"] * 100,
        agg["entropy_weighted_score"] * 100,
        agg["distribution_similarity"] * 100,
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    bars = ax_bar.bar(score_labels, score_values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, score_values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax_bar.set_ylabel("Score (%)", fontsize=10)
    ax_bar.set_title("Similarity score breakdown", fontsize=11)
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(axis="y", alpha=0.3)

    # ---- Bottom‑left: Top discriminating features ----
    ax_dist = fig.add_axes([0.05, 0.08, 0.40, 0.36])
    top_idx = np.argsort(per_dim_dist)[-8:][::-1]
    y_pos = np.arange(len(top_idx))
    ax_dist.barh(
        y_pos, per_dim_dist[top_idx],
        color="#d62728", alpha=0.7, edgecolor="black",
    )
    ax_dist.set_yticks(y_pos)
    ax_dist.set_yticklabels(
        [feature_names[i][:30] for i in top_idx], fontsize=7
    )
    ax_dist.set_xlabel("Wasserstein distance", fontsize=9)
    ax_dist.set_title("Top discriminating features", fontsize=11)
    ax_dist.invert_yaxis()
    ax_dist.grid(axis="x", alpha=0.3)

    # ---- Bottom‑right: Annotation table text ----
    ax_table = fig.add_axes([0.52, 0.08, 0.44, 0.36])
    ax_table.axis("off")
    ax_table.set_title("Top high‑similarity segment pairs", fontsize=11, loc="left")

    if annotations:
        n_show = min(8, len(annotations))
        col_labels = ["T₁ (s)", "T₂ (s)", "Sim", "Rhythm", "Pitch"]
        cell_text = []
        for ann in annotations[:n_show]:
            cell_text.append([
                f"{ann['time_a']:.2f}",
                f"{ann['time_b']:.2f}",
                f"{ann['similarity']:.3f}",
                ann["rhythm"].split("/")[0].strip(),
                ann["dominant_pitch"],
            ])

        tbl = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colWidths=[0.12, 0.12, 0.10, 0.42, 0.12],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.3)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#404040")
                cell.set_text_props(color="white", fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# 6. Report generator
# ---------------------------------------------------------------------------

def generate_similarity_report_plots(
    results: Dict,
    output_dir: str = "outputs/similarity",
    y1: Optional[np.ndarray] = None,
    y2: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> List[str]:
    """
    Generate all similarity visualizations and save them to *output_dir*.

    Parameters
    ----------
    results : dict   Output of ``analyze_similarity()``.
    output_dir : str Directory to save PNG files.
    y1, y2 : ndarray, optional  Raw waveforms (for segment overlay).
    sr : int, optional          Sample rate.

    Returns
    -------
    saved_files : list of str   Basenames of saved PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    # 1. Similarity matrix
    path = os.path.join(output_dir, "similarity_matrix.png")
    plot_similarity_matrix(
        results["local_similarity_matrix"],
        results["window_times_1"],
        results["window_times_2"],
        annotations=results["annotations"],
        aggregation=results["aggregation"],
        save_path=path,
    )
    saved.append("similarity_matrix.png")

    # 2. Feature distributions
    path = os.path.join(output_dir, "feature_distributions.png")
    plot_feature_distributions(
        results["feature_matrix_1"],
        results["feature_matrix_2"],
        results["per_dim_distance"],
        results["feature_names"],
        top_n=12,
        save_path=path,
    )
    saved.append("feature_distributions.png")

    # 3. Pattern annotations for audio 1
    path = os.path.join(output_dir, "patterns_audio1.png")
    plot_pattern_annotations(
        y1 if y1 is not None else np.zeros(1000),
        sr or 16000,
        results["window_times_1"],
        results["patterns_1"],
        results["feature_list_1"],
        audio_label="Audio 1",
        save_path=path,
    )
    saved.append("patterns_audio1.png")

    # 4. Pattern annotations for audio 2
    path = os.path.join(output_dir, "patterns_audio2.png")
    plot_pattern_annotations(
        y2 if y2 is not None else np.zeros(1000),
        sr or 16000,
        results["window_times_2"],
        results["patterns_2"],
        results["feature_list_2"],
        audio_label="Audio 2",
        save_path=path,
    )
    saved.append("patterns_audio2.png")

    # 5. High‑similarity segments overlay
    if y1 is not None and y2 is not None and sr is not None:
        path = os.path.join(output_dir, "similar_segments.png")
        plot_similar_segments(
            y1, sr, y2, sr,
            results["annotations"],
            window_size=results["params"]["window_size"],
            save_path=path,
        )
        saved.append("similar_segments.png")

    # 6. Summary dashboard
    path = os.path.join(output_dir, "similarity_summary.png")
    plot_similarity_summary(results, save_path=path)
    saved.append("similarity_summary.png")

    plt.close("all")
    return saved


def save_similarity_plots(
    results: Dict,
    output_dir: str = "outputs/similarity",
    y1: Optional[np.ndarray] = None,
    y2: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
):
    """
    Convenience wrapper — generate and save all similarity plots.
    (Alias for generate_similarity_report_plots.)
    """
    return generate_similarity_report_plots(results, output_dir, y1, y2, sr)
