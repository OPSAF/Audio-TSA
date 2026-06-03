"""
Discovery Visualisation Module
==============================

Companion to ``audiots.discovery``.  Produces evidence-backed figures
that show *what* was discovered and *why*, rather than a single score.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

DIM_COLORS = {
    "energy":      "#e74c3c",
    "brightness":  "#3498db",
    "complexity":  "#2ecc71",
    "rhythm":      "#9b59b6",
    "structure":   "#e67e22",
}

SEGMENT_COLORS = {
    "climax":      "#e74c3c",
    "calm":        "#3498db",
    "buildup":     "#e67e22",
    "release":     "#1abc9c",
    "transition":  "#9b59b6",
    "sustained":   "#95a5a6",
}


# ---------------------------------------------------------------------------
# 1. Self-discovery: character + segments
# ---------------------------------------------------------------------------

def plot_self_discovery(
    self_result: Dict,
    save_path: Optional[str] = None,
    title: str = "Audio Self-Discovery",
) -> plt.Figure:
    """
    Show a single audio's structural segments, character profile,
    and repeated motifs overlaid on its trend curves.
    """
    dyn = self_result["dynamics"]
    segments_raw = self_result.get("segments_raw", {})
    character = self_result.get("character")
    motifs = self_result.get("motifs", [])
    times = dyn["times"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # ---- Top panel: 4 normalised trends + segment shading ----
    ax_trends = fig.add_axes([0.05, 0.45, 0.65, 0.50])
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    for key in trend_keys:
        norm_key = f"{key}_norm"
        data = dyn.get(norm_key, dyn[key])
        ax_trends.plot(times, data, color=DIM_COLORS[key],
                       linewidth=1.2, alpha=0.85, label=key)
    ax_trends.axhline(y=0, color="black", linewidth=0.5, linestyle=":")
    ax_trends.set_ylabel("Z-score", fontsize=10)
    ax_trends.set_xlabel("Time (s)", fontsize=10)
    ax_trends.legend(loc="upper right", fontsize=8, ncol=4)
    ax_trends.grid(alpha=0.2)
    ax_trends.set_title("Normalised trend curves + structural segments", fontsize=11)

    # Shade segments
    if segments_raw:
        labels = segments_raw.get("labels", [])
        hop = dyn["params"]["hop_size"]
        for i, lbl in enumerate(labels):
            if lbl in SEGMENT_COLORS and i < len(times):
                ax_trends.axvspan(
                    times[i] - hop/2, times[i] + hop/2,
                    alpha=0.10, color=SEGMENT_COLORS[lbl], edgecolor="none",
                )

    # ---- Top-right: Character profile text ----
    ax_char = fig.add_axes([0.73, 0.50, 0.25, 0.45])
    ax_char.axis("off")
    if character:
        lines = [
            ("Rhythm", character.rhythm_signature),
            ("Energy", character.energy_profile),
            ("Timbre", character.timbre_quality),
        ]
        y = 0.85
        ax_char.text(0.0, 0.95, "Character Profile",
                     fontsize=12, fontweight="bold", transform=ax_char.transAxes)
        for label, value in lines:
            ax_char.text(0.0, y, f"{label}:", fontsize=9, fontweight="bold",
                         transform=ax_char.transAxes, color=DIM_COLORS.get(label.lower(), "black"))
            # Word-wrap value
            words = value.split()
            line = ""
            line_y = y - 0.06
            for w in words:
                test = f"{line} {w}".strip()
                if len(test) > 40:
                    ax_char.text(0.02, line_y, line, fontsize=8,
                                 transform=ax_char.transAxes)
                    line_y -= 0.05
                    line = w
                else:
                    line = test
            if line:
                ax_char.text(0.02, line_y, line, fontsize=8,
                             transform=ax_char.transAxes)
            y = line_y - 0.08

    # ---- Bottom-left: Energy trend with peak/valley markers ----
    ax_energy = fig.add_axes([0.05, 0.08, 0.40, 0.30])
    energy = dyn["energy"]
    ax_energy.plot(times, energy, color="#e74c3c", linewidth=1.2)
    ax_energy.fill_between(times, 0, energy, alpha=0.08, color="#e74c3c")
    # Mark climax points
    seg_data = self_result.get("segments", {})
    for ci in seg_data.get("climax", []):
        ax_energy.axvline(ci["time"], color="#e74c3c", linestyle="--", alpha=0.6, linewidth=0.8)
    for ci in seg_data.get("calm", []):
        ax_energy.axvline(ci["time"], color="#3498db", linestyle=":", alpha=0.5, linewidth=0.8)
    ax_energy.set_ylabel("RMS Energy", fontsize=10)
    ax_energy.set_xlabel("Time (s)", fontsize=10)
    ax_energy.set_title("Energy + structural markers", fontsize=11)
    ax_energy.grid(alpha=0.2)

    # ---- Bottom-right: Motif summary ----
    ax_motif = fig.add_axes([0.55, 0.08, 0.40, 0.30])
    ax_motif.axis("off")
    if motifs:
        ax_motif.text(0.0, 0.95, f"Repeated Motifs ({len(motifs)} found)",
                      fontsize=11, fontweight="bold", transform=ax_motif.transAxes)
        n_show = min(6, len(motifs))
        for i, m in enumerate(motifs[:n_show]):
            y = 0.82 - i * 0.12
            ax_motif.text(
                0.0, y,
                f"  {m['start_a']:.1f}s <-> {m['start_b']:.1f}s  "
                f"(lag {m.get('lag_seconds', 0):.1f}s, sim={m['similarity']:.3f})",
                fontsize=8, transform=ax_motif.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", alpha=0.7),
            )
    else:
        ax_motif.text(0.5, 0.5, "No repeated motifs found",
                      ha="center", va="center", transform=ax_motif.transAxes,
                      fontsize=10, color="gray")

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 2. Cross-discovery: segment mapping
# ---------------------------------------------------------------------------

def plot_segment_mapping(
    discoveries: List,
    dyn1: Dict,
    dyn2: Dict,
    save_path: Optional[str] = None,
    title: str = "Cross-Audio — Discovered Segment Correspondences",
) -> plt.Figure:
    """
    For each dimension, show the trend curves of both audios and draw
    arrows between corresponding segments that were discovered.
    """
    # Filter to discoveries with actual matches
    dims_with_matches = [
        d for d in discoveries
        if d.segment_matches and d.meta.get("n_matches", 0) > 0
    ]

    if not dims_with_matches:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No corresponding segments discovered between the two audios.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig

    n = len(dims_with_matches)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    times1 = dyn1["times"]
    times2 = dyn2["times"]

    for row, disc in enumerate(dims_with_matches):
        ax = axes[row, 0]
        dim_key = disc.dimension
        color = DIM_COLORS.get(dim_key, "#333333")

        a = dyn1[dim_key]
        b = dyn2[dim_key]

        # Normalize for overlay comparison
        a_norm = (a - a.min()) / (a.max() - a.min() + 1e-12)
        b_norm = (b - b.min()) / (b.max() - b.min() + 1e-12)

        ax.plot(times1, a_norm, color=color, linewidth=1.2,
                alpha=0.9, label=f"Audio A ({dim_key})")
        ax.plot(times2, b_norm, color="gray", linewidth=1.0,
                alpha=0.6, linestyle="--", label=f"Audio B ({dim_key})")

        # Draw connecting lines between matched segments
        for sm in disc.segment_matches:
            mid_a = (sm.a.start + sm.a.end) / 2
            mid_b = (sm.b.start + sm.b.end) / 2
            val_a = np.interp(mid_a, times1, a_norm)
            val_b = np.interp(mid_b, times2, b_norm)

            ax.annotate(
                "", xy=(mid_b, val_b), xytext=(mid_a, val_a),
                arrowprops=dict(
                    arrowstyle="->", color=color, alpha=0.4,
                    lw=1.0, connectionstyle="arc3,rad=0.15",
                ),
            )
            # Shade matched regions
            ax.axvspan(sm.a.start, sm.a.end, alpha=0.12, color=color, edgecolor="none")
            ax.axvspan(sm.b.start, sm.b.end, alpha=0.12, color=color, edgecolor="none")

        ax.set_ylabel(f"{dim_key} (norm)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_title(
            f"{disc.title} — {disc.meta.get('n_matches', 0)} matches, "
            f"avg confidence {disc.meta.get('avg_confidence', 0):.2f}",
            fontsize=11,
        )

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 3. Contrast visualisation
# ---------------------------------------------------------------------------

def plot_contrast_panel(
    contrasts: List,
    dyn1: Dict,
    dyn2: Dict,
    save_path: Optional[str] = None,
    title: str = "Audio Contrast — Where They Differ",
) -> plt.Figure:
    """
    Side-by-side distribution comparison for each dimension,
    highlighting where the two audios diverge.
    """
    n = len(contrasts)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for col, contrast in enumerate(contrasts):
        ax = axes[0, col]
        dim_key = contrast.dimension
        color = DIM_COLORS.get(dim_key, "#333333")

        a = dyn1[dim_key]
        b = dyn2[dim_key]

        bins = max(8, min(25, int(np.sqrt(min(len(a), len(b))))))
        ax.hist(a, bins=bins, alpha=0.6, color=color, label="Audio A", density=True,
                edgecolor="white", linewidth=0.5)
        ax.hist(b, bins=bins, alpha=0.4, color="gray", label="Audio B", density=True,
                edgecolor="white", linewidth=0.5)

        # Annotate with key stats
        overlap = contrast.meta.get("distribution_overlap", 1.0)
        diff_ratio = contrast.meta.get("diff_ratio", 0)
        ax.text(
            0.98, 0.95,
            f"overlap: {overlap:.1%}\ndiff ratio: {diff_ratio:.2f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

        ax.set_title(contrast.title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylabel("Density", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 4. Discovery summary dashboard
# ---------------------------------------------------------------------------

def plot_discovery_summary(
    report,
    save_path: Optional[str] = None,
    title: str = "Audio Discovery Summary",
) -> plt.Figure:
    """
    Single-page dashboard: character profiles, key discoveries, contrasts,
    and the natural-language overview.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ---- Left panel (0-0.5): Overview text ----
    ax_overview = fig.add_axes([0.03, 0.78, 0.45, 0.18])
    ax_overview.axis("off")
    overview = report.overview if report.overview else "No overview available."
    # Word-wrap
    words = overview.split()
    lines = []
    line = ""
    for w in words:
        test = f"{line} {w}".strip()
        if len(test) > 75:
            lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)

    ax_overview.text(0.0, 0.95, "Overview", fontsize=13, fontweight="bold",
                     transform=ax_overview.transAxes)
    for i, ln in enumerate(lines):
        ax_overview.text(0.0, 0.85 - i * 0.08, ln, fontsize=9,
                         transform=ax_overview.transAxes)

    # ---- Left panel (0.45-0.78): Character profiles side-by-side ----
    ax_profile_a = fig.add_axes([0.03, 0.52, 0.22, 0.23])
    ax_profile_a.axis("off")
    ax_profile_a.set_title("Audio A — Character", fontsize=11, fontweight="bold")
    _draw_character_card(ax_profile_a, report.audio_a_profile)

    ax_profile_b = fig.add_axes([0.26, 0.52, 0.22, 0.23])
    ax_profile_b.axis("off")
    ax_profile_b.set_title("Audio B — Character", fontsize=11, fontweight="bold")
    _draw_character_card(ax_profile_b, report.audio_b_profile)

    # ---- Left panel (0-0.52): Contrast summary ----
    ax_contrast = fig.add_axes([0.03, 0.02, 0.45, 0.47])
    ax_contrast.axis("off")
    contrasts = report.contrasts
    if contrasts:
        ax_contrast.text(0.0, 0.95, "Key Contrasts", fontsize=12, fontweight="bold",
                         transform=ax_contrast.transAxes)
        y = 0.85
        for i, c in enumerate(contrasts[:5]):
            dim_key = c.dimension
            color = DIM_COLORS.get(dim_key, "#333")
            # Coloured indicator
            ax_contrast.add_patch(plt.Rectangle(
                (0.0, y - 0.02), 0.015, 0.04,
                facecolor=color, transform=ax_contrast.transAxes, clip_on=False,
            ))
            diff_r = c.meta.get("diff_ratio", 0)
            overlap = c.meta.get("distribution_overlap", 1.0)
            ax_contrast.text(
                0.03, y,
                f"{c.title}: diff={diff_r:.2f}, overlap={overlap:.1%}",
                fontsize=9, transform=ax_contrast.transAxes,
            )
            # Truncated summary
            summary_short = c.summary[:100] + "..." if len(c.summary) > 100 else c.summary
            ax_contrast.text(
                0.03, y - 0.04,
                summary_short,
                fontsize=7, transform=ax_contrast.transAxes, color="#555555",
            )
            y -= 0.12
            if y < 0.05:
                break

    # ---- Right panel (0.52-1.0): Discoveries table ----
    ax_disc = fig.add_axes([0.52, 0.02, 0.46, 0.94])
    ax_disc.axis("off")
    ax_disc.set_title("Cross-Audio Discoveries", fontsize=13, fontweight="bold",
                      loc="left")

    discoveries = report.discoveries
    if discoveries:
        # Build a visual table
        n_show = min(len(discoveries), 6)
        y_start = 0.90
        row_h = 0.13
        for i, disc in enumerate(discoveries[:n_show]):
            y = y_start - i * row_h
            color = DIM_COLORS.get(disc.dimension, "#333")

            # Dimension badge
            ax_disc.add_patch(FancyBboxPatch(
                (0.0, y - 0.01), 0.22, 0.04,
                boxstyle="round,pad=0.1", facecolor=color, alpha=0.8,
                transform=ax_disc.transAxes,
            ))
            ax_disc.text(0.02, y, disc.title, fontsize=9, fontweight="bold",
                         color="white", transform=ax_disc.transAxes, va="center")

            # Match count badge
            n_m = disc.meta.get("n_matches", 0)
            ax_disc.text(
                0.24, y,
                f"{n_m} matches" if n_m > 0 else "no matches",
                fontsize=8, transform=ax_disc.transAxes, va="center",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#f8f8f8" if n_m > 0 else "#fff0f0",
                          alpha=0.9),
            )

            # Confidence
            avg_c = disc.meta.get("avg_confidence", 0)
            ax_disc.text(
                0.42, y,
                f"conf: {avg_c:.2f}",
                fontsize=8, transform=ax_disc.transAxes, va="center",
                color="#2ecc71" if avg_c > 0.7 else "#e67e22" if avg_c > 0.5 else "#e74c3c",
            )

            # Summary (truncated)
            summary_short = disc.summary[:120] + "..." if len(disc.summary) > 120 else disc.summary
            ax_disc.text(0.0, y - 0.045, summary_short, fontsize=7,
                         transform=ax_disc.transAxes, color="#444444")

            # Segment match preview
            if disc.segment_matches:
                preview = "  ".join(
                    f"A:{sm.a.start:.1f}s–B:{sm.b.start:.1f}s"
                    for sm in disc.segment_matches[:3]
                )
                ax_disc.text(0.0, y - 0.07, f"  {preview}", fontsize=6.5,
                             transform=ax_disc.transAxes, color="#888888")

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def _draw_character_card(ax, profile):
    """Draw a character profile inside an existing axes."""
    if profile is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
        return

    items = [
        ("Rhythm", profile.rhythm_signature, DIM_COLORS["rhythm"]),
        ("Energy", profile.energy_profile, DIM_COLORS["energy"]),
        ("Timbre", profile.timbre_quality, DIM_COLORS["brightness"]),
    ]
    y = 0.88
    for label, value, color in items:
        ax.text(0.0, y, label, fontsize=9, fontweight="bold",
                transform=ax.transAxes, color=color)
        # Truncate long values
        val_short = value[:75] + "..." if len(value) > 75 else value
        ax.text(0.0, y - 0.06, val_short, fontsize=7.5,
                transform=ax.transAxes, color="#333333")
        y -= 0.18


# ---------------------------------------------------------------------------
# 5. Report generator
# ---------------------------------------------------------------------------

def generate_discovery_report_plots(
    report,
    output_dir: str = "outputs/discovery",
    y1: Optional[np.ndarray] = None,
    y2: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    dyn1: Optional[Dict] = None,
    dyn2: Optional[Dict] = None,
) -> List[str]:
    """
    Generate all discovery visualizations and save to *output_dir*.

    Returns list of saved filenames (relative to output_dir).
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    # 1. Self-discovery for Audio A
    # Build a minimal self_result dict from what we have
    if dyn1 is not None:
        from audiots.dynamics import detect_structural_segments
        seg1 = detect_structural_segments(dyn1)

        self_a = {
            "dynamics": dyn1,
            "segments_raw": seg1,
            "character": report.audio_a_profile,
            "motifs": report.audio_a_motifs,
            "segments": {
                "climax": [{"time": float(dyn1["times"][i]), "energy": float(dyn1["energy"][i])}
                           for i in seg1.get("climax_indices", [])],
                "calm": [{"time": float(dyn1["times"][i]), "energy": float(dyn1["energy"][i])}
                         for i in seg1.get("calm_indices", [])],
            },
        }
        path = os.path.join(output_dir, "self_discovery_a.png")
        plot_self_discovery(self_a, save_path=path, title="Audio A — Self-Discovery")
        saved.append("self_discovery_a.png")

    # 2. Self-discovery for Audio B
    if dyn2 is not None:
        from audiots.dynamics import detect_structural_segments
        seg2 = detect_structural_segments(dyn2)

        self_b = {
            "dynamics": dyn2,
            "segments_raw": seg2,
            "character": report.audio_b_profile,
            "motifs": report.audio_b_motifs,
            "segments": {
                "climax": [{"time": float(dyn2["times"][i]), "energy": float(dyn2["energy"][i])}
                           for i in seg2.get("climax_indices", [])],
                "calm": [{"time": float(dyn2["times"][i]), "energy": float(dyn2["energy"][i])}
                         for i in seg2.get("calm_indices", [])],
            },
        }
        path = os.path.join(output_dir, "self_discovery_b.png")
        plot_self_discovery(self_b, save_path=path, title="Audio B — Self-Discovery")
        saved.append("self_discovery_b.png")

    # 3. Cross-discovery segment mapping
    if report.discoveries and dyn1 is not None and dyn2 is not None:
        path = os.path.join(output_dir, "segment_mapping.png")
        plot_segment_mapping(
            report.discoveries, dyn1, dyn2,
            save_path=path,
        )
        saved.append("segment_mapping.png")

    # 4. Contrast panel
    if report.contrasts and dyn1 is not None and dyn2 is not None:
        path = os.path.join(output_dir, "contrast_panel.png")
        plot_contrast_panel(
            report.contrasts, dyn1, dyn2,
            save_path=path,
        )
        saved.append("contrast_panel.png")

    # 5. Summary dashboard
    path = os.path.join(output_dir, "discovery_summary.png")
    plot_discovery_summary(report, save_path=path)
    saved.append("discovery_summary.png")

    plt.close("all")
    return saved
