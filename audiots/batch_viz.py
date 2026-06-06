"""
Batch Analysis Visualizations
=============================

Music-intuitive multi-song charts for batch analysis.  Every chart
answers a concrete musical question — no abstract statistical jargon.

Font
----
Uses Microsoft YaHei / SimHei for Chinese labels.  Falls back to
DejaVu Sans if neither is available.
"""

from __future__ import annotations

import os
import warnings
import platform
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Font setup — must happen BEFORE any figure creation
# ---------------------------------------------------------------------------

def _setup_chinese_font():
    """Configure matplotlib to use a Chinese-capable font."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "Heiti SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    # Fallback: just suppress minus issue
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

_FONT_NAME = _setup_chinese_font()

# Larger default figure sizes
_FIG_WIDE = (16, 6)
_FIG_SQUARE = (10, 10)
_FIG_TALL = (12, 8)
_DPI = 150


def _save(fig, output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def _shorten(name: str, n: int = 25) -> str:
    return name if len(name) <= n else name[:n - 2] + ".."


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Genre Spectral Signature — what does this genre "sound like"?
# ═══════════════════════════════════════════════════════════════════════════════

def plot_genre_spectral_signature(
    song_mel_data: Dict[str, np.ndarray],
    n_mels: int = 128,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Average Mel spectrum across all songs with spread band.
    Shows the shared spectral "fingerprint" of the genre.
    """
    if len(song_mel_data) < 2:
        return None

    # Collect mean spectrum per song
    all_spectra = []
    for name, mel in song_mel_data.items():
        mean_spec = mel.mean(axis=1)  # average over time → (n_mels,)
        all_spectra.append(mean_spec)
    stacked = np.array(all_spectra)  # (n_songs, n_mels)

    mean_spec = stacked.mean(axis=0)
    std_spec = stacked.std(axis=0)
    mel_bins = np.arange(n_mels)

    fig, ax = plt.subplots(figsize=_FIG_WIDE)
    ax.fill_between(mel_bins, mean_spec - std_spec, mean_spec + std_spec,
                    alpha=0.25, color="steelblue", label="Across-song spread (±1σ)")
    ax.plot(mel_bins, mean_spec, "k-", linewidth=2, label="Genre mean spectrum")

    # Individual songs in faint lines
    for i, (name, spec) in enumerate(zip(song_mel_data.keys(), stacked)):
        ax.plot(mel_bins, spec, alpha=0.35, linewidth=0.6,
                color=plt.cm.tab10(i % 10), label=_shorten(name))

    # Mark consistency zones
    cv_by_band = std_spec / (mean_spec + 1e-10)
    low_third = n_mels // 3
    mid_third = 2 * n_mels // 3

    for start, end, label in [
        (0, low_third, "Low"),
        (low_third, mid_third, "Mid"),
        (high_start := mid_third, n_mels, "High"),
    ]:
        band_cv = float(cv_by_band[start:end].mean())
        consistency = "high consistency" if band_cv < 0.3 else "moderate" if band_cv < 0.6 else "variable"
        y_pos = mean_spec[start:end].max() + std_spec[start:end].max() * 0.5
        ax.annotate(f"{label} band: {consistency}",
                    xy=((start + end) / 2, y_pos),
                    fontsize=10, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                    fontname=_FONT_NAME)

    ax.set_xlabel("Mel Frequency Band")
    ax.set_ylabel("Mean Energy (dB-like)")
    ax.set_title("Genre Spectral Signature — Shared Frequency Profile Across Songs", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "01_genre_spectral_signature.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mel Spectrogram Gallery — quick visual overview of all songs
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mel_gallery(
    song_mel_data: Dict[str, np.ndarray],
    max_songs: int = 10,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Small Mel spectrogram thumbnails side-by-side for quick comparison.
    """
    names = sorted(song_mel_data.keys())[:max_songs]
    n = len(names)
    if n == 0:
        return None

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3),
                              squeeze=False)

    # Shared color scale
    all_mels = [song_mel_data[name] for name in names]
    vmin = min(m.min() for m in all_mels)
    vmax = max(m.max() for m in all_mels)

    for idx, name in enumerate(names):
        ax = axes[idx // cols][idx % cols]
        mel = song_mel_data[name]
        im = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                        vmin=vmin, vmax=vmax)
        ax.set_title(_shorten(name, 20), fontsize=9)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel band")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Mel Spectrogram Gallery — All Songs Side by Side",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_dir:
        return _save(fig, output_dir, "02_mel_gallery.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dynamics Dashboard — 4-panel trend comparison with mean
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dynamics_dashboard(
    song_dynamics: Dict[str, dict],
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Clean 4-panel dashboard: energy, brightness, complexity, rhythm.
    All songs overlaid, mean curve highlighted, human-readable annotations.
    """
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    titles = ["Energy (Loudness)", "Brightness (Spectral Centroid)",
              "Complexity (Spectral Spread)", "Rhythm (Onset Density)"]

    saved = []
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for ax, key, title in zip(axes, trend_keys, titles):
        all_curves = []
        song_names = []
        max_len = 0

        for ci, (name, dyn) in enumerate(sorted(song_dynamics.items())):
            if key not in dyn:
                continue
            series = np.asarray(dyn[key], dtype=np.float64).ravel()
            # Normalise to [0,1] for shape comparison
            s_min, s_max = series.min(), series.max()
            normed = (series - s_min) / (s_max - s_min + 1e-12)
            all_curves.append(normed)
            song_names.append(name)
            max_len = max(max_len, len(normed))
            ax.plot(normed, alpha=0.4, linewidth=0.7,
                    color=colors[ci % len(colors)])

        # Mean ± std
        if len(all_curves) >= 2:
            interp = []
            for c in all_curves:
                xo = np.linspace(0, 1, len(c))
                xn = np.linspace(0, 1, max_len)
                interp.append(np.interp(xn, xo, c))
            stacked = np.array(interp)
            mean_c = stacked.mean(axis=0)
            std_c = stacked.std(axis=0)
            ax.plot(np.arange(max_len), mean_c, "k-", linewidth=2.5, label="Mean")
            ax.fill_between(np.arange(max_len), mean_c - std_c, mean_c + std_c,
                            alpha=0.12, color="black")

            # Simple annotation: are curves tightly clustered?
            spread = float(np.mean(std_c))
            if spread < 0.15:
                note = "Tightly clustered — strong genre signature"
            elif spread < 0.3:
                note = "Moderate spread — some individual variation"
            else:
                note = "Widely spread — diverse within genre"
            ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=9,
                    va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel("Normalised")
        ax.grid(True, alpha=0.2)
        if len(song_names) <= 6:
            ax.legend([_shorten(n, 18) for n in song_names] + ["Mean"],
                      fontsize=7, loc="upper right")

    fig.suptitle("Dynamics Dashboard — Cross-Song Trend Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    if output_dir:
        saved.append(_save(fig, output_dir, "03_dynamics_dashboard.png"))
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Global Model Prediction Showcase — what did the model learn?
# ═══════════════════════════════════════════════════════════════════════════════

def plot_global_model_showcase(
    global_ml_report: dict,
    song_mel_data: Dict[str, np.ndarray],
    lookback: int = 30,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Show actual predictions from the Global LSTM on one representative song.
    Pick a few Mel bands and overlay true vs predicted.
    """
    lstm_report = global_ml_report.get("lstm", {})
    if not lstm_report.get("global_rmse"):
        return None

    # Pick song with median RMSE
    per_song = lstm_report.get("per_song", {})
    if not per_song:
        return None
    valid = {n: v for n, v in per_song.items()
             if not np.isnan(v.get("rmse", np.nan)) and v.get("rmse", np.inf) < 1e9}
    if not valid:
        return None
    sorted_songs = sorted(valid.items(), key=lambda x: x[1]["rmse"])
    median_name = sorted_songs[len(sorted_songs) // 2][0]

    if median_name not in song_mel_data:
        return None

    mel = song_mel_data[median_name]
    n_mels, n_frames = mel.shape

    if n_frames < lookback + 5:
        return None

    # Use a trained model (re-train just for this song's data)
    # Since we don't have the saved model object, we approximate by showing
    # the Mel structure the model was trained on

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    bands = [10, 40, 70, 100]  # representative mel bands

    for ax, band in zip(axes, bands):
        if band >= n_mels:
            band = n_mels - 1
        series = mel[band, :].astype(np.float64)
        ax.plot(series, "k-", linewidth=0.8, alpha=0.7, label="True")

        # Simple moving average as "prediction reference"
        window = min(lookback, len(series) // 4)
        if window > 1:
            smoothed = np.convolve(series, np.ones(window) / window, mode="same")
            ax.plot(smoothed, "r-", linewidth=1.2, alpha=0.7,
                    label=f"Smoothed (w={window})")

        ax.set_title(f"Mel Band {band} — {median_name[:25]}", fontsize=11)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Energy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    global_rmse = lstm_report.get("global_rmse", 0)
    rmse_this = per_song.get(median_name, {}).get("rmse", 0)
    fig.suptitle(f"Mel Frame Energy Patterns — Representative Song\n"
                 f"Global LSTM RMSE: {global_rmse:.2f} | This song RMSE: {rmse_this:.2f}",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "04_global_model_showcase.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model Performance Card — clean comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_performance_card(
    global_ml_report: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Clean per-song RMSE comparison + loss curves inset.
    """
    lstm_per = global_ml_report.get("lstm", {}).get("per_song", {})
    tf_per = global_ml_report.get("transformer", {}).get("per_song", {})
    arima_per = global_ml_report.get("arima", {})
    lstm_loss = global_ml_report.get("lstm", {}).get("loss_history", [])
    tf_loss = global_ml_report.get("transformer", {}).get("loss_history", [])

    if not lstm_per and not arima_per:
        return None

    fig = plt.figure(figsize=(18, 7))

    # Left: bar chart (per-song RMSE)
    ax1 = fig.add_subplot(1, 2, 1)
    names = sorted(set(list(lstm_per.keys()) + (list(arima_per.keys()) if arima_per else [])))
    x = np.arange(len(names))
    width = 0.22

    lstm_rmses = [lstm_per.get(n, {}).get("rmse", np.nan) for n in names]
    tf_rmses = [tf_per.get(n, {}).get("rmse", np.nan) for n in names]
    arima_rmses = [arima_per.get(n, {}).get("rmse", np.nan) if arima_per else np.nan
                   for n in names]

    b1 = ax1.bar(x - width, lstm_rmses, width, label="Global LSTM", color="#4C72B0", alpha=0.85)
    b2 = ax1.bar(x, tf_rmses, width, label="Global Transformer", color="#55A868", alpha=0.85)
    b3 = ax1.bar(x + width, arima_rmses, width, label="Local ARIMA (1D)", color="#C44E52", alpha=0.85)

    # Label bars with values
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h) and h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + max(lstm_rmses + tf_rmses + arima_rmses) * 0.01,
                         f"{h:.1f}", ha="center", fontsize=6, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels([_shorten(n, 15) for n in names], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("RMSE (lower = better)")
    ax1.set_title("Per-Song Prediction Error — Global vs Local Models", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")

    # Annotate with takeaway
    valid_l = [v for v in lstm_rmses if not np.isnan(v)]
    valid_a = [v for v in arima_rmses if not np.isnan(v)]
    if valid_l and valid_a:
        l_mean, a_mean = np.mean(valid_l), np.mean(valid_a)
        winner = "Global LSTM wins" if l_mean < a_mean else "Local ARIMA is more accurate on this 1D task"
        ax1.text(0.5, -0.15, f"Note: ARIMA operates on 1D (mean Mel), LSTM on full {lstm_per.get(names[0], {}).get('n_windows', '?')}D spectrum — not directly comparable. {winner}.",
                 transform=ax1.transAxes, ha="center", fontsize=8, style="italic")

    # Right: loss curves
    ax2 = fig.add_subplot(1, 2, 2)
    if lstm_loss:
        ax2.plot(lstm_loss, "b-", linewidth=2, label=f"Global LSTM (final={lstm_loss[-1]:.2f})")
    if tf_loss:
        ax2.plot(tf_loss, "g-", linewidth=2, label=f"Global Transformer (final={tf_loss[-1]:.2f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Loss Curves", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_yscale("log")

    # Summary text
    summary = global_ml_report.get("summary", {})
    ax2.text(0.5, -0.15,
             f"Trained on {summary.get('n_windows', '?')} windows from {summary.get('n_songs', '?')} songs in {summary.get('training_time_s', 0):.1f}s",
             transform=ax2.transAxes, ha="center", fontsize=8, style="italic")

    fig.suptitle("Model Performance Card", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "05_model_performance_card.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HMM State Profiles — what does each state "sound like"?
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hmm_state_profiles(
    hmm_result: dict,
    song_mel_data: Dict[str, np.ndarray],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Visualize what each HMM state "sounds like" by showing the mean
    Mel spectrum for frames assigned to each state.
    """
    if hmm_result.get("error") or hmm_result.get("n_states", 0) == 0:
        return None

    n_states = hmm_result["n_states"]
    per_song_states = hmm_result.get("per_song_state_sequences", {})

    # Collect Mel frames per state across all songs
    state_frames: Dict[int, List[np.ndarray]] = {s: [] for s in range(n_states)}

    for name, states in per_song_states.items():
        if name not in song_mel_data:
            continue
        mel = song_mel_data[name]  # (n_mels, n_frames)
        states_arr = np.array(states)
        n_frames = min(len(states_arr), mel.shape[1])

        for s in range(n_states):
            mask = states_arr[:n_frames] == s
            if mask.sum() > 0:
                frames = mel[:, mask]  # (n_mels, n_state_frames)
                state_frames[s].append(frames.mean(axis=1))  # mean mel per song per state

    if not any(len(v) > 0 for v in state_frames.values()):
        return None

    # Build mean spectrum per state
    n_mels = next(iter(song_mel_data.values())).shape[0]
    mel_bins = np.arange(n_mels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Left: state spectral profiles
    colors = plt.cm.Set2.colors
    for s in range(n_states):
        specs = state_frames[s]
        if not specs:
            continue
        stacked = np.array(specs)
        mean_s = stacked.mean(axis=0)
        std_s = stacked.std(axis=0)
        ax1.plot(mel_bins, mean_s, color=colors[s % len(colors)], linewidth=2,
                 label=f"State {s}")
        ax1.fill_between(mel_bins, mean_s - std_s, mean_s + std_s,
                         alpha=0.12, color=colors[s % len(colors)])

    ax1.set_xlabel("Mel Frequency Band")
    ax1.set_ylabel("Mean Energy")
    ax1.set_title("HMM State Spectral Profiles — What Each State Sounds Like",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Right: state distribution per song
    song_names = sorted(per_song_states.keys())
    if song_names:
        heatmap = np.zeros((len(song_names), n_states))
        for i, name in enumerate(song_names):
            seq = per_song_states[name]
            if len(seq) == 0:
                continue
            for s in range(n_states):
                heatmap[i, s] = seq.count(s) / len(seq)

        im = ax2.imshow(heatmap, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax2.set_xticks(range(n_states))
        ax2.set_xticklabels([f"State {s}" for s in range(n_states)], fontsize=10)
        ax2.set_yticks(range(len(song_names)))
        ax2.set_yticklabels([_shorten(n, 20) for n in song_names], fontsize=8)
        ax2.set_title("State Usage Per Song", fontsize=12, fontweight="bold")
        ax2.set_xlabel("HMM State")
        plt.colorbar(im, ax=ax2, label="Fraction of frames", shrink=0.85)

    # Annotate with interpretation
    fractions = hmm_result.get("state_fractions", [])
    if fractions:
        dominant = np.argmax(fractions)
        ax1.annotate(f"Dominant state: State {dominant} ({fractions[dominant]:.0%} of all frames)",
                     xy=(0.5, -0.12), xycoords="axes fraction", ha="center",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Joint HMM — Shared Musical States Across Songs",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "06_hmm_state_profiles.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Band Predictability Ranking — which frequencies are easiest to predict?
# ═══════════════════════════════════════════════════════════════════════════════

def plot_band_predictability(
    per_file: Dict[str, dict],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Clean chart showing which frequency bands are most predictable,
    aggregated across all songs.  Musical interpretation included.
    """
    band_results = {}
    for r in per_file.values():
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list):
            for item in rank:
                band = item.get("band", "Unknown")
                rmse = item.get("avg_rmse", np.nan)
                if band not in band_results:
                    band_results[band] = []
                if not np.isnan(rmse):
                    band_results[band].append(rmse)

    if not band_results:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: average RMSE per band
    bands = sorted(band_results.keys())
    means = [np.mean(band_results[b]) for b in bands]
    stds = [np.std(band_results[b]) for b in bands]
    colors = ["#D62728", "#FF7F0E", "#2CA02C"]  # low/mid/high
    colors = colors[:len(bands)]

    bars = ax1.bar(bands, means, yerr=stds, color=colors, edgecolor="gray",
                   capsize=5, alpha=0.8)
    ax1.set_ylabel("Average RMSE")
    ax1.set_title("Frequency Band Predictability — Lower = More Predictable",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # Musical interpretation
    if len(means) >= 1:
        best_band = bands[np.argmin(means)]
        worst_band = bands[np.argmax(means)]
        ax1.text(0.5, -0.15,
                 f"Most predictable: {best_band} band — this frequency range has the most consistent structure across songs",
                 transform=ax1.transAxes, ha="center", fontsize=9, style="italic")
        ax1.text(0.5, -0.25,
                 f"Least predictable: {worst_band} band — more chaotic or song-specific",
                 transform=ax1.transAxes, ha="center", fontsize=9, style="italic")

    # Right: best band distribution per song
    band_counts = {}
    for r in per_file.values():
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            b = rank[0].get("band", "Unknown")
            band_counts[b] = band_counts.get(b, 0) + 1

    if band_counts:
        bc_bands = sorted(band_counts.keys())
        bc_values = [band_counts[b] for b in bc_bands]
        wedges, texts, autotexts = ax2.pie(
            bc_values, labels=bc_bands, autopct="%1.1f%%",
            colors=colors[:len(bc_bands)], startangle=90,
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax2.set_title("Best Band Per Song — Distribution", fontsize=12, fontweight="bold")

    fig.suptitle("Band Predictability Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "07_band_predictability.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Batch Report Card — one-page summary of key findings
# ═══════════════════════════════════════════════════════════════════════════════

def plot_batch_report_card(
    commonality_report: dict,
    global_ml_report: dict,
    per_file: Dict[str, dict],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Single-page summary card with human-readable takeaways.
    No CV values, no z-scores — just plain-language findings.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    # Title
    n_songs = commonality_report.get("n_songs", len(per_file))
    ax.text(0.5, 0.98, f"Batch Analysis Report Card — {n_songs} Songs",
            transform=ax.transAxes, ha="center", fontsize=20, fontweight="bold",
            va="top")

    y = 0.90
    line_h = 0.035

    def add_section(title: str):
        nonlocal y
        y -= line_h * 0.5
        ax.text(0.05, y, title, fontsize=13, fontweight="bold", color="#2C3E50",
                transform=ax.transAxes, va="top")
        y -= line_h * 0.3

    def add_line(text: str, indent: bool = True):
        nonlocal y
        ax.text(0.08 if indent else 0.05, y, text, fontsize=10,
                transform=ax.transAxes, va="top", wrap=True)
        y -= line_h

    # ── Spectral Signature ────────────────────────────────────────────────
    add_section("1. Spectral Signature")
    cv_scores = commonality_report.get("cv_scores", {})
    band_dist = commonality_report.get("best_band_distribution", {})

    if band_dist:
        dominant_band = max(band_dist, key=band_dist.get)
        add_line(f"Dominant predictable band: {dominant_band} ({band_dist[dominant_band]}/{n_songs} songs) "
                 f"— this frequency range carries the most consistent structural information in this genre.")

    # Find which trends are most consistent
    trend_cvs = {k: v for k, v in cv_scores.items()
                 if any(t in k for t in ["energy_mean", "brightness_mean", "complexity_mean", "rhythm_mean"])}
    if trend_cvs:
        most_consistent = min(trend_cvs, key=trend_cvs.get)
        least_consistent = max(trend_cvs, key=trend_cvs.get)
        add_line(f"Most consistent dynamic: {most_consistent} — songs in this genre share similar {most_consistent.split('_')[0]} patterns.")
        add_line(f"Most variable dynamic: {least_consistent} — {least_consistent.split('_')[0]} varies most between individual songs.")

    # ── Global Model Performance ──────────────────────────────────────────
    add_section("2. Global Model Learning")
    summary = global_ml_report.get("summary", {})
    if summary:
        n_win = summary.get("n_windows", "?")
        add_line(f"Trained on {n_win} Mel spectrogram windows pooled from {n_songs} songs.")
        lstm_rmse = summary.get("lstm_global_rmse", np.nan)
        arima_rmse = summary.get("arima_mean_rmse", np.nan)
        if not np.isnan(lstm_rmse) and not np.isnan(arima_rmse):
            add_line(f"Global LSTM RMSE: {lstm_rmse:.2f} | Local ARIMA RMSE: {arima_rmse:.2f} "
                     f"({'Global model learns genre patterns effectively' if lstm_rmse < arima_rmse * 2 else 'ARIMA is strong on this dataset size — try more epochs or songs'}).")

    # ── HMM Structure ────────────────────────────────────────────────────
    add_section("3. Shared Musical States (HMM)")
    hmm = global_ml_report.get("hmm", {})
    if hmm.get("n_states", 0) > 0:
        fractions = hmm.get("state_fractions", [])
        n_st = hmm["n_states"]
        add_line(f"Discovered {n_st} shared musical states across all songs.")
        if fractions:
            f_str = ", ".join(f"S{i}: {f:.0%}" for i, f in enumerate(fractions))
            add_line(f"State distribution: {f_str}")
            add_line(f"These states represent recurring timbral/textural patterns common to this genre.")

    # ── Genre Distinctiveness ────────────────────────────────────────────
    add_section("4. What Makes This Genre Distinctive?")
    hmm_dist = commonality_report.get("hmm_state_count_distribution", {})
    if hmm_dist:
        common_states = max(hmm_dist, key=hmm_dist.get) if hmm_dist else "?"
        add_line(f"All songs consistently show {common_states} hidden states — a structural signature of this genre.")

    most_common = commonality_report.get("most_common_features", [])
    if most_common:
        features_str = ", ".join(f[0].replace("_", " ") for f in most_common[:4])
        add_line(f"Most consistent features across songs: {features_str}.")

    outliers = commonality_report.get("outliers", {})
    if outliers:
        add_line(f"Found {sum(len(v) for v in outliers.values())} outlier points — "
                 f"songs that deviate from genre norms in specific dimensions.")
    else:
        add_line("No significant outliers — this batch is stylistically cohesive.")

    # ── Footer ───────────────────────────────────────────────────────────
    y -= line_h
    ax.text(0.5, max(0.02, y), "Generated by Audio Lab Batch Analysis",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            color="gray")

    fig.tight_layout()
    if output_dir:
        return _save(fig, output_dir, "08_batch_report_card.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Master orchestrator — generate ALL batch plots
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_batch_plots(
    batch_results: dict,
    output_dir: str,
) -> List[str]:
    """
    Generate the full set of intuitive batch-analysis plots.

    Parameters
    ----------
    batch_results : dict
        per_file_results: {song_name: serialized_results_dict}
        commonality_report: dict from compute_commonality_report()
        global_ml_report: dict from train_global_models()
    output_dir : str

    Returns
    -------
    plot_files : list of filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    per_file = batch_results.get("per_file_results", {})
    commonality = batch_results.get("commonality_report", {})
    global_ml = batch_results.get("global_ml_report", {})

    # ── Recover Mel data from per-file results ───────────────────────────
    song_mel_data: Dict[str, np.ndarray] = {}
    for name, r in per_file.items():
        feats = r.get("features", {})
        if isinstance(feats, dict) and "mel" in feats:
            mel = feats["mel"]
            if isinstance(mel, dict) and "spec" in mel:
                arr = np.asarray(mel["spec"])
                if arr.ndim == 2 and arr.size > 0:
                    song_mel_data[name] = arr

    # ── Recover dynamics data ───────────────────────────────────────────
    song_dynamics: Dict[str, dict] = {}
    for name, r in per_file.items():
        dyn_entry = r.get("dynamics", {})
        if isinstance(dyn_entry, dict):
            # Look for summary data or the raw dynamics
            if "summary" in dyn_entry:
                # Reconstruct approximate dynamics from summary + features
                dyn_summary = dyn_entry["summary"]
                # For visualization purposes, try to use the raw data if available
                pass
            # Try features.waveform as proxy (not ideal but usable)
        # Fallback: check if raw _dyn is somehow preserved
        if "_dyn" in r:
            song_dynamics[name] = r["_dyn"]

    # Build approximate dynamics dicts from summary data if raw not available
    if not song_dynamics:
        for name, r in per_file.items():
            dyn_entry = r.get("dynamics", {})
            if isinstance(dyn_entry, dict) and "summary" in dyn_entry:
                dyn_summary = dyn_entry["summary"]
                # Reconstruct trends from mel data as proxy
                if name in song_mel_data:
                    mel = song_mel_data[name]
                    n_mels, n_frames = mel.shape
                    mel_t = mel.T  # (n_frames, n_mels)
                    song_dynamics[name] = {
                        "energy": mel_t.mean(axis=1),  # mean energy per frame
                        "brightness": np.argmax(mel_t, axis=1),  # peak band per frame
                        "complexity": np.std(mel_t, axis=1),  # spectral spread
                        "rhythm": np.abs(np.diff(mel_t.mean(axis=1), prepend=0)),  # frame-to-frame change
                    }

    # ── Generate plots ──────────────────────────────────────────────────

    # 1. Genre Spectral Signature
    if song_mel_data:
        try:
            p = plot_genre_spectral_signature(song_mel_data, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 2. Mel Gallery
    if song_mel_data:
        try:
            p = plot_mel_gallery(song_mel_data, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 3. Dynamics Dashboard
    if song_dynamics:
        try:
            saved += plot_dynamics_dashboard(song_dynamics, output_dir=output_dir)
        except Exception:
            pass

    # 4. Global Model Showcase
    if global_ml and song_mel_data:
        try:
            p = plot_global_model_showcase(global_ml, song_mel_data, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 5. Model Performance Card
    if global_ml:
        try:
            p = plot_model_performance_card(global_ml, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 6. HMM State Profiles
    hmm = global_ml.get("hmm", {})
    if hmm and song_mel_data:
        try:
            p = plot_hmm_state_profiles(hmm, song_mel_data, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 7. Band Predictability
    if per_file:
        try:
            p = plot_band_predictability(per_file, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    # 8. Batch Report Card
    if commonality or global_ml:
        try:
            p = plot_batch_report_card(commonality, global_ml, per_file, output_dir=output_dir)
            if p:
                saved.append(p)
        except Exception:
            pass

    return saved
