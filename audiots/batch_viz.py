"""
Batch Analysis Visualizations
=============================

Comprehensive cross-song comparison dashboards. Each figure is a large
multi-panel layout covering one analysis dimension across all songs.

Design principles
-----------------
- Large figures (18-22 inches), DPI 200, readable at any zoom level
- Every panel annotated with statistical insights in plain language
- Mirrors the richness of app/main per-file output, but cross-song
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Font ────────────────────────────────────────────────────────────────────

def _init_font():
    for name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"

_FONT = _init_font()

# ── Global style ────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "figure.dpi": 200,
    "figure.figsize": (20, 12),
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

COLORS = plt.cm.tab10.colors
BAND_COLORS = {"Low Band": "#D62728", "Mid Band": "#FF7F0E", "High Band": "#2CA02C"}

# ── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, d: str, name: str) -> str:
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name

def _short(name: str, n: int = 22) -> str:
    return name if len(name) <= n else name[:n - 2] + ".."

def _safe(v, default=np.nan):
    if v is None: return default
    if isinstance(v, (int, float, np.floating)): return float(v)
    try: return float(str(v))
    except: return default

def _cv(arr):
    """Coefficient of variation, handles edge cases."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2 or np.abs(np.mean(arr)) < 1e-12:
        return np.nan
    return float(np.std(arr) / np.abs(np.mean(arr)))

def _annotate_consistency(ax, values, x=0.98, y=0.95):
    """Add a consistency label to a plot."""
    cv = _cv(values)
    if np.isnan(cv): return
    if cv < 0.2: label, color = "high consistency", "green"
    elif cv < 0.5: label, color = "moderate variation", "orange"
    else: label, color = "high diversity", "red"
    ax.text(x, y, f"CV={cv:.2f} ({label})", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# ── Data extraction helpers ─────────────────────────────────────────────────

def _extract_dynamics_series(per_file: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract trend series from per-file results (reconstruct from mel if needed)."""
    out = {}
    for name, r in per_file.items():
        # Try raw dynamics first
        dyn_entry = r.get("dynamics", {})
        if isinstance(dyn_entry, dict):
            summary = dyn_entry.get("summary", dyn_entry)
            if isinstance(summary, dict) and all(k in summary for k in ["energy", "brightness", "complexity", "rhythm"]):
                # Reconstruct approximate series from mel data
                feats = r.get("features", {})
                mel = feats.get("mel", {}).get("spec") if isinstance(feats, dict) else None
                if mel is not None:
                    mel_arr = np.asarray(mel)  # (n_mels, n_frames)
                    mel_t = mel_arr.T
                    out[name] = {
                        "energy": mel_t.mean(axis=1),
                        "brightness": np.argmax(mel_t, axis=1).astype(float),
                        "complexity": np.std(mel_t, axis=1),
                        "rhythm": np.abs(np.diff(mel_t.mean(axis=1), prepend=0)),
                    }
                continue
    return out

def _extract_scalar_features(per_file: dict) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build (n_songs, n_features) scalar feature matrix from per-file results."""
    names = sorted(per_file.keys())
    rows = []

    for name in names:
        r = per_file[name]
        row = {}

        # Dynamics summaries
        dyn_entry = r.get("dynamics", {})
        if isinstance(dyn_entry, dict):
            ds = dyn_entry.get("summary", dyn_entry)
        else:
            ds = {}
        if isinstance(ds, dict):
            for t in ["energy", "brightness", "complexity", "rhythm"]:
                td = ds.get(t, {})
                if isinstance(td, dict):
                    row[f"{t}_mean"] = _safe(td.get("mean"))
                    row[f"{t}_std"] = _safe(td.get("std"))

        # Volatility
        vol = r.get("volatility", {})
        if isinstance(vol, dict):
            vs = vol.get("summary", vol)
        else:
            vs = {}
        if isinstance(vs, dict):
            for t in ["energy", "brightness", "complexity", "rhythm"]:
                vd = vs.get(t, {})
                if isinstance(vd, dict):
                    row[f"{t}_vol_mean"] = _safe(vd.get("mean_vol"))
                    row[f"{t}_garch_p"] = _safe(vd.get("garch_persistence"))

        # Complexity
        ts = r.get("timeseries", {})
        if isinstance(ts, dict):
            cpx = ts.get("complexity", {})
            if isinstance(cpx, dict):
                row["zcr"] = _safe(cpx.get("zero_crossing_rate"))
                row["sample_entropy"] = _safe(cpx.get("sample_entropy"))
            row["spectral_flatness"] = _safe(ts.get("spectral_flatness"))
            per = ts.get("periodicity", {})
            if isinstance(per, dict):
                row["dominant_freq"] = _safe(per.get("dominant_frequency"))

        # Model analysis
        ma = r.get("model_analysis", {})
        if isinstance(ma, dict):
            row["hmm_n_states"] = _safe(ma.get("hmm_n_states", 0))
            row["lstm_lookback_s"] = _safe(ma.get("lstm_optimal_lookback_s"))
            row["tf_n_layers"] = _safe(ma.get("transformer_n_layers", 0))

        # Prediction
        preds = r.get("prediction", {})
        if isinstance(preds, dict):
            for mn in ["ARIMA", "HMM", "LSTM", "Transformer"]:
                arr = preds.get(mn, [])
                if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                    row[f"pred_{mn}_rmse"] = _safe(arr[1].get("RMSE"))
                    row[f"pred_{mn}_mae"] = _safe(arr[1].get("MAE"))
                else:
                    row[f"pred_{mn}_rmse"] = np.nan

        # Audio duration
        ai = r.get("audio_info", {})
        if isinstance(ai, dict):
            row["duration_s"] = _safe(ai.get("duration"))

        rows.append(row)

    feat_names = list(rows[0].keys()) if rows else []
    mat = np.array([[r.get(f, np.nan) for f in feat_names] for r in rows], dtype=np.float64)
    return mat, feat_names, names


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Spectral Analysis Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spectral_dashboard(
    per_file: dict,
    song_mel_data: Dict[str, np.ndarray],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Multi-panel spectral comparison across all songs."""
    if len(per_file) < 2:
        return None

    names = sorted(song_mel_data.keys()) if song_mel_data else sorted(per_file.keys())
    n_songs = len(names)
    mat, feat_names, _ = _extract_scalar_features(per_file)

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1.1 Mean Mel spectrum overlay (spans top row left 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    if song_mel_data:
        for ci, name in enumerate(names):
            if name in song_mel_data:
                mel = song_mel_data[name]
                mean_s = mel.mean(axis=1)
                ax1.plot(mean_s, linewidth=0.8, alpha=0.7,
                         color=COLORS[ci % len(COLORS)], label=_short(name))
        # Mean of means
        all_means = [song_mel_data[n].mean(axis=1) for n in names if n in song_mel_data]
        if len(all_means) >= 2:
            stacked = np.array(all_means)
            mean_all = stacked.mean(axis=0)
            std_all = stacked.std(axis=0)
            ax1.fill_between(range(len(mean_all)), mean_all - std_all, mean_all + std_all,
                             alpha=0.15, color="black")
            ax1.plot(mean_all, "k-", linewidth=2.5, label="Genre mean")
        _annotate_consistency(ax1, [np.mean(song_mel_data[n]) for n in names if n in song_mel_data])
    ax1.set_xlabel("Mel frequency band")
    ax1.set_ylabel("Mean energy")
    ax1.set_title("1. Mean Mel Spectrum — Genre Spectral Fingerprint", fontweight="bold")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.2)

    # 1.2 Mel gallery thumbnails (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if song_mel_data:
        sample_name = names[0]
        if sample_name in song_mel_data:
            ax2.imshow(song_mel_data[sample_name], aspect="auto", origin="lower",
                       cmap="magma")
            ax2.set_title(f"Mel Spectrogram — {_short(sample_name, 18)}", fontsize=10)
            ax2.set_xlabel("Frame")
            ax2.set_ylabel("Mel band")

    # 1.3 Spectral flatness comparison (row 2, col 1)
    ax3 = fig.add_subplot(gs[1, 0])
    flat_vals = []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        sf = _safe(ts.get("spectral_flatness"))
        if not np.isnan(sf):
            flat_vals.append(sf)
        else:
            flat_vals.append(0)
    bars = ax3.bar(range(n_songs), flat_vals,
                   color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
                   edgecolor="gray", alpha=0.8)
    ax3.set_xticks(range(n_songs))
    ax3.set_xticklabels([_short(n, 12) for n in names], rotation=45, ha="right")
    ax3.set_ylabel("Spectral Flatness")
    ax3.set_title("2. Spectral Flatness (lower = more tonal)", fontweight="bold")
    ax3.grid(True, alpha=0.2, axis="y")
    _annotate_consistency(ax3, flat_vals)

    # 1.4 Zero-crossing rate comparison (row 2, col 2)
    ax4 = fig.add_subplot(gs[1, 1])
    zcr_vals = []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        cpx = ts.get("complexity", {}) if isinstance(ts, dict) else {}
        z = _safe(cpx.get("zero_crossing_rate")) if isinstance(cpx, dict) else np.nan
        zcr_vals.append(z if not np.isnan(z) else 0)
    ax4.bar(range(n_songs), zcr_vals,
            color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
            edgecolor="gray", alpha=0.8)
    ax4.set_xticks(range(n_songs))
    ax4.set_xticklabels([_short(n, 12) for n in names], rotation=45, ha="right")
    ax4.set_ylabel("ZCR")
    ax4.set_title("3. Zero-Crossing Rate (signal noisiness)", fontweight="bold")
    ax4.grid(True, alpha=0.2, axis="y")
    _annotate_consistency(ax4, zcr_vals, x=0.98, y=0.2)

    # 1.5 Sample entropy (row 2, col 3)
    ax5 = fig.add_subplot(gs[1, 2])
    se_vals = []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        cpx = ts.get("complexity", {}) if isinstance(ts, dict) else {}
        se = _safe(cpx.get("sample_entropy")) if isinstance(cpx, dict) else np.nan
        se_vals.append(se if not np.isnan(se) else 0)
    ax5.bar(range(n_songs), se_vals,
            color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
            edgecolor="gray", alpha=0.8)
    ax5.set_xticks(range(n_songs))
    ax5.set_xticklabels([_short(n, 12) for n in names], rotation=45, ha="right")
    ax5.set_ylabel("Sample Entropy")
    ax5.set_title("4. Sample Entropy (structural complexity)", fontweight="bold")
    ax5.grid(True, alpha=0.2, axis="y")
    _annotate_consistency(ax5, se_vals)

    # 1.6 Duration comparison (row 3, col 1)
    ax6 = fig.add_subplot(gs[2, 0])
    dur_vals = []
    for name in names:
        r = per_file.get(name, {})
        ai = r.get("audio_info", {}) if isinstance(r.get("audio_info"), dict) else {}
        dur_vals.append(_safe(ai.get("duration", 0)))
    ax6.barh(range(n_songs), dur_vals,
             color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
             edgecolor="gray", alpha=0.8)
    ax6.set_yticks(range(n_songs))
    ax6.set_yticklabels([_short(n, 15) for n in names])
    ax6.set_xlabel("Duration (s)")
    ax6.set_title("5. Audio Duration", fontweight="bold")
    ax6.grid(True, alpha=0.2, axis="x")

    # 1.7 Dominant frequency / periodicity (row 3, col 2)
    ax7 = fig.add_subplot(gs[2, 1])
    domf_vals = []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        per = ts.get("periodicity", {}) if isinstance(ts, dict) else {}
        domf_vals.append(_safe(per.get("dominant_frequency", 0)) if isinstance(per, dict) else 0)
    ax7.bar(range(n_songs), domf_vals,
            color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
            edgecolor="gray", alpha=0.8)
    ax7.set_xticks(range(n_songs))
    ax7.set_xticklabels([_short(n, 12) for n in names], rotation=45, ha="right")
    ax7.set_ylabel("Dominant Frequency (Hz)")
    ax7.set_title("6. Dominant Frequency", fontweight="bold")
    ax7.grid(True, alpha=0.2, axis="y")

    # 1.8 Spectral feature correlation (row 3, col 3)
    ax8 = fig.add_subplot(gs[2, 2])
    spec_feats = ["spectral_flatness", "zcr", "sample_entropy", "duration_s"]
    available = [f for f in spec_feats if f in feat_names]
    if len(available) >= 2:
        idxs = [feat_names.index(f) for f in available]
        sub_mat = mat[:, idxs]
        sub_mat = sub_mat[~np.all(np.isnan(sub_mat), axis=1)]
        if sub_mat.shape[0] >= 3:
            corr = np.corrcoef(sub_mat.T)
            corr = np.nan_to_num(corr)
            im = ax8.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax8.set_xticks(range(len(available)))
            ax8.set_xticklabels([a.replace("_", "\n") for a in available], fontsize=8)
            ax8.set_yticks(range(len(available)))
            ax8.set_yticklabels(available)
            for i in range(len(available)):
                for j in range(len(available)):
                    ax8.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                             fontsize=7, color="white" if abs(corr[i, j]) > 0.6 else "black")
            plt.colorbar(im, ax=ax8, shrink=0.8)
    ax8.set_title("7. Spectral Feature Correlation", fontweight="bold")

    fig.suptitle("Spectral Analysis — Cross-Song Comparison", fontsize=16, fontweight="bold", y=1.01)
    if output_dir:
        return _save(fig, output_dir, "B01_spectral_dashboard.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dynamics & Volatility Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dynamics_dashboard(
    per_file: dict,
    song_mel_data: Dict[str, np.ndarray],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """4-panel trend overlay + 4-panel volatility overlay + GARCH summary."""
    dyn_series = _extract_dynamics_series(per_file)
    if not dyn_series:
        return None

    names = sorted(dyn_series.keys())
    n_songs = len(names)
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    trend_titles = ["Energy (Loudness)", "Brightness (Centroid)", "Complexity (Spread)", "Rhythm (Onset Δ)"]

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Row 1: Trend overlays (4 panels)
    for col, (key, title) in enumerate(zip(trend_keys, trend_titles)):
        ax = fig.add_subplot(gs[0, col])
        max_len = 0
        all_curves = []
        for ci, name in enumerate(names):
            series = dyn_series[name].get(key)
            if series is None:
                continue
            s = np.asarray(series, dtype=np.float64).ravel()
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)
            all_curves.append(s)
            max_len = max(max_len, len(s))
            ax.plot(s, linewidth=0.6, alpha=0.5, color=COLORS[ci % len(COLORS)])

        if len(all_curves) >= 2:
            interp = []
            for c in all_curves:
                xo = np.linspace(0, 1, len(c))
                xn = np.linspace(0, 1, max_len)
                interp.append(np.interp(xn, xo, c))
            st = np.array(interp)
            ax.plot(np.arange(max_len), st.mean(axis=0), "k-", linewidth=2.5)
            ax.fill_between(np.arange(max_len), st.mean(axis=0) - st.std(axis=0),
                            st.mean(axis=0) + st.std(axis=0), alpha=0.12, color="black")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Window"); ax.set_ylabel("Norm")
        ax.grid(True, alpha=0.2)

    # Row 2: Volatility overlays (4 panels)
    for col, (key, title) in enumerate(zip(trend_keys, trend_titles)):
        ax = fig.add_subplot(gs[1, col])
        vol_curves = []
        for ci, name in enumerate(names):
            r = per_file.get(name, {})
            vol = r.get("volatility", {}) if isinstance(r.get("volatility"), dict) else {}
            vs = vol.get("summary", vol)
            if isinstance(vs, dict):
                vd = vs.get(key, {})
                if isinstance(vd, dict):
                    mv = _safe(vd.get("mean_vol"))
                    if not np.isnan(mv):
                        vol_curves.append(mv)

        if vol_curves:
            ax.bar(range(len(vol_curves)), vol_curves,
                   color=[COLORS[i % len(COLORS)] for i in range(len(vol_curves))],
                   edgecolor="gray", alpha=0.8)
            ax.set_xticks(range(len(vol_curves)))
            ax.set_xticklabels([_short(names[i], 10) for i in range(len(vol_curves))],
                               rotation=45, ha="right", fontsize=7)
            _annotate_consistency(ax, vol_curves)
        ax.set_title(f"{title} — Volatility", fontweight="bold")
        ax.set_ylabel("Mean Volatility")
        ax.grid(True, alpha=0.2, axis="y")

    # Row 3: GARCH persistence + volatility regime summary
    # Col 0-1: GARCH persistence per trend
    ax_g = fig.add_subplot(gs[2, :2])
    garch_data = {t: [] for t in trend_keys}
    garch_names = []
    for name in names:
        garch_names.append(_short(name, 12))
        r = per_file.get(name, {})
        vol = r.get("volatility", {}) if isinstance(r.get("volatility"), dict) else {}
        vs = vol.get("summary", vol)
        if isinstance(vs, dict):
            for t in trend_keys:
                vd = vs.get(t, {})
                if isinstance(vd, dict):
                    gp = _safe(vd.get("garch_persistence"))
                    garch_data[t].append(gp if not np.isnan(gp) else 0)
                else:
                    garch_data[t].append(0)
        else:
            for t in trend_keys:
                garch_data[t].append(0)

    x = np.arange(len(garch_names))
    width = 0.2
    for ti, t in enumerate(trend_keys):
        ax_g.bar(x + ti * width, garch_data[t], width, label=t,
                 color=COLORS[ti % len(COLORS)], alpha=0.8, edgecolor="gray")
    ax_g.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="α+β=1 (unit root)")
    ax_g.set_xticks(x + width * 1.5)
    ax_g.set_xticklabels(garch_names, rotation=45, ha="right", fontsize=8)
    ax_g.set_ylabel("GARCH Persistence (α+β)")
    ax_g.set_title("GARCH(1,1) Persistence by Trend (higher = longer volatility memory)", fontweight="bold")
    ax_g.legend(fontsize=8); ax_g.grid(True, alpha=0.2, axis="y")

    # Col 2-3: Volatility regime summary text
    ax_regime = fig.add_subplot(gs[2, 2:])
    ax_regime.axis("off")
    regime_text = ["Volatility Regime Summary", "=" * 35, ""]
    for name in names:
        r = per_file.get(name, {})
        vol = r.get("volatility", {}) if isinstance(r.get("volatility"), dict) else {}
        vs = vol.get("summary", vol)
        regimes = []
        if isinstance(vs, dict):
            for t in trend_keys:
                vd = vs.get(t, {})
                if isinstance(vd, dict):
                    regimes.append(f"{t}={vd.get('volatility_regime', '?')}")
        regime_text.append(f"{_short(name, 18)}: {', '.join(regimes)}")

    # GARCH converged?
    regime_text.append("")
    regime_text.append("GARCH Convergence:")
    for name in names:
        r = per_file.get(name, {})
        vol = r.get("volatility", {}) if isinstance(r.get("volatility"), dict) else {}
        vs = vol.get("summary", vol)
        conv = []
        if isinstance(vs, dict):
            for t in trend_keys:
                vd = vs.get(t, {})
                if isinstance(vd, dict):
                    conv.append(f"{t}={vd.get('garch_converged', '?')}")
        regime_text.append(f"{_short(name, 18)}: {', '.join(conv)}")

    ax_regime.text(0.05, 0.95, "\n".join(regime_text), transform=ax_regime.transAxes,
                   fontsize=8, va="top", fontfamily="monospace")

    fig.suptitle("Dynamics & Volatility — Cross-Song Comparison", fontsize=16, fontweight="bold", y=1.01)
    if output_dir:
        return _save(fig, output_dir, "B02_dynamics_dashboard.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model Ensemble Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_ensemble_dashboard(
    per_file: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """ARIMA types + HMM states + LSTM memory + Transformer layers across songs."""
    if len(per_file) < 2:
        return None

    names = sorted(per_file.keys())
    n_songs = len(names)

    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    # Row 1, Col 0-1: Per-song per-model RMSE heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    models = ["ARIMA", "HMM", "LSTM", "Transformer"]
    heatmap = np.zeros((n_songs, len(models)))
    for i, name in enumerate(names):
        r = per_file.get(name, {})
        preds = r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}
        for j, mn in enumerate(models):
            arr = preds.get(mn, [])
            if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                heatmap[i, j] = _safe(arr[1].get("RMSE", np.nan))
            else:
                heatmap[i, j] = np.nan

    # Replace NaN with 0 for display
    heatmap_disp = np.nan_to_num(heatmap, nan=0)
    im = ax1.imshow(heatmap_disp, aspect="auto", cmap="YlOrRd")
    ax1.set_xticks(range(len(models))); ax1.set_xticklabels(models, fontsize=10)
    ax1.set_yticks(range(n_songs)); ax1.set_yticklabels([_short(n, 18) for n in names], fontsize=8)
    for i in range(n_songs):
        for j in range(len(models)):
            v = heatmap[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) and v > 0 else "N/A"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="white" if (not np.isnan(v) and v > heatmap_disp.max() * 0.5) else "black")
    ax1.set_title("1. Per-Song Per-Model RMSE (single-song prediction)", fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.85, label="RMSE")

    # Row 1, Col 2: HMM state counts
    ax2 = fig.add_subplot(gs[0, 2])
    hmm_counts = {}
    for name in names:
        r = per_file.get(name, {})
        ma = r.get("model_analysis", {}) if isinstance(r.get("model_analysis"), dict) else {}
        ns = ma.get("hmm_n_states", 0)
        hmm_counts[str(ns)] = hmm_counts.get(str(ns), 0) + 1
    if hmm_counts:
        labels = sorted(hmm_counts.keys(), key=lambda x: int(x))
        vals = [hmm_counts[l] for l in labels]
        ax2.pie(vals, labels=[f"{l} states" for l in labels], autopct="%1.1f%%",
                colors=plt.cm.Set2.colors[:len(labels)], startangle=90)
    ax2.set_title("2. HMM State Count Distribution", fontweight="bold")

    # Row 1, Col 3: Best band distribution
    ax3 = fig.add_subplot(gs[0, 3])
    band_dist = {}
    for name in names:
        r = per_file.get(name, {})
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            b = rank[0].get("band", "Unknown")
            band_dist[b] = band_dist.get(b, 0) + 1
    if band_dist:
        bl = sorted(band_dist.keys())
        bv = [band_dist[b] for b in bl]
        bc = [BAND_COLORS.get(b, "#888888") for b in bl]
        ax3.bar(bl, bv, color=bc, edgecolor="gray", alpha=0.85)
        ax3.set_ylabel("Number of songs")
        ax3.set_title("3. Most Predictable Frequency Band", fontweight="bold")
        ax3.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 0-1: LSTM learnability & Transformer layers
    ax4 = fig.add_subplot(gs[1, :2])
    lstm_lb, lstm_ml = [], []
    tf_layers = []
    song_labels = []
    for name in names:
        r = per_file.get(name, {})
        ma = r.get("model_analysis", {}) if isinstance(r.get("model_analysis"), dict) else {}
        lstm_lb.append(_safe(ma.get("lstm_optimal_lookback_s", 0)))
        lstm_ml.append(ma.get("lstm_most_learnable", "?"))
        tf_layers.append(_safe(ma.get("transformer_n_layers", 0)))
        song_labels.append(_short(name, 14))

    x = np.arange(len(song_labels))
    width = 0.3
    ax4.bar(x - width / 2, lstm_lb, width, label="LSTM Optimal Lookback (s)",
            color="#4C72B0", alpha=0.85, edgecolor="gray")
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + width / 2, tf_layers, width, label="Transformer Distinct Layers",
                  color="#55A868", alpha=0.85, edgecolor="gray")
    ax4.set_xticks(x)
    ax4.set_xticklabels(song_labels, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("LSTM Lookback (seconds)", color="#4C72B0")
    ax4_twin.set_ylabel("Transformer Layers", color="#55A868")
    ax4.set_title("4. LSTM Memory Length vs Transformer Layer Diversity", fontweight="bold")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 2-3: Ensemble summary text
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis("off")
    summary_lines = ["Model Ensemble Summary", "=" * 30, ""]
    for name in names:
        r = per_file.get(name, {})
        ma = r.get("model_analysis", {}) if isinstance(r.get("model_analysis"), dict) else {}
        es = ma.get("ensemble_summary", "")
        # Truncate long summaries
        if isinstance(es, str) and len(es) > 200:
            es = es[:197] + "..."
        summary_lines.append(f"{_short(name, 16)}:")
        summary_lines.append(f"  HMM: {ma.get('hmm_n_states', '?')} states")
        summary_lines.append(f"  LSTM: lookback={_safe(ma.get('lstm_optimal_lookback_s')):.2f}s, "
                             f"learnable={ma.get('lstm_most_learnable', '?')}")
        summary_lines.append(f"  Transformer: {ma.get('transformer_n_layers', '?')} layers")
        summary_lines.append(f"  Ensemble: {es}")
        summary_lines.append("")

    ax5.text(0.05, 0.95, "\n".join(summary_lines), transform=ax5.transAxes,
             fontsize=8, va="top", fontfamily="monospace")

    fig.suptitle("Model Ensemble — Cross-Song Structural Analysis", fontsize=16, fontweight="bold", y=1.01)
    if output_dir:
        return _save(fig, output_dir, "B03_model_ensemble_dashboard.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Global ML Performance Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_global_ml_dashboard(
    global_ml_report: dict,
    song_mel_data: Dict[str, np.ndarray],
    lookback: int = 30,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Comprehensive Global ML evaluation: RMSE, loss curves, HMM profiles, prediction samples."""
    lstm = global_ml_report.get("lstm", {})
    tf = global_ml_report.get("transformer", {})
    arima = global_ml_report.get("arima", {})
    hmm = global_ml_report.get("hmm", {})
    summary = global_ml_report.get("summary", {})

    if not summary:
        return None

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Row 1, Col 0-1: Per-song RMSE comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, :2])
    lstm_per = lstm.get("per_song", {})
    tf_per = tf.get("per_song", {})
    arima_per = arima if isinstance(arima, dict) else {}

    all_names = sorted(set(list(lstm_per.keys()) + list(arima_per.keys())))
    if not all_names:
        all_names = sorted(song_mel_data.keys())

    x = np.arange(len(all_names))
    width = 0.22
    lstm_r = [lstm_per.get(n, {}).get("rmse", np.nan) for n in all_names]
    tf_r = [tf_per.get(n, {}).get("rmse", np.nan) for n in all_names]
    arima_r = [arima_per.get(n, {}).get("rmse", np.nan) for n in all_names]

    ax1.bar(x - width, lstm_r, width, label="Global LSTM (full Mel spectrum)", color="#4C72B0", alpha=0.85)
    ax1.bar(x, tf_r, width, label="Global Transformer", color="#55A868", alpha=0.85)
    ax1.bar(x + width, arima_r, width, label="Local ARIMA (1D mean Mel)", color="#C44E52", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_short(n, 14) for n in all_names], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("RMSE")
    ax1.set_title("1. Per-Song Prediction Error — Global vs Local", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    # Add mean lines
    for vals, color, label in [(lstm_r, "#4C72B0", "LSTM mean"),
                                 (tf_r, "#55A868", "TF mean"),
                                 (arima_r, "#C44E52", "ARIMA mean")]:
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            ax1.axhline(y=np.mean(valid), color=color, linestyle="--", alpha=0.4)

    # Annotation
    valid_l = [v for v in lstm_r if not np.isnan(v)]
    valid_a = [v for v in arima_r if not np.isnan(v)]
    if valid_l and valid_a:
        l_mean, a_mean = np.mean(valid_l), np.mean(valid_a)
        note = (f"Global LSTM learns {summary.get('n_mels', '?')}-dim spectral patterns; "
                f"ARIMA operates on 1D.  {'Global model shows genre-level learning' if l_mean < a_mean * 3 else 'ARIMA strong on small datasets — try more songs or epochs'}.")
        ax1.text(0.5, -0.18, note, transform=ax1.transAxes, ha="center", fontsize=8, style="italic")

    # Row 1, Col 2: Loss curves
    ax2 = fig.add_subplot(gs[0, 2])
    lstm_loss = lstm.get("loss_history", [])
    tf_loss = tf.get("loss_history", [])
    if lstm_loss:
        ax2.plot(lstm_loss, "b-", linewidth=2, label=f"LSTM (final={lstm_loss[-1]:.2f})")
    if tf_loss:
        ax2.plot(tf_loss, "g-", linewidth=2, label=f"Transformer (final={tf_loss[-1]:.2f})")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
    ax2.set_title("2. Training Loss Curves", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)
    ax2.set_yscale("log")

    # Row 2: HMM state spectral profiles
    if hmm.get("n_states", 0) > 0 and song_mel_data:
        n_states = hmm["n_states"]
        per_song_states = hmm.get("per_song_state_sequences", {})

        # Collect Mel frames per state
        state_specs = {s: [] for s in range(n_states)}
        for name, states in per_song_states.items():
            if name not in song_mel_data:
                continue
            mel = song_mel_data[name]
            states_arr = np.array(states)
            n_frames = min(len(states_arr), mel.shape[1])
            for s in range(n_states):
                mask = states_arr[:n_frames] == s
                if mask.sum() > 0:
                    state_specs[s].append(mel[:, mask].mean(axis=1))

        n_mels = next(iter(song_mel_data.values())).shape[0]
        for s in range(min(n_states, 3)):  # up to 3 state panels
            ax_s = fig.add_subplot(gs[1, s]) if n_states <= 3 else fig.add_subplot(gs[1, s % 4])
            specs = state_specs.get(s, [])
            if specs:
                stacked = np.array(specs)
                mean_s = stacked.mean(axis=0)
                std_s = stacked.std(axis=0)
                ax_s.plot(mean_s, "k-", linewidth=2)
                ax_s.fill_between(range(n_mels), mean_s - std_s, mean_s + std_s,
                                  alpha=0.2, color="steelblue")
                ax_s.set_title(f"HMM State {s} — Spectral Profile", fontweight="bold")
                ax_s.set_xlabel("Mel band"); ax_s.set_ylabel("Energy")
                ax_s.grid(True, alpha=0.2)

    # Row 3, Col 0: Per-song state distribution
    ax4 = fig.add_subplot(gs[2, 0])
    if hmm.get("n_states", 0) > 0 and per_song_states:
        n_st = hmm["n_states"]
        sn = sorted(per_song_states.keys())
        hm = np.zeros((len(sn), n_st))
        for i, name in enumerate(sn):
            seq = per_song_states[name]
            if len(seq) > 0:
                for s in range(n_st):
                    hm[i, s] = seq.count(s) / len(seq)
        im = ax4.imshow(hm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax4.set_xticks(range(n_st)); ax4.set_xticklabels([f"S{s}" for s in range(n_st)])
        ax4.set_yticks(range(len(sn))); ax4.set_yticklabels([_short(n, 14) for n in sn], fontsize=7)
        ax4.set_title("4. Per-Song HMM State Usage", fontweight="bold")
        ax4.set_xlabel("HMM State")
        plt.colorbar(im, ax=ax4, shrink=0.85)

    # Row 3, Col 1: Prediction sample from one song
    ax5 = fig.add_subplot(gs[2, 1])
    if song_mel_data and all_names:
        sample_name = all_names[len(all_names) // 2]
        if sample_name in song_mel_data:
            mel = song_mel_data[sample_name]
            mid_band = mel.shape[0] // 2
            series = mel[mid_band, :200] if mel.shape[1] > 200 else mel[mid_band, :]
            ax5.plot(series, "k-", linewidth=0.8, alpha=0.7, label="True Mel band")
            # Smoothed reference
            if len(series) > 5:
                smoothed = np.convolve(series, np.ones(min(10, len(series)//4))/min(10, len(series)//4), mode="same")
                ax5.plot(smoothed, "r-", linewidth=1.2, alpha=0.7, label="Smoothed ref")
            ax5.set_title(f"5. Mel Energy — {_short(sample_name, 16)} (band {mid_band})", fontweight="bold")
            ax5.set_xlabel("Frame"); ax5.set_ylabel("Energy")
            ax5.legend(fontsize=8); ax5.grid(True, alpha=0.2)

    # Row 3, Col 2: Summary text
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")
    stats_text = [
        "Global ML Training Summary",
        "=" * 30, "",
        f"Songs: {summary.get('n_songs', '?')}",
        f"Windows pooled: {summary.get('n_windows', '?')}",
        f"Lookback: {summary.get('lookback', '?')} frames",
        f"Mel bands: {summary.get('n_mels', '?')}",
        f"Epochs: {summary.get('epochs', '?')}",
        f"Training time: {summary.get('training_time_s', 0):.1f}s",
        "",
        f"Global LSTM RMSE: {summary.get('lstm_global_rmse', np.nan):.3f}",
        f"Global Transformer RMSE: {summary.get('transformer_global_rmse', np.nan):.3f}",
        f"Local ARIMA mean RMSE: {summary.get('arima_mean_rmse', np.nan):.3f}",
        f"HMM states: {summary.get('hmm_n_states', '?')}",
        "",
        "Interpretation:",
    ]

    valid_l = [v for v in lstm_r if not np.isnan(v)]
    valid_a = [v for v in arima_r if not np.isnan(v)]
    if valid_l and valid_a:
        if np.mean(valid_l) < np.mean(valid_a) * 2:
            stats_text.append("  Global LSTM shows meaningful genre-level learning.")
        else:
            stats_text.append("  Try more songs or epochs for better global model performance.")
    stats_text.append("  HMM states represent shared musical textures across the genre.")

    ax6.text(0.05, 0.95, "\n".join(stats_text), transform=ax6.transAxes,
             fontsize=9, va="top", fontfamily="monospace")

    fig.suptitle("Global ML — Multi-Song Model Evaluation", fontsize=16, fontweight="bold", y=1.01)
    if output_dir:
        return _save(fig, output_dir, "B04_global_ml_dashboard.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Statistical Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def plot_statistical_summary(
    commonality_report: dict,
    per_file: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """Statistical summary: commonality ranking, distributions, outlier detection."""
    cv_scores = commonality_report.get("cv_scores", {})
    if not cv_scores:
        return None

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Top row, col 0: Feature consistency ranking (horizontal bar)
    ax1 = fig.add_subplot(gs[0, 0])
    valid_cv = {k: v for k, v in cv_scores.items() if not np.isnan(v) and v < 10 and v >= 0}
    sorted_cv = sorted(valid_cv.items(), key=lambda x: x[1])
    if sorted_cv:
        labels = [f[0].replace("_", " ") for f in sorted_cv]
        values = [f[1] for f in sorted_cv]
        colors = [plt.cm.RdYlGn_r(min(1.0, max(0.0, 1.0 - v))) for v in values]
        ax1.barh(range(len(labels)), values, color=colors, edgecolor="gray", alpha=0.85)
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=7)
        ax1.axvline(x=0.3, color="green", linestyle="--", alpha=0.5)
        ax1.axvline(x=0.6, color="orange", linestyle="--", alpha=0.5)
        ax1.set_xlabel("CV (lower = more consistent across songs)")
        ax1.set_title("1. Feature Consistency Ranking", fontweight="bold")
        ax1.grid(True, alpha=0.2, axis="x")

    # Top row, col 1: Genre highlights (most common features)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    most_common = commonality_report.get("most_common_features", [])
    most_diverse = commonality_report.get("most_diverse_features", [])

    highlight_text = ["Genre Signature Features", "=" * 30, ""]
    highlight_text.append("Most consistent (genre-defining):")
    for fn, cv in most_common[:5]:
        highlight_text.append(f"  + {fn.replace('_', ' ')}: CV={cv:.3f}")
    highlight_text.append("")
    highlight_text.append("Most variable (song-specific):")
    for fn, cv in most_diverse[:5]:
        highlight_text.append(f"  ~ {fn.replace('_', ' ')}: CV={cv:.3f}")
    highlight_text.append("")
    highlight_text.append("Interpretation:")
    if most_common:
        top = [f[0].split("_")[0] for f in most_common[:2]]
        highlight_text.append(f"  Songs in this group share similar {', '.join(top)} characteristics.")
    if most_diverse:
        bot = [f[0].split("_")[0] for f in most_diverse[:2]]
        highlight_text.append(f"  {', '.join(bot)} vary most between individual songs.")

    ax2.text(0.05, 0.95, "\n".join(highlight_text), transform=ax2.transAxes,
             fontsize=9, va="top", fontfamily="monospace")

    # Top row, col 2: Distribution summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    band_dist = commonality_report.get("best_band_distribution", {})
    hmm_dist = commonality_report.get("hmm_state_count_distribution", {})

    dist_text = ["Distributions Across Songs", "=" * 30, ""]
    dist_text.append("Best Predictable Band:")
    for band, count in sorted(band_dist.items()):
        dist_text.append(f"  {band}: {count} songs")
    dist_text.append("")
    dist_text.append("HMM State Count:")
    for ns, count in sorted(hmm_dist.items()):
        dist_text.append(f"  {ns} states: {count} songs")
    dist_text.append("")

    # Outlier summary
    outliers = commonality_report.get("outliers", {})
    if outliers:
        dist_text.append("Outliers Detected (|z| > 2):")
        for fn, songs in sorted(outliers.items()):
            dist_text.append(f"  {fn}: {', '.join(_short(s, 12) for s in songs)}")
    else:
        dist_text.append("No significant outliers — stylistically cohesive group.")

    ax3.text(0.05, 0.95, "\n".join(dist_text), transform=ax3.transAxes,
             fontsize=9, va="top", fontfamily="monospace")

    # Bottom row: Distribution plots per main feature group
    names = sorted(per_file.keys())
    n_songs = len(names)

    # Row 2, Col 0-1: Dynamics distributions (box plot style)
    ax4 = fig.add_subplot(gs[1, :2])
    dyn_data = {"energy": [], "brightness": [], "complexity": [], "rhythm": []}
    for name in names:
        r = per_file.get(name, {})
        dyn_entry = r.get("dynamics", {})
        if isinstance(dyn_entry, dict):
            ds = dyn_entry.get("summary", dyn_entry)
            if isinstance(ds, dict):
                for t in dyn_data:
                    td = ds.get(t, {})
                    if isinstance(td, dict):
                        dyn_data[t].append(_safe(td.get("mean"), 0))
                    else:
                        dyn_data[t].append(0)
            else:
                for t in dyn_data:
                    dyn_data[t].append(0)

    positions = []
    all_vals = []
    labels = []
    for ti, t in enumerate(["energy", "brightness", "complexity", "rhythm"]):
        vals = dyn_data[t]
        if vals:
            positions.append(ti + 1)
            all_vals.append(vals)
            labels.append(t)

    bp = ax4.boxplot(all_vals, positions=positions, labels=labels, patch_artist=True,
                      widths=0.4)
    for patch, color in zip(bp["boxes"], COLORS[:len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    # Overlay individual points
    for i, vals in enumerate(all_vals):
        ax4.scatter([positions[i]] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.7)
    ax4.set_ylabel("Mean Value")
    ax4.set_title("2. Dynamic Trend Distribution Across Songs", fontweight="bold")
    ax4.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 2: Prediction error distribution
    ax5 = fig.add_subplot(gs[1, 2])
    model_rmses = {m: [] for m in ["ARIMA", "HMM", "LSTM", "Transformer"]}
    for name in names:
        r = per_file.get(name, {})
        preds = r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}
        for mn in model_rmses:
            arr = preds.get(mn, [])
            if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                v = _safe(arr[1].get("RMSE"))
                if not np.isnan(v):
                    model_rmses[mn].append(v)

    positions2 = []
    all_vals2 = []
    labels2 = []
    for ti, mn in enumerate(["ARIMA", "HMM", "LSTM", "Transformer"]):
        vals = model_rmses[mn]
        if vals:
            positions2.append(ti + 1)
            all_vals2.append(vals)
            labels2.append(mn)

    if all_vals2:
        bp2 = ax5.boxplot(all_vals2, positions=positions2, labels=labels2, patch_artist=True, widths=0.4)
        for patch, color in zip(bp2["boxes"], COLORS[:len(labels2)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for i, vals in enumerate(all_vals2):
            ax5.scatter([positions2[i]] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.7)
    ax5.set_ylabel("RMSE")
    ax5.set_title("3. Prediction Error Distribution (single-song models)", fontweight="bold")
    ax5.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Statistical Summary — Genre Cohesion & Variation", fontsize=16, fontweight="bold", y=1.01)
    if output_dir:
        return _save(fig, output_dir, "B05_statistical_summary.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Batch Report Card
# ═══════════════════════════════════════════════════════════════════════════════

def plot_batch_report_card(
    commonality_report: dict,
    global_ml_report: dict,
    per_file: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """One-page text summary with key takeaways."""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")

    n_songs = commonality_report.get("n_songs", len(per_file))
    summary = global_ml_report.get("summary", {})

    y = 0.96
    lh = 0.022

    def add(title, size=14, bold=True, color="#2C3E50"):
        nonlocal y
        ax.text(0.04, y, title, fontsize=size, fontweight="bold" if bold else "normal",
                color=color, transform=ax.transAxes, va="top")
        y -= lh * (1.2 if size > 12 else 1)

    def line(text, size=10, indent=0):
        nonlocal y
        ax.text(0.04 + indent * 0.02, y, text, fontsize=size, transform=ax.transAxes, va="top")
        y -= lh

    # Title
    add(f"Batch Analysis Report — {n_songs} Songs", 18)
    y -= lh

    # 1. Overview
    add("1. Dataset Overview", 13)
    dur_vals = []
    for r in per_file.values():
        ai = r.get("audio_info", {}) if isinstance(r.get("audio_info"), dict) else {}
        d = ai.get("duration", 0)
        if d: dur_vals.append(d)
    if dur_vals:
        line(f"Total duration: {sum(dur_vals):.0f}s | Mean: {np.mean(dur_vals):.1f}s | Range: {min(dur_vals):.1f}-{max(dur_vals):.1f}s", indent=4)

    # 2. Spectral Signature
    add("2. Genre Spectral Signature", 13)
    cv_scores = commonality_report.get("cv_scores", {})
    band_dist = commonality_report.get("best_band_distribution", {})
    if band_dist:
        dom = max(band_dist, key=band_dist.get)
        line(f"Dominant predictable band: {dom} ({band_dist[dom]}/{n_songs} songs) — this frequency range has the most consistent temporal structure.", indent=4)
    most_common = commonality_report.get("most_common_features", [])
    if most_common:
        top3 = [f[0].replace("_", " ") for f in most_common[:3]]
        line(f"Most consistent features: {', '.join(top3)} — these define the genre's core identity.", indent=4)

    # 3. Dynamics & Structure
    add("3. Dynamics & Musical Structure", 13)
    hmm_dist = commonality_report.get("hmm_state_count_distribution", {})
    if hmm_dist:
        common_hmm = max(hmm_dist, key=hmm_dist.get)
        line(f"HMM hidden states: {common_hmm} states consistently found across songs — represents shared structural organization.", indent=4)

    arima_types = commonality_report.get("arima_trend_types", {})
    if arima_types:
        # Count trend types
        type_counts = {}
        for song, trends in arima_types.items():
            for t, tt in trends.items():
                type_counts[tt] = type_counts.get(tt, 0) + 1
        if type_counts:
            dom_type = max(type_counts, key=type_counts.get)
            line(f"Dominant ARIMA trend type: {dom_type} ({type_counts[dom_type]} occurrences) — most trends follow this pattern.", indent=4)

    # 4. Global ML
    add("4. Global Model Learning", 13)
    if summary:
        line(f"Training data: {summary.get('n_windows', '?')} Mel spectrogram windows pooled from {n_songs} songs.", indent=4)
        lstm_rmse = summary.get("lstm_global_rmse", np.nan)
        arima_rmse = summary.get("arima_mean_rmse", np.nan)
        if not np.isnan(lstm_rmse):
            line(f"Global LSTM RMSE: {lstm_rmse:.3f} | The shared model captures genre-level time-frequency patterns.", indent=4)
        if not np.isnan(arima_rmse):
            line(f"Local ARIMA RMSE: {arima_rmse:.3f} | Per-song baseline for comparison.", indent=4)

    # 5. HMM
    hmm = global_ml_report.get("hmm", {})
    if hmm.get("n_states", 0) > 0:
        add("5. Shared Musical States (Joint HMM)", 13)
        fractions = hmm.get("state_fractions", [])
        line(f"Discovered {hmm['n_states']} shared states across all songs.", indent=4)
        if fractions:
            line(f"State distribution: " + ", ".join(f"S{i}: {f:.0%}" for i, f in enumerate(fractions)), indent=4)
            line("These states represent recurring timbral patterns — the 'vocabulary' of this genre.", indent=4)

    # 6. Outliers
    add("6. Anomalies & Outliers", 13)
    outliers = commonality_report.get("outliers", {})
    if outliers:
        total_outliers = sum(len(v) for v in outliers.values())
        line(f"Found {total_outliers} outlier points across {len(outliers)} feature dimensions.", indent=4)
        for fn, songs in sorted(outliers.items(), key=lambda x: -len(x[1]))[:3]:
            line(f"  - {fn.replace('_', ' ')}: {', '.join(_short(s, 15) for s in songs)}", indent=6)
    else:
        line("No significant outliers — this group is stylistically cohesive.", indent=4)

    # Footer
    y -= lh * 2
    line("Generated by Audio Lab Batch Analysis", 8)
    line(f"Analysis dimensions: spectral + dynamics + volatility + model ensemble + prediction + HMM", 8)

    if output_dir:
        return _save(fig, output_dir, "B06_batch_report_card.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_batch_plots(
    batch_results: dict,
    output_dir: str,
) -> List[str]:
    """
    Generate ALL cross-song batch plots.

    Parameters
    ----------
    batch_results : dict
        per_file_results : {song_name: serialized_results}
        commonality_report : dict
        global_ml_report : dict
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    per_file = batch_results.get("per_file_results", {})
    commonality = batch_results.get("commonality_report", {})
    global_ml = batch_results.get("global_ml_report", {})

    # Recover Mel data from per-file results
    song_mel_data: Dict[str, np.ndarray] = {}
    for name, r in per_file.items():
        feats = r.get("features", {})
        if isinstance(feats, dict) and "mel" in feats:
            mel = feats["mel"]
            if isinstance(mel, dict) and "spec" in mel:
                arr = np.asarray(mel["spec"])
                if arr.ndim == 2 and arr.size > 0:
                    song_mel_data[name] = arr

    # 1. Spectral Dashboard
    if per_file:
        try:
            p = plot_spectral_dashboard(per_file, song_mel_data, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Spectral dashboard failed: {e}")

    # 2. Dynamics & Volatility Dashboard
    if per_file:
        try:
            p = plot_dynamics_dashboard(per_file, song_mel_data, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Dynamics dashboard failed: {e}")

    # 3. Model Ensemble Dashboard
    if per_file:
        try:
            p = plot_model_ensemble_dashboard(per_file, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Model ensemble dashboard failed: {e}")

    # 4. Global ML Dashboard
    if global_ml and song_mel_data:
        try:
            p = plot_global_ml_dashboard(global_ml, song_mel_data, output_dir=output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Global ML dashboard failed: {e}")

    # 5. Statistical Summary
    if commonality and per_file:
        try:
            p = plot_statistical_summary(commonality, per_file, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Statistical summary failed: {e}")

    # 6. Report Card
    if commonality or global_ml:
        try:
            p = plot_batch_report_card(commonality, global_ml, per_file, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Report card failed: {e}")

    return saved
