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
                   fontsize=8, va="top")

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
             fontsize=8, va="top")

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
             fontsize=9, va="top")

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
             fontsize=9, va="top")

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
             fontsize=9, va="top")

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
# 7. Genre Structure Decoder — trend analysis + GARCH + structural segments
# ═══════════════════════════════════════════════════════════════════════════════

def plot_genre_structure_decoder(
    per_file: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Decode what makes this genre structurally distinctive using:
    - ARIMA trend types per song (4 trends × N songs matrix)
    - GARCH persistence matrix
    - Structural segment composition (climax/calm/buildup/transition)
    - Trend directions
    """
    if len(per_file) < 2:
        return None

    names = sorted(per_file.keys())
    n_songs = len(names)
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    trend_cn = ["Energy", "Brightness", "Complexity", "Rhythm"]

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Row 1: ARIMA trend type matrix (text grid)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5, "Genre Structure Decoder — Multi-Dimensional Trend & GARCH Analysis",
                  transform=ax_title.transAxes, ha="center", fontsize=15, fontweight="bold")

    # Extract ARIMA types
    arima_matrix = {t: [] for t in trend_keys}
    garch_matrix = {t: [] for t in trend_keys}
    segment_data = {"climax": [], "calm": [], "buildup": [], "transition": []}
    trend_dirs = {t: [] for t in trend_keys}
    stationary_data = {t: [] for t in trend_keys}

    for name in names:
        r = per_file.get(name, {})
        extra = r.get("_batch_extra", {})

        # ARIMA types
        arima_types = extra.get("arima_trend_types", {})
        for t in trend_keys:
            arima_matrix[t].append(arima_types.get(t, "?"))

        # Stationarity
        arima_stat = extra.get("arima_stationary", {})
        for t in trend_keys:
            stationary_data[t].append(arima_stat.get(t, False))

        # GARCH persistence
        vol = r.get("volatility", {}) if isinstance(r.get("volatility"), dict) else {}
        vs = vol.get("summary", vol)
        if isinstance(vs, dict):
            for t in trend_keys:
                vd = vs.get(t, {})
                garch_matrix[t].append(_safe(vd.get("garch_persistence", np.nan)) if isinstance(vd, dict) else np.nan)
        else:
            for t in trend_keys:
                garch_matrix[t].append(np.nan)

        # Structural segments
        seg = extra.get("structural_segments", {})
        if seg:
            total = sum(seg.values()) + 1
            for k in segment_data:
                segment_data[k].append(seg.get(f"n_{k}", 0) / total * 100)
        else:
            for k in segment_data:
                segment_data[k].append(0)

        # Trend directions
        dyn = r.get("dynamics", {}) if isinstance(r.get("dynamics"), dict) else {}
        ds = dyn.get("summary", dyn)
        if isinstance(ds, dict):
            for t in trend_keys:
                td = ds.get(t, {})
                trend_dirs[t].append(td.get("trend_direction", "?") if isinstance(td, dict) else "?")
        else:
            for t in trend_keys:
                trend_dirs[t].append("?")

    # Col 0: ARIMA type grid
    ax1 = fig.add_subplot(gs[1, 0])
    # Map types to numbers for display
    type_map = {"mean-reverting": 0, "moving-average": 1, "oscillating": 2,
                "random-walk": 3, "trending": 4, "white-noise": 5,
                "unclear": 6, "too_short": 7, "no_model": 8, "?": 6}
    type_names = ["mean-reverting", "moving-avg", "oscillating", "random-walk",
                  "trending", "white-noise", "unclear"]
    type_colors = ["#4C72B0", "#55A868", "#F9A65A", "#C44E52", "#8B0000", "#937860", "#AAAAAA"]

    arima_num = np.zeros((n_songs, len(trend_keys)))
    for j, t in enumerate(trend_keys):
        for i, at in enumerate(arima_matrix[t]):
            arima_num[i, j] = type_map.get(at, 6)

    im1 = ax1.imshow(arima_num, aspect="auto", cmap="tab10", vmin=0, vmax=9)
    ax1.set_xticks(range(len(trend_keys))); ax1.set_xticklabels(trend_cn, fontsize=9)
    ax1.set_yticks(range(n_songs)); ax1.set_yticklabels([_short(n, 14) for n in names], fontsize=7)
    for i in range(n_songs):
        for j in range(len(trend_keys)):
            ax1.text(j, i, arima_matrix[trend_keys[j]][i][:10], ha="center", va="center", fontsize=6)
    ax1.set_title("ARIMA Trend Types", fontweight="bold")

    # Col 1: GARCH persistence heatmap
    ax2 = fig.add_subplot(gs[1, 1])
    gp_arr = np.zeros((n_songs, len(trend_keys)))
    for j, t in enumerate(trend_keys):
        for i, v in enumerate(garch_matrix[t]):
            gp_arr[i, j] = v if not np.isnan(v) else 0
    im2 = ax2.imshow(gp_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax2.set_xticks(range(len(trend_keys))); ax2.set_xticklabels(trend_cn, fontsize=9)
    ax2.set_yticks(range(n_songs)); ax2.set_yticklabels([_short(n, 14) for n in names], fontsize=7)
    for i in range(n_songs):
        for j in range(len(trend_keys)):
            v = garch_matrix[trend_keys[j]][i]
            txt = f"{v:.2f}" if not np.isnan(v) else "N/A"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=6,
                     color="white" if (not np.isnan(v) and v > 0.6) else "black")
    ax2.set_title("GARCH Persistence (α+β)", fontweight="bold")
    plt.colorbar(im2, ax=ax2, shrink=0.85)

    # Col 2: Structural segment composition
    ax3 = fig.add_subplot(gs[1, 2])
    x = np.arange(n_songs)
    bottom = np.zeros(n_songs)
    seg_colors = {"climax": "#C44E52", "buildup": "#F9A65A", "transition": "#937860", "calm": "#4C72B0"}
    for k in ["climax", "buildup", "transition", "calm"]:
        vals = segment_data[k]
        ax3.bar(x, vals, bottom=bottom, label=k.capitalize(),
                color=seg_colors[k], alpha=0.8, edgecolor="gray")
        bottom += vals
    ax3.set_xticks(x); ax3.set_xticklabels([_short(n, 12) for n in names], rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("% of windows")
    ax3.set_title("Structural Segment Composition", fontweight="bold")
    ax3.legend(fontsize=7)

    # Col 3: Trend directions summary text
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.axis("off")
    dir_text = ["Trend Direction Summary", "=" * 25, ""]
    for name in names:
        dirs = []
        for t in trend_keys:
            d = trend_dirs[t][names.index(name)] if names.index(name) < len(trend_dirs[t]) else "?"
            dirs.append(f"{t[0]}:{d}")
        dir_text.append(f"{_short(name, 16)}: {', '.join(dirs)}")

    # Add stationarity summary
    dir_text.append("")
    dir_text.append("Stationarity (ADF test):")
    for name in names:
        stats = []
        for t in trend_keys:
            idx = names.index(name)
            s = stationary_data[t][idx] if idx < len(stationary_data[t]) else False
            stats.append(f"{t[0]}:{'Y' if s else 'N'}")
        dir_text.append(f"{_short(name, 16)}: {', '.join(stats)}")

    ax4.text(0.05, 0.95, "\n".join(dir_text), transform=ax4.transAxes,
             fontsize=8, va="top")

    # Row 2: Genre-level summary
    # Count dominant ARIMA type per trend
    ax5 = fig.add_subplot(gs[2, :2])
    dom_types = {}
    for t in trend_keys:
        types_t = arima_matrix[t]
        counts = {}
        for tt in types_t:
            counts[tt] = counts.get(tt, 0) + 1
        dom = max(counts, key=counts.get) if counts else "?"
        dom_types[t] = (dom, counts[dom] / n_songs if counts else 0)

    # Bar chart of dominant types
    for ti, t in enumerate(trend_keys):
        counts = {}
        for tt in arima_matrix[t]:
            counts[tt] = counts.get(tt, 0) + 1
        x_pos = np.arange(len(type_names)) + ti * 0.2
        vals = [counts.get(tn, 0) for tn in type_names]
        ax5.bar(x_pos, vals, width=0.18, label=trend_cn[ti],
                color=COLORS[ti % len(COLORS)], alpha=0.8)
    ax5.set_xticks(np.arange(len(type_names)) + 0.3)
    ax5.set_xticklabels(type_names, rotation=45, ha="right", fontsize=7)
    ax5.set_ylabel("Count")
    ax5.set_title("ARIMA Trend Type Distribution by Dimension", fontweight="bold")
    ax5.legend(fontsize=8)

    # GARCH summary
    ax6 = fig.add_subplot(gs[2, 2:])
    gp_valid = {}
    for t in trend_keys:
        vals = [v for v in garch_matrix[t] if not np.isnan(v)]
        if vals:
            gp_valid[t] = vals
    if gp_valid:
        bp_data = list(gp_valid.values())
        bp_labels = list(gp_valid.keys())
        bp = ax6.boxplot(bp_data, labels=bp_labels, patch_artist=True, widths=0.4)
        for patch, color in zip(bp["boxes"], COLORS[:len(bp_labels)]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        # Overlay individual points
        for i, vals in enumerate(bp_data):
            ax6.scatter([i + 1] * len(vals), vals, color="black", s=20, zorder=5, alpha=0.7)
    ax6.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Unit root (α+β=1)")
    ax6.set_ylabel("GARCH Persistence (α+β)")
    ax6.set_title("GARCH(1,1) Persistence Distribution (higher = longer vol memory)", fontweight="bold")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.2, axis="y")
    _annotate_consistency(ax6, [np.mean(v) for v in gp_valid.values()], x=0.98, y=0.98)

    if output_dir:
        return _save(fig, output_dir, "B07_genre_structure_decoder.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Audio Signal Deep Dive — white noise, periodicity, complexity, unsupervised
# ═══════════════════════════════════════════════════════════════════════════════

def plot_audio_signal_deep_dive(
    per_file: dict,
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Deep dive into audio signal properties:
    - White noise test results (6 tests per song)
    - Periodicity comparison
    - Complexity scatter
    - Unsupervised measures (RQA, change points, motifs, clusters)
    """
    if len(per_file) < 2:
        return None

    names = sorted(per_file.keys())
    n_songs = len(names)

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.text(0.5, 0.5, "Audio Signal Deep Dive — White Noise, Periodicity & Unsupervised Patterns",
              transform=ax_t.transAxes, ha="center", fontsize=15, fontweight="bold")

    # Col 0: White noise test summary matrix
    ax1 = fig.add_subplot(gs[1, :2])
    test_names = ["ljung_box", "box_pierce", "jarque_bera", "acf_test",
                  "variance_stationarity", "runs_test"]
    test_labels = ["Ljung-Box", "Box-Pierce", "Jarque-Bera", "ACF Test",
                   "Var Stationarity", "Runs Test"]
    wn_matrix = np.zeros((n_songs, len(test_names)))
    for i, name in enumerate(names):
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        wn = ts.get("white_noise_test", {})
        if isinstance(wn, dict):
            for j, tn in enumerate(test_names):
                t = wn.get(tn, {})
                passed = False
                if isinstance(t, dict):
                    if tn == "jarque_bera":
                        passed = t.get("is_normal") == "True"
                    elif tn == "variance_stationarity":
                        passed = t.get("is_stationary") == True or t.get("is_stationary") == "True"
                    elif tn in ("acf_test",):
                        passed = t.get("is_white_noise") == True or t.get("is_white_noise") == "True"
                    else:
                        passed = t.get("is_white_noise") == "True" or t.get("is_white_noise") == True
                wn_matrix[i, j] = 1 if passed else 0

    im1 = ax1.imshow(wn_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_xticks(range(len(test_labels))); ax1.set_xticklabels(test_labels, fontsize=9)
    ax1.set_yticks(range(n_songs)); ax1.set_yticklabels([_short(n, 14) for n in names], fontsize=8)
    for i in range(n_songs):
        for j in range(len(test_labels)):
            ax1.text(j, i, "PASS" if wn_matrix[i, j] > 0.5 else "FAIL",
                     ha="center", va="center", fontsize=7, fontweight="bold",
                     color="white" if wn_matrix[i, j] > 0.5 else "black")
    ax1.set_title("White Noise Test Results (6 tests) — Green=PASS, Red=FAIL", fontweight="bold")
    plt.colorbar(im1, ax=ax1, shrink=0.85, ticks=[0, 1])
    ax1.set_xlabel("Higher pass rate = more unpredictable / noisy signal")

    # Col 2: Overall white noise summary
    ax2 = fig.add_subplot(gs[1, 2])
    overall = []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        wn = ts.get("white_noise_test", {})
        ov = wn.get("overall", {}) if isinstance(wn, dict) else {}
        tests_passed = ov.get("tests_passed", 0)
        total_tests = ov.get("total_tests", 5)
        overall.append(tests_passed / max(total_tests, 1) * 100)
    ax2.barh(range(n_songs), overall,
             color=[COLORS[i % len(COLORS)] for i in range(n_songs)],
             edgecolor="gray", alpha=0.8)
    ax2.set_yticks(range(n_songs)); ax2.set_yticklabels([_short(n, 14) for n in names], fontsize=8)
    ax2.set_xlabel("% Tests Passed (higher = closer to white noise)")
    ax2.set_title("White Noise Score\n(higher = more random signal)", fontweight="bold", fontsize=10)
    ax2.grid(True, alpha=0.2, axis="x")
    _annotate_consistency(ax2, overall, x=0.98, y=0.2)

    # Col 3: Complexity scatter
    ax3 = fig.add_subplot(gs[1, 3])
    zcr_vals, se_vals, sf_vals = [], [], []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        cpx = ts.get("complexity", {}) if isinstance(ts, dict) else {}
        zcr_vals.append(_safe(cpx.get("zero_crossing_rate", 0) if isinstance(cpx, dict) else 0))
        se_vals.append(_safe(cpx.get("sample_entropy", 0) if isinstance(cpx, dict) else 0))
        sf_vals.append(_safe(ts.get("spectral_flatness", 0)))
    for i, name in enumerate(names):
        ax3.scatter(zcr_vals[i], se_vals[i], s=100, color=COLORS[i % len(COLORS)],
                    edgecolor="black", alpha=0.8, label=_short(name, 14))
    ax3.set_xlabel("Zero-Crossing Rate"); ax3.set_ylabel("Sample Entropy")
    ax3.set_title("Complexity Space — ZCR vs Entropy\n(higher = more complex)", fontweight="bold", fontsize=10)
    ax3.legend(fontsize=6, loc="upper left"); ax3.grid(True, alpha=0.2)

    # Row 2, Col 0-1: Periodicity comparison
    ax4 = fig.add_subplot(gs[2, :2])
    domf_vals, domp_vals = [], []
    for name in names:
        r = per_file.get(name, {})
        ts = r.get("timeseries", {}) if isinstance(r.get("timeseries"), dict) else {}
        per = ts.get("periodicity", {}) if isinstance(ts, dict) else {}
        domf_vals.append(_safe(per.get("dominant_frequency", 0) if isinstance(per, dict) else 0))
        domp_vals.append(_safe(per.get("dominant_period", 0) if isinstance(per, dict) else 0))
    x = np.arange(n_songs)
    width = 0.35
    ax4.bar(x - width / 2, domf_vals, width, label="Dominant Frequency (Hz)", color="#4C72B0", alpha=0.85)
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + width / 2, domp_vals, width, label="Dominant Period (samples)", color="#C44E52", alpha=0.85)
    ax4.set_xticks(x); ax4.set_xticklabels([_short(n, 14) for n in names], rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Frequency (Hz)", color="#4C72B0")
    ax4_twin.set_ylabel("Period (samples)", color="#C44E52")
    ax4.set_title("Periodicity Analysis — Dominant Frequency & Period", fontweight="bold")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 2-3: Unsupervised measures
    ax5 = fig.add_subplot(gs[2, 2:])
    unsup_metrics = ["n_change_points", "n_segments", "n_motifs", "n_clusters",
                     "silhouette_score", "recurrence_rate", "determinism", "laminarity"]
    unsup_labels = ["Change\nPoints", "Segments", "Motifs", "N Clusters",
                    "Silhouette\nScore", "Recurrence\nRate", "Determinism", "Laminarity"]
    unsup_data = {m: [] for m in unsup_metrics}
    for name in names:
        r = per_file.get(name, {})
        extra = r.get("_batch_extra", {})
        unsup = extra.get("unsupervised", {})
        for m in unsup_metrics:
            v = unsup.get(m, np.nan) if isinstance(unsup, dict) else np.nan
            unsup_data[m].append(v if not np.isnan(v) else 0)

    x_u = np.arange(len(unsup_labels))
    width_u = 0.8 / n_songs
    for si, name in enumerate(names):
        vals = [unsup_data[m][si] if si < len(unsup_data[m]) else 0 for m in unsup_metrics]
        # Normalize to [0,1] for comparison
        max_vals = [max(unsup_data[m]) + 1e-12 for m in unsup_metrics]
        normed = [v / mv if mv > 0 else 0 for v, mv in zip(vals, max_vals)]
        ax5.bar(x_u + si * width_u, normed, width_u,
                color=COLORS[si % len(COLORS)], alpha=0.8, label=_short(name, 12))
    ax5.set_xticks(x_u + width_u * (n_songs - 1) / 2)
    ax5.set_xticklabels(unsup_labels, fontsize=7)
    ax5.set_ylabel("Normalized Value")
    ax5.set_title("Unsupervised Pattern Discovery — Cross-Song Comparison", fontweight="bold")
    ax5.legend(fontsize=6, ncol=2)
    ax5.grid(True, alpha=0.2, axis="y")

    if output_dir:
        return _save(fig, output_dir, "B08_audio_signal_deep_dive.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Unsupervised Commonality Discovery — PCA + Clustering + NMF
# ═══════════════════════════════════════════════════════════════════════════════

def plot_unsupervised_discovery(
    per_file: dict,
    song_mel_data: Dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    feature_names: List[str],
    song_names: List[str],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Unsupervised discovery of latent commonalities:
    - PCA 2D projection of songs in feature space
    - Hierarchical clustering dendrogram
    - NMF shared spectral components
    - Silhouette analysis
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    if len(song_names) < 3:
        return None

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.text(0.5, 0.5, "Unsupervised Commonality Discovery — Latent Structure Across Songs",
              transform=ax_t.transAxes, ha="center", fontsize=15, fontweight="bold")

    # Prepare data: remove NaN-heavy features and rows
    mat, fn, sn = feature_matrix, list(feature_names), list(song_names)
    # Keep features with < 50% NaN
    valid_cols = [j for j in range(mat.shape[1])
                  if np.sum(np.isnan(mat[:, j])) < mat.shape[0] * 0.5]
    mat_clean = mat[:, valid_cols]
    fn_clean = [fn[j] for j in valid_cols]

    # Fill remaining NaN with column median
    for j in range(mat_clean.shape[1]):
        col = mat_clean[:, j]
        mask = np.isnan(col)
        if mask.any():
            mat_clean[mask, j] = np.nanmedian(col)

    # Standardize
    mat_std = np.zeros_like(mat_clean)
    for j in range(mat_clean.shape[1]):
        col = mat_clean[:, j]
        std_j = np.std(col)
        if std_j > 1e-12:
            mat_std[:, j] = (col - np.mean(col)) / std_j
        else:
            mat_std[:, j] = 0.0

    # Col 0-1: PCA 2D projection
    ax1 = fig.add_subplot(gs[1, :2])
    pca = PCA(n_components=min(5, mat_std.shape[0], mat_std.shape[1]))
    pca_result = pca.fit_transform(mat_std)

    if pca_result.shape[1] >= 2:
        # Plot songs in PCA space
        for i, name in enumerate(sn):
            ax1.scatter(pca_result[i, 0], pca_result[i, 1],
                        s=250, color=COLORS[i % len(COLORS)],
                        edgecolor="black", linewidth=1.5, alpha=0.85,
                        label=_short(name, 16))
            ax1.annotate(_short(name, 14), (pca_result[i, 0], pca_result[i, 1]),
                         textcoords="offset points", xytext=(0, 12),
                         fontsize=8, ha="center")

        # Confidence ellipse (2σ for bivariate normal)
        from matplotlib.patches import Ellipse
        cov = np.cov(pca_result[:, :2].T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues) * 2.0  # 2σ
        ellipse = Ellipse(xy=np.mean(pca_result[:, :2], axis=0),
                          width=width, height=height, angle=angle,
                          facecolor="none", edgecolor="gray", linestyle="--",
                          linewidth=2, alpha=0.6, label="95% confidence ellipse")
        ax1.add_patch(ellipse)

        # Explained variance
        ev = pca.explained_variance_ratio_
        ax1.set_xlabel(f"PC1 ({ev[0]:.0%} variance)")
        ax1.set_ylabel(f"PC2 ({ev[1]:.0%} variance)" if len(ev) > 1 else "PC2")
    ax1.set_title("PCA Projection — Songs in Feature Space\n(closer = more similar)", fontweight="bold")
    ax1.legend(fontsize=7, loc="best", ncol=2)
    ax1.grid(True, alpha=0.2)
    ax1.axhline(y=0, color="gray", alpha=0.3); ax1.axvline(x=0, color="gray", alpha=0.3)

    # Col 2: Hierarchical clustering dendrogram
    ax2 = fig.add_subplot(gs[1, 2])
    if mat_std.shape[0] >= 3:
        dists = pdist(mat_std, metric="euclidean")
        Z = linkage(dists, method="ward")
        dendrogram(Z, labels=[_short(n, 14) for n in sn],
                   ax=ax2, leaf_font_size=8, color_threshold=0.7 * max(Z[:, 2]))
        ax2.set_title("Hierarchical Clustering\n(Ward linkage, Euclidean)", fontweight="bold")
        ax2.set_ylabel("Distance")
        # Rotate labels
        for label in ax2.get_xmajorticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    # Col 3: Automatic discovery summary
    ax3 = fig.add_subplot(gs[1, 3])
    ax3.axis("off")

    # Compute silhouette-like score for the group
    if mat_std.shape[0] >= 3:
        within_dists = []
        for i in range(mat_std.shape[0]):
            for j in range(i + 1, mat_std.shape[0]):
                within_dists.append(np.sqrt(np.sum((mat_std[i] - mat_std[j]) ** 2)))
        mean_within = np.mean(within_dists) if within_dists else 0
        # Within/between ratio as cohesion metric
        total_var = np.sum(np.var(mat_std, axis=0))
        cohesion = 1.0 / (1.0 + mean_within) if mean_within > 0 else 1.0
    else:
        cohesion = 0.0
        mean_within = 0.0

    # Find top 3 most similar song pairs
    similar_pairs = []
    if mat_std.shape[0] >= 2:
        for i in range(mat_std.shape[0]):
            for j in range(i + 1, mat_std.shape[0]):
                d = np.sqrt(np.sum((mat_std[i] - mat_std[j]) ** 2))
                similar_pairs.append((d, i, j))
        similar_pairs.sort()

    # Find NMF shared components from Mel data
    nmf_text = []
    if song_mel_data and len(song_mel_data) >= 2:
        try:
            from sklearn.decomposition import NMF
            # Pool all Mel spectra (time-averaged per song)
            all_means = []
            for name in sn:
                if name in song_mel_data:
                    all_means.append(song_mel_data[name].mean(axis=1))
            if len(all_means) >= 3:
                stacked = np.array(all_means)
                nmf = NMF(n_components=min(3, len(all_means)), random_state=42)
                W = nmf.fit_transform(stacked)  # (n_songs, n_components)
                H = nmf.components_  # (n_components, n_mels)
                nmf_text.append(f"NMF found {H.shape[0]} shared spectral components:")
                for c in range(H.shape[0]):
                    # Which songs use this component most?
                    top_song = sn[np.argmax(W[:, c])]
                    nmf_text.append(f"  Component {c+1}: dominant in {_short(top_song, 14)} "
                                    f"(weight={W[:, c].max():.2f})")
        except Exception:
            pass

    report = [
        "Unsupervised Discovery Report",
        "=" * 30, "",
        f"Group cohesion: {cohesion:.3f} (1.0 = perfectly cohesive)",
        f"Mean pairwise distance: {mean_within:.3f}",
        f"PCA explains {pca_result.shape[1]} dims",
        f"  PC1: {pca.explained_variance_ratio_[0]:.1%}",
        f"  PC2: {pca.explained_variance_ratio_[1]:.1%}" if pca_result.shape[1] > 1 else "",
        f"  Total: {sum(pca.explained_variance_ratio_):.1%}",
        "",
        "Most similar song pairs:",
    ]
    for d, i, j in similar_pairs[:3]:
        report.append(f"  {_short(sn[i], 12)} ↔ {_short(sn[j], 12)} (d={d:.3f})")
    if nmf_text:
        report.append("")
        report.extend(nmf_text)

    ax3.text(0.05, 0.95, "\n".join(report), transform=ax3.transAxes,
             fontsize=8.5, va="top")

    if output_dir:
        return _save(fig, output_dir, "B09_unsupervised_discovery.png")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Statistical Confidence Report — bootstrap CIs + effect sizes
# ═══════════════════════════════════════════════════════════════════════════════

def plot_statistical_confidence(
    per_file: dict,
    feature_matrix: np.ndarray,
    feature_names: List[str],
    song_names: List[str],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Statistical confidence for each commonality finding:
    - Bootstrap 95% CI waterfall chart
    - Effect size (Cohen's d) ranking
    - Permutation test p-value
    - Consensus score per feature group
    """
    from audiots.batch_stats import (
        feature_consensus_report, permutation_cluster_test, bootstrap_ci,
    )

    if len(song_names) < 3:
        return None

    consensus = feature_consensus_report(feature_matrix, feature_names)
    perm_test = permutation_cluster_test(feature_matrix)

    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.text(0.5, 0.5, "Statistical Confidence Report — How Confident Are These Commonalities?",
              transform=ax_t.transAxes, ha="center", fontsize=15, fontweight="bold")

    per_feat = consensus["per_feature"]

    # Col 0: Top features with bootstrap CI (waterfall)
    ax1 = fig.add_subplot(gs[1, 0])
    top_n = min(12, len(per_feat))
    top_features = per_feat[:top_n]
    labels = [f["feature"].replace("_", " ")[:25] for f in top_features]
    means = [f["ci_mean"] for f in top_features]
    lowers = [f["ci_lower"] for f in top_features]
    uppers = [f["ci_upper"] for f in top_features]

    y_pos = range(len(labels))
    # Normalize for display
    max_range = max(abs(np.nanmax(uppers)), abs(np.nanmin(lowers)), 1e-12)
    means_n = [m / max_range for m in means]
    lowers_n = [(m - l) / max_range for m, l in zip(means, lowers)]
    uppers_n = [(u - m) / max_range for m, u in zip(means, uppers)]

    colors = [plt.cm.RdYlGn(1.0 - f["consensus_score"]) for f in top_features]
    ax1.barh(y_pos, means_n, color=colors, edgecolor="gray", alpha=0.85,
             xerr=[lowers_n, uppers_n], capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel("Normalized mean with 95% CI")
    ax1.set_title("Feature Consensus Ranking\n(bar = mean, whisker = 95% bootstrap CI)", fontweight="bold", fontsize=10)
    ax1.axvline(x=0, color="gray", linewidth=0.5)
    ax1.grid(True, alpha=0.2, axis="x")

    # Col 1: Effect size ranking
    ax2 = fig.add_subplot(gs[1, 1])
    d_vals = [abs(f["cohens_d"]) if np.isfinite(abs(f["cohens_d"])) else 0 for f in per_feat[:top_n]]
    d_colors = []
    for d in d_vals:
        if d >= 0.8: d_colors.append("#2CA02C")  # large
        elif d >= 0.5: d_colors.append("#FF7F0E")  # medium
        elif d >= 0.2: d_colors.append("#D62728")  # small
        else: d_colors.append("#AAAAAA")  # negligible
    ax2.barh(y_pos, d_vals, color=d_colors, edgecolor="gray", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.axvline(x=0.2, color="orange", linestyle="--", alpha=0.5, label="small (0.2)")
    ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="medium (0.5)")
    ax2.axvline(x=0.8, color="green", linestyle="--", alpha=0.5, label="large (0.8)")
    ax2.set_xlabel("|Cohen's d|")
    ax2.set_title("Effect Size Ranking\n(higher = stronger genre signal)", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7)

    # Col 2: Consensus score by category
    ax3 = fig.add_subplot(gs[1, 2])
    # Group features by category
    groups = {"dynamics": [], "volatility": [], "spectral": [], "model": [], "complexity": [], "other": []}
    for f in per_feat:
        fn = f["feature"]
        if any(t in fn for t in ["energy", "brightness", "complex", "rhythm"]):
            if "vol" in fn or "garch" in fn:
                groups["volatility"].append(f["consensus_score"])
            elif "mean" in fn or "std" in fn or "peaks" in fn:
                groups["dynamics"].append(f["consensus_score"])
            else:
                groups["model"].append(f["consensus_score"])
        elif any(t in fn for t in ["zcr", "entropy", "flatness", "spectral"]):
            groups["spectral"].append(f["consensus_score"])
        elif any(t in fn for t in ["pred_", "rmse", "mae", "hmm_", "lstm_"]):
            groups["model"].append(f["consensus_score"])
        elif any(t in fn for t in ["duration", "change_point", "segment", "motif"]):
            groups["complexity"].append(f["consensus_score"])
        else:
            groups["other"].append(f["consensus_score"])

    group_labels = []
    group_means = []
    group_cis = []
    for g, vals in groups.items():
        if vals:
            group_labels.append(g)
            group_means.append(np.mean(vals))
            # Bootstrap CI for group mean
            if len(vals) >= 3:
                bs = bootstrap_ci(np.array(vals), n_bootstrap=500)
                group_cis.append((bs["mean"] - bs["lower"], bs["upper"] - bs["mean"]))
            else:
                group_cis.append((0, 0))

    group_cis_lower = [c[0] for c in group_cis]
    group_cis_upper = [c[1] for c in group_cis]
    g_colors = [plt.cm.Set2(i / len(group_labels)) for i in range(len(group_labels))]
    ax3.barh(range(len(group_labels)), group_means, color=g_colors, edgecolor="gray", alpha=0.85,
             xerr=[group_cis_lower, group_cis_upper], capsize=4)
    ax3.set_yticks(range(len(group_labels)))
    ax3.set_yticklabels(group_labels, fontsize=10)
    ax3.set_xlabel("Mean Consensus Score (0-1)")
    ax3.set_title("Consensus by Feature Group\n(higher = more genre-consistent)", fontweight="bold", fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.grid(True, alpha=0.2, axis="x")

    # Col 3: Summary text
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.axis("off")

    # Permutation test interpretation
    p_val = perm_test.get("p_value", np.nan)
    sig = perm_test.get("significant", False)

    report = [
        "Statistical Confidence Summary",
        "=" * 30, "",
        f"Features analyzed: {len(per_feat)}",
        f"Songs: {len(song_names)}",
        "",
        "Permutation Test:",
        f"  p = {p_val:.4f} {'*' if sig else '(n.s.)'}",
        f"  Observed cohesion: {perm_test.get('observed_dist', np.nan):.3f}",
        f"  Null expectation: {perm_test.get('null_mean', np.nan):.3f} +/- {perm_test.get('null_std', np.nan):.3f}",
    ]
    if sig:
        report.append("  => The observed similarity is statistically")
        report.append("     significant — songs are more similar than")
        report.append("     expected by chance.")
    else:
        report.append("  => Cannot reject null — need more songs or")
        report.append("     features to establish significance.")
    report.append("")
    report.append("Consensus Score Distribution:")
    report.append(f"  High (>0.7): {consensus['n_high_consensus']} features")
    report.append(f"  Medium (0.4-0.7): {consensus['n_medium_consensus']} features")
    report.append(f"  Low (<0.4): {consensus['n_low_consensus']} features")
    report.append("")
    report.append("Interpretation Guide:")
    report.append("  |d| > 0.8: strong genre-defining feature")
    report.append("  p < 0.05: statistically significant clustering")
    report.append("  Consensus > 0.7: highly reliable commonality")
    if per_feat and per_feat[0]["consensus_score"] > 0.7:
        report.append("")
        report.append(f"Strongest finding: {per_feat[0]['feature'].replace('_', ' ')}")
        report.append(f"  d={per_feat[0]['cohens_d']:.2f}, CI=[{per_feat[0]['ci_lower']:.3f}, {per_feat[0]['ci_upper']:.3f}]")
        report.append(f"  Consensus={per_feat[0]['consensus_score']:.3f}")

    ax4.text(0.05, 0.95, "\n".join(report), transform=ax4.transAxes,
             fontsize=8.5, va="top")

    if output_dir:
        return _save(fig, output_dir, "B10_statistical_confidence.png")
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

    # 7. Genre Structure Decoder (ARIMA types + GARCH + segments)
    if per_file:
        try:
            p = plot_genre_structure_decoder(per_file, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Genre structure decoder failed: {e}")

    # 8. Audio Signal Deep Dive (white noise + periodicity + unsupervised)
    if per_file:
        try:
            p = plot_audio_signal_deep_dive(per_file, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Audio signal deep dive failed: {e}")

    # ── Build feature matrix once for B09 and B10 ──────────────────────
    mat, feat_names, song_names_ordered = _extract_scalar_features(per_file)

    # 9. Unsupervised Commonality Discovery (PCA + clustering + NMF)
    if per_file and len(song_names_ordered) >= 3:
        try:
            p = plot_unsupervised_discovery(
                per_file, song_mel_data, mat, feat_names, song_names_ordered, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Unsupervised discovery failed: {e}")

    # 10. Statistical Confidence Report (bootstrap + effect sizes)
    if per_file and len(song_names_ordered) >= 3 and mat.shape[0] >= 3:
        try:
            p = plot_statistical_confidence(
                per_file, mat, feat_names, song_names_ordered, output_dir)
            if p: saved.append(p)
        except Exception as e:
            print(f"  [WARN] Statistical confidence failed: {e}")

    return saved
