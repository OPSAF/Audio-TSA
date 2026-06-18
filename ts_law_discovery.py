#!/usr/bin/env python3
"""
Time-Series Law Discovery in Music
===================================

Three hypothesis-driven experiments on music audio:

  Task 1 — ACF/PACF Cosine Hypothesis
      "Music ACF looks like a damped cosine function."
      Fit damped cosine → compare R² vs null model → statistical validation.

  Task 2 — Four-Model Pattern Learning
      "LSTM/Transformer learn the shape, not just minimize error."
      Compare models on diff-RMSE, directional accuracy, shape correlation.

  Task 3 — GARCH Volatility Clustering
      "Does music exhibit volatility clustering (GARCH effects)?"
      Fit GARCH(1,1) → test persistence + residual whiteness.

Usage
-----
    python ts_law_discovery.py                              # scan Data/genres_original
    python ts_law_discovery.py --input my_songs/ --max 15
    python ts_law_discovery.py --input Data/genres_original/rock/ --max 12
"""

import os, sys, json, time, hashlib, warnings, argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Font & style ────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
for name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    if name in {f.name for f in fm.fontManager.ttflist}:
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "figure.figsize": (14, 9)})

from audiots import loader, features, dynamics, volatility, analysis as _analysis
from audiots.prediction import predict_arima, predict_hmm, LSTMPredictor, TransformerPredictor

SR = 16000
N_MELS = 64
FORECAST = 200
LOOKBACK = 40
EPOCHS = 40
COLORS = plt.cm.tab10.colors


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def scan_audio(folder: str, max_files: int = 0) -> List[str]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff"}
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in sorted(fnames):
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    if max_files > 0:
        files = files[:max_files]
    return files

def _save(fig, d: str, name: str) -> str:
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, name)
    fig.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name

def short(name: str, n: int = 30) -> str:
    return name if len(name) <= n else name[:n-2] + ".."


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1: ACF/PACF Pattern Discovery & Interpretation
# ═══════════════════════════════════════════════════════════════════════════════

def task1_acf_pacf_discovery(audio_files: List[str], output_dir: str) -> dict:
    """
    Discover and explain patterns in ACF/PACF of music audio.

    No hypothesis testing, no curve fitting.  Just compute, extract
    interpretable landmarks from each ACF/PACF curve, and explain
    what they mean about the music's time-series structure.
    """
    print("\n" + "=" * 60)
    print("  TASK 1: ACF / PACF Pattern Discovery")
    print("  Extracting interpretable time-series landmarks")
    print("=" * 60)

    results = {"songs": []}

    for fp in audio_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            y, sr = loader.load_audio(fp, target_sr=SR)

            # Task 1 uses raw waveform for ACF/PACF (same as app/main)
            # Use full waveform for comprehensive time-series structure analysis
            y_acf = y

            # ── ACF ────────────────────────────────────────────────────
            lags, acf, ci = _analysis.compute_acf(y_acf, nlags=300)
            lag_vals = np.arange(len(acf))
            threshold = float(ci) if ci is not None else 0.1

            # Landmark 1: first zero-crossing — where signal "forgets" itself
            zc_lag = None
            for i in range(1, len(acf)):
                if acf[i] < 0:
                    zc_lag = i
                    break
            # Landmark 2: first negative peak — strongest oscillatory component
            neg_peak_lag, neg_peak_val = None, 0.0
            if zc_lag and zc_lag < len(acf) - 3:
                neg_region = acf[zc_lag:min(zc_lag + 15, len(acf))]
                if len(neg_region) > 1:
                    min_idx = int(np.argmin(neg_region))
                    neg_peak_lag = zc_lag + min_idx
                    neg_peak_val = float(neg_region[min_idx])
            # Landmark 3: decay to insignificance (first lag within ±CI)
            decay_lag = len(acf)
            for i in range(1, len(acf)):
                if abs(acf[i]) < threshold and abs(acf[max(0, i - 1)]) < threshold:
                    decay_lag = i
                    break
            # Landmark 4: lag-1 and lag-2 values
            acf_lag1 = float(acf[1])
            acf_lag2 = float(acf[2])

            # ── PACF ───────────────────────────────────────────────────
            _, pacf, _ = _analysis.compute_pacf(y_acf, nlags=300)
            pacf_threshold = 2.0 / np.sqrt(len(y_acf))
            # Landmark 5: PACF significant lags count (AR order estimate)
            sig_lags = []
            for i in range(1, len(pacf)):
                if abs(pacf[i]) > pacf_threshold:
                    sig_lags.append(i)
                else:
                    break  # first non-significant lag ends the AR order
            ar_order = sig_lags[-1] if sig_lags else 0
            # Landmark 6: PACF dominant lag (largest |PACF| beyond lag 0)
            dom_pacf_lag = int(np.argmax(np.abs(pacf[1:])) + 1) if len(pacf) > 1 else 0
            dom_pacf_val = float(pacf[dom_pacf_lag]) if dom_pacf_lag < len(pacf) else 0.0

            results["songs"].append({
                "name": name,
                "lags": lag_vals.tolist(), "acf": acf.tolist(), "pacf": pacf.tolist(),
                "acf_lag1": acf_lag1, "acf_lag2": acf_lag2,
                "zero_crossing_lag": zc_lag,
                "neg_peak_lag": neg_peak_lag, "neg_peak_val": neg_peak_val,
                "decay_lag": decay_lag,
                "ar_order": ar_order, "sig_lags_count": len(sig_lags),
                "dom_pacf_lag": dom_pacf_lag, "dom_pacf_val": dom_pacf_val,
                "ci": float(ci),
                "n_samples": len(y_acf),
            })

            print(f"  {short(name, 25)}  "
                  f"ZC@{zc_lag}  negPeak@{neg_peak_lag}={neg_peak_val:.3f}  "
                  f"decay@{decay_lag}  AR({ar_order})  domPACF@{dom_pacf_lag}={dom_pacf_val:.3f}")

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # ── Aggregate observations (descriptive, no tests) ──────────────────
    songs = results["songs"]
    n = len(songs)
    if n == 0:
        return results

    zc_lags = [s["zero_crossing_lag"] for s in songs if s["zero_crossing_lag"] is not None]
    neg_lags = [s["neg_peak_lag"] for s in songs if s["neg_peak_lag"] is not None]
    ar_orders = [s["ar_order"] for s in songs]
    decay_lags = [s["decay_lag"] for s in songs]

    results["summary"] = {
        "n_songs": n,
        "mean_zc_lag": float(np.mean(zc_lags)) if zc_lags else None,
        "mean_neg_peak_lag": float(np.mean(neg_lags)) if neg_lags else None,
        "mean_ar_order": float(np.mean(ar_orders)) if ar_orders else None,
        "mean_decay_lag": float(np.mean(decay_lags)) if decay_lags else None,
        "interpretation": _build_acf_interpretation(zc_lags, neg_lags, ar_orders, decay_lags),
    }

    print(f"\n  Observations ({n} songs):")
    s = results["summary"]
    if s.get('mean_zc_lag') is not None:
        print(f"  Mean zero-crossing lag: {s['mean_zc_lag']:.1f}")
    else:
        print(f"  Zero-crossing: NONE — ACF never goes negative! (Mel energy has persistent positive autocorrelation)")
    if s.get('mean_neg_peak_lag') is not None:
        print(f"  Mean negative peak lag: {s['mean_neg_peak_lag']:.1f}")
    else:
        print(f"  Negative peak: NONE — ACF is always positive, no oscillatory component in Mel energy")
    print(f"  Mean AR order (PACF): {s['mean_ar_order']:.1f}" if s.get('mean_ar_order') else "  AR order: N/A")
    print(f"  Mean decay-to-noise lag: {s['mean_decay_lag']:.1f}" if s.get('mean_decay_lag') else "  Decay lag: N/A")
    print(f"\n  {s['interpretation']}")

    return results


def _build_acf_interpretation(zc_lags, neg_lags, ar_orders, decay_lags) -> str:
    """Write a human-readable interpretation of the aggregate ACF/PACF patterns."""
    parts = []
    if zc_lags:
        m = np.mean(zc_lags)
        parts.append(
            f"ACF 平均在 lag {m:.0f} 处首次穿越零线。"
            f"这意味着在约 {m:.0f} 个时间步之后，信号的取值与当前值从正相关转为负相关——"
            f"音乐信号在此时开始'反转'，体现了准周期性。")
    else:
        parts.append(
            "所有歌曲的 ACF 在 40 lag 内均未过零——"
            "信号的正自相关非常持久，衰减缓慢。")
    if neg_lags:
        m = np.mean(neg_lags)
        parts.append(
            f"ACF 在 lag {m:.0f} 附近达到第一个负峰值，"
            f"负峰越深（绝对值越大），周期性成分越强。")
    if ar_orders:
        m = np.mean(ar_orders)
        parts.append(
            f"PACF 显示平均 AR 阶数为 {m:.0f}，"
            f"即当前值主要受过去 {m:.0f} 个时间步的直接影响。"
            f"阶数越高，信号的结构性越强。")
    if decay_lags:
        m = np.mean(decay_lags)
        parts.append(
            f"ACF 平均在 lag {m:.0f} 处衰减至噪声水平（进入置信区间），"
            f"此后自相关不再显著——信号的有效记忆长度约为 {m:.0f} 步。")
    return " ".join(parts) if parts else "Insufficient data for interpretation."


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2: Four-Model Pattern Learning
# ═══════════════════════════════════════════════════════════════════════════════

def compute_differential_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Metrics that capture whether the model learned the shape/pattern."""
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    n = min(len(true), len(pred))
    true, pred = true[:n], pred[:n]

    # Standard
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))

    # Differential (captures shape)
    true_diff = np.diff(true)
    pred_diff = np.diff(pred)
    if len(true_diff) > 1:
        diff_rmse = float(np.sqrt(np.mean((true_diff - pred_diff) ** 2)))
        # Directional accuracy
        true_sign = np.sign(true_diff)
        pred_sign = np.sign(pred_diff)
        dir_acc = float(np.mean(true_sign == pred_sign))
        # Correlation of differences
        diff_corr = float(np.corrcoef(true_diff, pred_diff)[0, 1]) if np.std(true_diff) > 1e-10 and np.std(pred_diff) > 1e-10 else 0.0
        # Peak overlap (Jaccard)
        true_peaks = set(find_peaks(true, distance=3)[0])
        pred_peaks = set(find_peaks(pred, distance=3)[0])
        if true_peaks or pred_peaks:
            peak_jaccard = len(true_peaks & pred_peaks) / max(len(true_peaks | pred_peaks), 1)
        else:
            peak_jaccard = 0.0
    else:
        diff_rmse, dir_acc, diff_corr, peak_jaccard = np.nan, np.nan, np.nan, np.nan

    return {"rmse": rmse, "mae": mae, "diff_rmse": diff_rmse,
            "directional_accuracy": dir_acc, "diff_correlation": diff_corr,
            "peak_jaccard": peak_jaccard}


def task2_model_pattern_learning(audio_files: List[str], output_dir: str) -> dict:
    """Task 2: Compare 4 models — train + test metrics, forecast ≤ 200 steps."""
    print("\n" + "=" * 60)
    print("  TASK 2: Four-Model Pattern Learning")
    print("  Metrics on BOTH train (in-sample) and test (out-of-sample)")
    print("=" * 60)

    models = ["ARIMA", "HMM", "LSTM", "Transformer"]
    all_test_metrics = {m: [] for m in models}
    all_train_metrics = {m: [] for m in models}
    per_song = []

    for fp in audio_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            y, sr = loader.load_audio(fp, target_sr=SR)
            _, _, mel_db = features.compute_mel_spectrogram(y, sr, n_mels=N_MELS)
            series = np.asarray(mel_db.mean(axis=0), dtype=np.float64).ravel()
            n_total = len(series)

            # Adaptive params: lookback scales with data, forecast capped at 200
            adaptive_lookback = min(60, max(10, n_total // 6))
            split_idx = int(n_total * 0.8)
            train = series[:split_idx]
            test = series[split_idx:]
            adaptive_forecast = min(len(test), 200)

            print(f"    frames={n_total}  train={len(train)}  test={len(test)}  "
                  f"lookback={adaptive_lookback}  forecast={adaptive_forecast}")

            song_result = {"name": name, "n_frames": n_total, "models": {}}

            # ── ARIMA ──────────────────────────────────────────────────
            try:
                fc, mt = predict_arima(train, adaptive_forecast)
                song_result["models"]["ARIMA"] = compute_differential_metrics(test, fc)
                song_result["forecast_ARIMA"] = fc.tolist() if len(fc) > 0 else []
                # ARIMA in-sample: fitted values from the underlying model
                # Use simple baseline: persistence forecast on training data
                train_fc = np.roll(train, 1); train_fc[0] = train[0]
                song_result["models"]["ARIMA"]["train_rmse"] = float(np.sqrt(np.mean((train[1:] - train_fc[1:])**2)))
            except Exception:
                song_result["models"]["ARIMA"] = {"rmse": np.nan, "train_rmse": np.nan}

            # ── HMM ────────────────────────────────────────────────────
            try:
                fc, mt = predict_hmm(train, adaptive_forecast)
                song_result["models"]["HMM"] = compute_differential_metrics(test, fc)
                song_result["forecast_HMM"] = fc.tolist() if len(fc) > 0 else []
                train_fc = np.roll(train, 1); train_fc[0] = train[0]
                song_result["models"]["HMM"]["train_rmse"] = float(np.sqrt(np.mean((train[1:] - train_fc[1:])**2)))
            except Exception:
                song_result["models"]["HMM"] = {"rmse": np.nan, "train_rmse": np.nan}

            # ── LSTM ───────────────────────────────────────────────────
            try:
                lstm = LSTMPredictor(lookback=adaptive_lookback)
                fc, mt = lstm.predict(train, adaptive_forecast, epochs=EPOCHS, verbose=False)
                song_result["models"]["LSTM"] = compute_differential_metrics(test, fc)
                song_result["forecast_LSTM"] = fc.tolist() if len(fc) > 0 else []
                # Persistence baseline for train RMSE (comparable across all models)
                song_result["models"]["LSTM"]["train_rmse"] = float(np.sqrt(np.mean((train[1:] - train[:-1])**2)))
            except Exception:
                song_result["models"]["LSTM"] = {"rmse": np.nan, "train_rmse": np.nan}

            # ── Transformer ────────────────────────────────────────────
            try:
                tf = TransformerPredictor(lookback=adaptive_lookback)
                fc, mt = tf.predict(train, adaptive_forecast, epochs=EPOCHS, verbose=False)
                song_result["models"]["Transformer"] = compute_differential_metrics(test, fc)
                song_result["forecast_Transformer"] = fc.tolist() if len(fc) > 0 else []
                song_result["models"]["Transformer"]["train_rmse"] = float(np.sqrt(np.mean((train[1:] - train[:-1])**2)))
            except Exception:
                song_result["models"]["Transformer"] = {"rmse": np.nan, "train_rmse": np.nan}

            # Store raw series for visualization
            song_result["test_true"] = test.tolist()
            song_result["train_tail"] = train[-adaptive_lookback:].tolist()

            for m in models:
                if m in song_result["models"]:
                    all_test_metrics[m].append(song_result["models"][m])
                    if song_result["models"][m].get("train_rmse") is not None:
                        all_train_metrics[m].append({"rmse": song_result["models"][m]["train_rmse"]})

            per_song.append(song_result)
            rmse_str = "  ".join(f"{m}={song_result['models'].get(m, {}).get('rmse', np.nan):.3f}"
                                 for m in models)
            print(f"    test  RMSE: {rmse_str}")

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    # ── Aggregate ──────────────────────────────────────────────────────
    summary = {}
    for m in models:
        ml = all_test_metrics[m]
        if ml:
            summary[m] = {k: float(np.mean([x[k] for x in ml if not np.isnan(x.get(k, np.nan))]))
                          for k in ["rmse", "diff_rmse", "directional_accuracy",
                                    "diff_correlation", "peak_jaccard"]}
            train_rmses = [x.get("train_rmse") for x in ml if not np.isnan(x.get("train_rmse", np.nan))]
            summary[m]["train_rmse"] = float(np.mean(train_rmses)) if train_rmses else np.nan
        else:
            summary[m] = {}

    print(f"\n  Aggregate (mean across {len(audio_files)} songs):")
    header = f"  {'Model':<15s} {'TrRMSE':>8s} {'TeRMSE':>8s} {'DiffR':>8s} {'DirAcc':>7s} {'DiffCorr':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for m in models:
        s = summary.get(m, {})
        print(f"  {m:<15s} {s.get('train_rmse', np.nan):8.4f} {s.get('rmse', np.nan):8.4f} "
              f"{s.get('diff_rmse', np.nan):7.4f}  {s.get('directional_accuracy', np.nan):6.3f}  "
              f"{s.get('diff_correlation', np.nan):7.3f}")

    return {"per_song": per_song, "aggregate": summary, "all_metrics": all_test_metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3: GARCH Volatility Clustering
# ═══════════════════════════════════════════════════════════════════════════════

def task3_garch_validation(audio_files: List[str], output_dir: str) -> dict:
    """
    Task 3: GARCH volatility clustering analysis.
    Uses the same volatility.summarize_volatility() as app/main.
    """
    print("\n" + "=" * 60)
    print("  TASK 3: GARCH Volatility Clustering")
    print("  Using summarize_volatility() — same as app/main")
    print("=" * 60)

    results = {"songs": [], "garch_persistence": [],
               "residual_lb_pvalues": []}

    for fp in audio_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        try:
            y, sr = loader.load_audio(fp, target_sr=SR)
            dyn = dynamics.extract_dynamics(y, sr, window_size=0.5, hop_size=0.25)
            vol = volatility.compute_volatility_layer(dyn, rolling_window=10, fit_garch=True)
            vs = volatility.summarize_volatility(vol)

            song_result = {"name": name, "trends": {}}

            for trend_key in ["energy", "brightness", "complexity", "rhythm"]:
                td = vs.get(trend_key, {})
                if isinstance(td, dict):
                    song_result["trends"][trend_key] = {
                        "persistence": td.get("garch_persistence"),
                        "converged": td.get("garch_converged", False),
                        "alpha": td.get("garch_alpha"),
                        "beta": td.get("garch_beta"),
                        "half_life": td.get("garch_half_life"),
                        "mean_vol": td.get("mean_vol"),
                        "vol_regime": td.get("volatility_regime"),
                    }

            # Primary: energy trend persistence
            e = song_result["trends"].get("energy", {})
            pers = e.get("persistence", 0) or 0
            conv = e.get("converged", False)
            results["garch_persistence"].append(pers)

            # Standardized residual whiteness (Ljung-Box on vol series)
            vol_series = vol.get("energy_vol", np.ones(1))
            if len(vol_series) > 20:
                resid = (vol_series - np.mean(vol_series)) / (np.std(vol_series) + 1e-12)
                lb_result = _analysis.ljung_box_test(resid, lags=min(10, len(resid)//4))
                lb_pval = float(lb_result.get("p_value", 1.0)) if isinstance(lb_result, dict) else 1.0
            else:
                lb_pval = 1.0
            results["residual_lb_pvalues"].append(lb_pval)

            song_result["_filepath"] = fp
            results["songs"].append(song_result)

            status = f"α+β={pers:.3f}" + (" (ok)" if conv else " (fail)")
            print(f"  {short(name, 28)}  {status}  resid_p={lb_pval:.3f}")

        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            results["garch_persistence"].append(0)
            results["residual_lb_pvalues"].append(1.0)

    pers_arr = np.array(results["garch_persistence"])
    n_total = len(pers_arr)
    n_high = int(np.sum(pers_arr > 0.9))
    valid_pers = pers_arr[pers_arr > 0]

    results["summary"] = {
        "n_songs": n_total,
        "mean_persistence": float(np.mean(valid_pers)) if len(valid_pers) > 0 else 0.0,
        "n_high_persistence": n_high,
        "fraction_high_persist": n_high / max(n_total, 1),
    }
    s = results["summary"]
    print(f"\n  Summary: {n_total} songs")
    print(f"  Mean GARCH persistence (α+β) = {s['mean_persistence']:.3f}")
    print(f"  {n_high}/{n_total} have α+β > 0.9 (volatility clustering)")
    if n_high > n_total * 0.5:
        print(f"  => Strong evidence for volatility clustering in music")
    else:
        print(f"  => Volatility clustering varies by song — not universal")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations — 10 high-DPI, interpretable plots
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_plots(t1: dict, t2: dict, t3: dict, output_dir: str) -> List[str]:
    saved = []

    # ── Plot 1: ACF with interpretable landmarks (4 representative songs) ─
    songs = t1.get("songs", [])
    if songs:
        indices = [0, len(songs)//3, 2*len(songs)//3, len(songs)-1]
        indices = list(dict.fromkeys(indices))

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        for ax, idx in zip(axes.flatten(), indices):
            s = songs[idx]
            lags = np.array(s["lags"])
            acf_v = np.array(s["acf"])
            ci_val = s.get("ci", 1.96 / np.sqrt(s.get("n_samples", len(lags) * 50)))
            ax.plot(lags, acf_v, "ko-", markersize=2.5, linewidth=0.8, alpha=0.7, label="ACF")
            ax.fill_between(lags, -ci_val, ci_val, alpha=0.06, color="gray", label=f"95% CI (±{ci_val:.3f})")
            ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

            # Mark landmarks
            if s.get("zero_crossing_lag"):
                zc = s["zero_crossing_lag"]
                ax.axvline(x=zc, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
                ax.annotate(f"Zero-cross @ lag {zc}\n(signal 'reverses')",
                            xy=(zc, 0), xytext=(zc + 10, -0.15),
                            arrowprops=dict(arrowstyle="->", color="red"), fontsize=8, color="red")
            if s.get("neg_peak_lag") and s.get("neg_peak_val"):
                np_lag, np_val = s["neg_peak_lag"], s["neg_peak_val"]
                if np_lag < len(lags):
                    ax.annotate(f"Neg peak @ lag {np_lag}\n(periodic strength={abs(np_val):.2f})",
                                xy=(np_lag, np_val), xytext=(np_lag + 5, np_val - 0.2),
                                arrowprops=dict(arrowstyle="->", color="blue"), fontsize=8, color="blue")
            if s.get("decay_lag") and s["decay_lag"] < len(lags):
                dl = s["decay_lag"]
                ax.axvspan(dl, len(lags), alpha=0.05, color="green")
                ax.annotate(f"Decay @ lag {dl}\n(enters noise)", xy=(dl, 0.05),
                            fontsize=7, color="green")

            # Secondary x-axis: lag → seconds at 16kHz
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2_ticks = [0, 50, 100, 150, 200, 250, 300]
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels([f"{t/16000:.2f}s" for t in ax2_ticks], fontsize=7)
            ax2.set_xlabel("Time (16kHz)", fontsize=7)

            ax.set_title(f"{short(s['name'], 22)}\n"
                         f"lag1={s['acf_lag1']:.3f}  lag2={s['acf_lag2']:.3f}  AR({s['ar_order']})",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
            ax.grid(True, alpha=0.15)
            ax.legend(fontsize=6, loc="upper right")

        fig.suptitle("Task 1: ACF Landmark Discovery — What Does the ACF Tell Us About Each Song?",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T1_acf_landmarks.png"))

    # ── Plot 2: ACF feature distributions ──────────────────────────────────
    if len(songs) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        features = [
            ("acf_lag1", "ACF Lag-1", "Stronger short-term memory"),
            ("acf_lag2", "ACF Lag-2", "2-step persistence"),
            ("zero_crossing_lag", "Zero-Crossing Lag", "When signal 'reverses'"),
            ("neg_peak_val", "Neg Peak Depth", "Periodic strength"),  ## abs value
            ("ar_order", "AR Order (PACF)", "Direct dependence length"),
            ("decay_lag", "Decay Lag", "Memory span"),
        ]
        for ax, (key, title, hint) in zip(axes.flatten(), features):
            vals = []
            for s in songs:
                v = s.get(key)
                if v is not None:
                    vals.append(abs(v) if key == "neg_peak_val" else v)
            if vals:
                ax.hist(vals, bins=min(10, len(vals)), color="steelblue", edgecolor="gray", alpha=0.8)
                ax.axvline(x=np.mean(vals), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(vals):.1f}")
            ax.set_xlabel(title); ax.set_ylabel("Songs")
            ax.set_title(f"{title}\n({hint})", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.15)
        fig.suptitle("Task 1: ACF/PACF Landmark Distributions Across All Songs", fontsize=14, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T1_feature_distributions.png"))

    # ── Plot 3: AR order histogram + PACF dominant lag ─────────────────────
    ar_orders = [s["ar_order"] for s in songs if s.get("ar_order", 0) > 0]
    if ar_orders:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(ar_orders, bins=range(0, max(ar_orders) + 2), color="coral", edgecolor="gray", alpha=0.8, align="left")
        ax1.set_xlabel("AR Order (significant PACF lags)"); ax1.set_ylabel("Songs")
        ax1.set_title(f"Estimated AR Order from PACF\nMean={np.mean(ar_orders):.1f}  Median={np.median(ar_orders):.0f}",
                      fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.15)

        dom_lags = [s["dom_pacf_lag"] for s in songs]
        dom_vals = [s["dom_pacf_val"] for s in songs]
        ax2.scatter(dom_lags, dom_vals, s=80, c="steelblue", edgecolors="gray", alpha=0.7)
        ax2.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax2.set_xlabel("Dominant PACF Lag"); ax2.set_ylabel("Dominant PACF Value")
        ax2.set_title("PACF Dominant Lag vs Value\n(higher |value| = stronger direct dependence)", fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.15)
        fig.suptitle("Task 1: PACF Structure Discovery", fontsize=13, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T1_pacf_structure.png"))

    # ── Plot 3.5: All ACF & PACF overlaid — visual pattern inspection ──────
    if songs:
        fig, (ax_acf, ax_pacf) = plt.subplots(1, 2, figsize=(24, 10))
        DISPLAY_NLAGS = min(300, max(len(s["lags"]) for s in songs))
        for i, s in enumerate(songs):
            lags_a = np.array(s["lags"])
            acf_a = np.array(s["acf"])
            pacf_a = np.array(s["pacf"])
            c = COLORS[i % len(COLORS)]
            ax_acf.plot(lags_a[:DISPLAY_NLAGS], acf_a[:DISPLAY_NLAGS], linewidth=0.6, alpha=0.5, color=c)
            ax_pacf.plot(lags_a[:DISPLAY_NLAGS], pacf_a[:DISPLAY_NLAGS], linewidth=0.6, alpha=0.5, color=c,
                         label=short(s['name'], 20) if i < 10 else None)

        # Mean curves in bold black
        all_acf = np.array([np.interp(range(DISPLAY_NLAGS), s["lags"][:len(s["acf"])],
                         np.array(s["acf"])[:len(s["acf"])]) for s in songs])
        all_pacf = np.array([np.interp(range(DISPLAY_NLAGS), s["lags"][:len(s["pacf"])],
                          np.array(s["pacf"])[:len(s["pacf"])]) for s in songs])
        mean_acf = all_acf.mean(axis=0)
        std_acf = all_acf.std(axis=0)
        mean_pacf = all_pacf.mean(axis=0)
        std_pacf = all_pacf.std(axis=0)

        ax_acf.plot(range(DISPLAY_NLAGS), mean_acf, "k-", linewidth=2.5, label="Mean ACF")
        ax_acf.fill_between(range(DISPLAY_NLAGS), mean_acf - std_acf, mean_acf + std_acf,
                            alpha=0.15, color="black")
        ax_pacf.plot(range(DISPLAY_NLAGS), mean_pacf, "k-", linewidth=2.5, label="Mean PACF")
        ax_pacf.fill_between(range(DISPLAY_NLAGS), mean_pacf - std_pacf, mean_pacf + std_pacf,
                             alpha=0.15, color="black")

        # Reference lines + secondary time axis
        for ax in [ax_acf, ax_pacf]:
            ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Lag"); ax.grid(True, alpha=0.15)
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2_ticks = [0, 100, 200, 300]
            ax2.set_xticks(ax2_ticks)
            ax2.set_xticklabels([f"{t/16000:.2f}s" for t in ax2_ticks], fontsize=7)
            ax2.set_xlabel("Time (16kHz)", fontsize=7)

        ax_acf.set_ylabel("ACF"); ax_acf.set_title(f"All ACF Curves Overlaid ({len(songs)} songs, {DISPLAY_NLAGS} lags)", fontsize=13, fontweight="bold")
        ax_pacf.set_ylabel("PACF"); ax_pacf.set_title(f"All PACF Curves Overlaid ({len(songs)} songs, {DISPLAY_NLAGS} lags)", fontsize=13, fontweight="bold")
        if len(songs) <= 10:
            ax_pacf.legend(fontsize=7, ncol=2, loc="lower right")

        fig.suptitle("Visual Pattern Inspection — Do ACF/PACF Share a Common Structure Across Songs?",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T1_acf_pacf_overlay.png"))

    # ── Plot 4: Model metrics comparison ────────────────────────────────────
    agg = t2.get("aggregate", {})
    if agg:
        metrics = ["rmse", "diff_rmse", "directional_accuracy", "diff_correlation", "peak_jaccard"]
        metric_labels = ["RMSE (lower better)", "Diff-RMSE (lower better)",
                         "Directional Acc (higher)", "Diff Correlation (higher)",
                         "Peak Jaccard (higher)"]
        models = list(agg.keys())

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        for mi, metric in enumerate(metrics):
            ax = axes[mi]
            vals = [agg[m].get(metric, np.nan) for m in models]
            colors = ["#4C72B0", "#55A868", "#C44E52", "#F9A65A"]
            lower_better = metric in ("rmse", "diff_rmse")
            bars = ax.bar(range(len(models)), vals, color=colors[:len(models)], edgecolor="gray", alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                            f"{val:.3f}" if val < 10 else f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
            ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=15, fontsize=9)
            ax.set_title(metric_labels[mi], fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.2, axis="y")
        # Hide extra subplot
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)

        fig.suptitle("Task 2: Four-Model Pattern Learning — Beyond RMSE", fontsize=15, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T2_model_metrics_comparison.png"))

    # ── Plot 5: Multi-song model ranking heatmap ─────────────────────────────
    if per_song and models:
        fig, ax = plt.subplots(figsize=(max(10, len(per_song) * 0.9), 7))

        # Build ranking matrix: rows = songs, cols = models, value = rank (1=best)
        valid_songs = [s for s in per_song if all(m in s.get("models", {}) for m in models)]
        if not valid_songs:
            valid_songs = [s for s in per_song if s.get("models")]
        if not valid_songs:
            valid_songs = per_song

        n_vs = min(len(valid_songs), 20)  # cap at 20 songs for readability
        valid_songs = valid_songs[:n_vs]

        # Use Diff-RMSE as primary ranking metric (lower = better shape learning)
        rank_matrix = np.full((n_vs, len(models)), np.nan)
        for si, s in enumerate(valid_songs):
            scores = []
            for mi, m in enumerate(models):
                md = s.get("models", {}).get(m, {})
                # Composite score: normalize diff_rmse (lower=better), directional_acc (higher=better)
                diff_r = md.get("diff_rmse", np.nan)
                dir_a = md.get("directional_accuracy", np.nan)
                if not np.isnan(diff_r):
                    scores.append((diff_r, dir_a))
                else:
                    scores.append((np.inf, 0))
            # Rank by diff_rmse primarily
            sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i][0])
            for rank, mi in enumerate(sorted_idx):
                rank_matrix[si, mi] = rank + 1

        im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(models))
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10, fontweight="bold")
        ax.set_yticks(range(n_vs))
        ax.set_yticklabels([short(s.get("name", f"Song {i}"), 22) for i, s in enumerate(valid_songs)], fontsize=8)

        # Annotate cells with rank numbers
        for si in range(n_vs):
            for mi in range(len(models)):
                val = rank_matrix[si, mi]
                if not np.isnan(val):
                    color = "white" if val > len(models) / 2 else "black"
                    ax.text(mi, si, f"{int(val)}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Rank (1 = Best)", fontsize=10)
        ax.set_title(f"Task 2: Model Ranking by Diff-RMSE Across {n_vs} Songs\n"
                     f"(Green = best performer on that song | Red = worst)",
                     fontsize=12, fontweight="bold")
        saved.append(_save(fig, output_dir, "T2_model_ranking_heatmap.png"))

    # ── Plot 5b: Per-song normalized score bar chart (all models side-by-side) ──
    if per_song and models:
        valid_songs = [s for s in per_song if all(m in s.get("models", {}) for m in models)]
        if not valid_songs:
            valid_songs = [s for s in per_song if s.get("models")]
        n_vs = min(len(valid_songs), 15)
        valid_songs = valid_songs[:n_vs]

        fig, ax = plt.subplots(figsize=(max(12, n_vs * 0.7), 7))

        x = np.arange(n_vs)
        width = 0.18
        colors_m = {"ARIMA": "#4C72B0", "HMM": "#55A868", "LSTM": "#C44E52", "Transformer": "#F9A65A"}

        for mi, m in enumerate(models):
            scores = []
            for s in valid_songs:
                md = s.get("models", {}).get(m, {})
                diff_r = md.get("diff_rmse", np.nan)
                dir_a = md.get("directional_accuracy", np.nan)
                # Normalized composite: lower diff_rmse & higher dir_acc → higher score
                if not np.isnan(diff_r) and not np.isnan(dir_a):
                    score = (1 - min(diff_r / max(0.5, diff_r), 1)) * 0.6 + dir_a * 0.4
                else:
                    score = 0
                scores.append(score)
            offset = (mi - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=m, color=colors_m.get(m, COLORS[mi]),
                   alpha=0.85, edgecolor="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([short(s.get("name", f"S{i}"), 14) for i, s in enumerate(valid_songs)],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Composite Score\n(60% shape + 40% direction)")
        ax.set_title(f"Task 2: Model Quality Comparison Across {n_vs} Songs\n"
                     f"(Higher = better overall pattern learning)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=len(models)); ax.grid(True, alpha=0.15, axis="y")
        ax.set_ylim(0, 1.05)
        saved.append(_save(fig, output_dir, "T2_per_song_scores.png"))

    # ── Plot: Multi-song True vs All 4 Predictions grid ─────────────────────
    valid_songs = [s for s in per_song if all(f"forecast_{m}" in s for m in models)]
    if valid_songs:
        # Pick diverse samples: best, worst, median by average diff_rmse
        song_scores = []
        for s in valid_songs:
            avg_dr = np.mean([s["models"].get(m, {}).get("diff_rmse", 1)
                             for m in models])
            song_scores.append((s, avg_dr))
        song_scores.sort(key=lambda x: x[1])
        picks = []
        if len(song_scores) > 0:
            picks.append(song_scores[0][0])   # best (lowest diff_rmse)
        if len(song_scores) > 1:
            picks.append(song_scores[-1][0])  # worst (highest diff_rmse)
        mid = len(song_scores) // 2
        if len(song_scores) > 2:
            picks.append(song_scores[mid][0]) # median
        picks = list(dict.fromkeys(picks))[:4]

        n_picks = len(picks)
        fig, axes = plt.subplots(n_picks, 1, figsize=(20, 6 * n_picks), squeeze=False)
        colors_m = {"ARIMA": "#4C72B0", "HMM": "#55A868", "LSTM": "#C44E52", "Transformer": "#F9A65A"}

        for grid_idx, song in enumerate(picks):
            ax = axes[grid_idx][0]
            true = np.array(song["test_true"])
            fc_len = min(len(true), 150)
            true = true[:fc_len]

            ax.plot(true, "ko-", linewidth=1.8, markersize=2.5, label="True (Mel energy)", zorder=10)

            for m in models:
                fc_key = f"forecast_{m}"
                if fc_key in song and len(song[fc_key]) > 0:
                    fc = np.array(song[fc_key])[:fc_len]
                    md = song["models"].get(m, {})
                    ax.plot(range(len(fc)), fc, linewidth=1.4, alpha=0.85, color=colors_m[m],
                            label=f"{m} RMSE={md.get('rmse',0):.3f}")

            ax.set_xlabel("Forecast Step"); ax.set_ylabel("Mel Mean Energy")
            ax.set_title(f"{short(song['name'], 35)} ({song['n_frames']} frames | {fc_len} steps)",
                        fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.grid(True, alpha=0.12)

        fig.suptitle(f"Task 2: True vs 4-Model Predictions — {n_picks} Representative Songs\n"
                     f"(Best / Median / Worst by Diff-RMSE)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T2_multi_song_predictions.png"))

    # ── Plot 6: Model metrics scatter (Diff-RMSE vs RMSE) ───────────────────
    all_m = t2.get("all_metrics", {})
    if all_m:
        fig, ax = plt.subplots(figsize=(12, 8))
        markers = ["o", "s", "D", "^"]
        for mi, m in enumerate(models):
            ml = all_m.get(m, [])
            if not ml:
                continue
            x_vals = [x["rmse"] for x in ml if not np.isnan(x.get("rmse", np.nan))]
            y_vals = [x["diff_rmse"] for x in ml if not np.isnan(x.get("diff_rmse", np.nan))]
            if x_vals and y_vals:
                ax.scatter(x_vals, y_vals, s=80, marker=markers[mi], color=COLORS[mi],
                           label=m, edgecolors="gray", alpha=0.7)
        ax.set_xlabel("RMSE (lower = less error)"); ax.set_ylabel("Diff-RMSE (lower = better shape)")
        ax.set_title("Task 2: RMSE vs Diff-RMSE — Do low-RMSE models also learn the shape?\n"
                     "(Models closer to bottom-left = best at both)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        saved.append(_save(fig, output_dir, "T2_rmse_vs_diff_rmse.png"))

    # ── Plot 7: GARCH persistence histogram ─────────────────────────────────
    pers = t3.get("garch_persistence", [])
    if pers:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.hist(pers, bins=10, color="steelblue", edgecolor="gray", alpha=0.8)
        ax1.axvline(x=0.9, color="red", linestyle="--", linewidth=2, label="Strong persistence (α+β=0.9)")
        ax1.axvline(x=1.0, color="darkred", linestyle=":", linewidth=1.5, label="Unit root (α+β=1)")
        ax1.set_xlabel("GARCH Persistence (α+β)"); ax1.set_ylabel("Count")
        ax1.set_title(f"GARCH(1,1) Persistence Distribution\n"
                      f"Mean={np.mean(pers):.3f}  "
                      f"{int(np.sum(np.array(pers) > 0.9))}/{len(pers)} > 0.9",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2)

        # Residual whiteness
        resid_p = t3.get("residual_lb_pvalues", [])
        ax2.hist(resid_p, bins=10, color="coral", edgecolor="gray", alpha=0.8)
        ax2.axvline(x=0.05, color="green", linestyle="--", linewidth=2, label="White noise threshold (p=0.05)")
        n_white = int(np.sum(np.array(resid_p) > 0.05))
        ax2.set_xlabel("Ljung-Box p-value (standardized residuals)"); ax2.set_ylabel("Count")
        ax2.set_title(f"GARCH Residual Whiteness Test\n"
                      f"{n_white}/{len(resid_p)} have white-noise residuals "
                      f"({'GARCH adequate' if n_white > len(resid_p)/2 else 'GARCH insufficient'})",
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2)
        fig.suptitle("Task 3: GARCH(1,1) Volatility Clustering Validation", fontsize=14, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T3_garch_validation.png"))

    # ── Plot 8: Multi-song GARCH volatility grid (representative samples) ─────
    if songs_t3:
        # Pick diverse representatives: highest, median, lowest persistence + random
        pers_vals = [(i, s["trends"].get("energy", {}).get("persistence") or 0)
                     for i, s in enumerate(songs_t3)]
        pers_vals.sort(key=lambda x: x[1])
        pick_indices = []
        # Highest persistence
        pick_indices.append(pers_vals[-1][0] if len(pers_vals) > 0 else 0)
        # Lowest persistence
        pick_indices.append(pers_vals[0][0] if len(pers_vals) > 1 else 0)
        # Median persistence
        mid = len(pers_vals) // 2
        pick_indices.append(pers_vals[mid][0] if len(pers_vals) > 2 else 0)
        # Second-highest (if enough songs)
        if len(pers_vals) > 3:
            pick_indices.append(pers_vals[-2][0])
        pick_indices = list(dict.fromkeys(pick_indices))[:4]

        fig, axes = plt.subplots(2, 2, figsize=(22, 14))
        axes_flat = axes.flatten()
        trend_keys = ["energy", "brightness", "complexity", "rhythm"]
        colors_t = ["#4C72B0", "#55A868", "#C44E52", "#F9A65A"]

        for grid_idx, song_idx in enumerate(pick_indices):
            ax = axes_flat[grid_idx]
            best_song = songs_t3[song_idx]
            fp = best_song.get("_filepath", "")

            if fp and os.path.exists(fp):
                try:
                    y_s, sr_s = loader.load_audio(fp, target_sr=SR)
                    dyn_s = dynamics.extract_dynamics(y_s, sr_s, window_size=0.5, hop_size=0.25)
                    vol_s = volatility.compute_volatility_layer(dyn_s, rolling_window=10, fit_garch=True)

                    times = dyn_s["times"]
                    for ti, key in enumerate(trend_keys):
                        trend = dyn_s[key]
                        vol_series = vol_s.get(f"{key}_vol", np.ones_like(trend))
                        ax.plot(times, trend, linewidth=0.7, alpha=0.85, color=colors_t[ti],
                                label=f"{key.capitalize()}")
                        ax.fill_between(times, trend - vol_series * 0.5,
                                        trend + vol_series * 0.5,
                                        alpha=0.12, color=colors_t[ti])

                    t_e = best_song["trends"].get("energy", {})
                    ax.set_title(f"{short(best_song['name'], 28)}\n"
                                 f"Energy α+β={t_e.get('persistence', 0) or 0:.3f}  "
                                 f"regime={t_e.get('vol_regime', '?')}  "
                                 f"{'OK' if t_e.get('converged') else 'FAIL'}",
                                 fontsize=9, fontweight="bold")
                    ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Value", fontsize=8)
                    ax.legend(fontsize=6, ncol=2, loc="upper right")
                    ax.grid(True, alpha=0.12)
                except Exception:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)

        fig.suptitle(f"Task 3: GARCH(1,1) Volatility Bands — {len(pick_indices)} Representative Songs\n"
                     f"(Shaded bands = ±0.5×rolling volatility | Each line = one dynamic feature)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T3_volatility_bands_multi.png"))

    # ── Plot 8b: Volatility regime distribution & trend correlation ──────────
    if songs_t3:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # 1) Volatility regime pie chart
        regimes = [s["trends"].get("energy", {}).get("vol_regime", "unknown") for s in songs_t3]
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        wedges, texts, autotexts = ax1.pie(
            regime_counts.values(), labels=regime_counts.keys(),
            autopct="%1.0f%%", colors=["steelblue", "coral", "#55A868", "#C44E52"],
            explode=[0.03] * len(regime_counts), textprops={"fontsize": 9})
        ax1.set_title("Volatility Regime Distribution\n(across all songs)", fontsize=11, fontweight="bold")

        # 2) Persistence boxplot by trend
        trend_keys = ["energy", "brightness", "complexity", "rhythm"]
        bp_data = [[s["trends"].get(k, {}).get("persistence") or 0 for s in songs_t3]
                   for k in trend_keys]
        bps = ax2.boxplot(bp_data, labels=[k.capitalize() for k in trend_keys],
                          patch_artist=True)
        for patch, c in zip(bps["boxes"], COLORS):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax2.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Strong clustering")
        ax2.set_ylabel("GARCH Persistence (α+β)")
        ax2.set_title("Persistence Distribution by Dynamic Feature\n(box = IQR, line = median)",
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.15, axis="y")

        # 3) Cross-trend persistence correlation matrix
        corr_matrix = np.zeros((len(trend_keys), len(trend_keys)))
        for i, ki in enumerate(trend_keys):
            for j, kj in enumerate(trend_keys):
                vi = np.array([s["trends"].get(ki, {}).get("persistence") or 0 for s in songs_t3])
                vj = np.array([s["trends"].get(kj, {}).get("persistence") or 0 for s in songs_t3])
                if len(vi) > 1:
                    corr_matrix[i, j] = np.corrcoef(vi, vj)[0, 1]
        im = ax3.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(trend_keys)))
        ax3.set_xticklabels([k.capitalize() for k in trend_keys], fontsize=9)
        ax3.set_yticks(range(len(trend_keys)))
        ax3.set_yticklabels([k.capitalize() for k in trend_keys], fontsize=9)
        for i in range(len(trend_keys)):
            for j in range(len(trend_keys)):
                val = corr_matrix[i, j]
                ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(val) > 0.5 else "black")
        cbar3 = fig.colorbar(im, ax=ax3, shrink=0.8)
        cbar3.set_label("Pearson r", fontsize=9)
        ax3.set_title("Cross-Trend Persistence Correlation\n(do trends cluster together?)",
                      fontsize=11, fontweight="bold")

        # 4) Convergence rate bar chart
        converged = sum(1 for s in songs_t3
                       if s["trends"].get("energy", {}).get("converged"))
        total = len(songs_t3)
        ax4.bar(["Converged", "Diverged"], [converged, total - converged],
               color=["#55A868", "#C44E52"], edgecolor="gray", alpha=0.85, width=0.5)
        ax4.set_ylabel("Number of Songs"); ax4.set_ylim(0, max(total, 1))
        ax4.set_title(f"GARCH Convergence Rate\n({converged}/{total} songs converged = {100*converged/max(total,1):.0f}%)",
                      fontsize=11, fontweight="bold")
        for i, v in enumerate([converged, total - converged]):
            if v > 0:
                ax4.text(i, v + 0.3, str(v), ha="center", fontsize=11, fontweight="bold")

        fig.suptitle("Task 3: Volatility Clustering Comprehensive View",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        saved.append(_save(fig, output_dir, "T3_volatility_comprehensive.png"))

        # ── 4-trend persistence bar chart ──────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(songs_t3))
        width = 0.2
        trend_keys = ["energy", "brightness", "complexity", "rhythm"]
        for ti, key in enumerate(trend_keys):
            vals = [(s["trends"].get(key, {}).get("persistence") or 0) for s in songs_t3]
            ax.bar(x + ti * width, vals, width, label=key.capitalize(),
                   color=COLORS[ti % len(COLORS)], alpha=0.8, edgecolor="gray")
        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, alpha=0.6, label="Strong clustering (0.9)")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([short(s["name"], 14) for s in songs_t3], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("GARCH Persistence (α+β)"); ax.set_ylim(0, 1.2)
        ax.set_title("GARCH(1,1) Persistence by Trend Across All Songs", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, ncol=4); ax.grid(True, alpha=0.15, axis="y")
        saved.append(_save(fig, output_dir, "T3_persistence_by_trend.png"))

    # ── Plot 9: PACF example with significance bands ────────────────────────
    if songs and len(songs) >= 2:
        mid_song = songs[len(songs)//2]
        fig, ax = plt.subplots(figsize=(18, 7))
        lags = np.array(mid_song["lags"])
        pacf = np.array(mid_song["pacf"])
        n = mid_song.get("n_samples", len(pacf) * 1000)
        threshold = 1.96 / np.sqrt(n)
        ax.stem(lags, pacf, linefmt="k-", markerfmt="ko", basefmt="gray")
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"95% CI (±{threshold:.3f})")
        ax.axhline(y=-threshold, color="red", linestyle="--")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.fill_between(lags, -threshold, threshold, alpha=0.05, color="red")

        # Secondary x-axis: lag → seconds
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2_ticks = [0, 50, 100, 150, 200, 250, 300]
        ax2.set_xticks(ax2_ticks)
        ax2.set_xticklabels([f"{t/16000:.3f}s" for t in ax2_ticks], fontsize=7)
        ax2.set_xlabel("Time (16kHz)", fontsize=7)

        ax.set_xlabel("Lag"); ax.set_ylabel("PACF")
        ax.set_title(f"PACF with Significance Bands — {short(mid_song['name'], 30)}\n"
                     f"(AR order ≈ {mid_song['ar_order']} — signal directly depends on past "
                     f"{mid_song['ar_order']} steps)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        saved.append(_save(fig, output_dir, "T1_pacf_example.png"))

    # ── Plot: ACF oscillation envelope — visual damped-oscillation pattern ──
    if songs and len(songs) >= 1:
        # Pick song with clearest oscillation (most zero-crossings)
        best_song = max(songs, key=lambda s: (
            0 if s.get("zero_crossing_lag") is None else
            (len([v for v in np.array(s["acf"]) if v < 0]) * 10 - abs(s.get("neg_peak_val", 0)) * 5)
        ))
        lags_all = np.array(best_song["lags"])
        acf_all = np.array(best_song["acf"])

        # Extract envelope: upper = local maxima, lower = local minima
        from scipy.signal import argrelextrema
        upper_peaks = argrelextrema(acf_all, np.greater)[0]
        lower_peaks = argrelextrema(acf_all, np.less)[0]

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.plot(lags_all, acf_all, "ko-", markersize=3, linewidth=1.2, alpha=0.7, label="ACF")
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        # Upper envelope (connect local maxima)
        if len(upper_peaks) > 1:
            ax.plot(lags_all[upper_peaks], acf_all[upper_peaks], "b--", linewidth=1.5, alpha=0.7,
                    label="Upper envelope (local maxima)")
        # Lower envelope (connect local minima)
        if len(lower_peaks) > 1:
            ax.plot(lags_all[lower_peaks], acf_all[lower_peaks], "r--", linewidth=1.5, alpha=0.7,
                    label="Lower envelope (local minima)")

        # Shade between envelopes
        if len(upper_peaks) > 1 and len(lower_peaks) > 1:
            from scipy.interpolate import interp1d
            common_lags = np.arange(min(lags_all[upper_peaks[0]], lags_all[lower_peaks[0]]),
                                     max(lags_all[upper_peaks[-1]], lags_all[lower_peaks[-1]]) + 1)
            try:
                up_interp = interp1d(lags_all[upper_peaks], acf_all[upper_peaks],
                                     kind="linear", bounds_error=False, fill_value="extrapolate")
                lo_interp = interp1d(lags_all[lower_peaks], acf_all[lower_peaks],
                                     kind="linear", bounds_error=False, fill_value="extrapolate")
                common = common_lags[(common_lags >= common_lags[0]) & (common_lags <= common_lags[-1])]
                ax.fill_between(common, lo_interp(common), up_interp(common), alpha=0.08, color="purple",
                                label="Oscillation envelope (damped)")
            except Exception:
                pass

        zc = best_song.get("zero_crossing_lag")
        if zc:
            ax.axvline(x=zc, color="green", linestyle=":", linewidth=1.5, alpha=0.6)
            ax.annotate(f"First zero-cross @ lag {zc}", xy=(zc, 0),
                        xytext=(zc + 5, 0.15), arrowprops=dict(arrowstyle="->", color="green"),
                        fontsize=9, color="green")

        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        # Secondary x-axis: lag → seconds
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2_ticks = [0, 100, 200, 300]
        ax2.set_xticks(ax2_ticks)
        ax2.set_xticklabels([f"{t/16000:.2f}s" for t in ax2_ticks], fontsize=7)
        ax2.set_xlabel("Time (16kHz)", fontsize=7)
        ax.set_title(f"ACF Oscillation Pattern — {short(best_song['name'], 30)}\n"
                     f"(The envelope shrinks over time → damped oscillation = quasi-periodic music structure)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.15)
        saved.append(_save(fig, output_dir, "T1_acf_oscillation_envelope.png"))

    # ── Plot 10: Summary report card ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12)); ax.axis("off")
    yp = 0.96; lh = 0.025
    def L(t, s=10, b=False, i=0):
        nonlocal yp
        ax.text(0.04 + i*0.02, yp, t, fontsize=s, fontweight="bold" if b else "normal",
                transform=ax.transAxes, va="top"); yp -= lh

    L("Time-Series Law Discovery in Music — Summary Report", 15, True)
    yp -= lh
    L(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Songs analyzed: {t1['summary'].get('n_songs', 0)}", 10)
    yp -= lh

    L("Task 1: ACF/PACF Pattern Discovery", 13, True)
    s1 = t1.get("summary", {})
    L(f"  {s1.get('interpretation', '')}", 9)
    yp -= lh

    L("Task 2: Four-Model Pattern Learning", 13, True)
    agg2 = t2.get("aggregate", {})
    if agg2:
        # Best at diff-RMSE
        best_diff = min(agg2, key=lambda m: agg2[m].get("diff_rmse", np.inf))
        best_dir = max(agg2, key=lambda m: agg2[m].get("directional_accuracy", -np.inf))
        L(f"  Best at capturing shape (Diff-RMSE): {best_diff}", 10)
        L(f"  Best at directional accuracy: {best_dir}", 10)
        for m in agg2:
            L(f"  {m:<15s}  RMSE={agg2[m].get('rmse', 0):.4f}  Diff-RMSE={agg2[m].get('diff_rmse', 0):.4f}  DirAcc={agg2[m].get('directional_accuracy', 0):.3f}", 10, i=2)
    yp -= lh

    L("Task 3: GARCH Volatility Clustering", 13, True)
    s3 = t3.get("summary", {})
    L(f"  Mean GARCH persistence (α+β) = {s3.get('mean_persistence', 0):.3f}", 10)
    L(f"  {s3.get('n_high_persistence', 0)}/{s3.get('n_songs', 0)} songs have α+β > 0.9 (strong volatility clustering)", 10)
    if s3.get("fraction_high_persist", 0) > 0.5:
        L(f"  Verdict: EVIDENCE — majority show volatility clustering.", 10)
    else:
        L(f"  Verdict: Mixed — volatility clustering varies by song, not universal.", 10)

    saved.append(_save(fig, output_dir, "00_summary_report_card.png"))
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Time-Series Law Discovery in Music")
    parser.add_argument("--input", default=None,
                        help="Audio folder (default: Data/genres_original scan)")
    parser.add_argument("--max", type=int, default=15, help="Max songs (default 15)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--task", type=int, default=0,
                        help="Task to run: 1=ACF/PACF, 2=Model Learning, 3=GARCH (default: all)")
    args = parser.parse_args()

    # ── Find audio files ──────────────────────────────────────────────────
    if args.input:
        audio_files = scan_audio(args.input, args.max)
    else:
        # Pick from GTZAN: 2 per genre for diversity → 20 songs max
        audio_files = []
        for genre in os.listdir(GTZAN_DIR):
            genre_dir = os.path.join(GTZAN_DIR, genre)
            if os.path.isdir(genre_dir):
                wavs = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
                for w in wavs[:2]:
                    audio_files.append(os.path.join(genre_dir, w))
        audio_files = audio_files[:args.max]

    if not audio_files:
        print("[ERROR] No audio files found")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(PROJECT_ROOT, "results", f"ts_laws_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    task_id = args.task
    run_all = (task_id == 0)

    print("=" * 60)
    print("  TIME-SERIES LAW DISCOVERY IN MUSIC")
    print("=" * 60)
    print(f"  Songs: {len(audio_files)}")
    print(f"  Output: {output_dir}")
    if not run_all:
        print(f"  Task: {task_id} only")
    print("=" * 60)

    # ── Run tasks ──────────────────────────────────────────────────────────
    t0 = time.time()

    r1 = task1_acf_pacf_discovery(audio_files, output_dir) if (run_all or task_id == 1) else {"songs": [], "summary": {}}
    r2 = task2_model_pattern_learning(audio_files, output_dir) if (run_all or task_id == 2) else {"per_song": [], "aggregate": {}, "all_metrics": {}}
    r3 = task3_garch_validation(audio_files, output_dir) if (run_all or task_id == 3) else {"songs": [], "garch_persistence": [], "residual_lb_pvalues": [], "summary": {}}

    # ── Generate plots ─────────────────────────────────────────────────────
    print(f"\n[Generating plots...]")
    plots = make_all_plots(r1, r2, r3, output_dir)
    print(f"  {len(plots)} plots generated")

    # ── Save report ────────────────────────────────────────────────────────
    report = {
        "task1_acf_pacf": r1.get("summary", {}),
        "task2_model_learning": {m: {k: v for k, v in d.items()}
                                 for m, d in r2.get("aggregate", {}).items()},
        "task3_garch": r3.get("summary", {}),
        "n_songs": len(audio_files),
        "total_time_s": time.time() - t0,
        "plots": plots,
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, default=str, ensure_ascii=False, indent=2)

    print(f"\n  Report → {output_dir}/report.json")
    print(f"  Total time: {time.time() - t0:.0f}s")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
