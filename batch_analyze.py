#!/usr/bin/env python3
"""
Batch Audio Time-Series Analysis CLI
=====================================

Comprehensive multi-song statistical analysis for same-genre audio files.
Reuses the shared ``audiots.pipeline`` for per-file analysis (zero duplication),
then adds cross-song commonality mining, Global-Model ML training, and
rich batch visualisations.

Usage
-----
    python batch_analyze.py data/rock/ -o results/rock/          # full analysis
    python batch_analyze.py data/rock/ -o results/rock/ --fast   # reduced epochs
    python batch_analyze.py data/rock/ --max-files 30            # limit files
    python batch_analyze.py data/rock/ --no-global-ml            # skip ML training
    python batch_analyze.py data/rock/ --resume                  # skip done files

Features
--------
* Phase A — Per-file analysis (shared pipeline, same as web/CLI)
* Phase B — Cross-song "same-window" commonality analysis
  - Feature matrix assembly (dynamics + volatility + complexity + band metrics)
  - CV-based commonality ranking
  - ARIMA trend-type / HMM-state / best-band distribution analysis
  - Outlier detection (z-score > 2.0)
* Phase C — Global Model ML training (NEW)
  - Mel Spectrogram sliding-window dataset (all songs pooled)
  - Global LSTM  — one model, trained on ALL songs' windows
  - Global Transformer — one model, trained on ALL songs' windows
  - ARIMA per song (Local baseline)
  - Joint HMM on all songs' Mel features
* Phase D — Batch visualisations (trend overlays, radar, model comparison, …)
* Output — per_file/*.json + aggregate_summary.json + commonality_report.json
           + global_ml_report.json + figures/*.png + batch_report.csv
"""

import os
import sys
import json
import time
import uuid
import argparse
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audiots import (
    loader, features, dynamics, volatility, model_analysis,
    discovery, unsupervised, analysis as _analysis, prediction, band_analysis,
    config as _cfg,
)
from audiots.pipeline import (
    run_full_analysis, generate_all_plots, serialize_results,
    DEFAULT_OPTIONS,
)

from audiots.batch_ml import (
    prepare_mel_windows_all_songs,
    train_global_models,
    train_arima_baselines,
    train_joint_hmm,
)

from audiots.batch_viz import generate_all_batch_plots

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 128
DEFAULT_FORECAST_HORIZON = 100
DEFAULT_MAX_FILES = 0           # 0 = no limit
DEFAULT_LOOKBACK = 30
DEFAULT_ML_EPOCHS = 30
DEFAULT_HMM_STATES = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Phase A — Per-file analysis (shared pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def run_per_file_analysis(
    filepath: str,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    n_mels: int = DEFAULT_N_MELS,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    fast: bool = False,
) -> dict:
    """
    Run full analysis on one audio file using the SHARED pipeline.

    Returns raw results dict (with ndarrays, model objects, etc.).
    Serialization happens later.
    """
    y, sr = loader.load_audio(filepath, target_sr=target_sr)

    # Build analysis options — same as web app defaults minus comparison
    analysis_options = [
        "features", "dynamics", "dynamics_analysis", "model_analysis",
        "timeseries", "unsupervised", "prediction", "band",
    ]

    results = run_full_analysis(
        y1=y, sr=sr,
        forecast_horizon=forecast_horizon,
        n_mels=n_mels,
        analysis_options=analysis_options,
        progress_callback=None,  # silent for batch
        fast=fast,
    )

    return results


def _safe_serialize_results(results: dict, task_id: str,
                            task_info: dict) -> dict:
    """Call pipeline.serialize_results and strip what can't be JSON-encoded."""
    return serialize_results(results, task_id=task_id, task_info=task_info,
                             plot_files=[])


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B — Cross-song commonality analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=np.nan):
    """Extract float from nested structures."""
    if val is None:
        return default
    if isinstance(val, (int, float, np.floating)):
        return float(val)
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return default


def build_feature_matrix(per_file: Dict[str, dict]) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Assemble (N_songs, D_features) matrix from per-file results.

    Returns
    -------
    feature_matrix : ndarray
    feature_names : list of str
    meta : dict with extra categorical data (ARIMA types, etc.)
    """
    song_names = sorted(per_file.keys())
    rows = []
    arima_trend_types: Dict[str, Dict[str, str]] = {}

    for name in song_names:
        r = per_file[name]

        # ── Dynamics summaries ───────────────────────────────────────────
        dyn_entry = r.get("dynamics", {})
        if isinstance(dyn_entry, dict):
            dyn_summary = dyn_entry.get("summary", dyn_entry)
        else:
            dyn_summary = {}

        row = {}
        for trend in ["energy", "brightness", "complexity", "rhythm"]:
            ds = dyn_summary.get(trend, {}) if isinstance(dyn_summary, dict) else {}
            row[f"{trend}_mean"] = _safe_float(ds.get("mean"))
            row[f"{trend}_std"] = _safe_float(ds.get("std"))
            row[f"{trend}_n_peaks"] = _safe_float(ds.get("n_peaks"))

        # ── Volatility summaries ─────────────────────────────────────────
        vol = r.get("volatility", {})
        if isinstance(vol, dict):
            vol_summary = vol.get("summary", vol)
        else:
            vol_summary = {}
        for trend in ["energy", "brightness", "complexity", "rhythm"]:
            vs = vol_summary.get(trend, {}) if isinstance(vol_summary, dict) else {}
            row[f"{trend}_vol_mean"] = _safe_float(vs.get("mean_vol"))
            row[f"{trend}_vol_of_vol"] = _safe_float(vs.get("vol_of_vol"))
            gp = vs.get("garch_persistence")
            row[f"{trend}_garch_persist"] = _safe_float(gp) if gp is not None else np.nan

        # ── Complexity ───────────────────────────────────────────────────
        ts = r.get("timeseries", {})
        if isinstance(ts, dict):
            cpx = ts.get("complexity", {})
            if isinstance(cpx, dict):
                row["zero_crossing_rate"] = _safe_float(cpx.get("zero_crossing_rate"))
                row["sample_entropy"] = _safe_float(cpx.get("sample_entropy"))
            else:
                row["zero_crossing_rate"] = np.nan
                row["sample_entropy"] = np.nan
            row["spectral_flatness"] = _safe_float(ts.get("spectral_flatness"))
        else:
            row["zero_crossing_rate"] = np.nan
            row["sample_entropy"] = np.nan
            row["spectral_flatness"] = np.nan

        # ── Model analysis ───────────────────────────────────────────────
        ma = r.get("model_analysis", {})
        if isinstance(ma, dict):
            row["hmm_n_states"] = _safe_float(ma.get("hmm_n_states", 0))
            row["lstm_optimal_lookback_s"] = _safe_float(ma.get("lstm_optimal_lookback_s"))
            row["transformer_n_layers"] = _safe_float(ma.get("transformer_n_layers", 0))
            # Per-trend ARIMA types (categorical)
            if ma.get("arima_summary"):
                # ARIMA types come from per_trend dict
                pass
        else:
            row["hmm_n_states"] = np.nan
            row["lstm_optimal_lookback_s"] = np.nan
            row["transformer_n_layers"] = np.nan

        # ── Prediction metrics ───────────────────────────────────────────
        preds = r.get("prediction", {})
        if isinstance(preds, dict):
            for mn in ["ARIMA", "HMM", "LSTM", "Transformer"]:
                arr = preds.get(mn, [])
                if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                    row[f"pred_{mn}_rmse"] = _safe_float(arr[1].get("RMSE"))
                    row[f"pred_{mn}_mae"] = _safe_float(arr[1].get("MAE"))
                else:
                    row[f"pred_{mn}_rmse"] = np.nan
                    row[f"pred_{mn}_mae"] = np.nan
        else:
            for mn in ["ARIMA", "HMM", "LSTM", "Transformer"]:
                row[f"pred_{mn}_rmse"] = np.nan
                row[f"pred_{mn}_mae"] = np.nan

        # ── Band ─────────────────────────────────────────────────────────
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            row["best_band_rmse"] = _safe_float(rank[0].get("avg_rmse"))

        # ── Audio info ───────────────────────────────────────────────────
        ai = r.get("audio_info", {})
        if isinstance(ai, dict):
            row["duration_s"] = _safe_float(ai.get("duration"))

        # ── Unsupervised ─────────────────────────────────────────────────
        u = r.get("unsupervised", {})
        if isinstance(u, dict):
            row["n_change_points"] = _safe_float(u.get("n_change_points", 0))
            row["n_segments"] = _safe_float(u.get("n_segments", 0))
            row["n_motifs"] = _safe_float(u.get("n_motifs", 0))

        # ── Collect ARIMA trend types (from model_analysis) ──────────────
        song_arima_types: Dict[str, str] = {}
        ma = r.get("model_analysis", {})
        if isinstance(ma, dict):
            # ARIMA insights in model_analysis
            hmm_profiles = ma.get("hmm_state_profiles", [])
            for trend in ["energy", "brightness", "complexity", "rhythm"]:
                # Try to get type from per_trend ARIMA
                song_arima_types[trend] = "unclear"
        arima_trend_types[name] = song_arima_types

        rows.append(row)

    # Build feature names from first row
    feature_names = list(rows[0].keys()) if rows else []

    # Fill matrix
    feature_matrix = np.array([[r.get(fn, np.nan) for fn in feature_names]
                                for r in rows], dtype=np.float64)

    meta = {
        "arima_trend_types": arima_trend_types,
    }

    return feature_matrix, feature_names, meta


def compute_commonality_report(
    feature_matrix: np.ndarray,
    feature_names: List[str],
    song_names: List[str],
    per_file: Dict[str, dict],
    meta: dict,
) -> dict:
    """
    Compute cross-song commonality metrics.

    Returns a comprehensive report with:
    - cv_scores: {feature: CV} — lower = more common
    - outliers: {feature: [song_names]}
    - genre_highlights: most consistent features
    - diversity_points: most variable features
    - distributions: categorical feature distributions
    """
    n_songs, n_features = feature_matrix.shape
    report: dict = {
        "n_songs": n_songs,
        "n_features": n_features,
    }

    # ── CV per feature ────────────────────────────────────────────────────
    cv_scores = {}
    outlier_report = {}
    mean_vals = {}
    std_vals = {}

    for j, fn in enumerate(feature_names):
        col = feature_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            cv_scores[fn] = np.nan
            continue
        mean_j = float(np.mean(valid))
        std_j = float(np.std(valid))
        cv_j = std_j / abs(mean_j) if abs(mean_j) > 1e-12 else 0.0
        cv_scores[fn] = float(cv_j)
        mean_vals[fn] = mean_j
        std_vals[fn] = std_j

        # Outliers: |z| > 2
        if std_j > 1e-12:
            z_scores = np.abs((col - mean_j) / std_j)
            outlier_songs = [song_names[i] for i in range(n_songs)
                             if not np.isnan(z_scores[i]) and z_scores[i] > 2.0]
            if outlier_songs:
                outlier_report[fn] = outlier_songs

    report["cv_scores"] = cv_scores
    report["mean_values"] = mean_vals
    report["std_values"] = std_vals
    report["outliers"] = outlier_report
    report["feature_names"] = feature_names
    report["song_names"] = song_names
    report["feature_matrix"] = feature_matrix.tolist()

    # ── Genre highlights (lowest CV = most common) ────────────────────────
    valid_cv = {k: v for k, v in cv_scores.items() if not np.isnan(v) and v < 10}
    sorted_cv = sorted(valid_cv.items(), key=lambda x: x[1])
    report["most_common_features"] = sorted_cv[:5] if len(sorted_cv) >= 5 else sorted_cv
    report["most_diverse_features"] = sorted_cv[-5:] if len(sorted_cv) >= 5 else sorted_cv

    # ── Categorical distributions ────────────────────────────────────────
    # Best band distribution
    band_dist = {}
    for name in song_names:
        r = per_file.get(name, {})
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            bb = rank[0].get("band", "unknown")
            band_dist[bb] = band_dist.get(bb, 0) + 1
    report["best_band_distribution"] = band_dist

    # HMM state count distribution
    hmm_dist = {}
    for name in song_names:
        r = per_file.get(name, {})
        ma = r.get("model_analysis", {})
        if isinstance(ma, dict):
            ns = ma.get("hmm_n_states", 0)
            hmm_dist[str(ns)] = hmm_dist.get(str(ns), 0) + 1
    report["hmm_state_count_distribution"] = hmm_dist

    # ARIMA trend types
    report["arima_trend_types"] = meta.get("arima_trend_types", {})

    # ── Synthesis ────────────────────────────────────────────────────────
    highlights_parts = []
    for fn, cv in sorted_cv[:3]:
        highlights_parts.append(f"{fn} (CV={cv:.3f})")
    diversity_parts = []
    for fn, cv in sorted_cv[-3:]:
        diversity_parts.append(f"{fn} (CV={cv:.3f})")

    report["overview"] = (
        f"对 {n_songs} 首歌曲的 {n_features} 个特征进行了跨曲共性分析。\n"
        f"流派共性最强特征: {', '.join(highlights_parts)}。\n"
        f"流派内差异最大特征: {', '.join(diversity_parts)}。\n"
        f"检测到 {sum(len(v) for v in outlier_report.values())} 个离群点 "
        f"({len(outlier_report)} 个特征维度)。\n"
        f"最佳可预测频带分布: {band_dist}。\n"
        f"HMM 状态数分布: {hmm_dist}。"
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C — Global ML orchestration
# ═══════════════════════════════════════════════════════════════════════════════

def run_global_ml(
    audio_files: List[str],
    per_file: Dict[str, dict],
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = DEFAULT_ML_EPOCHS,
    n_hmm_states: int = DEFAULT_HMM_STATES,
    n_mels: int = DEFAULT_N_MELS,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    verbose: bool = True,
) -> dict:
    """
    Load Mel Spectrograms for all songs and run Global ML training.

    Uses the Mel Spectrogram already computed in per-file results if available,
    otherwise recomputes from audio.
    """
    song_mel_specs: Dict[str, np.ndarray] = {}

    for fp in audio_files:
        name = os.path.splitext(os.path.basename(fp))[0]

        # Try to get Mel from per_file results first
        r = per_file.get(name, {})
        feats = r.get("features", {})
        if isinstance(feats, dict) and "mel" in feats:
            mel_data = feats["mel"]
            if isinstance(mel_data, dict) and "spec" in mel_data:
                mel_arr = np.asarray(mel_data["spec"])
                if mel_arr.ndim == 2:
                    song_mel_specs[name] = mel_arr
                    continue

        # Fallback: recompute Mel Spectrogram from audio
        try:
            y, sr = loader.load_audio(fp, target_sr=target_sr)
            _, _, mel_spec = features.compute_mel_spectrogram(y, sr, n_mels=n_mels)
            song_mel_specs[name] = mel_spec
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Cannot load Mel for {name}: {exc}")

    if len(song_mel_specs) < 1:
        return {"error": "No Mel Spectrograms available"}

    return train_global_models(
        song_mel_specs,
        lookback=lookback,
        epochs=epochs,
        n_hmm_states=n_hmm_states,
        verbose=verbose,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation (enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(per_file: Dict[str, dict]) -> dict:
    """Compute basic aggregate statistics across all analysed files."""
    song_names = sorted(per_file.keys())
    agg: dict = {
        "n_files": len(song_names),
        "aggregated_at": datetime.now().isoformat(),
    }

    if not song_names:
        return agg

    # ── Audio info ────────────────────────────────────────────────────────
    durations = []
    for r in per_file.values():
        ai = r.get("audio_info", {})
        if isinstance(ai, dict) and ai.get("duration"):
            durations.append(ai["duration"])
    agg["audio"] = {
        "total_duration_s": sum(durations),
        "mean_duration_s": float(np.mean(durations)) if durations else np.nan,
        "n_files_with_audio_info": len(durations),
    }

    # ── Dynamics averages ─────────────────────────────────────────────────
    dyn_keys = ["energy", "brightness", "complexity", "rhythm"]
    for key in dyn_keys:
        means, stds = [], []
        for r in per_file.values():
            dyn_entry = r.get("dynamics", {})
            if isinstance(dyn_entry, dict):
                ds = dyn_entry.get("summary", dyn_entry)
            else:
                ds = {}
            if isinstance(ds, dict):
                td = ds.get(key, {})
                if isinstance(td, dict):
                    m = td.get("mean")
                    s = td.get("std")
                    if m is not None:
                        means.append(_safe_float(m))
                    if s is not None:
                        stds.append(_safe_float(s))
        agg[f"dynamics_{key}_mean"] = float(np.mean(means)) if means else np.nan
        agg[f"dynamics_{key}_std"] = float(np.std(stds)) if stds else np.nan

    # ── Complexity ────────────────────────────────────────────────────────
    zcrs, sampens, flatnesses = [], [], []
    for r in per_file.values():
        ts = r.get("timeseries", {})
        if isinstance(ts, dict):
            cpx = ts.get("complexity", {})
            if isinstance(cpx, dict):
                zcrs.append(_safe_float(cpx.get("zero_crossing_rate")))
                sampens.append(_safe_float(cpx.get("sample_entropy")))
            flatnesses.append(_safe_float(ts.get("spectral_flatness")))
    agg["complexity"] = {
        "mean_zcr": float(np.mean(zcrs)) if zcrs else np.nan,
        "mean_sample_entropy": float(np.mean(sampens)) if sampens else np.nan,
        "mean_spectral_flatness": float(np.mean(flatnesses)) if flatnesses else np.nan,
    }

    # ── Model performance (per-song prediction) ───────────────────────────
    for mn in ["ARIMA", "HMM", "LSTM", "Transformer"]:
        rmses, maes = [], []
        for r in per_file.values():
            preds = r.get("prediction", {})
            if isinstance(preds, dict):
                arr = preds.get(mn, [])
                if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                    rmses.append(_safe_float(arr[1].get("RMSE")))
                    maes.append(_safe_float(arr[1].get("MAE")))
        agg[f"pred_{mn}_mean_rmse"] = float(np.mean(rmses)) if rmses else np.nan
        agg[f"pred_{mn}_std_rmse"] = float(np.std(rmses)) if rmses else np.nan

    # ── HMM ──────────────────────────────────────────────────────────────
    hmm_states = []
    for r in per_file.values():
        ma = r.get("model_analysis", {})
        if isinstance(ma, dict):
            ns = ma.get("hmm_n_states", 0)
            if ns:
                hmm_states.append(ns)
    agg["hmm_mean_n_states"] = float(np.mean(hmm_states)) if hmm_states else np.nan

    # ── Best band ────────────────────────────────────────────────────────
    band_best = {}
    for r in per_file.values():
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            b = rank[0].get("band", "unknown")
            band_best[b] = band_best.get(b, 0) + 1
    agg["best_band_distribution"] = band_best

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════════════════════

def _write_csv(per_file: Dict[str, dict], commonality: dict,
               global_ml: dict, csv_path: str):
    """Write a flat CSV combining per-file + commonality + global ML metrics."""
    import csv

    song_names = sorted(per_file.keys())
    if not song_names:
        return

    rows = []
    for name in song_names:
        r = per_file[name]
        dyn_entry = r.get("dynamics", {})
        dyn_summary = dyn_entry.get("summary", dyn_entry) if isinstance(dyn_entry, dict) else {}
        ts = r.get("timeseries", {})
        cpx = ts.get("complexity", {}) if isinstance(ts, dict) else {}
        ma = r.get("model_analysis", {}) if isinstance(r.get("model_analysis"), dict) else {}
        ai = r.get("audio_info", {}) if isinstance(r.get("audio_info"), dict) else {}

        row = {
            "file": name,
            "duration_s": ai.get("duration", ""),
            "energy_mean": dyn_summary.get("energy", {}).get("mean", "") if isinstance(dyn_summary, dict) else "",
            "brightness_mean": dyn_summary.get("brightness", {}).get("mean", "") if isinstance(dyn_summary, dict) else "",
            "complexity_mean": dyn_summary.get("complexity", {}).get("mean", "") if isinstance(dyn_summary, dict) else "",
            "rhythm_mean": dyn_summary.get("rhythm", {}).get("mean", "") if isinstance(dyn_summary, dict) else "",
            "zcr": cpx.get("zero_crossing_rate", ""),
            "sample_entropy": cpx.get("sample_entropy", ""),
            "spectral_flatness": ts.get("spectral_flatness", ""),
            "hmm_n_states": ma.get("hmm_n_states", ""),
            "lstm_most_learnable": ma.get("lstm_most_learnable", ""),
            "lstm_optimal_lookback_s": ma.get("lstm_optimal_lookback_s", ""),
        }

        # Prediction metrics per model
        preds = r.get("prediction", {}) if isinstance(r.get("prediction"), dict) else {}
        for mn in ["ARIMA", "HMM", "LSTM", "Transformer"]:
            arr = preds.get(mn, [])
            if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[1], dict):
                row[f"{mn}_rmse"] = arr[1].get("RMSE", "")
                row[f"{mn}_mae"] = arr[1].get("MAE", "")
            else:
                row[f"{mn}_rmse"] = ""
                row[f"{mn}_mae"] = ""

        # Best band
        rank = r.get("predictability_rank", [])
        if isinstance(rank, list) and len(rank) > 0:
            row["best_band"] = rank[0].get("band", "")
            row["best_band_rmse"] = rank[0].get("avg_rmse", "")
            row["best_band_model"] = rank[0].get("best_model", "")

        # Global ML per-song metrics
        lstm_per = global_ml.get("lstm", {}).get("per_song", {})
        tf_per = global_ml.get("transformer", {}).get("per_song", {})
        arima_per = global_ml.get("arima", {})
        if name in lstm_per:
            row["global_lstm_rmse"] = lstm_per[name].get("rmse", "")
        if name in tf_per:
            row["global_transformer_rmse"] = tf_per[name].get("rmse", "")
        if arima_per and name in arima_per:
            row["local_arima_rmse"] = arima_per[name].get("rmse", "")
            row["local_arima_order"] = arima_per[name].get("order", "")

        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# File scanning
# ═══════════════════════════════════════════════════════════════════════════════

def scan_audio_files(folder: str) -> List[str]:
    """Recursively find all WAV/MP3/FLAC files in folder."""
    exts = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in sorted(fnames):
            if os.path.splitext(fn)[1].lower() in exts:
                files.append(os.path.join(root, fn))
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch(
    input_folder: str,
    output_folder: str,
    max_files: int = DEFAULT_MAX_FILES,
    fast: bool = False,
    resume: bool = True,
    do_global_ml: bool = True,
    lookback: int = DEFAULT_LOOKBACK,
    ml_epochs: int = DEFAULT_ML_EPOCHS,
    n_hmm_states: int = DEFAULT_HMM_STATES,
    forecast_horizon: int = DEFAULT_FORECAST_HORIZON,
    n_mels: int = DEFAULT_N_MELS,
) -> dict:
    """
    Run batch analysis on all audio files in *input_folder*.

    Returns the aggregate summary dict.
    """
    os.makedirs(output_folder, exist_ok=True)
    per_file_dir = os.path.join(output_folder, "per_file")
    figures_dir = os.path.join(output_folder, "figures")
    os.makedirs(per_file_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ── Discover files ────────────────────────────────────────────────────
    audio_files = scan_audio_files(input_folder)
    if max_files > 0:
        audio_files = audio_files[:max_files]

    if not audio_files:
        print(f"[ERROR] No audio files found in: {input_folder}")
        return {}

    print(f"\n{'='*70}")
    print(f"  Audio Lab Batch Analysis")
    print(f"{'='*70}")
    print(f"  Input:       {input_folder}")
    print(f"  Output:      {output_folder}")
    print(f"  Files:       {len(audio_files)}")
    print(f"  Mode:        {'fast' if fast else 'full'}")
    print(f"  Global ML:   {do_global_ml}")
    print(f"  Resume:      {resume}")
    print(f"{'='*70}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase A — Per-file analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"{'─'*70}")
    print(f"  PHASE A — Per-file analysis (shared pipeline)")
    print(f"{'─'*70}")

    per_file_results: Dict[str, dict] = {}
    skipped = 0

    for i, fp in enumerate(audio_files):
        fname = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(per_file_dir, f"{fname}_results.json")

        # Resume check
        if resume and os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    per_file_results[fname] = json.load(fh)
                skipped += 1
                print(f"  [{i+1:4d}/{len(audio_files)}] SKIP (cached)  {os.path.basename(fp)}")
                continue
            except Exception:
                pass

        # Analyse via shared pipeline
        t_start = time.time()
        try:
            raw = run_per_file_analysis(fp, forecast_horizon=forecast_horizon,
                                         n_mels=n_mels, fast=fast)
            # Serialize with pipeline helper
            serialized = _safe_serialize_results(raw, task_id=fname, task_info={
                "experiment_name": fname,
                "forecast_horizon": forecast_horizon,
                "n_mels": n_mels,
                "audio1_name": os.path.basename(fp),
                "analysis_options": raw.get("analysis_options", []),
            })

            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(serialized, fh, default=str, ensure_ascii=False)

            per_file_results[fname] = serialized
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{len(audio_files)}] OK  {elapsed:5.1f}s  {os.path.basename(fp)}")

        except Exception as exc:
            elapsed = time.time() - t_start
            print(f"  [{i+1:4d}/{len(audio_files)}] FAIL {elapsed:5.1f}s  {os.path.basename(fp)}  — {exc}")
            traceback.print_exc()

    n_analysed = len(audio_files) - skipped
    print(f"\n  Per-file analysis done: {n_analysed} analysed, {skipped} cached, "
          f"{len(audio_files) - n_analysed - skipped} failed\n")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase B — Cross-song commonality analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"{'─'*70}")
    print(f"  PHASE B — Cross-song commonality analysis")
    print(f"{'─'*70}")

    commonality_report: dict = {}
    song_names = sorted(per_file_results.keys())

    if len(song_names) >= 2:
        feature_matrix, feature_names, meta = build_feature_matrix(per_file_results)
        commonality_report = compute_commonality_report(
            feature_matrix, feature_names, song_names,
            per_file_results, meta,
        )

        cv_scores = commonality_report.get("cv_scores", {})
        valid_cv = {k: v for k, v in cv_scores.items() if not np.isnan(v) and v < 10}
        sorted_cv = sorted(valid_cv.items(), key=lambda x: x[1])

        print(f"\n  Feature matrix: {commonality_report['n_songs']} songs × "
              f"{commonality_report['n_features']} features")
        print(f"\n  Most common features (lowest CV = genre signature):")
        for fn, cv in sorted_cv[:5]:
            print(f"    - {fn:<40s} CV={cv:.4f}")
        print(f"\n  Most diverse features (highest CV = intra-genre variation):")
        for fn, cv in sorted_cv[-5:]:
            print(f"    - {fn:<40s} CV={cv:.4f}")

        outliers = commonality_report.get("outliers", {})
        if outliers:
            print(f"\n  Outliers detected (|z| > 2.0):")
            for fn, songs in sorted(outliers.items()):
                print(f"    -{fn}: {', '.join(s[:25] for s in songs)}")

        print(f"\n  Best band distribution: {commonality_report.get('best_band_distribution', {})}")
        print(f"  HMM state count distribution: {commonality_report.get('hmm_state_count_distribution', {})}")

        # Save commonality report
        cp = os.path.join(output_folder, "commonality_report.json")
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(commonality_report, fh, default=str, ensure_ascii=False, indent=2)
        print(f"\n  Commonality report saved → {cp}")
    else:
        print(f"\n  (Skipped — need at least 2 songs for commonality analysis)")
        commonality_report["overview"] = "Insufficient songs for cross-song commonality analysis."

    # ═══════════════════════════════════════════════════════════════════════
    # Phase C — Global ML training
    # ═══════════════════════════════════════════════════════════════════════
    global_ml_report: dict = {}

    if do_global_ml and len(audio_files) >= 2:
        print(f"\n{'─'*70}")
        print(f"  PHASE C — Global Model ML Training")
        print(f"  (Mel Spectrogram sliding windows, all songs pooled)")
        print(f"{'─'*70}")

        global_ml_report = run_global_ml(
            audio_files=audio_files,
            per_file=per_file_results,
            lookback=lookback,
            epochs=ml_epochs,
            n_hmm_states=n_hmm_states,
            n_mels=n_mels,
            verbose=True,
        )

        # Save global ML report
        gmp = os.path.join(output_folder, "global_ml_report.json")
        # Strip non-serializable parts
        ml_serializable = {}
        for key in ["lstm", "transformer", "arima", "hmm", "summary"]:
            if key in global_ml_report:
                ml_serializable[key] = global_ml_report[key]

        # Remove per_song_state_sequences (too large for JSON)
        if "hmm" in ml_serializable:
            ml_serializable["hmm"] = {
                k: v for k, v in ml_serializable["hmm"].items()
                if k != "per_song_state_sequences"
                if k != "transition_matrix" or not isinstance(v, list)
            }
            # Keep transition_matrix but limit
            if "transition_matrix" in global_ml_report.get("hmm", {}):
                ml_serializable["hmm"]["transition_matrix"] = \
                    global_ml_report["hmm"]["transition_matrix"]

        with open(gmp, "w", encoding="utf-8") as fh:
            json.dump(ml_serializable, fh, default=str, ensure_ascii=False, indent=2)
        print(f"\n  Global ML report saved → {gmp}")
    elif do_global_ml:
        print(f"\n{'─'*70}")
        print(f"  PHASE C — Skipped (need at least 2 songs)")
        print(f"{'─'*70}")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase D — Visualizations
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE D — Batch visualizations")
    print(f"{'─'*70}")

    batch_results_for_viz = {
        "per_file_results": per_file_results,
        "commonality_report": commonality_report,
        "global_ml_report": global_ml_report,
    }

    plot_files = generate_all_batch_plots(batch_results_for_viz, figures_dir)
    if plot_files:
        print(f"\n  Generated {len(plot_files)} batch plots:")
        for pf in sorted(plot_files):
            print(f"    -figures/{pf}")
    else:
        print(f"\n  (No batch plots generated)")

    # ═══════════════════════════════════════════════════════════════════════
    # Phase E — Aggregation & Final output
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE E — Aggregation & output")
    print(f"{'─'*70}")

    agg = aggregate_results(per_file_results)

    # Attach meta
    agg["batch_meta"] = {
        "input_folder": input_folder,
        "output_folder": output_folder,
        "n_files_total": len(audio_files),
        "n_files_analysed": n_analysed,
        "fast_mode": fast,
        "global_ml_enabled": do_global_ml,
        "lookback": lookback if do_global_ml else None,
        "ml_epochs": ml_epochs if do_global_ml else None,
        "forecast_horizon": forecast_horizon,
        "n_mels": n_mels,
        "run_at": datetime.now().isoformat(),
    }

    # Add global ML summary to aggregate
    if global_ml_report.get("summary"):
        agg["global_ml_summary"] = global_ml_report["summary"]

    # Add commonality overview
    if commonality_report.get("overview"):
        agg["commonality_overview"] = commonality_report["overview"]

    # Save aggregate JSON
    agg_path = os.path.join(output_folder, "aggregate_summary.json")
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, default=str, ensure_ascii=False, indent=2)

    # Save CSV
    csv_path = os.path.join(output_folder, "batch_report.csv")
    _write_csv(per_file_results, commonality_report, global_ml_report, csv_path)

    print(f"\n{'='*70}")
    print(f"  Batch complete!")
    print(f"  Aggregate JSON     : {agg_path}")
    print(f"  Commonality report : {os.path.join(output_folder, 'commonality_report.json')}")
    if do_global_ml:
        print(f"  Global ML report   : {os.path.join(output_folder, 'global_ml_report.json')}")
    print(f"  Batch CSV          : {csv_path}")
    print(f"  Figures            : {figures_dir}/ ({len(plot_files)} plots)")
    print(f"{'='*70}\n")

    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch Audio Time-Series Analysis — CLI mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_analyze.py data/rock/ -o results/rock/
  python batch_analyze.py data/rock/ -o results/rock/ --fast
  python batch_analyze.py data/rock/ -o results/rock/ --no-global-ml
  python batch_analyze.py data/ --output results/ --max-files 30
        """,
    )
    parser.add_argument("input", help="Input folder containing audio files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output folder (default: input_folder + '_results')")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                        help=f"Max files to process (0=unlimited, default: {DEFAULT_MAX_FILES})")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce deep-learning epochs for speed")
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip already-analysed files")
    parser.add_argument("--no-global-ml", action="store_true",
                        help="Skip Global ML training (Phases C+D)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help=f"Sliding window size for Global ML (default: {DEFAULT_LOOKBACK})")
    parser.add_argument("--ml-epochs", type=int, default=DEFAULT_ML_EPOCHS,
                        help=f"Training epochs for Global ML (default: {DEFAULT_ML_EPOCHS})")
    parser.add_argument("--hmm-states", type=int, default=DEFAULT_HMM_STATES,
                        help=f"Joint HMM hidden states (default: {DEFAULT_HMM_STATES})")
    parser.add_argument("--forecast-horizon", type=int, default=DEFAULT_FORECAST_HORIZON,
                        help=f"Forecast horizon (default: {DEFAULT_FORECAST_HORIZON})")
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS,
                        help=f"Number of Mel bands (default: {DEFAULT_N_MELS})")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Target sample rate (default: {DEFAULT_SAMPLE_RATE})")

    args = parser.parse_args()

    output = args.output or (args.input.rstrip("/\\") + "_results")

    run_batch(
        input_folder=args.input,
        output_folder=output,
        max_files=args.max_files,
        fast=args.fast,
        resume=not args.no_resume,
        do_global_ml=not args.no_global_ml,
        lookback=args.lookback,
        ml_epochs=args.ml_epochs,
        n_hmm_states=args.hmm_states,
        forecast_horizon=args.forecast_horizon,
        n_mels=args.n_mels,
    )


if __name__ == "__main__":
    main()
