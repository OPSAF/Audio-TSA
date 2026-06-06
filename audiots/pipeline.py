"""
Shared Analysis Pipeline
========================

Single unified analysis core used by ALL entry points (web / CLI / batch).
Every caller gets the **same** result dict structure and the **same** plot
generation logic — no more divergence between app.py, main.py, and
batch_analyze.py.

Usage
-----
    from audiots.pipeline import run_full_analysis, serialize_results

    results = run_full_analysis(y, sr, y2=y2, sr2=sr,
                                forecast_horizon=20, n_mels=128,
                                analysis_options=['features', 'prediction', ...],
                                progress_callback=my_logger)

    serialized = serialize_results(results, task_id='xxx', task_info={...})
    # serialized is JSON-safe and ready to write to disk
"""

from __future__ import annotations

import os
import time
import json
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

from . import (
    loader, features, dynamics, volatility, model_analysis,
    discovery, unsupervised, analysis, prediction, band_analysis,
    visualization,
)
from . import discovery_viz

# ── Available options ────────────────────────────────────────────────────────

ALL_OPTIONS = [
    'features', 'dynamics', 'dynamics_analysis', 'model_analysis',
    'timeseries', 'unsupervised', 'prediction', 'band',
    'comparison', 'visualization',
]

DEFAULT_OPTIONS = [
    'features', 'dynamics', 'dynamics_analysis', 'model_analysis',
    'timeseries', 'unsupervised', 'prediction', 'band', 'visualization',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Features
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_features(results: dict, y: np.ndarray, sr: int,
                    n_mels: int, log: Callable) -> None:
    log("[1.1] 波形", 'info')
    t, wav = features.compute_waveform(y, sr)
    results['features'] = {}
    results['features']['waveform'] = {'t': t, 'y': wav}

    log("[1.2] FFT", 'info')
    freqs, mag = features.compute_fft(y, sr)
    results['features']['fft'] = {'freqs': freqs, 'mag': mag}

    log("[1.3] STFT", 'info')
    f_stft, t_stft, spec_stft = features.compute_stft(y, sr)
    results['features']['stft'] = {'freqs': f_stft, 'times': t_stft, 'spec': spec_stft}

    log("[1.4] Mel 频谱图", 'info')
    mel_f, mel_t, mel_s = features.compute_mel_spectrogram(y, sr, n_mels=n_mels)
    results['features']['mel'] = {'freqs': mel_f, 'times': mel_t, 'spec': mel_s}

    log("[1.5] MFCC", 'info')
    mfcc, mfcc_t = features.compute_mfcc(y, sr, n_mfcc=20)
    results['features']['mfcc'] = {'mfcc': mfcc, 'times': mfcc_t}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1.5-1.6: Dynamics + Volatility
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_dynamics(results: dict, y: np.ndarray, sr: int,
                    log: Callable) -> dict:
    log("[Dyn] 提取动态特征 (能量/亮度/复杂度/节奏)", 'info')
    dyn = dynamics.extract_dynamics(y, sr, window_size=0.5, hop_size=0.25)
    dyn_seg = dynamics.detect_structural_segments(dyn)
    results['dynamics'] = {'data': dyn, 'segments': dyn_seg}
    return dyn


def _phase_dynamics_analysis(results: dict, dyn: dict,
                              forecast_horizon: int, log: Callable) -> dict:
    log("[Vol] 滚动波动率 + GARCH(1,1)", 'info')
    vol = volatility.compute_volatility_layer(dyn, rolling_window=10, fit_garch=True)
    results['volatility'] = vol
    results['volatility_summary'] = volatility.summarize_volatility(vol)

    log("[TrendPred] 趋势预测 (ARIMA+HMM)", 'info')
    results['trend_predictions'] = prediction.predict_all_trends(
        dyn, forecast_horizon=forecast_horizon, models="ARIMA,HMM", verbose=False)

    log("[VolPred] 波动率预测 (HMM)", 'info')
    results['volatility_predictions'] = prediction.predict_all_volatilities(
        vol, forecast_horizon=min(10, forecast_horizon),
        models="ARIMA,HMM", verbose=False)

    return vol


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Timeseries + Model Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_model_analysis(results: dict, dyn: dict, log: Callable,
                          fast: bool = False) -> None:
    epochs = 10 if fast else 20
    log("[Model] ARIMA/HMM/LSTM/Transformer 结构侦探", 'info')
    mr = model_analysis.analyze_model_ensemble(
        dyn, n_hmm_states=3, lstm_epochs=epochs,
        transformer_epochs=epochs, verbose=False)
    results['model_analysis'] = mr


def _phase_timeseries(results: dict, y: np.ndarray, sr: int,
                      log: Callable) -> None:
    results['timeseries'] = {}

    wav = results.get('features', {}).get('waveform', {}).get('y', y)
    acf_in = wav[:min(len(wav), sr * 1)]

    log("[2.1] ACF & PACF", 'info')
    lags, acf_vals, ci = analysis.compute_acf(acf_in, nlags=40)
    _, pacf_vals, _ = analysis.compute_pacf(acf_in, nlags=40)
    results['timeseries']['acf_pacf'] = {
        'lags': lags, 'acf': acf_vals, 'pacf': pacf_vals, 'ci': ci}

    log("[2.2] 周期性", 'info')
    results['timeseries']['periodicity'] = analysis.analyze_periodicity(y, sr)

    log("[2.3] 复杂度", 'info')
    results['timeseries']['complexity'] = analysis.analyze_complexity(y)

    log("[2.4] 频谱平坦度", 'info')
    fft_mag = results.get('features', {}).get('fft', {}).get('mag',
                features.compute_fft(y, sr)[1])
    results['timeseries']['spectral_flatness'] = analysis.compute_spectral_flatness(fft_mag)

    log("[2.5] 白噪声检验", 'info')
    results['timeseries']['white_noise_test'] = analysis.test_white_noise(
        wav[:min(len(wav), sr * 2)])


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3-4: Unsupervised + Prediction + Band
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_unsupervised(results: dict, y: np.ndarray, sr: int,
                        log: Callable) -> None:
    log("[Unsup] 聚类和motif检测", 'info')
    results['unsupervised'] = unsupervised.explore_unsupervised(
        y, sr, n_components=4, n_clusters=4, verbose=False)


def _phase_prediction(results: dict, y: np.ndarray, sr: int,
                       forecast_horizon: int, n_mels: int, log: Callable,
                       fast: bool = False) -> None:
    if 'features' in results and 'mel' in results['features']:
        mel_spec = results['features']['mel']['spec']
    else:
        _, _, mel_spec = features.compute_mel_spectrogram(y, sr, n_mels=n_mels)

    epochs = 10 if fast else 30
    log("[Pred] ARIMA/HMM/LSTM/Transformer 预测", 'info')
    results['prediction'] = prediction.run_all_predictions(
        mel_spec, forecast_horizon=forecast_horizon, epochs=epochs,
        verbose=False)


def _phase_band(results: dict, y: np.ndarray, sr: int,
                 forecast_horizon: int, n_mels: int, log: Callable,
                 fast: bool = False) -> None:
    if 'features' in results and 'mel' in results['features']:
        mel_spec = results['features']['mel']['spec']
    else:
        _, _, mel_spec = features.compute_mel_spectrogram(y, sr, n_mels=n_mels)

    epochs = 10 if fast else 30
    log("[Band] 频带可预测性", 'info')
    band_results = band_analysis.analyze_band_predictability(
        mel_spec, forecast_horizon=forecast_horizon, epochs=epochs)
    results['band'] = {
        'results': band_results,
        'summary': band_analysis.compute_band_error_summary(band_results),
        'rank': band_analysis.get_predictability_rank(
            band_analysis.compute_band_error_summary(band_results)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Dual Audio Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_comparison(results: dict, y1: np.ndarray, sr1: int,
                       y2: np.ndarray, sr2: int,
                       vol1: Optional[dict], dyn1: Optional[dict],
                       log: Callable) -> None:
    results['comparison'] = {}

    log("[Comp] 多维探索分析", 'info')
    disc = discovery.explore(y1, sr1, y2, sr2,
                             window_size=0.5, hop_size=0.25, verbose=False)
    results['comparison']['discovery'] = disc
    results['discovery'] = disc  # top-level alias for web compat

    if vol1 is not None and dyn1 is not None:
        # Extract dyn2 + vol2
        dyn2 = dynamics.extract_dynamics(y2, sr2, window_size=0.5, hop_size=0.25)
        vol2 = volatility.compute_volatility_layer(dyn2, rolling_window=10, fit_garch=True)
        results['_dyn2'] = dyn2
        results['_vol2'] = vol2

        log("[Comp] 波动率相似度", 'info')
        results['volatility_similarity'] = volatility.compute_volatility_similarity(
            results.get('volatility') or vol1, vol2)

        log("[Comp] 动态趋势相似度", 'info')
        results['dynamics_similarity'] = dynamics.compute_dynamics_similarity(dyn1, dyn2)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: Visualization (shared!)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(results: dict, output_dir: str,
                       y1: np.ndarray, sr: int,
                       y2: Optional[np.ndarray] = None,
                       log: Optional[Callable] = None) -> List[str]:
    """
    Generate ALL plots from analysis results and save to *output_dir*.

    This is the SINGLE function used by every entry point.  It produces
    the identical set of images regardless of web / CLI / batch.
    """
    if log is None:
        log = lambda msg, level='info': None

    os.makedirs(output_dir, exist_ok=True)
    plot_files: List[str] = []

    def _save(fig, name: str, subdir: str = '') -> str:
        d = os.path.join(output_dir, subdir) if subdir else output_dir
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        import matplotlib.pyplot as plt
        plt.close(fig)
        rel = f"{subdir}/{name}" if subdir else name
        return rel

    log("[Viz] 生成所有图表...", 'info')

    # ── Feature plots ─────────────────────────────────────────────────────
    feats = results.get('features', {})
    if feats:
        if feats.get('waveform'):
            fig = visualization.plot_waveform(feats['waveform']['t'],
                                              feats['waveform']['y'],
                                              title="Audio Waveform")
            plot_files.append(_save(fig, 'waveform.png'))
        if feats.get('fft'):
            fig = visualization.plot_fft(feats['fft']['freqs'],
                                         feats['fft']['mag'],
                                         title="FFT Magnitude Spectrum")
            plot_files.append(_save(fig, 'fft_spectrum.png'))
        if feats.get('stft'):
            fig = visualization.plot_stft(feats['stft']['freqs'],
                                          feats['stft']['times'],
                                          feats['stft']['spec'],
                                          title="STFT Spectrogram")
            plot_files.append(_save(fig, 'stft_spectrogram.png'))
        if feats.get('mel'):
            fig = visualization.plot_mel_spectrogram(
                feats['mel']['freqs'], feats['mel']['times'],
                feats['mel']['spec'], title="Mel Spectrogram")
            plot_files.append(_save(fig, 'mel_spectrogram.png'))
        if feats.get('mfcc'):
            fig = visualization.plot_mfcc(feats['mfcc']['mfcc'],
                                          feats['mfcc']['times'],
                                          title="MFCC Coefficients")
            plot_files.append(_save(fig, 'mfcc_heatmap.png'))

    # ── Timeseries plots ──────────────────────────────────────────────────
    ts = results.get('timeseries', {})
    if ts:
        if ts.get('acf_pacf'):
            a = ts['acf_pacf']
            fig = visualization.plot_acf_pacf(
                a['lags'], a['acf'], a['pacf'], a['ci'],
                title="ACF & PACF Analysis")
            plot_files.append(_save(fig, 'acf_pacf.png'))
        if ts.get('periodicity'):
            fig = visualization.plot_periodicity(
                y1, sr, ts['periodicity'], title="Periodicity Analysis")
            plot_files.append(_save(fig, 'periodicity.png'))

    # ── Prediction plots ──────────────────────────────────────────────────
    pred = results.get('prediction', {})
    if pred:
        try:
            fig = visualization.plot_prediction_comparison(
                pred, forecast_horizon=100, title="Prediction Comparison")
            plot_files.append(_save(fig, 'prediction_comparison.png'))
        except Exception:
            pass
        try:
            fig = visualization.plot_model_error_comparison(
                pred, title="Model Error Comparison")
            plot_files.append(_save(fig, 'error_comparison.png'))
        except Exception:
            pass

    # ── Band plots ────────────────────────────────────────────────────────
    band = results.get('band', {})
    if band and band.get('results'):
        try:
            fig = visualization.plot_band_error_heatmap(
                band['results'], title="Band Error Heatmap")
            plot_files.append(_save(fig, 'band_errors.png'))
        except Exception:
            pass
        try:
            fig = visualization.plot_band_error_bars(
                band['results'], title="Band Error Comparison")
            plot_files.append(_save(fig, 'band_error_bars.png'))
        except Exception:
            pass

    # ── Dynamics plots ────────────────────────────────────────────────────
    dyn_entry = results.get('dynamics', {})
    dyn_data = dyn_entry.get('data') if isinstance(dyn_entry, dict) else None
    dyn_seg = dyn_entry.get('segments') if isinstance(dyn_entry, dict) else None

    if dyn_data is not None:
        try:
            fig = visualization.plot_dynamics_trends(
                dyn_data, segments=dyn_seg, title="Dynamics Trend Analysis")
            plot_files.append(_save(fig, 'trends.png', 'dynamics'))
            fig = visualization.plot_dynamics_summary(
                dyn_data, segments=dyn_seg, title="Dynamics Summary")
            plot_files.append(_save(fig, 'summary.png', 'dynamics'))
        except Exception:
            pass
        # Dual audio dynamics comparison
        dyn2 = results.get('_dyn2')
        if dyn2 is not None:
            try:
                dyn_sim = results.get('dynamics_similarity')
                if dyn_sim is None and dyn2 is not None:
                    dyn_sim = dynamics.compute_dynamics_similarity(dyn_data, dyn2)
                fig = visualization.plot_dynamics_dual(
                    dyn_data, dyn2, sim_result=dyn_sim,
                    title="Dual Audio — Dynamics Comparison")
                plot_files.append(_save(fig, 'dynamics_dual.png', 'dynamics'))
            except Exception:
                pass

    # ── Volatility plots ──────────────────────────────────────────────────
    vol_data = results.get('volatility')
    if vol_data is not None and dyn_data is not None:
        try:
            fig = visualization.plot_volatility_layer(
                dyn_data, vol_data, title="Volatility Layer")
            plot_files.append(_save(fig, 'volatility_layer.png', 'volatility'))
            fig = visualization.plot_garch_diagnostics(
                vol_data, trend_key='energy',
                title="GARCH(1,1) — Energy Trend Volatility")
            plot_files.append(_save(fig, 'garch_energy.png', 'volatility'))

            da = {'trend_summary': dynamics.summarize_dynamics(dyn_data),
                  'volatility_summary': volatility.summarize_volatility(vol_data)}
            fig = visualization.plot_dynamics_analysis_summary(
                da, title="Dynamics Analysis — Trend + Volatility Summary")
            plot_files.append(_save(fig, 'dynamics_analysis_summary.png', 'volatility'))
        except Exception:
            pass

    # ── Volatility comparison ─────────────────────────────────────────────
    vol2 = results.get('_vol2')
    if vol_data is not None and vol2 is not None:
        try:
            vol_sim = results.get('volatility_similarity')
            fig = visualization.plot_volatility_comparison(
                vol_data, vol2, sim_result=vol_sim,
                title="Dual Audio — Volatility Comparison")
            plot_files.append(_save(fig, 'volatility_comparison.png', 'volatility'))
        except Exception:
            pass

    # ── Discovery plots ───────────────────────────────────────────────────
    disc = results.get('discovery')
    if disc is not None:
        try:
            disc_dir = os.path.join(output_dir, 'discovery')
            dyn_b = results.get('_dyn2')
            disc_plots = discovery_viz.generate_discovery_report_plots(
                disc, output_dir=disc_dir, y1=y1, y2=y2, sr=sr,
                dyn1=dyn_data, dyn2=dyn_b)
            plot_files += [f"discovery/{p}" for p in disc_plots]

            fig = discovery_viz.plot_discovery_summary(
                disc, title="Audio Discovery Summary")
            plot_files.append(_save(fig, 'discovery_summary.png'))
        except Exception:
            pass

    log(f"[完成] 生成了 {len(plot_files)} 个图表", 'info')
    return plot_files


# ═══════════════════════════════════════════════════════════════════════════════
# Master orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    y1: np.ndarray,
    sr: int,
    y2: Optional[np.ndarray] = None,
    sr2: Optional[int] = None,
    forecast_horizon: int = 20,
    n_mels: int = 128,
    analysis_options: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    fast: bool = False,
) -> dict:
    """
    Run the complete analysis pipeline on one audio file.

    Parameters
    ----------
    y1, sr : primary audio
    y2, sr2 : optional second audio for comparison
    forecast_horizon, n_mels : analysis parameters
    analysis_options : list of phases to run (default: all except comparison)
    progress_callback : fn(message, level) — called for logging
    fast : reduce deep-learning epochs for speed

    Returns
    -------
    results : dict with keys:
        audio_info, features, dynamics, volatility, volatility_summary,
        trend_predictions, volatility_predictions, model_analysis,
        timeseries, unsupervised, prediction, band,
        comparison (if dual audio), discovery (if dual audio),
        volatility_similarity (if dual audio), dynamics_similarity (if dual audio),
        _y1, _sr, _y2, _vol2, _dyn2  (internal)
    """
    if analysis_options is None:
        analysis_options = list(DEFAULT_OPTIONS)

    def log(msg: str, level: str = 'info'):
        if progress_callback:
            progress_callback(msg, level)

    dual_audio = y2 is not None
    results: dict = {
        'audio_info': {
            'duration': len(y1) / sr,
            'sample_rate': sr,
            'samples': len(y1),
        },
        '_y1': y1,
        '_sr': sr,
    }
    if y2 is not None:
        results['_y2'] = y2

    dyn1: Optional[dict] = None
    vol1: Optional[dict] = None

    # ── PHASE 1 : Features ─────────────────────────────────────────────
    if 'features' in analysis_options:
        results['_phase'] = 'features'
        _phase_features(results, y1, sr, n_mels, log)

    # ── PHASE 1.5 : Dynamics ───────────────────────────────────────────
    if 'dynamics' in analysis_options:
        results['_phase'] = 'dynamics'
        dyn1 = _phase_dynamics(results, y1, sr, log)

    # ── PHASE 1.6 : Dynamics Analysis (Volatility) ─────────────────────
    if 'dynamics_analysis' in analysis_options:
        results['_phase'] = 'dynamics_analysis'
        if dyn1 is None:
            dyn1 = _phase_dynamics(results, y1, sr, log)
        vol1 = _phase_dynamics_analysis(results, dyn1, forecast_horizon, log)

    # ── PHASE 2a : Model Analysis ──────────────────────────────────────
    if 'model_analysis' in analysis_options:
        results['_phase'] = 'model_analysis'
        if dyn1 is None:
            dyn1 = _phase_dynamics(results, y1, sr, log)
        _phase_model_analysis(results, dyn1, log, fast=fast)

    # ── PHASE 2 : Timeseries ───────────────────────────────────────────
    if 'timeseries' in analysis_options:
        results['_phase'] = 'timeseries'
        _phase_timeseries(results, y1, sr, log)

    # ── PHASE 3 : Unsupervised ─────────────────────────────────────────
    if 'unsupervised' in analysis_options:
        results['_phase'] = 'unsupervised'
        _phase_unsupervised(results, y1, sr, log)

    # ── PHASE 4 : Prediction ───────────────────────────────────────────
    if 'prediction' in analysis_options:
        results['_phase'] = 'prediction'
        _phase_prediction(results, y1, sr, forecast_horizon, n_mels, log,
                          fast=fast)

    # ── PHASE 4.5 : Band ───────────────────────────────────────────────
    if 'band' in analysis_options:
        results['_phase'] = 'band'
        _phase_band(results, y1, sr, forecast_horizon, n_mels, log,
                    fast=fast)

    # ── PHASE 5 : Comparison ───────────────────────────────────────────
    if 'comparison' in analysis_options and dual_audio and y2 is not None:
        results['_phase'] = 'comparison'
        _phase_comparison(results, y1, sr, y2, sr2 or sr, vol1, dyn1, log)

    results['_phase'] = 'done'
    results['analysis_options'] = analysis_options
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization helper — call AFTER generating plots
# ═══════════════════════════════════════════════════════════════════════════════

def serialize_results(results: dict, task_id: str = '',
                      task_info: Optional[dict] = None,
                      plot_files: Optional[List[str]] = None) -> dict:
    """
    Convert the rich results dict (with ndarrays, model objects, etc.)
    into a JSON-safe dict ready for disk / API.

    Call this AFTER ``generate_all_plots()`` so that plot_files are known.
    """
    out: dict = {}

    # ── Simple top-level keys ──────────────────────────────────────────────
    for key in ['audio_info', 'analysis_options', 'experiment_name',
                'spectral_flatness']:
        if key in results:
            out[key] = results[key]

    out['task_id'] = task_id
    out['plot_files'] = plot_files or []

    if task_info:
        out['params'] = {
            'experiment_name': task_info.get('experiment_name'),
            'forecast_horizon': task_info.get('forecast_horizon'),
            'n_mels': task_info.get('n_mels'),
            'audio1_name': task_info.get('audio1_name', ''),
            'audio2_name': task_info.get('audio2_name'),
            'analysis_options': task_info.get('analysis_options', []),
        }
        out['experiment_name'] = task_info.get('experiment_name') or ''

    # ── features → strip ndarrays ─────────────────────────────────────────
    if 'features' in results:
        out['features'] = _serialize_dict(results['features'])

    # ── timeseries → keep as-is (already small dicts of floats) ───────────
    if 'timeseries' in results:
        out['timeseries'] = _serialize_dict(results['timeseries'])

    # ── dynamics → summary only (raw data too large for JSON) ─────────────
    dyn_entry = results.get('dynamics', {})
    dyn_data = dyn_entry.get('data') if isinstance(dyn_entry, dict) else None
    dyn_seg = dyn_entry.get('segments') if isinstance(dyn_entry, dict) else None
    if dyn_data is not None:
        from .dynamics import summarize_dynamics
        seg_dict = dyn_seg if isinstance(dyn_seg, dict) else {}
        out['dynamics'] = {
            'summary': summarize_dynamics(dyn_data),
            'n_climax': len(seg_dict.get('climax_indices', [])),
            'n_calm': len(seg_dict.get('calm_indices', [])),
        }

    # ── volatility → summary only ─────────────────────────────────────────
    if 'volatility' in results:
        from .volatility import summarize_volatility
        vs = results.get('volatility_summary') or summarize_volatility(results['volatility'])
        out['volatility'] = {
            'summary': {
                k: {sk: sv for sk, sv in v.items()
                    if not isinstance(sv, np.ndarray)}
                for k, v in vs.items()
            },
            'global_vol_similarity': results.get('volatility_similarity', {}).get(
                'global_volatility_similarity'),
        }

    # ── model_analysis ────────────────────────────────────────────────────
    mr = results.get('model_analysis')
    if mr is not None:
        out['model_analysis'] = {
            'arima_summary': mr.arima.summary if mr.arima else None,
            'hmm_summary': mr.hmm.summary if mr.hmm else None,
            'hmm_n_states': mr.hmm.n_states if mr.hmm else 0,
            'hmm_state_profiles': [
                {'id': p.state_id, 'label': p.label, 'fraction': p.fraction,
                 'description': p.description}
                for p in (mr.hmm.state_profiles if mr.hmm else [])
            ],
            'lstm_summary': mr.lstm.summary if mr.lstm else None,
            'lstm_optimal_lookback_s': mr.lstm.optimal_lookback_seconds if mr.lstm else None,
            'lstm_most_learnable': mr.lstm.most_learnable if mr.lstm else None,
            'transformer_summary': mr.transformer.summary if mr.transformer else None,
            'transformer_n_layers': mr.transformer.n_distinct_layers if mr.transformer else 0,
            'ensemble_summary': mr.ensemble_summary,
        }

    # ── trend predictions → simplified ────────────────────────────────────
    if 'trend_predictions' in results:
        tps = {}
        for tk, preds in results['trend_predictions'].items():
            tps[tk] = {}
            for mn, (forecast, metrics, true) in preds.items():
                tps[tk][mn] = {
                    'rmse': float(metrics.get('RMSE', np.nan)) if not np.isnan(metrics.get('RMSE', np.nan)) else None,
                    'mae': float(metrics.get('MAE', np.nan)) if not np.isnan(metrics.get('MAE', np.nan)) else None,
                }
        out['trend_predictions'] = tps

    # ── prediction → keep ─────────────────────────────────────────────────
    if 'prediction' in results:
        out['prediction'] = _serialize_dict(results['prediction'])

    # ── band → serialized ─────────────────────────────────────────────────
    band = results.get('band', {})
    if band and band.get('results'):
        br = {}
        for bk, bd in band['results'].items():
            br[bk] = {
                'info': bd['info'],
                'predictions': {
                    mn: {'metrics': md['metrics']}
                    for mn, md in bd['predictions'].items()
                },
            }
        out['band_results'] = br
        out['band_summary'] = _serialize_dict(band.get('summary', {}))
        out['predictability_rank'] = _serialize_dict(band.get('rank', []))

    # ── discovery → simplified ────────────────────────────────────────────
    disc = results.get('discovery')
    if disc is not None:
        def _p(p):
            if p is None: return None
            return {'rhythm_signature': p.rhythm_signature,
                    'energy_profile': p.energy_profile,
                    'timbre_quality': p.timbre_quality,
                    'standout_features': p.standout_features}
        out['discovery'] = {
            'audio_a_profile': _p(disc.audio_a_profile),
            'audio_b_profile': _p(disc.audio_b_profile),
            'n_discoveries': len(disc.discoveries),
            'n_contrasts': len(disc.contrasts),
            'overview': disc.overview,
            'params': disc.params,
        }

    # ── unsupervised → simplified ─────────────────────────────────────────
    u = results.get('unsupervised')
    if u is not None:
        out['unsupervised'] = {
            'n_change_points': len(u.change_points),
            'n_segments': len(u.segments),
            'n_motifs': len(u.motifs),
            'overview': u.overview,
        }

    # ── similarities (if dual audio) ──────────────────────────────────────
    if 'volatility_similarity' in results:
        out['volatility_similarity'] = _serialize_dict(results['volatility_similarity'])
    if 'dynamics_similarity' in results:
        out['dynamics_similarity'] = _serialize_dict(results['dynamics_similarity'])

    return out


def _serialize_dict(d):
    """Recursively convert ndarrays → lists in a dict/list structure."""
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, dict):
        return {k: _serialize_dict(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_serialize_dict(v) for v in d]
    return d
