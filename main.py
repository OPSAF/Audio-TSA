"""
Audio Time Series Analysis & Prediction
========================================
Course project for Time Series Analysis.
Supports: 1 or 2 audio files (WAV/MP3) or auto-generated synthetic audio.

Usage:
    python main.py                          # Use synthetic audio
    python main.py --audio1 path/to/file.wav
    python main.py --audio1 a.wav --audio2 b.wav
"""

import os
os.environ['NUMBA_DISABLE_CACHE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.numba_cache')

import sys
import argparse
import numpy as np

# Import from audiots package
from audiots import (
    loader, features, dynamics, discovery, analysis, prediction,
    band_analysis, visualization, similarity, similarity_viz, discovery_viz
)


def main():
    parser = argparse.ArgumentParser(
        description='Audio Time Series Analysis & Prediction')
    parser.add_argument('--audio1', type=str, default=None,
                        help='Path to first audio file (wav/mp3)')
    parser.add_argument('--audio2', type=str, default=None,
                        help='Path to second audio file (wav/mp3) for dual analysis')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory for figures')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Target sample rate (default: 16000)')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Duration of synthetic audio in seconds')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save figures to disk')
    args = parser.parse_args()

    print("=" * 70)
    print("  AUDIO TIME SERIES ANALYSIS & PREDICTION")
    print("=" * 70)

    dual_audio = args.audio2 is not None

    if args.audio1 and args.audio2:
        print(f"\n[Loading] Audio 1: {args.audio1}")
        print(f"[Loading] Audio 2: {args.audio2}")
        y1, sr1 = loader.load_audio(args.audio1, target_sr=args.sr)
        y2, sr2 = loader.load_audio(args.audio2, target_sr=args.sr)
        sr = args.sr
        dual_audio = True
    elif args.audio1:
        print(f"\n[Loading] Audio 1: {args.audio1}")
        y1, sr = loader.load_audio(args.audio1, target_sr=args.sr)
        print(f"[Info]  Generating related synthetic audio for dual analysis...")
        y2, _ = loader.generate_sample_audio(duration=len(y1) / sr if sr > 0 else 3.0, sr=sr)
        dual_audio = True
    else:
        print("\n[Info]  No audio files provided. Using synthetic audio samples.")
        print(f"[Generating] {args.duration}s audio at {args.sr}Hz...")
        y1, sr = loader.generate_sample_audio(duration=args.duration, sr=args.sr)
        y2, _ = loader.generate_sample_audio(duration=args.duration, sr=args.sr)
        dual_audio = True

    print(f"[Info]  Sample rate: {sr} Hz, Length: {len(y1)} samples ({len(y1) / sr:.2f}s)")
    print()

    sim_results = None
    dual_results = {}
    dyn1 = None
    dyn2 = None
    dyn_seg1 = None

    # ============================================================
    # Phase 1: Feature Extraction
    # ============================================================
    print("=" * 70)
    print("  PHASE 1: FEATURE EXTRACTION")
    print("=" * 70)

    print("[1.1] Extracting waveform...")
    t, waveform = features.compute_waveform(y1, sr)

    print("[1.2] Computing FFT...")
    freqs, fft_mag = features.compute_fft(y1, sr)

    print("[1.3] Computing STFT...")
    stft_freqs, stft_times, stft_mag = features.compute_stft(y1, sr)

    print("[1.4] Computing Mel Spectrogram...")
    mel_freqs, mel_times, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=128)

    print("[1.5] Computing MFCC...")
    mfcc, mfcc_times = features.compute_mfcc(y1, sr, n_mfcc=20)

    n_mels = mel_spec.shape[0]
    print(f"[Done]  Mel shape: {mel_spec.shape}, MFCC shape: {mfcc.shape}")
    print()

    # ============================================================
    # Phase 1.5: Audio Dynamics / Trend Analysis
    # ============================================================
    print("=" * 70)
    print("  PHASE 1.5: AUDIO DYNAMICS / TREND ANALYSIS")
    print("=" * 70)

    print("[Dyn] Extracting dynamics (energy, brightness, complexity, rhythm)...")
    dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
    dyn_seg1 = dynamics.detect_structural_segments(dyn1)
    dynamics.print_dynamics_report(dyn1, dyn_seg1)

    # ============================================================
    # Phase 2: Time Series Analysis
    # ============================================================
    print("=" * 70)
    print("  PHASE 2: TIME SERIES ANALYSIS")
    print("=" * 70)

    print("[2.1] Computing ACF & PACF...")
    lags, acf_vals, ci = analysis.compute_acf(waveform[:min(len(waveform), sr * 1)], nlags=40)
    _, pacf_vals, _ = analysis.compute_pacf(waveform[:min(len(waveform), sr * 1)], nlags=40)

    print("[2.2] Periodicity analysis...")
    period_info = analysis.analyze_periodicity(y1, sr)
    print(f"      Dominant frequency: {period_info['dominant_frequency']:.1f} Hz")
    print(f"      Dominant period: {period_info['dominant_period_seconds']:.4f}s")
    if period_info['acf_peaks']:
        print(f"      ACF peaks: {period_info['acf_peaks']}")

    print("[2.3] Complexity analysis...")
    complexity = analysis.analyze_complexity(y1)
    print(f"      Zero-crossing rate: {complexity['zero_crossing_rate']:.4f}")
    print(f"      Sample entropy: {complexity['sample_entropy']:.4f}")
    print(f"      Is tonal: {complexity['is_tonal']}")

    print("[2.4] Spectral flatness...")
    flatness = analysis.compute_spectral_flatness(fft_mag)
    print(f"      Spectral flatness: {flatness:.4f} (0=tonal, 1=noise)")

    mel_mean = np.mean(mel_spec, axis=0)
    _, mel_acf, _ = analysis.compute_acf(mel_mean, nlags=40)
    print(f"      Mel energy ACF lag-1: {mel_acf[1]:.4f}")
    print()

    # ============================================================
    # Phase 3: Prediction Models
    # ============================================================
    print("=" * 70)
    print("  PHASE 3: PREDICTION MODEL COMPARISON")
    print("=" * 70)

    forecast_horizon = 20
    print(f"[Prediction] Forecasting {forecast_horizon} future frames...")
    print()
    pred_results = prediction.run_all_predictions(mel_spec, forecast_horizon=forecast_horizon, verbose=True)

    print("\n[Summary] Prediction Model Comparison:")
    print("-" * 50)
    print(f"  {'Model':<14} {'RMSE':<12} {'MAE':<12}")
    print("-" * 50)
    for model in ['ARIMA', 'HMM', 'LSTM', 'Transformer']:
        _, metrics, _ = pred_results[model]
        rmse_str = f"{metrics['RMSE']:.4f}" if not np.isnan(metrics['RMSE']) else "N/A"
        mae_str = f"{metrics['MAE']:.4f}" if not np.isnan(metrics['MAE']) else "N/A"
        print(f"  {model:<14} {rmse_str:<12} {mae_str:<12}")
        if model == 'ARIMA' and 'error' in metrics and metrics['error']:
            print(f"         Error: {metrics['error']}")
    print("-" * 50)
    print()

    # ============================================================
    # Phase 4: Frequency Band Predictability Analysis
    # ============================================================
    print("=" * 70)
    print("  PHASE 4: FREQUENCY BAND PREDICTABILITY ANALYSIS")
    print("=" * 70)

    band_results = band_analysis.analyze_band_predictability(
        mel_spec, forecast_horizon=forecast_horizon, 
        parallel=True, epochs=15)
    
    band_summary = band_analysis.compute_band_error_summary(band_results)
    band_analysis.print_band_summary(band_summary)
    band_analysis.print_detailed_band_results(band_results)
    print()

    # ============================================================
    # Phase 5: Dual Audio Discovery (Multi-Dimensional Exploration)
    # ============================================================
    disc_report = None
    if dual_audio:
        print("=" * 70)
        print("  PHASE 5: DUAL AUDIO DISCOVERY (Multi-Dimensional Exploration)")
        print("=" * 70)

        disc_report = discovery.explore(
            y1, sr, y2, sr,
            window_size=0.5, hop_size=0.25, verbose=True,
        )
        discovery.print_discovery_report(disc_report)

        # Also keep simple DTW for backward compatibility
        dual_results = {}
        try:
            from dtw import dtw
            from scipy.spatial.distance import euclidean
            mel2_freqs, mel2_times, mel2_spec = features.compute_mel_spectrogram(y2, sr, n_mels=128)
            mel1_mean = np.mean(mel_spec, axis=0)[:100]
            mel2_mean = np.mean(mel2_spec, axis=0)[:100]
            dtw_result = dtw(
                mel1_mean.reshape(-1, 1), mel2_mean.reshape(-1, 1), dist_method=euclidean
            )
            dist = dtw_result.distance
            path = np.column_stack((dtw_result.index1, dtw_result.index2))
            dtw_similarity = 1 - dist / (len(mel1_mean) * np.std(mel1_mean))
            dual_results = {
                'dtw_distance': dist,
                'similarity': dtw_similarity,
                'dtw_path': path.tolist(),
            }
            print(f"\n  [Reference] DTW Distance: {dist:.4f}, Similarity: {dtw_similarity*100:.2f}%")
        except ImportError:
            print("  [Info] DTW library not installed")
            dual_results = {}
        except Exception as e:
            print(f"  [Warning] DTW analysis failed: {e}")
            dual_results = {}
        print()

    # ============================================================
    # Phase 6: Visualization
    # ============================================================
    print("=" * 70)
    print("  PHASE 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig_dict = {}

    print("  [6.1] Waveform plot...")
    fig_dict['01_waveform'] = visualization.plot_waveform(t, waveform, title="Audio Waveform")

    print("  [6.2] FFT magnitude spectrum...")
    fig_dict['02_fft_spectrum'] = visualization.plot_fft(freqs, fft_mag, title="FFT Magnitude Spectrum")

    print("  [6.3] STFT spectrogram...")
    fig_dict['03_stft_spectrogram'] = visualization.plot_stft(
        stft_freqs, stft_times, stft_mag, title="STFT Spectrogram")

    print("  [6.4] Mel spectrogram...")
    fig_dict['04_mel_spectrogram'] = visualization.plot_mel_spectrogram(
        mel_freqs, mel_times, mel_spec, title="Mel Spectrogram")

    print("  [6.5] MFCC heatmap...")
    fig_dict['05_mfcc'] = visualization.plot_mfcc(mfcc, mfcc_times, title="MFCC Coefficients")

    print("  [6.6] ACF & PACF...")
    fig_dict['06_acf_pacf'] = visualization.plot_acf_pacf(
        lags, acf_vals, pacf_vals, ci, title="ACF & PACF Analysis")

    print("  [6.7] Periodicity plot...")
    fig_dict['07_periodicity'] = visualization.plot_periodicity(
        y1, sr, period_info, title="Periodicity Analysis")

    print("  [6.8] Prediction comparison (4 models)...")
    fig_dict['08_prediction_comparison'] = visualization.plot_prediction_comparison(
        pred_results, forecast_horizon=forecast_horizon,
        title="Mel Energy Prediction: Model Comparison")

    print("  [6.9] Model error bar chart...")
    fig_dict['09_model_error_bars'] = visualization.plot_model_error_comparison(
        pred_results, title="Prediction Error Comparison Across Models")

    print("  [6.10] Band error heatmap...")
    fig_dict['10_band_error_heatmap'] = visualization.plot_band_error_heatmap(
        band_results, title="Frequency Band Prediction Error Heatmap")

    print("  [6.11] Band error bar chart...")
    fig_dict['11_band_error_bars'] = visualization.plot_band_error_bars(
        band_results, title="Band-wise Prediction Error Comparison")

    if dual_audio and dual_results:
        print("  [6.12] DTW alignment plot...")
        series1_ds = y1[::max(1, len(y1) // 200)]
        series2_ds = y2[::max(1, len(y2) // 200)]
        min_len = min(len(series1_ds), len(series2_ds))
        fig_dict['12_dtw_alignment'] = visualization.plot_dtw_alignment(
            series1_ds[:min_len], series2_ds[:min_len],
            dual_results.get('dtw_path', []),
            title=f"DTW Alignment (distance={dual_results.get('dtw_distance', 0):.2f})")

    # Generate discovery plots (replaces similarity scoring)
    if disc_report is not None:
        print("\n  [6.13] Generating discovery visualizations ...")
        disc_output_dir = os.path.join(args.output, 'discovery')
        os.makedirs(disc_output_dir, exist_ok=True)
        disc_plot_files = discovery_viz.generate_discovery_report_plots(
            disc_report,
            output_dir=disc_output_dir,
            y1=y1, y2=y2, sr=sr,
            dyn1=dyn1, dyn2=dyn2 if dual_audio else None,
        )
        for dp in disc_plot_files:
            print(f"         discovery/{dp}")
        fig_dict['13_discovery_summary'] = discovery_viz.plot_discovery_summary(
            disc_report, title="Audio Discovery Summary")

    # ---- Dynamics / Trend plots ----
    if dyn1 is not None:
        print("\n  [6.15] Generating dynamics trend plots...")
        fig_dict['14_dynamics_trends'] = visualization.plot_dynamics_trends(
            dyn1, segments=dyn_seg1,
            title="Audio 1 — Dynamics Trend Analysis")
        fig_dict['15_dynamics_summary'] = visualization.plot_dynamics_summary(
            dyn1, segments=dyn_seg1,
            title="Audio 1 — Dynamics Summary")

        if dual_audio and dyn2 is not None:
            dyn_sim_viz = dynamics.compute_dynamics_similarity(dyn1, dyn2)
            fig_dict['16_dynamics_dual'] = visualization.plot_dynamics_dual(
                dyn1, dyn2, sim_result=dyn_sim_viz,
                title="Dual Audio — Dynamics Comparison")

    if not args.no_save:
        print(f"\n  Saving figures to '{args.output}'...")
        visualization.save_all_figures(fig_dict, output_dir=args.output)
    else:
        print("\n  [Skip] --no-save flag set, not saving figures.")

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Total figures generated: {len(fig_dict)}")
    if not args.no_save:
        print(f"  Output directory: {os.path.abspath(args.output)}")
    print()

    import matplotlib.pyplot as plt
    plt.close('all')


if __name__ == '__main__':
    main()