"""
Audio Time Series Analysis & Prediction
========================================
Course project for Time Series Analysis.
Supports: 1 or 2 audio files (WAV/MP3) or auto-generated synthetic audio.

Usage:
    python main.py                          # Use synthetic audio, all analysis
    python main.py --audio1 path/to/file.wav
    python main.py --audio1 a.wav --audio2 b.wav
    python main.py --audio1 a.wav --analysis features,timeseries,prediction
    
Analysis options (comma-separated):
    features     - Feature extraction (waveform, FFT, STFT, Mel, MFCC)
    dynamics     - Audio dynamics/trend analysis
    timeseries   - Time series analysis (ACF, PACF, periodicity, complexity)
    unsupervised - Unsupervised pattern discovery
    prediction   - Machine learning prediction (ARIMA, HMM, LSTM, Transformer)
    band         - Frequency band predictability analysis
    comparison   - Dual audio comparison (DTW, discovery) [requires 2 audios]
    visualization- Generate visualizations
    
If --analysis is not specified, all applicable analyses are performed.
"""

import os
os.environ['NUMBA_DISABLE_CACHE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.numba_cache')

import sys
import argparse
import numpy as np

# Import from audiots package
from audiots import (
    loader, features, dynamics, volatility, model_analysis, discovery, unsupervised,
    analysis, prediction, band_analysis, visualization, similarity, similarity_viz,
    discovery_viz
)


def parse_analysis_options(args):
    """Parse analysis options from command line."""
    available_options = [
        'features', 'dynamics', 'dynamics_analysis', 'model_analysis', 'timeseries',
        'unsupervised', 'prediction', 'band', 'comparison', 'visualization'
    ]
    
    if args.analysis:
        selected = [a.strip().lower() for a in args.analysis.split(',')]
        # Validate selected options
        invalid = [a for a in selected if a not in available_options]
        if invalid:
            print(f"[Error] Invalid analysis options: {', '.join(invalid)}")
            print(f"Available options: {', '.join(available_options)}")
            sys.exit(1)
        return selected
    
    # Default: all analyses except comparison (unless dual audio)
    defaults = ['features', 'dynamics', 'dynamics_analysis', 'model_analysis',
                'timeseries', 'unsupervised', 'prediction', 'band', 'visualization']
    return defaults


def main():
    parser = argparse.ArgumentParser(
        description='Audio Time Series Analysis & Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis options (--analysis):
  features     - Feature extraction (waveform, FFT, STFT, Mel, MFCC)
  dynamics     - Audio dynamics/trend analysis
  dynamics_analysis - Trend Layer + Volatility Layer (ARCH/GARCH) + predictions
  timeseries   - Time series analysis (ACF, PACF, periodicity, complexity)
  unsupervised - Unsupervised pattern discovery
  prediction   - ML prediction (ARIMA, HMM, LSTM, Transformer)
  band         - Frequency band predictability analysis
  comparison   - Dual audio comparison [requires 2 audio files]
  visualization- Generate visualizations

Example:
  python main.py --audio1 song.wav --analysis features,timeseries,prediction
        """
    )
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
    parser.add_argument('--analysis', type=str, default=None,
                        help='Comma-separated list of analyses to perform')
    args = parser.parse_args()

    print("=" * 70)
    print("  AUDIO TIME SERIES ANALYSIS & PREDICTION")
    print("=" * 70)

    # Check if dual audio mode
    dual_audio = args.audio2 is not None
    
    # Load audio files
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
        y2 = None
        sr2 = None
        dual_audio = False
    else:
        print("\n[Info]  No audio files provided. Using synthetic audio samples.")
        print(f"[Generating] {args.duration}s audio at {args.sr}Hz...")
        y1, sr = loader.generate_sample_audio(duration=args.duration, sr=args.sr)
        y2, _ = loader.generate_sample_audio(duration=args.duration, sr=args.sr)
        dual_audio = True

    print(f"[Info]  Sample rate: {sr} Hz, Length: {len(y1)} samples ({len(y1) / sr:.2f}s)")
    print(f"[Info]  Mode: {'Dual Audio' if dual_audio else 'Single Audio'}")
    print()

    # Parse analysis options
    analysis_options = parse_analysis_options(args)
    print(f"[Info]  Selected analyses: {', '.join(analysis_options)}")
    
    # Check for comparison without dual audio
    if 'comparison' in analysis_options and not dual_audio:
        print("[Warning] Comparison analysis requires two audio files. Disabling comparison.")
        analysis_options.remove('comparison')
    
    print()

    # Initialize variables
    sim_results = None
    dual_results = {}
    dyn1 = None
    dyn2 = None
    dyn_seg1 = None
    
    # Results storage
    results = {
        'features': {},
        'dynamics': None,
        'timeseries': {},
        'unsupervised': None,
        'prediction': None,
        'band': None,
        'comparison': {},
        'visualization': {}
    }

    # ============================================================
    # Phase 1: Feature Extraction
    # ============================================================
    if 'features' in analysis_options:
        print("=" * 70)
        print("  PHASE 1: FEATURE EXTRACTION")
        print("=" * 70)

        print("[1.1] Extracting waveform...")
        t, waveform = features.compute_waveform(y1, sr)
        results['features']['waveform'] = {'t': t, 'y': waveform}

        print("[1.2] Computing FFT...")
        freqs, fft_mag = features.compute_fft(y1, sr)
        results['features']['fft'] = {'freqs': freqs, 'mag': fft_mag}

        print("[1.3] Computing STFT...")
        stft_freqs, stft_times, stft_mag = features.compute_stft(y1, sr)
        results['features']['stft'] = {'freqs': stft_freqs, 'times': stft_times, 'mag': stft_mag}

        print("[1.4] Computing Mel Spectrogram...")
        mel_freqs, mel_times, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=128)
        results['features']['mel'] = {'freqs': mel_freqs, 'times': mel_times, 'spec': mel_spec}

        print("[1.5] Computing MFCC...")
        mfcc, mfcc_times = features.compute_mfcc(y1, sr, n_mfcc=20)
        results['features']['mfcc'] = {'mfcc': mfcc, 'times': mfcc_times}

        n_mels = mel_spec.shape[0]
        print(f"[Done]  Mel shape: {mel_spec.shape}, MFCC shape: {mfcc.shape}")
        print()

    # ============================================================
    # Phase 1.5: Audio Dynamics / Trend Analysis
    # ============================================================
    if 'dynamics' in analysis_options:
        print("=" * 70)
        print("  PHASE 1.5: AUDIO DYNAMICS / TREND ANALYSIS")
        print("=" * 70)

        print("[Dyn] Extracting dynamics (energy, brightness, complexity, rhythm)...")
        dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
        dyn_seg1 = dynamics.detect_structural_segments(dyn1)
        results['dynamics'] = {'data': dyn1, 'segments': dyn_seg1}
        dynamics.print_dynamics_report(dyn1, dyn_seg1)

        # Extract dynamics for second audio if dual mode
        if dual_audio and y2 is not None:
            dyn2 = dynamics.extract_dynamics(y2, sr, window_size=0.5, hop_size=0.25)
        print()

    # ============================================================
    # Phase 1.6: Dynamics Analysis — Volatility Layer
    # ============================================================
    vol1 = None
    if 'dynamics_analysis' in analysis_options:
        print("=" * 70)
        print("  PHASE 1.6: AUDIO DYNAMICS ANALYSIS (Volatility + GARCH)")
        print("=" * 70)

        # Ensure dynamics are extracted first
        if dyn1 is None:
            print("[Dyn] Extracting dynamics first...")
            dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
            results['dynamics'] = {'data': dyn1, 'segments': None}

        # ---- Volatility Layer ----
        print("[Vol] Computing rolling volatility + GARCH(1,1)...")
        vol1 = volatility.compute_volatility_layer(dyn1, rolling_window=10, fit_garch=True)
        volatility.print_volatility_report(vol1, show_garch=True)

        # ---- Trend predictions ----
        print("[TrendPred] Predicting all 4 trends...")
        trend_preds = prediction.predict_all_trends(dyn1, forecast_horizon=20, verbose=True)
        results['trend_predictions'] = trend_preds

        # ---- Volatility predictions ----
        print("[VolPred] Predicting volatility...")
        vol_preds = prediction.predict_all_volatilities(vol1, forecast_horizon=10, verbose=True)
        results['volatility_predictions'] = vol_preds

        results['volatility'] = vol1
        print()

    # ============================================================
    # Phase 2a: Model Ensemble Structural Analysis
    # ============================================================
    if 'model_analysis' in analysis_options:
        print("=" * 70)
        print("  PHASE 2a: MODEL ENSEMBLE STRUCTURAL ANALYSIS")
        print("=" * 70)

        # Ensure dynamics are extracted
        if dyn1 is None:
            print("[Dyn] Extracting dynamics first...")
            dyn1 = dynamics.extract_dynamics(y1, sr, window_size=0.5, hop_size=0.25)
            results['dynamics'] = {'data': dyn1, 'segments': None}

        print("[Model] Running 4-model structural detective analysis...")
        model_report = model_analysis.analyze_model_ensemble(
            dyn1, n_hmm_states=3, lstm_epochs=20, transformer_epochs=20, verbose=True)
        model_analysis.print_model_ensemble_report(model_report)
        results['model_analysis'] = model_report
        print()

    # ============================================================
    # Phase 2: Time Series Analysis
    # ============================================================
    if 'timeseries' in analysis_options:
        print("=" * 70)
        print("  PHASE 2: TIME SERIES ANALYSIS")
        print("=" * 70)

        waveform_for_ts = results['features']['waveform']['y'] if 'features' in analysis_options else y1
        
        print("[2.1] Computing ACF & PACF...")
        lags, acf_vals, ci = analysis.compute_acf(waveform_for_ts[:min(len(waveform_for_ts), sr * 1)], nlags=40)
        _, pacf_vals, _ = analysis.compute_pacf(waveform_for_ts[:min(len(waveform_for_ts), sr * 1)], nlags=40)
        results['timeseries']['acf_pacf'] = {'lags': lags, 'acf': acf_vals, 'pacf': pacf_vals, 'ci': ci}

        print("[2.2] Periodicity analysis...")
        period_info = analysis.analyze_periodicity(y1, sr)
        results['timeseries']['periodicity'] = period_info
        print(f"      Dominant frequency: {period_info['dominant_frequency']:.1f} Hz")
        print(f"      Dominant period: {period_info['dominant_period_seconds']:.4f}s")
        if period_info['acf_peaks']:
            print(f"      ACF peaks: {period_info['acf_peaks']}")

        print("[2.3] Complexity analysis...")
        complexity = analysis.analyze_complexity(y1)
        results['timeseries']['complexity'] = complexity
        print(f"      Zero-crossing rate: {complexity['zero_crossing_rate']:.4f}")
        print(f"      Sample entropy: {complexity['sample_entropy']:.4f}")
        print(f"      Is tonal: {complexity['is_tonal']}")

        print("[2.4] Spectral flatness...")
        fft_mag_for_sf = results['features']['fft']['mag'] if 'features' in analysis_options else features.compute_fft(y1, sr)[1]
        flatness = analysis.compute_spectral_flatness(fft_mag_for_sf)
        results['timeseries']['spectral_flatness'] = flatness
        print(f"      Spectral flatness: {flatness:.4f} (0=tonal, 1=noise)")

        print("[2.5] White noise testing...")
        white_noise_results = analysis.test_white_noise(waveform_for_ts[:min(len(waveform_for_ts), sr * 2)])
        results['timeseries']['white_noise_test'] = white_noise_results
        analysis.print_white_noise_report(white_noise_results)

        if 'features' in analysis_options:
            mel_mean = np.mean(results['features']['mel']['spec'], axis=0)
            _, mel_acf, _ = analysis.compute_acf(mel_mean, nlags=40)
            print(f"      Mel energy ACF lag-1: {mel_acf[1]:.4f}")
        print()

    # ============================================================
    # Phase 3: Unsupervised Pattern Discovery
    # ============================================================
    if 'unsupervised' in analysis_options:
        print("=" * 70)
        print("  PHASE 3: UNSUPERVISED PATTERN DISCOVERY")
        print("=" * 70)

        unsup_report = unsupervised.explore_unsupervised(
            y1, sr, n_components=4, n_clusters=4, verbose=True,
        )
        results['unsupervised'] = unsup_report
        unsupervised.print_unsupervised_report(unsup_report)
        print()

    # ============================================================
    # Phase 4: Machine Learning Prediction (includes all 4 models)
    # ============================================================
    if 'prediction' in analysis_options:
        print("=" * 70)
        print("  PHASE 4: MACHINE LEARNING PREDICTION")
        print("=" * 70)
        
        if 'features' in analysis_options:
            mel_spec = results['features']['mel']['spec']
        else:
            _, _, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=128)

        print("[Pred] Running all prediction models (ARIMA, HMM, LSTM, Transformer)...")
        predictions = prediction.run_all_predictions(mel_spec, forecast_horizon=20, verbose=True)
        results['prediction'] = predictions
        
        print("\n[Pred] Model Performance Summary:")
        for model_name, (forecast, metrics, true) in predictions.items():
            if metrics:
                print(f"  {model_name}: RMSE={metrics.get('RMSE', 'N/A'):.4f}, MAE={metrics.get('MAE', 'N/A'):.4f}")
        print()

    # ============================================================
    # Phase 4.5: Band Analysis
    # ============================================================
    if 'band' in analysis_options:
        print("=" * 70)
        print("  PHASE 4.5: FREQUENCY BAND ANALYSIS")
        print("=" * 70)
        
        if 'features' in analysis_options:
            mel_spec = results['features']['mel']['spec']
        else:
            _, _, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=128)

        print("[Band] Analyzing frequency band predictability...")
        band_results = band_analysis.analyze_band_predictability(mel_spec, forecast_horizon=20)
        band_summary = band_analysis.compute_band_error_summary(band_results)
        predictability_rank = band_analysis.get_predictability_rank(band_summary)
        results['band'] = {'results': band_results, 'summary': band_summary, 'rank': predictability_rank}
        
        print("\n[Band] Band Predictability Ranking:")
        for i, item in enumerate(predictability_rank, 1):
            print(f"  {i}. {item['band']} - Best: {item['best_model']}, RMSE: {item['avg_rmse']:.4f}")
        print()

    # ============================================================
    # Phase 5: Dual Audio Comparison (Discovery)
    # ============================================================
    disc_report = None
    if 'comparison' in analysis_options and dual_audio and y2 is not None:
        print("=" * 70)
        print("  PHASE 5: DUAL AUDIO COMPARISON")
        print("=" * 70)

        print("[Comp] Running multi-dimensional discovery...")
        disc_report = discovery.explore(
            y1, sr, y2, sr,
            window_size=0.5, hop_size=0.25, verbose=True,
        )
        discovery.print_discovery_report(disc_report)
        results['comparison']['discovery'] = disc_report

        # ---- Volatility similarity (if dynamics_analysis was also run) ----
        if vol1 is not None and dyn2 is not None:
            print("-" * 70)
            print("  [Comp] Volatility similarity...")
            vol2 = volatility.compute_volatility_layer(dyn2, rolling_window=10, fit_garch=True)
            vol_sim = volatility.compute_volatility_similarity(vol1, vol2)
            volatility.print_volatility_similarity_report(vol_sim)
            results['comparison']['volatility_similarity'] = vol_sim

            print("  [Comp] Dynamics similarity...")
            dyn_sim = dynamics.compute_dynamics_similarity(dyn1, dyn2)
            dynamics.print_dynamics_similarity_report(dyn_sim)
            results['comparison']['dynamics_similarity'] = dyn_sim

        print()

    # ============================================================
    # Phase 6: Visualization
    # ============================================================
    if 'visualization' in analysis_options:
        print("=" * 70)
        print("  PHASE 6: GENERATING VISUALIZATIONS")
        print("=" * 70)

        fig_dict = {}

        if 'features' in analysis_options:
            print("  [6.1] Waveform plot...")
            fig_dict['01_waveform'] = visualization.plot_waveform(
                results['features']['waveform']['t'], 
                results['features']['waveform']['y'], 
                title="Audio Waveform")

            print("  [6.2] FFT magnitude spectrum...")
            fig_dict['02_fft_spectrum'] = visualization.plot_fft(
                results['features']['fft']['freqs'], 
                results['features']['fft']['mag'], 
                title="FFT Magnitude Spectrum")

            print("  [6.3] STFT spectrogram...")
            fig_dict['03_stft_spectrogram'] = visualization.plot_stft(
                results['features']['stft']['freqs'], 
                results['features']['stft']['times'], 
                results['features']['stft']['mag'], 
                title="STFT Spectrogram")

            print("  [6.4] Mel spectrogram...")
            fig_dict['04_mel_spectrogram'] = visualization.plot_mel_spectrogram(
                results['features']['mel']['freqs'], 
                results['features']['mel']['times'], 
                results['features']['mel']['spec'], 
                title="Mel Spectrogram")

            print("  [6.5] MFCC heatmap...")
            fig_dict['05_mfcc'] = visualization.plot_mfcc(
                results['features']['mfcc']['mfcc'], 
                results['features']['mfcc']['times'], 
                title="MFCC Coefficients")

        if 'timeseries' in analysis_options:
            print("  [6.6] ACF & PACF...")
            ts = results['timeseries']
            fig_dict['06_acf_pacf'] = visualization.plot_acf_pacf(
                ts['acf_pacf']['lags'], 
                ts['acf_pacf']['acf'], 
                ts['acf_pacf']['pacf'], 
                ts['acf_pacf']['ci'], 
                title="ACF & PACF Analysis")

            print("  [6.7] Periodicity plot...")
            fig_dict['07_periodicity'] = visualization.plot_periodicity(
                y1, sr, ts['periodicity'], title="Periodicity Analysis")

        if 'prediction' in analysis_options:
            print("  [6.8] Prediction comparison...")
            if 'features' in analysis_options:
                mel_spec = results['features']['mel']['spec']
            else:
                _, _, mel_spec = features.compute_mel_spectrogram(y1, sr, n_mels=128)
            fig_dict['08_prediction_comparison'] = visualization.plot_prediction_comparison(
                results['prediction'], forecast_horizon=20, title="Prediction Comparison")
            
            print("  [6.9] Model error bars...")
            fig_dict['09_model_error_bars'] = visualization.plot_model_error_comparison(
                results['prediction'], title="Model Error Comparison")

        if 'band' in analysis_options:
            print("  [6.10] Band error heatmap...")
            fig_dict['10_band_error_heatmap'] = visualization.plot_band_error_heatmap(
                results['band']['results'], title="Band Error Heatmap")
            
            print("  [6.11] Band error bars...")
            fig_dict['11_band_error_bars'] = visualization.plot_band_error_bars(
                results['band']['results'], title="Band Error Comparison")

        if disc_report is not None:
            print("\n  [6.13] Generating discovery visualizations...")
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

        if 'dynamics' in analysis_options and dyn1 is not None:
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

        if 'dynamics_analysis' in analysis_options and vol1 is not None and dyn1 is not None:
            print("\n  [6.16] Generating volatility analysis plots...")
            fig_dict['17_volatility_layer'] = visualization.plot_volatility_layer(
                dyn1, vol1,
                title="Audio 1 — Volatility Layer")
            fig_dict['18_garch_energy'] = visualization.plot_garch_diagnostics(
                vol1, trend_key='energy',
                title="GARCH(1,1) — Energy Trend Volatility")
            fig_dict['19_dynamics_analysis_summary'] = visualization.plot_dynamics_analysis_summary(
                {'trend_summary': dynamics.summarize_dynamics(dyn1),
                 'volatility_summary': volatility.summarize_volatility(vol1)},
                title="Dynamics Analysis — Trend + Volatility Summary")

            if dual_audio and dyn2 is not None:
                vol2_for_viz = volatility.compute_volatility_layer(dyn2, rolling_window=10, fit_garch=True)
                vol_sim_for_viz = None
                if 'comparison' in results and isinstance(results['comparison'], dict):
                    vol_sim_for_viz = results['comparison'].get('volatility_similarity')
                if vol_sim_for_viz is None:
                    vol_sim_for_viz = volatility.compute_volatility_similarity(vol1, vol2_for_viz)
                fig_dict['20_volatility_comparison'] = visualization.plot_volatility_comparison(
                    vol1, vol2_for_viz,
                    sim_result=vol_sim_for_viz,
                    title="Dual Audio — Volatility Comparison")

        if not args.no_save:
            print(f"\n  Saving figures to '{args.output}'...")
            visualization.save_all_figures(fig_dict, output_dir=args.output)
        else:
            print("\n  [Skip] --no-save flag set, not saving figures.")

        results['visualization']['figures'] = fig_dict

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    
    completed_analyses = [a for a in analysis_options if a in results and results[a]]
    print(f"  Analyses performed: {len(completed_analyses)}")
    for a in completed_analyses:
        status = "✓"
        if a == 'visualization':
            status += f" ({len(results[a].get('figures', {}))} figures)"
        print(f"    • {a}: {status}")
    
    if 'visualization' in analysis_options and not args.no_save:
        print(f"  Output directory: {os.path.abspath(args.output)}")
    print()

    import matplotlib.pyplot as plt
    plt.close('all')


if __name__ == '__main__':
    main()