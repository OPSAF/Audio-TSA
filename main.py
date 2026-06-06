"""
Audio Time Series Analysis & Prediction — CLI Entry Point
==========================================================
Uses the shared ``audiots.pipeline`` module (same core as web & batch).

Usage:
    python main.py                                    # synthetic audio demo
    python main.py --audio1 song.wav
    python main.py --audio1 a.wav --audio2 b.wav
    python main.py --audio1 song.wav --output ./my_results
"""

import os
import sys
import argparse
import json
import numpy as np

# Shared pipeline + print-report modules
from audiots import loader, dynamics, volatility, model_analysis
from audiots import unsupervised as _unsup, analysis as _analysis, discovery as _discovery
from audiots.pipeline import (
    run_full_analysis, generate_all_plots, serialize_results,
    DEFAULT_OPTIONS, ALL_OPTIONS,
)

OUTPUT_DIR = './outputs'


def main():
    parser = argparse.ArgumentParser(
        description='Audio Time Series Analysis & Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available --analysis options (comma-separated):
  {', '.join(ALL_OPTIONS)}

Default: {', '.join(DEFAULT_OPTIONS)}

Examples:
  python main.py --audio1 song.wav
  python main.py --audio1 a.wav --audio2 b.wav
  python main.py --audio1 song.wav --analysis features,timeseries,prediction
        """
    )
    parser.add_argument('--audio1', default=None, help='Path to primary audio file')
    parser.add_argument('--audio2', default=None, help='Path to second audio file')
    parser.add_argument('--output', default=OUTPUT_DIR, help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    parser.add_argument('--analysis', default=None,
                        help='Comma-separated list of analyses to run')
    parser.add_argument('--forecast-horizon', type=int, default=20,
                        help='Forecast horizon in steps')
    parser.add_argument('--n-mels', type=int, default=128,
                        help='Number of Mel bands')
    parser.add_argument('--fast', action='store_true',
                        help='Reduce deep-learning epochs for speed')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving visualizations to disk')
    parser.add_argument('--save-json', action='store_true',
                        help='Also save results.json alongside figures')
    args = parser.parse_args()

    # ── Parse analysis options ──────────────────────────────────────────────
    if args.analysis:
        selected = [a.strip().lower() for a in args.analysis.split(',')]
        invalid = [a for a in selected if a not in ALL_OPTIONS]
        if invalid:
            print(f"[ERROR] Invalid options: {', '.join(invalid)}")
            print(f"        Available: {', '.join(ALL_OPTIONS)}")
            sys.exit(1)
        analysis_options = selected
    else:
        analysis_options = list(DEFAULT_OPTIONS)

    # Add comparison if dual audio
    if args.audio2 and 'comparison' not in analysis_options:
        analysis_options.append('comparison')

    # ── Load audio ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("  AUDIO TIME SERIES ANALYSIS & PREDICTION")
    print("=" * 70)

    if args.audio1:
        print(f"\n[Loading] Audio 1: {args.audio1}")
        y1, sr = loader.load_audio(args.audio1, target_sr=args.sr)
        y2, sr2 = None, None
        if args.audio2:
            print(f"[Loading] Audio 2: {args.audio2}")
            y2, sr2 = loader.load_audio(args.audio2, target_sr=args.sr)
            print(f"[Info]  Mode: Dual Audio")
    else:
        print("\n[Info]  No audio files provided. Using synthetic audio samples.")
        y1, sr = loader.generate_sample_audio(duration=3.0, sr=args.sr)
        y2, _ = loader.generate_sample_audio(duration=3.0, sr=args.sr)
        sr2 = sr
        dual_audio = True

    dual_audio = y2 is not None
    print(f"[Info]  Sample rate: {sr} Hz, Length: {len(y1)} samples ({len(y1)/sr:.2f}s)")
    print(f"[Info]  Mode: {'Dual Audio' if dual_audio else 'Single Audio'}")
    print(f"[Info]  Selected analyses: {', '.join(analysis_options)}")
    if args.fast:
        print(f"[Info]  Fast mode: ON")
    print()

    # ── Progress callback (print to terminal) ───────────────────────────────
    def term_log(msg: str, level: str = 'info'):
        prefix = {'phase': '\n' + '=' * 50,
                  'divider': '-' * 50}.get(level, ' ')
        if level in ('phase', 'divider'):
            print(f"{prefix}  {msg}")
        elif level == 'error':
            print(f"  [ERR] {msg}", file=sys.stderr)
        elif level == 'success':
            print(f"  [OK]  {msg}")
        else:
            print(f"  {msg}")

    # ── Run analysis (delegated to shared pipeline) ─────────────────────────
    results = run_full_analysis(
        y1=y1, sr=sr, y2=y2, sr2=sr2,
        forecast_horizon=args.forecast_horizon,
        n_mels=args.n_mels,
        analysis_options=analysis_options,
        progress_callback=term_log,
        fast=args.fast,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Rich terminal reports (restored from original main.py)
    # Each report uses the raw objects from the pipeline results dict.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Dynamics report ──────────────────────────────────────────────────────
    dyn_entry = results.get('dynamics', {})
    dyn_data = dyn_entry.get('data') if isinstance(dyn_entry, dict) else None
    dyn_seg = dyn_entry.get('segments') if isinstance(dyn_entry, dict) else None
    if dyn_data is not None and 'dynamics' in analysis_options:
        dynamics.print_dynamics_report(dyn_data, dyn_seg)

    # ── Volatility report ────────────────────────────────────────────────────
    vol_data = results.get('volatility')
    if vol_data is not None and 'dynamics_analysis' in analysis_options:
        volatility.print_volatility_report(vol_data, show_garch=True)

    # ── Model ensemble report ────────────────────────────────────────────────
    mr = results.get('model_analysis')
    if mr is not None and 'model_analysis' in analysis_options:
        model_analysis.print_model_ensemble_report(mr)

    # ── White noise report ───────────────────────────────────────────────────
    ts = results.get('timeseries', {})
    white_noise = ts.get('white_noise_test')
    if white_noise is not None and 'timeseries' in analysis_options:
        _analysis.print_white_noise_report(white_noise)

    # ── Unsupervised report ──────────────────────────────────────────────────
    unsup = results.get('unsupervised')
    if unsup is not None and 'unsupervised' in analysis_options:
        _unsup.print_unsupervised_report(unsup)

    # ── Discovery report ─────────────────────────────────────────────────────
    disc = results.get('discovery')
    if disc is not None and 'comparison' in analysis_options:
        _discovery.print_discovery_report(disc)

    # ── Volatility similarity report ─────────────────────────────────────────
    vol_sim = results.get('volatility_similarity')
    if vol_sim is not None and isinstance(vol_sim, dict):
        volatility.print_volatility_similarity_report(vol_sim)

    # ── Dynamics similarity report ───────────────────────────────────────────
    dyn_sim = results.get('dynamics_similarity')
    if dyn_sim is not None and isinstance(dyn_sim, dict):
        dynamics.print_dynamics_similarity_report(dyn_sim)

    # ═══════════════════════════════════════════════════════════════════════════
    # Prediction & Band summary
    # ═══════════════════════════════════════════════════════════════════════════
    pred = results.get('prediction')
    if pred:
        print("\n" + "=" * 70)
        print("  PREDICTION MODEL PERFORMANCE")
        print("=" * 70)
        for mn, (forecast, metrics, true) in pred.items():
            if metrics:
                print(f"  {mn:15s}: RMSE={metrics.get('RMSE', np.nan):.4f}, "
                      f"MAE={metrics.get('MAE', np.nan):.4f}, "
                      f"MSE={metrics.get('MSE', np.nan):.4f}")

    band = results.get('band', {})
    rank = band.get('rank', [])
    if rank:
        print("\n" + "=" * 70)
        print("  FREQUENCY BAND PREDICTABILITY RANKING")
        print("=" * 70)
        for i, item in enumerate(rank, 1):
            print(f"  {i}. {item['band']:12s} — Best: {item['best_model']:12s}, "
                  f"RMSE: {item['avg_rmse']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Visualization (shared pipeline)
    # ═══════════════════════════════════════════════════════════════════════════
    plot_files = []
    if 'visualization' in analysis_options and not args.no_save:
        print("\n" + "=" * 70)
        print("  PHASE 6: GENERATING VISUALIZATIONS")
        print("=" * 70)
        os.makedirs(args.output, exist_ok=True)
        plot_files = generate_all_plots(
            results, args.output,
            y1=y1, sr=sr, y2=y2,
        )
        print(f"\n  Figures generated: {len(plot_files)}")
        for pf in sorted(plot_files):
            print(f"    • {pf}")
    elif 'visualization' in analysis_options:
        print("\n  [Skip] --no-save flag set, not saving figures.")

    # ── Save JSON ───────────────────────────────────────────────────────────
    if args.save_json:
        out_dir = args.output
        os.makedirs(out_dir, exist_ok=True)
        exp_name = os.path.splitext(os.path.basename(args.audio1 or 'synthetic'))[0]
        serialized = serialize_results(
            results, task_id='cli',
            task_info={
                'experiment_name': exp_name,
                'forecast_horizon': args.forecast_horizon,
                'n_mels': args.n_mels,
                'audio1_name': os.path.basename(args.audio1 or 'synthetic'),
                'audio2_name': os.path.basename(args.audio2) if args.audio2 else None,
                'analysis_options': analysis_options,
            },
            plot_files=plot_files,
        )
        json_path = os.path.join(out_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, default=str, ensure_ascii=False)
        print(f"\n  Results saved → {json_path}")

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    completed = [a for a in analysis_options
                 if a in results or (a == 'visualization' and plot_files)]
    for a in completed:
        status = "✓"
        if a == 'visualization':
            status += f" ({len(plot_files)} figures)"
        print(f"    • {a}: {status}")
    print(f"\n  Output directory: {os.path.abspath(args.output)}")
    print()

    import matplotlib.pyplot as plt
    plt.close('all')


if __name__ == '__main__':
    main()
