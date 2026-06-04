"""Visualization module."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_waveform(t, y, save_path=None, title='Audio Waveform'):
    """Plot audio waveform."""
    fig = plt.figure(figsize=(12, 4))
    plt.plot(t, y, color='#1f77b4')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_fft(freqs, mag, save_path=None, title='FFT Spectrum'):
    """Plot FFT magnitude spectrum."""
    fig = plt.figure(figsize=(12, 4))
    plt.plot(freqs, mag, color='#ff7f0e')
    plt.title(title, fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(0, 5000)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_spectrogram(freqs, times, spec, save_path=None, title='Spectrogram'):
    """Plot spectrogram."""
    fig = plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, spec, cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_stft(freqs, times, spec, save_path=None, title='STFT Spectrogram'):
    """Plot STFT spectrogram."""
    fig = plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, np.abs(spec), cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_mel_spectrogram(mel_freqs, times, mel_spec, save_path=None, title='Mel Spectrogram'):
    """Plot Mel spectrogram."""
    fig = plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, mel_freqs, mel_spec, cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(label='dB')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_mfcc(mfcc, times, save_path=None, title='MFCC Coefficients'):
    """Plot MFCC heatmap."""
    fig = plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, np.arange(mfcc.shape[0]), mfcc, cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('MFCC Coefficient', fontsize=12)
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_periodicity(y, sr, period_info=None, save_path=None, title='Periodicity Analysis'):
    """Plot periodogram for periodicity analysis.

    Parameters
    ----------
    y : ndarray
        Raw audio waveform.
    sr : int
        Sample rate.
    period_info : dict, optional
        Periodicity analysis results from analysis.analyze_periodicity().
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    from scipy.fft import fft, fftfreq
    n = len(y)
    Y = fft(y)
    periodogram = np.abs(Y[:n // 2]) / n
    periodogram_freqs = fftfreq(n, 1 / sr)[:n // 2]

    fig = plt.figure(figsize=(12, 4))
    plt.plot(periodogram_freqs, periodogram, color='#9467bd')

    # Mark dominant frequency if period_info is provided
    if period_info is not None and 'dominant_frequency' in period_info:
        dom_freq = period_info['dominant_frequency']
        dom_idx = np.argmin(np.abs(periodogram_freqs - dom_freq))
        dom_power = periodogram[dom_idx]
        plt.axvline(x=dom_freq, color='red', linestyle='--', alpha=0.7,
                    label=f'Dominant: {dom_freq:.1f} Hz')
        plt.plot(dom_freq, dom_power, 'ro', markersize=8)
        plt.legend()

    plt.title(title, fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Power', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_model_error_comparison(results, save_path=None, title='Prediction Error Comparison'):
    """Plot model error comparison bar chart."""
    models = list(results.keys())
    rmse_values = [results[m][1].get('RMSE', np.nan) for m in models]
    mae_values = [results[m][1].get('MAE', np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig = plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, rmse_values, width, label='RMSE')
    rects2 = plt.bar(x + width/2, mae_values, width, label='MAE')

    plt.title(title, fontsize=14)
    plt.xticks(x, models)
    plt.legend()

    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_band_error_heatmap(band_results, save_path=None, title='Band Error Heatmap'):
    """Plot band error heatmap from detailed band results.

    Parameters
    ----------
    band_results : dict
        Output of band_analysis.analyze_band_predictability().
        Structure: {band_key: {'info': {...}, 'predictions': {model: {'metrics': {...}}}}}
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    bands = list(band_results.keys())
    models = ['ARIMA', 'HMM', 'LSTM', 'Transformer']

    rmse_matrix = []
    for band in bands:
        row = []
        predictions = band_results[band]['predictions']
        for model in models:
            if model in predictions:
                row.append(predictions[model]['metrics'].get('RMSE', np.nan))
            else:
                row.append(np.nan)
        rmse_matrix.append(row)

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(rmse_matrix, cmap='viridis', interpolation='nearest')
    plt.title(title, fontsize=14)
    plt.xticks(np.arange(len(models)), models)
    plt.yticks(np.arange(len(bands)), [band_results[b]['info']['name'] for b in bands])
    plt.colorbar(label='RMSE')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_band_error_bars(band_results, save_path=None, title='Band-wise Error Comparison'):
    """Plot band error comparison bars from detailed band results.

    Parameters
    ----------
    band_results : dict
        Output of band_analysis.analyze_band_predictability().
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    bands = list(band_results.keys())
    band_names = [band_results[b]['info']['name'] for b in bands]

    # Compute average RMSE per band from the prediction results
    avg_rmse = []
    for band in bands:
        predictions = band_results[band]['predictions']
        rmse_vals = [m['metrics']['RMSE'] for m in predictions.values()
                     if 'RMSE' in m['metrics'] and not np.isnan(m['metrics']['RMSE'])]
        avg_rmse.append(np.mean(rmse_vals) if rmse_vals else np.nan)

    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(band_names, avg_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title(title, fontsize=14)
    plt.ylabel('RMSE', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_acf_pacf(lags, acf_vals, pacf_vals, ci, save_path=None, title=None):
    """Plot ACF and PACF."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.stem(lags, acf_vals, basefmt='b-')
    ax1.axhline(y=ci, color='r', linestyle='--', label='95% CI')
    ax1.axhline(y=-ci, color='r', linestyle='--')
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=12)
    ax1.set_xlabel('Lag', fontsize=10)
    ax1.legend()

    ax2.stem(lags, pacf_vals, basefmt='b-')
    ax2.axhline(y=ci, color='r', linestyle='--', label='95% CI')
    ax2.axhline(y=-ci, color='r', linestyle='--')
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12)
    ax2.set_xlabel('Lag', fontsize=10)
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_prediction_comparison(results, forecast_horizon=20, save_path=None, title='Prediction Comparison'):
    """Plot prediction comparison across models.

    Parameters
    ----------
    results : dict
        Model prediction results {model_name: (forecast, metrics, true)}.
    forecast_horizon : int
        Number of forecast steps (used to align x-axis).
    save_path : str, optional
        Path to save the figure.
    title : str
        Suptitle for the figure.
    """
    fig = plt.figure(figsize=(14, 8))
    plt.suptitle(title, fontsize=14)

    models = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (model_name, (forecast, metrics, true)) in enumerate(results.items()):
        t = np.arange(len(true))
        plt.subplot(2, 2, i + 1)
        plt.plot(t, true, label='True', color='black', linestyle='--')
        plt.plot(t, forecast, label='Predicted', color=colors[i])
        plt.title(f'{model_name}\nRMSE: {metrics.get("RMSE", "N/A"):.3f}', fontsize=12)
        plt.xlabel('Time Step', fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_error_bar(results, save_path=None):
    """Plot error comparison bar chart."""
    models = list(results.keys())
    rmse_values = [results[m][1].get('RMSE', np.nan) for m in models]
    mae_values = [results[m][1].get('MAE', np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig = plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, rmse_values, width, label='RMSE')
    rects2 = plt.bar(x + width/2, mae_values, width, label='MAE')

    plt.title('Model Error Comparison', fontsize=14)
    plt.xticks(x, models)
    plt.legend()

    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def plot_band_errors(band_summary, save_path=None):
    """Plot band prediction errors."""
    bands = list(band_summary.keys())
    band_names = [band_summary[b]['name'] for b in bands]
    avg_rmse = [band_summary[b]['avg_rmse'] for b in bands]

    fig = plt.figure(figsize=(10, 6))
    bars = plt.bar(band_names, avg_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title('Band-wise Error Comparison', fontsize=14)
    plt.ylabel('RMSE', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def generate_report_plots(analysis_results, output_dir='outputs'):
    """Generate all visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    if 'waveform' in analysis_results:
        plot_waveform(analysis_results['waveform']['t'],
                      analysis_results['waveform']['y'],
                      os.path.join(output_dir, 'waveform.png'))

    if 'fft' in analysis_results:
        plot_fft(analysis_results['fft']['freqs'],
                 analysis_results['fft']['mag'],
                 os.path.join(output_dir, 'fft_spectrum.png'))

    if 'stft' in analysis_results:
        plot_spectrogram(analysis_results['stft']['freqs'],
                         analysis_results['stft']['times'],
                         analysis_results['stft']['spec'],
                         os.path.join(output_dir, 'stft_spectrogram.png'))

    if 'mel' in analysis_results:
        plot_mel_spectrogram(analysis_results['mel']['freqs'],
                             analysis_results['mel']['times'],
                             analysis_results['mel']['spec'],
                             os.path.join(output_dir, 'mel_spectrogram.png'))

    if 'acf_pacf' in analysis_results:
        plot_acf_pacf(analysis_results['acf_pacf']['lags'],
                      analysis_results['acf_pacf']['acf'],
                      analysis_results['acf_pacf']['pacf'],
                      analysis_results['acf_pacf']['ci'],
                      os.path.join(output_dir, 'acf_pacf.png'))

    if 'predictions' in analysis_results:
        plot_prediction_comparison(analysis_results['predictions'],
                                   os.path.join(output_dir, 'prediction_comparison.png'))
        plot_error_bar(analysis_results['predictions'],
                       os.path.join(output_dir, 'error_comparison.png'))

    if 'band_summary' in analysis_results:
        plot_band_errors(analysis_results['band_summary'],
                         os.path.join(output_dir, 'band_errors.png'))

    return [f for f in os.listdir(output_dir) if f.endswith('.png')]


# ---------------------------------------------------------------------------
# Dynamics / Trend visualizations
# ---------------------------------------------------------------------------


def plot_dynamics_trends(
    dynamics,
    segments=None,
    save_path=None,
    title="Audio Dynamics — Trend Analysis",
):
    """
    4-panel plot of energy, brightness, complexity, and rhythm trends.

    Parameters
    ----------
    dynamics : dict   Output of ``dynamics.extract_dynamics()``.
    segments : dict, optional  Output of ``dynamics.detect_structural_segments()``.
    save_path : str, optional
    title : str
    """
    times = dynamics["times"]
    energy = dynamics["energy"]
    brightness = dynamics["brightness"]
    complexity = dynamics["complexity"]
    rhythm = dynamics["rhythm"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    trends = [
        (axes[0], energy, "#e74c3c", "Energy (RMS)"),
        (axes[1], brightness, "#3498db", "Brightness (Spectral Centroid, Hz)"),
        (axes[2], complexity, "#2ecc71", "Complexity (Spectral Entropy)"),
        (axes[3], rhythm, "#9b59b6", "Rhythm Density (Onset Strength)"),
    ]

    for ax, data, color, ylabel in trends:
        ax.plot(times, data, color=color, linewidth=1.2, alpha=0.9)
        ax.fill_between(times, 0, data, alpha=0.08, color=color)
        ax.set_ylabel(ylabel, fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, None)

    # Overlay segment shading if provided
    if segments is not None:
        seg_colors = {
            "climax": "#e74c3c",
            "calm": "#3498db",
            "buildup": "#e67e22",
            "release": "#1abc9c",
            "transition": "#9b59b6",
        }
        labels = segments.get("labels", [])
        hop = dynamics["params"]["hop_size"]
        for i, lbl in enumerate(labels):
            if lbl in seg_colors and i < len(times):
                t_start = times[i] - hop / 2
                t_end = times[i] + hop / 2
                for ax in axes:
                    ax.axvspan(t_start, t_end, alpha=0.12,
                               color=seg_colors[lbl], edgecolor="none")

        # Legend for segment colours
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor=c, alpha=0.3, label=segments.get("label_descriptions", {}).get(k, k))
            for k, c in seg_colors.items()
        ]
        axes[0].legend(handles=legend_elems, loc="upper right", fontsize=7, ncol=3)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_dynamics_dual(
    dyn1,
    dyn2,
    sim_result=None,
    save_path=None,
    title="Dual Audio — Dynamics Comparison",
):
    """
    4-panel overlay plot comparing dynamics of two audio files.

    Parameters
    ----------
    dyn1, dyn2 : dict   Outputs of ``dynamics.extract_dynamics()``.
    sim_result : dict, optional  Output of ``dynamics.compute_dynamics_similarity()``.
    save_path : str, optional
    title : str
    """
    times1 = dyn1["times"]
    times2 = dyn2["times"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    trends = [
        (axes[0], "energy", "Energy (RMS)", "#e74c3c"),
        (axes[1], "brightness", "Brightness (Spectral Centroid)", "#3498db"),
        (axes[2], "complexity", "Complexity (Spectral Entropy)", "#2ecc71"),
        (axes[3], "rhythm", "Rhythm Density (Onset Strength)", "#9b59b6"),
    ]

    for ax, key, ylabel, color in trends:
        ax.plot(times1, dyn1[key], color=color, linewidth=1.2, alpha=0.9,
                label="Audio 1")
        ax.plot(times2, dyn2[key], color="gray", linewidth=1.0, alpha=0.7,
                linestyle="--", label="Audio 2")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.2)

        # Annotate similarity score if available
        if sim_result and key in sim_result.get("per_trend", {}):
            ss = sim_result["per_trend"][key]["structural_score"]
            ax.text(
                0.98, 0.95, f"struct sim: {ss:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_dynamics_summary(
    dynamics,
    segments=None,
    save_path=None,
    title="Audio Dynamics Summary",
):
    """
    Combined summary: trends + segment bar + statistics table.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1],
                           hspace=0.35, wspace=0.30)

    times = dynamics["times"]

    # Top-left: 4 trends
    ax_trends = fig.add_subplot(gs[0, :])
    ax_trends.plot(times, dynamics["energy_norm"], color="#e74c3c",
                   linewidth=1.2, label="Energy", alpha=0.85)
    ax_trends.plot(times, dynamics["brightness_norm"], color="#3498db",
                   linewidth=1.2, label="Brightness", alpha=0.85)
    ax_trends.plot(times, dynamics["complexity_norm"], color="#2ecc71",
                   linewidth=1.2, label="Complexity", alpha=0.85)
    ax_trends.plot(times, dynamics["rhythm_norm"], color="#9b59b6",
                   linewidth=1.2, label="Rhythm", alpha=0.85)
    ax_trends.axhline(y=0, color="black", linewidth=0.5, linestyle=":")
    ax_trends.set_ylabel("Z-score", fontsize=10)
    ax_trends.legend(loc="upper right", fontsize=8, ncol=4)
    ax_trends.grid(alpha=0.2)
    ax_trends.set_title("Normalised trend curves", fontsize=11)

    # Bottom-left: segment bar
    ax_bar = fig.add_subplot(gs[1, 0])
    if segments is not None:
        labels = segments.get("labels", [])
        label_map = {"climax": 4, "buildup": 3, "transition": 2,
                     "sustained": 1, "calm": 0, "release": 1}
        y_bar = np.array([label_map.get(l, 1) for l in labels])
        n = len(y_bar)
        # Create a coloured bar
        cmap_colors = ["#3498db", "#95a5a6", "#9b59b6", "#e67e22", "#e74c3c"]
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap_colors)
        ax_bar.pcolormesh(
            np.arange(n + 1), [0, 1],
            y_bar.reshape(1, -1),
            cmap=cmap, vmin=0, vmax=4, shading="flat",
        )
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("Window index", fontsize=10)
        ax_bar.set_title("Structural segments", fontsize=11)

        # Legend
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#e74c3c", label="Climax"),
            Patch(facecolor="#e67e22", label="Buildup"),
            Patch(facecolor="#9b59b6", label="Transition"),
            Patch(facecolor="#95a5a6", label="Sustained"),
            Patch(facecolor="#3498db", label="Calm"),
        ]
        ax_bar.legend(handles=legend_elems, loc="upper right",
                      fontsize=7, ncol=5)
    else:
        ax_bar.text(0.5, 0.5, "No segment data", ha="center", va="center",
                    transform=ax_bar.transAxes)
        ax_bar.set_title("Structural segments", fontsize=11)

    # Bottom-right: stats table
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    from audiots.dynamics import summarize_dynamics
    summary = summarize_dynamics(dynamics)
    col_labels = ["Trend", "Mean", "Std", "Peaks", "Direction"]
    cell_text = []
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = summary[key]
        cell_text.append([
            key.capitalize(),
            f"{s['mean']:.3f}",
            f"{s['std']:.3f}",
            str(s["n_peaks"]),
            s["trend_direction"],
        ])
    tbl = ax_table.table(
        cellText=cell_text, colLabels=col_labels, loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", fontweight="bold")
    ax_table.set_title("Trend statistics", fontsize=11)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def save_all_figures(fig_dict, output_dir='outputs'):
    """Save all figures in the dictionary to the output directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    for fig_name, fig in fig_dict.items():
        if fig is not None:
            filename = f"{fig_name}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
            plt.close(fig)


# ---------------------------------------------------------------------------
# Volatility Layer visualizations
# ---------------------------------------------------------------------------


def plot_volatility_layer(
    dynamics: dict,
    vol_layer: dict,
    save_path: str = None,
    title: str = "Volatility Layer — Rolling Volatility on Audio Dynamics",
) -> "plt.Figure":
    """
    4-panel plot: each trend overlaid with its rolling volatility band.

    Parameters
    ----------
    dynamics : dict     Output of ``dynamics.extract_dynamics()``.
    vol_layer : dict    Output of ``volatility.compute_volatility_layer()``.
    save_path : str, optional
    title : str
    """
    times = dynamics["times"]
    vol_times = vol_layer["times"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        (axes[0], "energy", "Energy (RMS)", "#e74c3c"),
        (axes[1], "brightness", "Brightness (Spectral Centroid, Hz)", "#3498db"),
        (axes[2], "complexity", "Complexity (Spectral Entropy)", "#2ecc71"),
        (axes[3], "rhythm", "Rhythm Density (Onset Strength)", "#9b59b6"),
    ]

    for ax, key, ylabel, color in panels:
        trend = dynamics[key]
        vol = vol_layer[f"{key}_vol"]

        # Trend line
        ax.plot(times, trend, color=color, linewidth=1.2, alpha=0.85,
                label=f"{key} trend")

        # Volatility band: ±1 rolling std
        ax.fill_between(
            vol_times,
            np.maximum(0, trend - vol),
            trend + vol,
            alpha=0.15, color=color, edgecolor="none",
            label=f"±1 rolling σ",
        )

        # Volatility overlay (on secondary axis-like scaling)
        ax2 = ax.twinx()
        ax2.plot(vol_times, vol, color="gray", linewidth=0.8, alpha=0.5,
                 linestyle="--", label="rolling vol")
        ax2.set_ylabel("Vol (σ)", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)

        ax.set_ylabel(ylabel, fontsize=10, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, None)

        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                  fontsize=7)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_garch_diagnostics(
    vol_layer: dict,
    trend_key: str = "energy",
    save_path: str = None,
    title: str = None,
) -> "plt.Figure":
    """
    GARCH model diagnostics for a single trend dimension.

    Shows: (a) original trend, (b) conditional volatility from GARCH,
    (c) standardised residuals.

    Parameters
    ----------
    vol_layer : dict    Output of ``volatility.compute_volatility_layer()``.
    trend_key : str     Which trend's GARCH model to display.
    save_path : str, optional
    title : str, optional
    """
    garch = vol_layer.get("garch_models", {}).get(trend_key, {})
    cond_vol = garch.get("conditional_volatility")
    persistence = garch.get("persistence", None)
    backend = garch.get("backend", "?")
    converged = garch.get("converged", False)

    if title is None:
        title = (
            f"GARCH Diagnostics — {trend_key} trend "
            f"(α+β={persistence:.3f}, backend={backend})"
            if persistence is not None
            else f"GARCH Diagnostics — {trend_key} trend"
        )

    times = vol_layer["times"]
    vol_series = vol_layer.get(f"{trend_key}_vol")
    n = len(times)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # (a) Rolling volatility
    ax = axes[0]
    ax.plot(times, vol_series, color="#e74c3c", linewidth=1.2, alpha=0.9,
            label="Rolling volatility (σ)")
    ax.fill_between(times, 0, vol_series, alpha=0.10, color="#e74c3c")
    ax.set_ylabel("Rolling Volatility", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # (b) GARCH conditional volatility
    ax = axes[1]
    if cond_vol is not None and len(cond_vol) > 0:
        # Align lengths
        cv = np.asarray(cond_vol).ravel()
        if len(cv) == n:
            ax.plot(times, cv, color="#3498db", linewidth=1.2, alpha=0.9,
                    label=f"GARCH conditional σ (converged={converged})")
            ax.fill_between(times, 0, cv, alpha=0.10, color="#3498db")
        else:
            ax.text(0.5, 0.5, "Conditional volatility length mismatch",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray")
    else:
        ax.text(0.5, 0.5, "GARCH model not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
    ax.set_ylabel("Conditional Volatility", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # (c) GARCH parameter summary as text
    ax = axes[2]
    ax.axis("off")
    param_text = (
        f"GARCH Model:  ω = {garch.get('omega', np.nan):.6f}\n"
        f"              α = {garch.get('alpha', np.nan):.4f}  "
        f"(ARCH term — reaction to shocks)\n"
        f"              β = {garch.get('beta', np.nan):.4f}  "
        f"(GARCH term — volatility persistence)\n"
        f"              α+β = {garch.get('persistence', np.nan):.4f}  "
        f"(persistence — closeness to 1 = long memory)\n"
        f"              Half-life = "
        f"{garch.get('half_life', np.inf):.1f} steps\n"
        f"              Backend: {backend}  |  Converged: {converged}\n"
        f"              Interpretation: "
        + (
            "High volatility persistence — shocks decay slowly"
            if persistence and persistence > 0.9
            else "Moderate persistence — volatility mean-reverts steadily"
            if persistence and persistence > 0.5
            else "Low persistence — volatility shocks are short-lived"
            if persistence is not None
            else "Insufficient data for reliable GARCH fit"
        )
    )
    ax.text(0.05, 0.5, param_text, fontsize=9, fontfamily="monospace",
            va="center", transform=ax.transAxes)

    axes[-1].set_xlabel("Window index / Time", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_volatility_comparison(
    vol1: dict,
    vol2: dict,
    sim_result: dict = None,
    save_path: str = None,
    title: str = "Dual Audio — Volatility Comparison",
) -> "plt.Figure":
    """
    4-panel overlay comparing volatility profiles of two audio files.

    Parameters
    ----------
    vol1, vol2 : dict        Outputs of ``volatility.compute_volatility_layer()``.
    sim_result : dict, optional  Output of ``volatility.compute_volatility_similarity()``.
    save_path : str, optional
    title : str
    """
    times1 = vol1["times"]
    times2 = vol2["times"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    panels = [
        (axes[0], "energy", "Energy Volatility", "#e74c3c"),
        (axes[1], "brightness", "Brightness Volatility", "#3498db"),
        (axes[2], "complexity", "Complexity Volatility", "#2ecc71"),
        (axes[3], "rhythm", "Rhythm Volatility", "#9b59b6"),
    ]

    for ax, key, ylabel, color in panels:
        a = vol1[f"{key}_vol"]
        b = vol2[f"{key}_vol"]

        ax.plot(times1, a, color=color, linewidth=1.2, alpha=0.9,
                label="Audio 1")
        ax.plot(times2, b, color="gray", linewidth=1.0, alpha=0.7,
                linestyle="--", label="Audio 2")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.2)

        # Annotate similarity if available
        if sim_result and key in sim_result.get("per_trend", {}):
            ss = sim_result["per_trend"][key]["structural_score"]
            ax.text(
                0.98, 0.95, f"vol sim: {ss:.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_dynamics_analysis_summary(
    dyn_analysis: dict,
    save_path: str = None,
    title: str = "Audio Dynamics Analysis — Trend + Volatility Summary",
) -> "plt.Figure":
    """
    Combined dashboard: trend stats table + volatility stats table + GARCH params.

    Parameters
    ----------
    dyn_analysis : dict   Output of ``volatility.analyze_audio_dynamics()``.
    save_path : str, optional
    title : str
    """
    trend_summary = dyn_analysis.get("trend_summary", {})
    vol_summary = dyn_analysis.get("volatility_summary", {})

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # ---- Left: Trend statistics table ----
    ax_trend = fig.add_axes([0.03, 0.08, 0.30, 0.82])
    ax_trend.axis("off")
    ax_trend.set_title("Trend Layer", fontsize=12, fontweight="bold", loc="left")

    col_labels = ["Trend", "Mean", "Std", "Peaks", "Direction"]
    cell_text = []
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = trend_summary.get(key, {})
        cell_text.append([
            key.capitalize(),
            f"{s.get('mean', 0):.3f}",
            f"{s.get('std', 0):.3f}",
            str(s.get("n_peaks", 0)),
            s.get("trend_direction", "?"),
        ])
    tbl = ax_trend.table(
        cellText=cell_text, colLabels=col_labels, loc="upper center",
        cellLoc="center", colWidths=[0.18, 0.18, 0.18, 0.14, 0.18],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", fontweight="bold")

    # ---- Middle: Volatility statistics table ----
    ax_vol = fig.add_axes([0.36, 0.08, 0.30, 0.82])
    ax_vol.axis("off")
    ax_vol.set_title("Volatility Layer", fontsize=12, fontweight="bold", loc="left")

    col_labels_vol = ["Trend", "MeanVol", "VoV", "Regime"]
    cell_text_vol = []
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = vol_summary.get(key, {})
        cell_text_vol.append([
            key.capitalize(),
            f"{s.get('mean_vol', 0):.5f}",
            f"{s.get('vol_of_vol', 0):.3f}",
            s.get("volatility_regime", "?"),
        ])
    tbl2 = ax_vol.table(
        cellText=cell_text_vol, colLabels=col_labels_vol, loc="upper center",
        cellLoc="center", colWidths=[0.18, 0.24, 0.18, 0.18],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1.0, 1.5)
    for (row, col), cell in tbl2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", fontweight="bold")

    # ---- Right: GARCH parameters ----
    ax_garch = fig.add_axes([0.69, 0.08, 0.29, 0.82])
    ax_garch.axis("off")
    ax_garch.set_title("GARCH(1,1) Parameters", fontsize=12, fontweight="bold",
                       loc="left")

    col_labels_g = ["Trend", "ω", "α", "β", "α+β"]
    cell_text_g = []
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = vol_summary.get(key, {})
        if s.get("garch_converged"):
            cell_text_g.append([
                key.capitalize(),
                f"{s.get('garch_omega', 0):.4f}",
                f"{s.get('garch_alpha', 0):.3f}",
                f"{s.get('garch_beta', 0):.3f}",
                f"{s.get('garch_persistence', 0):.3f}",
            ])
        else:
            cell_text_g.append([
                key.capitalize(), "—", "—", "—", "no fit",
            ])
    tbl3 = ax_garch.table(
        cellText=cell_text_g, colLabels=col_labels_g, loc="upper center",
        cellLoc="center", colWidths=[0.18, 0.18, 0.13, 0.13, 0.13],
    )
    tbl3.auto_set_font_size(False)
    tbl3.set_fontsize(9)
    tbl3.scale(1.0, 1.5)
    for (row, col), cell in tbl3.get_celld().items():
        if row == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig