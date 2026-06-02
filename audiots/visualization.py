"""Visualization module."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_waveform(t, y, save_path=None, title='Audio Waveform'):
    """Plot audio waveform."""
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, color='#1f77b4')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_fft(freqs, mag, save_path=None, title='FFT Spectrum'):
    """Plot FFT magnitude spectrum."""
    plt.figure(figsize=(12, 4))
    plt.plot(freqs, mag, color='#ff7f0e')
    plt.title(title, fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(0, 5000)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_spectrogram(freqs, times, spec, save_path=None, title='Spectrogram'):
    """Plot spectrogram."""
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, spec, cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_mel_spectrogram(mel_freqs, times, mel_spec, save_path=None, title='Mel Spectrogram'):
    """Plot Mel spectrogram."""
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, mel_freqs, mel_spec, cmap='viridis', shading='gouraud')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.colorbar(label='dB')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_acf_pacf(lags, acf_vals, pacf_vals, ci, save_path=None):
    """Plot ACF and PACF."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    ax1.stem(lags, acf_vals, basefmt='b-', use_line_collection=True)
    ax1.axhline(y=ci, color='r', linestyle='--', label='95% CI')
    ax1.axhline(y=-ci, color='r', linestyle='--')
    ax1.set_title('Autocorrelation Function (ACF)', fontsize=12)
    ax1.set_xlabel('Lag', fontsize=10)
    ax1.legend()

    ax2.stem(lags, pacf_vals, basefmt='b-', use_line_collection=True)
    ax2.axhline(y=ci, color='r', linestyle='--', label='95% CI')
    ax2.axhline(y=-ci, color='r', linestyle='--')
    ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12)
    ax2.set_xlabel('Lag', fontsize=10)
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_prediction_comparison(results, save_path=None):
    """Plot prediction comparison across models."""
    plt.figure(figsize=(14, 8))

    models = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (model_name, (forecast, metrics, true)) in enumerate(results.items()):
        t = np.arange(len(true))
        plt.subplot(2, 2, i + 1)
        plt.plot(t, true, label='真实值', color='black', linestyle='--')
        plt.plot(t, forecast, label='预测值', color=colors[i])
        plt.title(f'{model_name}\nRMSE: {metrics.get("RMSE", "N/A"):.3f}', fontsize=12)
        plt.xlabel('时间步', fontsize=10)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_error_bar(results, save_path=None):
    """Plot error comparison bar chart."""
    models = list(results.keys())
    rmse_values = [results[m][1].get('RMSE', np.nan) for m in models]
    mae_values = [results[m][1].get('MAE', np.nan) for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width/2, rmse_values, width, label='RMSE')
    rects2 = plt.bar(x + width/2, mae_values, width, label='MAE')

    plt.title('模型预测误差对比', fontsize=14)
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
    plt.close()


def plot_band_errors(band_summary, save_path=None):
    """Plot band prediction errors."""
    bands = list(band_summary.keys())
    band_names = [band_summary[b]['name'] for b in bands]
    avg_rmse = [band_summary[b]['avg_rmse'] for b in bands]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(band_names, avg_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title('各频带平均预测误差', fontsize=14)
    plt.ylabel('RMSE', fontsize=12)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_dtw_alignment(path, x, y, save_path=None):
    """Plot DTW alignment path."""
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((len(y), len(x))), cmap='gray', extent=[0, len(x), 0, len(y)])

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    plt.plot(path_x, path_y, color='red', linewidth=2)

    plt.title('DTW Alignment Path', fontsize=14)
    plt.xlabel('Sequence X', fontsize=12)
    plt.ylabel('Sequence Y', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


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

    if 'dtw' in analysis_results:
        plot_dtw_alignment(analysis_results['dtw']['path'],
                           analysis_results['dtw']['x'],
                           analysis_results['dtw']['y'],
                           os.path.join(output_dir, 'dtw_alignment.png'))

    return [f for f in os.listdir(output_dir) if f.endswith('.png')]