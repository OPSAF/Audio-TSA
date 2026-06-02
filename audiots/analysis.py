"""Time series analysis module."""

import numpy as np
from scipy import signal


def compute_acf(x, nlags=40):
    """Compute autocorrelation function (ACF)."""
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)

    lags = np.arange(nlags)
    acf_vals = np.zeros(nlags)
    for lag in lags:
        if lag == 0:
            acf_vals[lag] = 1.0
        else:
            acf_vals[lag] = np.corrcoef(x_centered[lag:], x_centered[:-lag])[0, 1] if var > 0 else 0

    ci = 1.96 / np.sqrt(n)
    return lags, acf_vals, ci


def compute_pacf(x, nlags=40):
    """Compute partial autocorrelation function (PACF) using Yule-Walker."""
    n = len(x)
    lags = np.arange(nlags)
    pacf_vals = np.zeros(nlags)
    acf_raw = np.zeros(nlags + 1)

    x_centered = x - np.mean(x)
    var = np.var(x)
    for lag in range(nlags + 1):
        if lag == 0:
            acf_raw[0] = 1.0
        else:
            acf_raw[lag] = np.corrcoef(x_centered[lag:], x_centered[:-lag])[0, 1] if var > 0 else 0

    pacf_vals[0] = 1.0
    phi = np.zeros((nlags, nlags))
    phi[0, 0] = acf_raw[1]
    pacf_vals[1] = phi[0, 0]

    for k in range(2, nlags):
        numerator = acf_raw[k]
        for j in range(1, k):
            numerator -= phi[k - 2, j - 1] * acf_raw[k - j]
        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k - 2, j - 1] * acf_raw[j]
        if abs(denominator) < 1e-10:
            phi[k - 1, k - 1] = 0
        else:
            phi[k - 1, k - 1] = numerator / denominator

        for j in range(1, k):
            phi[k - 1, j - 1] = phi[k - 2, j - 1] - phi[k - 1, k - 1] * phi[k - 2, k - j - 1]

        pacf_vals[k] = phi[k - 1, k - 1]

    ci = 1.96 / np.sqrt(n)
    return lags, pacf_vals, ci


def fft_analysis(y, sr):
    """Perform FFT analysis on waveform."""
    from scipy.fft import fft, fftfreq
    n = len(y)
    Y = fft(y)
    mag = np.abs(Y[:n // 2]) / n
    phase = np.angle(Y[:n // 2])
    freqs = fftfreq(n, 1 / sr)[:n // 2]
    return freqs, mag, phase


def stft_analysis(y, sr, n_fft=2048, hop_length=512):
    """Perform STFT analysis."""
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
                             nfft=n_fft, window='hann')
    return f, t, np.abs(Zxx)


def analyze_periodicity(y, sr, nlags=40):
    """Analyze periodicity of a time series."""
    freqs, mag, _ = fft_analysis(y, sr)
    dominant_idx = np.argmax(mag[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

    lags, acf_vals, ci = compute_acf(y[:min(len(y), sr * 2)], nlags)

    peaks = []
    for i in range(2, len(acf_vals) - 1):
        if acf_vals[i] > ci and acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1]:
            peaks.append((i, acf_vals[i]))

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:3]

    return {
        'dominant_frequency': dominant_freq,
        'dominant_period_seconds': period,
        'dominant_period_samples': int(period * sr) if period < np.inf else -1,
        'acf_peaks': peaks,
        'confidence_interval': ci
    }


def compute_spectral_flatness(mag_spec):
    """Compute spectral flatness (Wiener entropy)."""
    if np.max(mag_spec) < 1e-10:
        return 1.0
    geometric_mean = np.exp(np.mean(np.log(mag_spec + 1e-10)))
    arithmetic_mean = np.mean(mag_spec)
    return geometric_mean / (arithmetic_mean + 1e-10)


def analyze_complexity(y):
    """Analyze signal complexity using sample entropy approximation."""
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)

    m = 2
    r_scale = 0.2 * np.std(y)

    def _count_matches(data, m_val, r_val):
        count = 0
        n = len(data)
        for i in range(n - m_val):
            template = data[i:i + m_val]
            for j in range(i + 1, n - m_val):
                if np.max(np.abs(template - data[j:j + m_val])) <= r_val:
                    count += 1
        return count

    n = min(len(y), 500)
    downsampled = y[::max(1, len(y) // n)][:n]
    downsampled = (downsampled - np.mean(downsampled)) / (np.std(downsampled) + 1e-10)

    a = _count_matches(downsampled, m + 1, r_scale)
    b = _count_matches(downsampled, m, r_scale)
    sampen = -np.log((a + 1) / (b + 1)) if b > 0 else float('inf')

    return {
        'zero_crossing_rate': zero_crossings,
        'sample_entropy': sampen,
        'is_tonal': sampen < 0.5
    }