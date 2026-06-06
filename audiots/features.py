"""Feature extraction module for audio signals."""

import numpy as np
from scipy.signal import stft
from scipy.fft import fft


def compute_waveform(y, sr):
    """Returns the raw waveform time series."""
    t = np.linspace(0, len(y) / sr, len(y), endpoint=False)
    return t, y


def compute_fft(y, sr):
    """Compute magnitude spectrum via FFT."""
    n = len(y)
    Y = fft(y)
    mag = np.abs(Y[:n // 2]) / n
    freqs = np.fft.fftfreq(n, 1 / sr)[:n // 2]
    return freqs, mag


def compute_stft(y, sr, n_fft=2048, hop_length=512, win_length=None):
    """Compute Short-Time Fourier Transform."""
    if win_length is None:
        win_length = n_fft
    f, t, Zxx = stft(y, fs=sr, nperseg=win_length, noverlap=n_fft - hop_length,
                     nfft=n_fft, window='hann')
    mag = np.abs(Zxx)
    return f, t, mag


def compute_mel_spectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Compute Mel spectrogram."""
    try:
        import librosa
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        times = librosa.frames_to_time(
            np.arange(mel_spec_db.shape[1]), sr=sr, hop_length=hop_length)
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
        return mel_freqs, times, mel_spec_db
    except ImportError:
        f, t, Zxx = compute_stft(y, sr, n_fft=n_fft, hop_length=hop_length)
        mel_filters = _mel_filterbank(n_mels, n_fft // 2 + 1, sr)
        mel_spec = np.dot(mel_filters, Zxx)
        mel_spec_db = 20 * np.log10(mel_spec / (np.max(mel_spec) + 1e-8) + 1e-8)
        mel_freqs = np.linspace(0, sr / 2, n_mels)
        return mel_freqs, t, mel_spec_db


def compute_mfcc(y, sr, n_mfcc=20, n_mels=128, n_fft=2048, hop_length=512):
    """Compute MFCC features."""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels)
        times = librosa.frames_to_time(
            np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)
        return mfcc, times
    except ImportError:
        mel_freqs, times, mel_spec = compute_mel_spectrogram(
            y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        from scipy.linalg import dct
        mfcc = dct(mel_spec, axis=0, type=2, norm='ortho')[:n_mfcc]
        return mfcc, times


def _mel_filterbank(n_mels, n_fft_bins, sr):
    """Construct Mel filterbank matrix."""
    mel_freqs = np.linspace(2595 * np.log10(1 + 0 / 700),
                            2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
    mel_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
    bin_freqs = np.linspace(0, sr / 2, n_fft_bins)
    filters = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        for j in range(n_fft_bins):
            if mel_freqs[i] <= bin_freqs[j] <= mel_freqs[i + 1]:
                filters[i, j] = (bin_freqs[j] - mel_freqs[i]) / (mel_freqs[i + 1] - mel_freqs[i] + 1e-8)
            elif mel_freqs[i + 1] < bin_freqs[j] <= mel_freqs[i + 2]:
                filters[i, j] = (mel_freqs[i + 2] - bin_freqs[j]) / (mel_freqs[i + 2] - mel_freqs[i + 1] + 1e-8)
    return filters


def split_mel_bands(mel_spec, n_mels=128):
    """Split Mel spectrogram into low, mid, high frequency bands."""
    third = n_mels // 3
    return {
        'low': mel_spec[:third, :],
        'mid': mel_spec[third:2 * third, :],
        'high': mel_spec[2 * third:, :]
    }