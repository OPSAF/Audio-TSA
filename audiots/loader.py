"""Audio file loading utilities."""

import numpy as np
import soundfile as sf


def load_audio(filepath, target_sr=16000):
    """
    Load audio file (wav/mp3) and convert to mono with target sample rate.
    Returns (waveform, sample_rate).
    """
    try:
        y, sr = sf.read(filepath, dtype='float32')
    except Exception:
        import librosa
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return y, sr

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(sr, target_sr)
            y = resample_poly(y, target_sr // g, sr // g)
        sr = target_sr

    return y.astype(np.float32), sr


def generate_sample_audio(duration=3.0, sr=16000, freq=440.0):
    """Generate a synthetic audio sample for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.6 * np.sin(2 * np.pi * freq * t)
    y += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    y += 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    y += 0.4 * np.sin(2 * np.pi * 100 * t)
    y += 0.05 * np.random.randn(len(t))

    envelope = np.linspace(0, 1, int(sr * 0.05))
    envelope_rev = np.linspace(1, 0, int(sr * 0.05))
    y[:len(envelope)] *= envelope
    y[-len(envelope_rev):] *= envelope_rev

    y /= np.max(np.abs(y)) + 1e-8
    return y.astype(np.float32), sr


def generate_two_sample_audios(duration=3.0, sr=16000):
    """Generate two related synthetic audio samples for dual-audio analysis."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y1 = 0.6 * np.sin(2 * np.pi * 440 * t)
    y1 += 0.3 * np.sin(2 * np.pi * 880 * t)
    y1 += 0.15 * np.sin(2 * np.pi * 1320 * t)
    y1 += 0.05 * np.random.randn(len(t))

    envelope = np.linspace(0, 1, int(sr * 0.05))
    envelope_rev = np.linspace(1, 0, int(sr * 0.05))
    y1[:len(envelope)] *= envelope
    y1[-len(envelope_rev):] *= envelope_rev

    time_stretch_factor = 0.85
    indices = np.linspace(0, len(t) - 1, int(len(t) * time_stretch_factor))
    y2_short = np.interp(indices, np.arange(len(t)), y1)

    t_short = np.linspace(0, duration * time_stretch_factor, len(y2_short), endpoint=False)
    y2 = 0.7 * y2_short * np.sin(2 * np.pi * 1.5 * 440 * t_short * 0.001)
    y2 += 0.1 * np.random.randn(len(y2_short))

    y1 /= np.max(np.abs(y1)) + 1e-8
    y2 /= np.max(np.abs(y2)) + 1e-8
    return y1.astype(np.float32), y2.astype(np.float32), sr