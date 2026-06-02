"""
Audio Time Series Analysis Package
=================================

This package provides tools for audio time series analysis and prediction.

Modules:
- loader: Audio file loading utilities
- features: Feature extraction (waveform, Mel spectrogram, MFCC)
- analysis: Time series analysis (ACF, PACF, FFT, STFT)
- prediction: Prediction models (ARIMA, HMM, LSTM, Transformer)
- band_analysis: Frequency band predictability analysis
- visualization: Plotting and visualization utilities
"""

__version__ = "1.0.0"

from . import loader
from . import features
from . import analysis
from . import prediction
from . import band_analysis
from . import visualization

__all__ = ['loader', 'features', 'analysis', 'prediction', 'band_analysis', 'visualization']