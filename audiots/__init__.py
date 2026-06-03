"""
Audio Time Series Analysis Package
==================================

This package provides tools for audio time series analysis and prediction.

Modules:
- loader: Audio file loading utilities
- features: Feature extraction (waveform, Mel spectrogram, MFCC)
- dynamics: Audio dynamics/trend analysis (energy, brightness, complexity, rhythm)
- analysis: Time series analysis (ACF, PACF, FFT, STFT)
- prediction: Prediction models (ARIMA, HMM, LSTM, Transformer)
- band_analysis: Frequency band predictability analysis
- visualization: Plotting and visualization utilities
- similarity: Audio similarity analysis (length-independent, alignment-free)
- similarity_viz: Visualization for similarity analysis results
"""

__version__ = "2.1.0"

from . import loader
from . import features
from . import dynamics
from . import discovery
from . import analysis
from . import prediction
from . import band_analysis
from . import visualization
from . import similarity
from . import similarity_viz
from . import discovery_viz

__all__ = [
    'loader', 'features', 'dynamics', 'discovery', 'analysis', 'prediction',
    'band_analysis', 'visualization', 'similarity', 'similarity_viz',
    'discovery_viz',
]