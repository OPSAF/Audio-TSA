"""
Audio Time Series Analysis Package
==================================

This package provides tools for audio time series analysis and prediction.

Modules:
- loader: Audio file loading utilities
- features: Feature extraction (waveform, Mel spectrogram, MFCC)
- dynamics: Audio dynamics/trend analysis (energy, brightness, complexity, rhythm)
- volatility: Volatility layer — rolling volatility + ARCH/GARCH modelling
- model_analysis: Model ensemble structural analysis (ARIMA/HMM/LSTM/Transformer as structure detectives)
- analysis: Time series analysis (ACF, PACF, FFT, STFT)
- prediction: Prediction models (ARIMA, HMM, LSTM, Transformer)
- band_analysis: Frequency band predictability analysis
- visualization: Plotting and visualization utilities
- similarity: Audio similarity analysis (length-independent, alignment-free)
- similarity_viz: Visualization for similarity analysis results
- discovery: Audio discovery engine (self-discovery, cross-discovery, contrast)
- discovery_viz: Visualization for discovery results
- unsupervised: Unsupervised pattern discovery (change points, motifs, NMF, RQA)
"""

__version__ = "2.3.0"

from . import loader
from . import features
from . import dynamics
from . import volatility
from . import model_analysis
from . import discovery
from . import unsupervised
from . import analysis
from . import prediction
from . import band_analysis
from . import visualization
from . import similarity
from . import similarity_viz
from . import discovery_viz
from . import config
from . import prediction_cache
from . import batch_ml
from . import batch_viz

__all__ = [
    "loader", "features", "dynamics", "volatility", "model_analysis",
    "discovery", "unsupervised",
    "analysis", "prediction", "band_analysis", "visualization",
    "similarity", "similarity_viz", "discovery_viz",
    "config", "prediction_cache",
    "batch_ml", "batch_viz",
]
