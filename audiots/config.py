"""Centralised configuration constants for model training parameters.

All modules that train models should import defaults from here instead of
hard-coding magic numbers.  This makes it easy to tune the speed/accuracy
trade-off globally.
"""

# ---------------------------------------------------------------------------
# Deep model training defaults
# ---------------------------------------------------------------------------

DEFAULT_LSTM_EPOCHS = 20
DEFAULT_TRANSFORMER_EPOCHS = 20
DEFAULT_BAND_EPOCHS = 30
DEFAULT_PREDICTION_EPOCHS = 30
DEFAULT_MODEL_ANALYSIS_EPOCHS = 15

# ---------------------------------------------------------------------------
# HMM defaults
# ---------------------------------------------------------------------------

DEFAULT_HMM_COMPONENTS = 3          # was 8 — far too many for typical audio
DEFAULT_HMM_COVARIANCE = "diag"     # was "full" — over-parameterised for 1-D
MAX_HMM_COMPONENTS = 8
HMM_N_ITER = 200                    # sufficient for convergence with 3 states
HMM_TOL = 1e-4                      # original tolerance — necessary for stable fits

# ---------------------------------------------------------------------------
# Data-size thresholds (in number of windows / time-steps)
# ---------------------------------------------------------------------------

MIN_WINDOWS_DEEP_MODEL = 50         # minimum points for LSTM / Transformer
MIN_WINDOWS_HMM = 15                # minimum points for HMM fitting
MIN_WINDOWS_ARIMA = 10              # minimum points for ARIMA

# ---------------------------------------------------------------------------
# Model architecture defaults
# ---------------------------------------------------------------------------

DEFAULT_LSTM_LOOKBACK = 30
DEFAULT_LSTM_HIDDEN = 64
DEFAULT_LSTM_LAYERS = 1
DEFAULT_LSTM_DROPOUT = 0.1

DEFAULT_TRANSFORMER_LOOKBACK = 30
DEFAULT_TRANSFORMER_DMODEL = 64
DEFAULT_TRANSFORMER_NHEAD = 4
DEFAULT_TRANSFORMER_LAYERS = 2
DEFAULT_TRANSFORMER_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Analysis pipeline defaults
# ---------------------------------------------------------------------------

DEFAULT_FORECAST_HORIZON = 20
DEFAULT_N_MELS = 128
DEFAULT_TREND_PREDICTION_MODELS = "ARIMA,HMM"  # fast models for trend/vol prediction
