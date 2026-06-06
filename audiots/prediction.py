"""Prediction models module."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 抑制所有关于收敛的警告
warnings.filterwarnings("ignore", message=".*Model is not converging.*")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

from . import config as _cfg
from .prediction_cache import get_prediction_cache

# ---------------------------------------------------------------------------
# Safe stdout suppression for noisy third-party libs (e.g. hmmlearn)
# ---------------------------------------------------------------------------
import sys as _sys
from contextlib import redirect_stdout as _redirect_stdout
from io import StringIO as _StringIO

# 全局GPU检测
import torch
_GLOBAL_DEVICE = None
_GLOBAL_DEVICE_INFO = None

_VERBOSE_GPU = False  # set True to re-enable GPU detection logging


def get_device(force_check=False, verbose=None):
    """Get global device (GPU if available, else CPU) with detailed info."""
    global _GLOBAL_DEVICE, _GLOBAL_DEVICE_INFO

    if verbose is None:
        verbose = _VERBOSE_GPU

    if _GLOBAL_DEVICE is None or force_check:
        if torch.cuda.is_available():
            _GLOBAL_DEVICE = torch.device("cuda")
            _GLOBAL_DEVICE_INFO = {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0),
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                "memory_allocated": lambda: f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
                "memory_cached": lambda: f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB",
            }
            if verbose:
                print(f"[GPU INFO] Found CUDA device: {_GLOBAL_DEVICE_INFO['name']}")
                print(f"[GPU INFO] Total memory: {_GLOBAL_DEVICE_INFO['memory_total']}")
        else:
            _GLOBAL_DEVICE = torch.device("cpu")
            _GLOBAL_DEVICE_INFO = {"type": "cpu", "name": "CPU", "memory_total": "N/A"}
            if verbose:
                print("[GPU INFO] WARNING: CUDA not available! Using CPU (will be very slow)")

    return _GLOBAL_DEVICE, _GLOBAL_DEVICE_INFO

def print_gpu_usage():
    """Print current GPU memory usage."""
    if _GLOBAL_DEVICE_INFO and _GLOBAL_DEVICE_INFO["type"] == "cuda":
        print(
            f"[GPU USAGE] Allocated: {_GLOBAL_DEVICE_INFO['memory_allocated']()}, "
            f"Cached: {_GLOBAL_DEVICE_INFO['memory_cached']()}"
        )


def clear_gpu_cache(verbose=False):
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print("[GPU INFO] Cleared CUDA cache")


def prepare_sequence_data(series, lookback=30, forecast_horizon=10):
    """Prepare sliding window data for time series prediction."""
    X, y = [], []
    for i in range(len(series) - lookback - forecast_horizon + 1):
        X.append(series[i:i + lookback])
        y.append(series[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)


def compute_metrics(y_true, y_pred):
    """Compute RMSE and MAE between true and predicted."""
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    return {'RMSE': np.sqrt(mse), 'MAE': mae, 'MSE': mse}


def predict_arima(series, forecast_horizon=10):
    """ARIMA model prediction with improved auto-order selection."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        series_len = len(series)
        
        # Handle short series case - relaxed constraints
        min_total_len = forecast_horizon + 5  # Reduced from 10 to 5
        if series_len < min_total_len:
            return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 
                                                'error': f'Series too short ({series_len} < {min_total_len})'}

        train_size = int(len(series) * 0.8)
        if train_size < forecast_horizon:
            train_size = len(series) - forecast_horizon - 1
            
        min_train_size = max(10, forecast_horizon)  # Reduced from 20 to max(10, forecast_horizon)
        if train_size < min_train_size:
            return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 
                                                'error': f'Insufficient training data (train_size={train_size} < {min_train_size})'}

        train = series[:train_size]
        test = series[train_size:train_size + forecast_horizon]

        # Test stationarity and determine d parameter
        try:
            adf_result = adfuller(train)
            d = 0 if adf_result[1] < 0.05 else 1
        except:
            d = 1

        # Expanded order search space with simpler models first
        orders = [
            (1, d, 1), (1, d, 0), (0, d, 1),  # Simpler models first
            (1, d, 2), (2, d, 1), (2, d, 2),
            (3, d, 1), (3, d, 2), (3, d, 3),
            (4, d, 1), (4, d, 2),
            (0, d, 0), (2, d, 0), (0, d, 2)   # Additional simple models
        ]

        best_forecast = None
        best_metrics = None
        best_aic = np.inf
        best_bic = np.inf
        last_error = None

        for order in orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(train, order=order)
                    fitted = model.fit(method_kwargs={'maxiter': 500})

                    current_aic = fitted.aic
                    current_bic = fitted.bic
                    
                    # Prefer models with lower AIC, using BIC as tiebreaker
                    if current_aic < best_aic or (abs(current_aic - best_aic) < 10 and current_bic < best_bic):
                        best_aic = current_aic
                        best_bic = current_bic
                        forecast = fitted.forecast(steps=forecast_horizon)
                        best_forecast = forecast

                        if len(test) >= forecast_horizon:
                            best_metrics = compute_metrics(test[:forecast_horizon], forecast)
                        else:
                            best_metrics = {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}
            except Exception as e:
                last_error = str(e)
                continue

        if best_forecast is None:
            # Fallback: use simple moving average if ARIMA fails
            fallback_forecast = np.full(forecast_horizon, np.mean(train[-5:]) if len(train) >= 5 else np.mean(train))
            return fallback_forecast, {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 
                                        'method': 'fallback_mean', 'error': last_error if last_error else 'All ARIMA orders failed'}

        return best_forecast, best_metrics

    except Exception as e:
        # Fallback to simple prediction on any error
        fallback_forecast = np.full(forecast_horizon, np.mean(series[-5:]) if len(series) >= 5 else 0)
        return fallback_forecast, {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 'error': str(e)}


def predict_hmm(series, forecast_horizon=10, n_components=None):
    """HMM (Hidden Markov Model) with Gaussian emissions for prediction."""
    try:
        from hmmlearn import hmm

        # ── Data sanity ──────────────────────────────────────────────────
        n = len(series)
        if n < 20:
            # Too few points for meaningful HMM — use persistence fallback
            s = np.asarray(series, dtype=np.float64).ravel()
            last_val = float(s[-1]) if len(s) > 0 else 0.0
            return np.full(forecast_horizon, last_val), {
                "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan,
                "method": "persistence (series too short for HMM)",
            }

        # ── Auto-limit components based on data size ─────────────────────
        if n_components is None:
            n_components = _cfg.DEFAULT_HMM_COMPONENTS
        # Each state needs at least ~8 observations to be meaningful
        max_states = min(_cfg.MAX_HMM_COMPONENTS, max(2, n // 8))
        n_components = min(n_components, max_states)

        train_size = int(n * 0.8)
        train = series[:train_size].reshape(-1, 1)
        test = series[train_size:train_size + forecast_horizon]

        # ── Try state counts, from smallest to largest ──────────────────
        # Start with fewer states (more robust) and only go up if data supports it
        candidates = sorted(set([
            max(2, n_components - 1),
            n_components,
            min(max_states, n_components + 1),
        ]))
        best_model = None
        best_score = -np.inf

        for n_comp in candidates:
            # Limit attempts per state count — don't spin forever
            max_attempts = 2 if n_comp >= 4 else 3
            for attempt in range(max_attempts):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = hmm.GaussianHMM(
                            n_components=n_comp,
                            covariance_type=_cfg.DEFAULT_HMM_COVARIANCE,
                            n_iter=_cfg.HMM_N_ITER,
                            tol=_cfg.HMM_TOL,
                            random_state=np.random.randint(0, 10000),
                            init_params="mcs",  # initialise means/covars/startprob
                        )
                        with _redirect_stdout(_StringIO()):
                            model.fit(train)

                        # ── Quality checks ───────────────────────────────
                        transmat = model.transmat_
                        row_sums = transmat.sum(axis=1)
                        n_degenerate = int(np.sum(row_sums < 1e-10))

                        if n_degenerate > 0:
                            # Degenerate transmat = unreliable model.
                            # Penalty proportional to number of dead states
                            score = model.score(train) - (2000.0 * n_degenerate)
                        else:
                            # Additional check: are all states actually used?
                            hidden_states = model.predict(train)
                            _, counts = np.unique(hidden_states, return_counts=True)
                            n_rare = int(np.sum(counts < 3))  # states with <3 obs
                            score = model.score(train) - (500.0 * n_rare)

                        if score > best_score:
                            best_score = score
                            best_model = model

                except Exception:
                    pass  # single fit failure — try next attempt/state-count

        # ── Fallback ────────────────────────────────────────────────────
        if best_model is None:
            s = np.asarray(series, dtype=np.float64).ravel()
            last_val = float(s[-1]) if len(s) > 0 else 0.0
            return np.full(forecast_horizon, last_val), {
                "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan,
                "method": "persistence (all HMM fits failed)",
            }

        # ── Forecast via state-sequence walk ────────────────────────────
        last_hidden = best_model.predict(train[-min(30, len(train)):])
        last_state = last_hidden[-1]

        forecast = np.zeros(forecast_horizon)
        current_state = last_state
        for i in range(forecast_horizon):
            forecast[i] = best_model.means_[current_state].item()
            row = best_model.transmat_[current_state]
            if row.sum() > 1e-10:
                current_state = int(np.argmax(row))
            # else: stay in same state (row is all zeros — degenerate but handled)
            # else: stay in same state (row is all zeros)

        # ---- fallback for degenerate / constant forecasts ----
        fc_std = float(np.std(forecast)) if len(forecast) > 1 else 0.0
        if fc_std < 1e-10 or np.any(~np.isfinite(forecast)) or np.all(forecast == 0):
            # HMM collapsed to a single state — use persistence forecast
            last_vals = train[-1].ravel() if train.ndim > 1 else train[-1]
            last_val = float(last_vals.item() if hasattr(last_vals, "item") else last_vals)
            forecast = np.full(forecast_horizon, last_val)

        if len(test) >= forecast_horizon:
            metrics = compute_metrics(test[:forecast_horizon], forecast)
        else:
            metrics = {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan}

        return forecast, metrics

    except Exception as e:
        s = np.asarray(series, dtype=np.float64).ravel()
        last_val = float(s[-1]) if len(s) > 0 else 0.0
        return np.full(forecast_horizon, last_val), {
            "RMSE": np.nan, "MAE": np.nan, "MSE": np.nan,
            "error": str(e),
        }


class LSTMPredictor:
    """LSTM model for time series forecasting."""

    def __init__(self, lookback=30, hidden_size=64, num_layers=1, dropout=0.1):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = StandardScaler()
        self._model = None
        self.device = None

    def _get_device(self, verbose=False):
        """Get device using global detection."""
        self.device, device_info = get_device()
        if verbose:
            print(f"  [LSTM] Device: {device_info['type'].upper()} ({device_info['name']})")
        return self.device

    def _build_model(self, forecast_horizon):
        import torch
        import torch.nn as nn

        class LSTMForecaster(nn.Module):
            def __init__(self, lookback, hidden, layers, dropout, horizon):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1, hidden_size=hidden, num_layers=layers,
                    dropout=dropout if layers > 1 else 0, batch_first=True)
                self.fc = nn.Linear(hidden, horizon)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self._model = LSTMForecaster(
            self.lookback, self.hidden_size, self.num_layers,
            self.dropout, forecast_horizon)

    def predict(self, series, forecast_horizon=10, epochs=30, lr=0.001, verbose=False):
        import torch
        import torch.nn as nn

        # --- prediction cache ---
        cache = get_prediction_cache()
        cached = cache.get(series, "lstm", self.lookback, forecast_horizon, epochs)
        if cached is not None:
            if verbose:
                print("  [LSTM] Using cached prediction")
            return cached

        # --- data-size guard ---
        if len(series) < _cfg.MIN_WINDOWS_DEEP_MODEL:
            if verbose:
                print(f"  [LSTM] Skipped: series too short ({len(series)} < {_cfg.MIN_WINDOWS_DEEP_MODEL})")
            result = (np.zeros(forecast_horizon), {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan,
                       "error": f"Series too short for LSTM ({len(series)})"})
            cache.put(series, "lstm", self.lookback, forecast_horizon, epochs, result)
            return result

        train_size = int(len(series) * 0.8)
        train_segment = series[:train_size]
        test_segment = series[train_size:train_size + forecast_horizon]

        train_scaled = self.scaler.fit_transform(train_segment.reshape(-1, 1)).flatten()

        X, y = prepare_sequence_data(train_scaled, self.lookback, forecast_horizon)
        if len(X) == 0:
            result = (np.zeros(forecast_horizon), {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan})
            cache.put(series, "lstm", self.lookback, forecast_horizon, epochs, result)
            return result

        if self._model is None:
            self._build_model(forecast_horizon)

        device = self._get_device(verbose=verbose)
        use_amp = torch.cuda.is_available()

        if verbose:
            print(f"  [LSTM] Moving model to {device.type.upper()}...")
        model = self._model.to(device)

        if verbose:
            print(f"  [LSTM] Moving data to {device.type.upper()}...")
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        batch_size = min(256, len(X))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        pin_memory = torch.cuda.is_available() and X_t.device.type == "cpu"
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        if verbose:
            print(f"  [LSTM] Training for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)

                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                last_window = train_scaled[-self.lookback:]
                last_input = (
                    torch.tensor(last_window, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(device)
                )
                pred_scaled = model(last_input).cpu().numpy().flatten()

        clear_gpu_cache()

        forecast = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        if len(test_segment) >= forecast_horizon:
            metrics = compute_metrics(test_segment[:forecast_horizon], forecast)
        else:
            metrics = {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan}

        result = (forecast, metrics)
        cache.put(series, "lstm", self.lookback, forecast_horizon, epochs, result)
        return result


class TransformerPredictor:
    """Transformer Encoder for time series forecasting."""

    def __init__(self, lookback=30, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        self.lookback = lookback
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = StandardScaler()
        self._model = None
        self.device = None

    def _get_device(self, verbose=False):
        """Get device using global detection."""
        self.device, device_info = get_device()
        if verbose:
            print(f"  [Transformer] Device: {device_info['type'].upper()} ({device_info['name']})")
        return self.device

    def _build_model(self, forecast_horizon):
        import torch
        import torch.nn as nn

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=200):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len).unsqueeze(1).float()
                div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:x.size(1), :]

        class TransformerForecaster(nn.Module):
            def __init__(self, lookback, d_model, nhead, layers, dropout, horizon):
                super().__init__()
                self.input_proj = nn.Linear(1, d_model)
                self.pos_enc = PositionalEncoding(d_model, max_len=lookback + horizon)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dropout=dropout,
                    dim_feedforward=d_model * 4, batch_first=True)
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.fc = nn.Linear(d_model, horizon)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_enc(x)
                enc_out = self.encoder(x)
                return self.fc(enc_out.mean(dim=1))

        self._model = TransformerForecaster(
            self.lookback, self.d_model, self.nhead,
            self.num_layers, self.dropout, forecast_horizon)

    def predict(self, series, forecast_horizon=10, epochs=30, lr=0.001, verbose=False):
        import torch
        import torch.nn as nn

        # --- prediction cache ---
        cache = get_prediction_cache()
        cached = cache.get(series, "transformer", self.lookback, forecast_horizon, epochs)
        if cached is not None:
            if verbose:
                print("  [Transformer] Using cached prediction")
            return cached

        # --- data-size guard ---
        if len(series) < _cfg.MIN_WINDOWS_DEEP_MODEL:
            if verbose:
                print(f"  [Transformer] Skipped: series too short ({len(series)} < {_cfg.MIN_WINDOWS_DEEP_MODEL})")
            result = (np.zeros(forecast_horizon), {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan,
                       "error": f"Series too short for Transformer ({len(series)})"})
            cache.put(series, "transformer", self.lookback, forecast_horizon, epochs, result)
            return result

        train_size = int(len(series) * 0.8)
        train_segment = series[:train_size]
        test_segment = series[train_size:train_size + forecast_horizon]

        train_scaled = self.scaler.fit_transform(train_segment.reshape(-1, 1)).flatten()

        X, y = prepare_sequence_data(train_scaled, self.lookback, forecast_horizon)
        if len(X) == 0:
            result = (np.zeros(forecast_horizon), {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan})
            cache.put(series, "transformer", self.lookback, forecast_horizon, epochs, result)
            return result

        if self._model is None:
            self._build_model(forecast_horizon)

        device = self._get_device(verbose=verbose)
        use_amp = torch.cuda.is_available()

        if verbose:
            print(f"  [Transformer] Moving model to {device.type.upper()}...")
        model = self._model.to(device)

        if verbose:
            print(f"  [Transformer] Moving data to {device.type.upper()}...")
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        batch_size = min(256, len(X))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        pin_memory = torch.cuda.is_available() and X_t.device.type == "cpu"
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        if verbose:
            print(f"  [Transformer] Training for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                last_window = train_scaled[-self.lookback:]
                last_input = (
                    torch.tensor(last_window, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(device)
                )
                pred_scaled = model(last_input).cpu().numpy().flatten()

        clear_gpu_cache()

        forecast = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        if len(test_segment) >= forecast_horizon:
            metrics = compute_metrics(test_segment[:forecast_horizon], forecast)
        else:
            metrics = {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan}

        result = (forecast, metrics)
        cache.put(series, "transformer", self.lookback, forecast_horizon, epochs, result)
        return result


def run_all_predictions(mel_spec, forecast_horizon=20, epochs=None, verbose=True):
    """Run all four prediction models on mel spectrogram."""
    if epochs is None:
        epochs = _cfg.DEFAULT_PREDICTION_EPOCHS

    mel_mean = np.mean(mel_spec, axis=0)

    train_size = int(len(mel_mean) * 0.8)
    test_start = train_size
    true_values = mel_mean[test_start:test_start + forecast_horizon]
    train_series = mel_mean[:train_size]

    results = {}

    if verbose:
        print("  [1/4] Running ARIMA...")
    arima_forecast, arima_metrics = predict_arima(train_series, forecast_horizon)
    results["ARIMA"] = (arima_forecast, arima_metrics, true_values)

    if verbose:
        print("  [2/4] Running HMM...")
    hmm_forecast, hmm_metrics = predict_hmm(train_series, forecast_horizon)
    results["HMM"] = (hmm_forecast, hmm_metrics, true_values)

    if verbose:
        print("  [3/4] Running LSTM...")
    lstm = LSTMPredictor(lookback=30)
    lstm_forecast, lstm_metrics = lstm.predict(
        train_series, forecast_horizon, epochs=epochs, verbose=verbose
    )
    results["LSTM"] = (lstm_forecast, lstm_metrics, true_values)

    if verbose:
        print("  [4/4] Running Transformer...")
    transformer = TransformerPredictor(lookback=30)
    tf_forecast, tf_metrics = transformer.predict(
        train_series, forecast_horizon, epochs=epochs, verbose=verbose
    )
    results["Transformer"] = (tf_forecast, tf_metrics, true_values)

    return results


# ---------------------------------------------------------------------------
# Trend & Volatility prediction (consumes dynamics / volatility output)
# ---------------------------------------------------------------------------

def predict_trend_series(
    series: np.ndarray,
    forecast_horizon: int = 20,
    models: str = "all",
    epochs: int = 30,
    verbose: bool = False,
) -> Dict:
    """
    Run prediction models on a single trend time series.

    This is the bridge between the Trend/Volatility Layer and the
    prediction engines.  Rather than operating on mel-spectrogram means,
    it accepts ANY 1-D time series (energy trend, brightness trend,
    volatility series, etc.).

    Parameters
    ----------
    series : ndarray       1-D time series to forecast.
    forecast_horizon : int Forecast steps.
    models : str           "all", "arima", "hmm", "lstm", "transformer",
                            or a list of model names.
    epochs : int           Epochs for deep models.
    verbose : bool         Print progress.

    Returns
    -------
    results : dict         Same structure as ``run_all_predictions()``:
                            {model_name: (forecast, metrics, true_values)}
    """
    series = np.asarray(series, dtype=np.float64).ravel()

    if models == "all":
        model_list = ["ARIMA", "HMM", "LSTM", "Transformer"]
    elif isinstance(models, str):
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = list(models)

    # Map to canonical case
    model_list = [m.upper() if m.lower() != "transformer" else "Transformer"
                  for m in model_list]
    model_list = [m.capitalize() if m.lower() not in ("arima", "hmm", "lstm", "transformer")
                  else ("ARIMA" if m.upper() == "ARIMA" else
                         "HMM" if m.upper() == "HMM" else m)
                  for m in model_list]
    # Simplify:
    canonical = []
    for m in model_list:
        ml = m.lower()
        if ml in ("arima",): canonical.append("ARIMA")
        elif ml in ("hmm",): canonical.append("HMM")
        elif ml in ("lstm",): canonical.append("LSTM")
        elif ml in ("transformer",): canonical.append("Transformer")
        else: canonical.append(m)
    model_list = canonical

    train_size = int(len(series) * 0.8)
    train_segment = series[:train_size]
    true_values = series[train_size: train_size + forecast_horizon]

    # Pad true_values if series is too short
    if len(true_values) < forecast_horizon:
        true_values = np.pad(true_values, (0, forecast_horizon - len(true_values)),
                             mode="edge")

    results = {}

    for model_name in model_list:
        try:
            if model_name == "ARIMA":
                if verbose:
                    print(f"  [Trend Pred] ARIMA on series (len={len(train_segment)})...")
                forecast, metrics = predict_arima(train_segment, forecast_horizon)
                results["ARIMA"] = (forecast, metrics, true_values)

            elif model_name == "HMM":
                if verbose:
                    print("  [Trend Pred] HMM...")
                forecast, metrics = predict_hmm(train_segment, forecast_horizon)
                results["HMM"] = (forecast, metrics, true_values)

            elif model_name == "LSTM":
                if verbose:
                    print("  [Trend Pred] LSTM...")
                lstm = LSTMPredictor(lookback=min(30, max(5, len(train_segment) // 4)))
                forecast, metrics = lstm.predict(train_segment, forecast_horizon, epochs=epochs)
                results["LSTM"] = (forecast, metrics, true_values)

            elif model_name == "Transformer":
                if verbose:
                    print("  [Trend Pred] Transformer...")
                transformer = TransformerPredictor(
                    lookback=min(30, max(5, len(train_segment) // 4)))
                forecast, metrics = transformer.predict(
                    train_segment, forecast_horizon, epochs=epochs)
                results["Transformer"] = (forecast, metrics, true_values)

        except Exception as e:
            if verbose:
                print(f"  [Trend Pred] {model_name} failed: {e}")
            results[model_name] = (
                np.zeros(forecast_horizon),
                {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": str(e)},
                true_values,
            )

    return results


def predict_all_trends(
    dynamics: Dict,
    forecast_horizon: int = 20,
    models: str = "all",
    epochs: int = 30,
    verbose: bool = True,
) -> Dict:
    """
    Run prediction models on all four trend time series.

    Parameters
    ----------
    dynamics : dict          Output of ``dynamics.extract_dynamics()``.
    forecast_horizon : int   Forecast steps.
    models : str             Model selection (see ``predict_trend_series()``).
    epochs : int             Epochs for deep models.
    verbose : bool           Print progress.

    Returns
    -------
    trend_predictions : dict
        Keys: ``energy``, ``brightness``, ``complexity``, ``rhythm``.
        Each value is the standard prediction dict:
        {model_name: (forecast, metrics, true_values)}
    """
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    trend_names_cn = ["能量", "亮度", "复杂度", "节奏"]

    all_results = {}

    for key, name_cn in zip(trend_keys, trend_names_cn):
        series = np.asarray(dynamics[key], dtype=np.float64).ravel()

        if verbose:
            print(f"  [TrendPred] Predicting {name_cn} trend ({len(series)} points) ...")

        all_results[key] = predict_trend_series(
            series,
            forecast_horizon=forecast_horizon,
            models=models,
            epochs=epochs,
            verbose=verbose,
        )

    return all_results


def predict_volatility(
    vol_layer: Dict,
    forecast_horizon: int = 10,
    trend_key: str = "energy",
    models: str = "all",
    epochs: int = 20,
    verbose: bool = False,
) -> Dict:
    """
    Run prediction on a volatility series.

    Volatility is slower-changing than raw trends, so a shorter
    forecast horizon is recommended.

    Parameters
    ----------
    vol_layer : dict         Output of ``volatility.compute_volatility_layer()``.
    forecast_horizon : int   Forecast steps (keep modest — 5–15).
    trend_key : str          Which volatility dimension to forecast.
    models : str             Model selection.
    epochs : int             Epochs for deep models.
    verbose : bool           Print progress.

    Returns
    -------
    results : dict           Same structure as ``predict_trend_series()``.
    """
    vol_series = vol_layer.get(f"{trend_key}_vol")
    if vol_series is None:
        raise ValueError(
            f"Volatility key '{trend_key}_vol' not found in vol_layer. "
            f"Available keys: {[k for k in vol_layer if '_vol' in k]}"
        )

    if verbose:
        print(f"  [VolPred] Predicting {trend_key} volatility "
              f"({len(vol_series)} points) ...")

    return predict_trend_series(
        vol_series,
        forecast_horizon=forecast_horizon,
        models=models,
        epochs=epochs,
        verbose=verbose,
    )


def predict_all_volatilities(
    vol_layer: Dict,
    forecast_horizon: int = 10,
    models: str = "all",
    epochs: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    Run prediction on all four volatility series.

    Parameters
    ----------
    vol_layer : dict         Output of ``volatility.compute_volatility_layer()``.
    forecast_horizon : int   Forecast steps.
    models : str             Model selection.
    epochs : int             Epochs for deep models.
    verbose : bool           Print progress.

    Returns
    -------
    vol_predictions : dict
        Keys: ``energy``, ``brightness``, ``complexity``, ``rhythm``.
    """
    # Volatility series are near-white-noise by construction — ARIMA fits
    # on them are slow, noisy ("Model is not converging"), and add little
    # predictive value.  HMM alone is faster and more robust for this task.
    if models == "ARIMA,HMM" or models == "ARIMA,HMM,LSTM,Transformer":
        models = "HMM"

    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    trend_names_cn = ["能量", "亮度", "复杂度", "节奏"]

    all_results = {}

    for key, name_cn in zip(trend_keys, trend_names_cn):
        if verbose:
            print(f"  [VolPred] Predicting {name_cn} volatility ...")
        all_results[key] = predict_volatility(
            vol_layer,
            forecast_horizon=forecast_horizon,
            trend_key=key,
            models=models,
            epochs=epochs,
            verbose=False,
        )

    return all_results