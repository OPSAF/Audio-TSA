"""Prediction models module."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 全局GPU检测
import torch
_GLOBAL_DEVICE = None
_GLOBAL_DEVICE_INFO = None

def get_device(force_check=False):
    """Get global device (GPU if available, else CPU) with detailed info."""
    global _GLOBAL_DEVICE, _GLOBAL_DEVICE_INFO
    
    if _GLOBAL_DEVICE is None or force_check:
        if torch.cuda.is_available():
            _GLOBAL_DEVICE = torch.device('cuda')
            _GLOBAL_DEVICE_INFO = {
                'type': 'cuda',
                'name': torch.cuda.get_device_name(0),
                'memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
                'memory_allocated': lambda: f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
                'memory_cached': lambda: f"{torch.cuda.memory_reserved(0) / 1e9:.2f} GB"
            }
            print(f"[GPU INFO] Found CUDA device: {_GLOBAL_DEVICE_INFO['name']}")
            print(f"[GPU INFO] Total memory: {_GLOBAL_DEVICE_INFO['memory_total']}")
        else:
            _GLOBAL_DEVICE = torch.device('cpu')
            _GLOBAL_DEVICE_INFO = {'type': 'cpu', 'name': 'CPU', 'memory_total': 'N/A'}
            print("[GPU INFO] WARNING: CUDA not available! Using CPU (will be very slow)")
    
    return _GLOBAL_DEVICE, _GLOBAL_DEVICE_INFO

def print_gpu_usage():
    """Print current GPU memory usage."""
    if _GLOBAL_DEVICE_INFO and _GLOBAL_DEVICE_INFO['type'] == 'cuda':
        print(f"[GPU USAGE] Allocated: {_GLOBAL_DEVICE_INFO['memory_allocated']()}, "
              f"Cached: {_GLOBAL_DEVICE_INFO['memory_cached']()}")

def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def predict_hmm(series, forecast_horizon=10, n_components=8):
    """HMM (Hidden Markov Model) with Gaussian emissions for prediction."""
    try:
        from hmmlearn import hmm

        train_size = int(len(series) * 0.8)
        train = series[:train_size].reshape(-1, 1)
        test = series[train_size:train_size + forecast_horizon]

        best_model = None
        best_score = -np.inf

        for n_comp in [n_components]:
            for _ in range(3):
                model = hmm.GaussianHMM(
                    n_components=n_comp, covariance_type='full',
                    n_iter=200, tol=1e-4, random_state=np.random.randint(0, 10000))
                try:
                    model.fit(train)
                    score = model.score(train)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    pass

        if best_model is None:
            return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 'error': 'HMM fit failed'}

        last_hidden = best_model.predict(train[-30:])
        last_state = last_hidden[-1]

        forecast = np.zeros(forecast_horizon)
        current_state = last_state
        for i in range(forecast_horizon):
            forecast[i] = best_model.means_[current_state].item()
            current_state = np.argmax(best_model.transmat_[current_state])

        if len(test) >= forecast_horizon:
            metrics = compute_metrics(test[:forecast_horizon], forecast)
        else:
            metrics = {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}

        return forecast, metrics

    except Exception as e:
        return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan, 'error': str(e)}


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

    def _get_device(self):
        """Get device using global detection."""
        self.device, device_info = get_device()
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

    def predict(self, series, forecast_horizon=10, epochs=30, lr=0.001):
        import torch
        import torch.nn as nn

        train_size = int(len(series) * 0.8)
        train_segment = series[:train_size]
        test_segment = series[train_size:train_size + forecast_horizon]

        train_scaled = self.scaler.fit_transform(train_segment.reshape(-1, 1)).flatten()

        X, y = prepare_sequence_data(train_scaled, self.lookback, forecast_horizon)
        if len(X) == 0:
            return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}

        if self._model is None:
            self._build_model(forecast_horizon)

        device = self._get_device()
        use_amp = torch.cuda.is_available()
        
        print(f"  [LSTM] Moving model to {device.type.upper()}...")
        model = self._model.to(device)

        print(f"  [LSTM] Moving data to {device.type.upper()}...")
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        batch_size = min(256, len(X))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        pin_memory = torch.cuda.is_available() and X_t.device.type == 'cpu'
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        print(f"  [LSTM] Training for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                last_window = train_scaled[-self.lookback:]
                last_input = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                pred_scaled = model(last_input).cpu().numpy().flatten()
        
        clear_gpu_cache()

        forecast = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        if len(test_segment) >= forecast_horizon:
            metrics = compute_metrics(test_segment[:forecast_horizon], forecast)
        else:
            metrics = {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}

        return forecast, metrics


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

    def _get_device(self):
        """Get device using global detection."""
        self.device, device_info = get_device()
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

    def predict(self, series, forecast_horizon=10, epochs=30, lr=0.001):
        import torch
        import torch.nn as nn

        train_size = int(len(series) * 0.8)
        train_segment = series[:train_size]
        test_segment = series[train_size:train_size + forecast_horizon]

        train_scaled = self.scaler.fit_transform(train_segment.reshape(-1, 1)).flatten()

        X, y = prepare_sequence_data(train_scaled, self.lookback, forecast_horizon)
        if len(X) == 0:
            return np.zeros(forecast_horizon), {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}

        if self._model is None:
            self._build_model(forecast_horizon)

        device = self._get_device()
        use_amp = torch.cuda.is_available()
        
        print(f"  [Transformer] Moving model to {device.type.upper()}...")
        model = self._model.to(device)

        print(f"  [Transformer] Moving data to {device.type.upper()}...")
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).to(device)

        batch_size = min(256, len(X))
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        pin_memory = torch.cuda.is_available() and X_t.device.type == 'cpu'
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        print(f"  [Transformer] Training for {epochs} epochs...")
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                last_window = train_scaled[-self.lookback:]
                last_input = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                pred_scaled = model(last_input).cpu().numpy().flatten()
        
        clear_gpu_cache()

        forecast = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        if len(test_segment) >= forecast_horizon:
            metrics = compute_metrics(test_segment[:forecast_horizon], forecast)
        else:
            metrics = {'RMSE': np.nan, 'MAE': np.nan, 'MSE': np.nan}

        return forecast, metrics


def run_all_predictions(mel_spec, forecast_horizon=20, verbose=True):
    """Run all four prediction models on mel spectrogram."""
    mel_mean = np.mean(mel_spec, axis=0)

    train_size = int(len(mel_mean) * 0.8)
    test_start = train_size
    true_values = mel_mean[test_start:test_start + forecast_horizon]
    train_series = mel_mean[:train_size]

    results = {}

    if verbose:
        print("  [1/4] Running ARIMA...")
    arima_forecast, arima_metrics = predict_arima(train_series, forecast_horizon)
    results['ARIMA'] = (arima_forecast, arima_metrics, true_values)

    if verbose:
        print("  [2/4] Running HMM...")
    hmm_forecast, hmm_metrics = predict_hmm(train_series, forecast_horizon)
    results['HMM'] = (hmm_forecast, hmm_metrics, true_values)

    if verbose:
        print("  [3/4] Running LSTM...")
    lstm = LSTMPredictor(lookback=30)
    lstm_forecast, lstm_metrics = lstm.predict(train_series, forecast_horizon, epochs=60)
    results['LSTM'] = (lstm_forecast, lstm_metrics, true_values)

    if verbose:
        print("  [4/4] Running Transformer...")
    transformer = TransformerPredictor(lookback=30)
    tf_forecast, tf_metrics = transformer.predict(train_series, forecast_horizon, epochs=60)
    results['Transformer'] = (tf_forecast, tf_metrics, true_values)

    return results