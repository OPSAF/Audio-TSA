"""
Batch Global Model Training
============================

Global-model training for multi-song batch analysis.  Instead of training
one model per song (as ``prediction.py`` does), this module trains a **single
shared model** on sliding-window samples pooled from ALL songs in a folder.

Architecture
------------
::

    Song 1  Mel (n_mels, T1)  ─┐
    Song 2  Mel (n_mels, T2)  ─┤
    ...                        ├─► sliding windows ─► pooled dataset
    Song N  Mel (n_mels, TN)  ─┘        │
                               ┌────────┼────────┐
                               ▼        ▼        ▼
                            LSTM   Transformer   HMM
                          (Global)  (Global)   (Joint)
                               │        │        │
                               ▼        ▼        ▼
                          per-song    per-song   shared
                          metrics     metrics    states

ARIMA is also provided here as a **local-model baseline** (one ARIMA per
song, no parameter sharing).

Key insight
-----------
- **Global Model** learns *genre-level* time-frequency patterns shared
  across multiple songs of the same style.
- **Local ARIMA** serves as a sanity check: if the global deep model
  consistently outperforms per-song ARIMA, the genre has learnable
  shared structure.
- **Joint HMM** discovers latent musical states that recur across songs.
"""

from __future__ import annotations

import warnings
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as _scipy_stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from . import config as _cfg

# ---------------------------------------------------------------------------
# Sliding-window data preparation
# ---------------------------------------------------------------------------

def prepare_mel_windows_single(
    mel_spec: np.ndarray,
    lookback: int = 30,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window for a single Mel spectrogram.

    Parameters
    ----------
    mel_spec : ndarray, shape (n_mels, n_frames)
    lookback : int
        Number of past frames used as input.
    forecast_horizon : int
        Number of future frames to predict (default 1).

    Returns
    -------
    X : ndarray, shape (n_windows, lookback, n_mels)
    y : ndarray, shape (n_windows, n_mels)  — next frame
    """
    mel = np.asarray(mel_spec, dtype=np.float32)
    n_mels, n_frames = mel.shape

    total = lookback + forecast_horizon
    if n_frames < total:
        return np.empty((0, lookback, n_mels), dtype=np.float32), \
               np.empty((0, n_mels), dtype=np.float32)

    n_windows = n_frames - total + 1
    X = np.zeros((n_windows, lookback, n_mels), dtype=np.float32)
    y = np.zeros((n_windows, n_mels), dtype=np.float32)

    for i in range(n_windows):
        # X: frames i .. i+lookback-1, all mel bands → transpose to (lookback, n_mels)
        X[i] = mel[:, i:i + lookback].T
        # y: the frame at i+lookback (next frame), all mel bands
        y[i] = mel[:, i + lookback]

    return X, y


def prepare_mel_windows_all_songs(
    song_mel_specs: Dict[str, np.ndarray],
    lookback: int = 30,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool sliding windows from ALL songs into a single dataset.

    Parameters
    ----------
    song_mel_specs : dict
        {song_name: mel_spec (n_mels, n_frames)}
    lookback : int
    forecast_horizon : int

    Returns
    -------
    X_all : ndarray, shape (total_windows, lookback, n_mels)
    y_all : ndarray, shape (total_windows, n_mels)
    song_ids : ndarray of int, shape (total_windows,)
        Maps each window to its source song index (0..N-1).
    """
    X_list, y_list, sid_list = [], [], []
    song_names = sorted(song_mel_specs.keys())

    for sid, name in enumerate(song_names):
        mel = song_mel_specs[name]
        X_s, y_s = prepare_mel_windows_single(mel, lookback, forecast_horizon)
        if len(X_s) == 0:
            continue
        X_list.append(X_s)
        y_list.append(y_s)
        sid_list.append(np.full(len(X_s), sid, dtype=np.int32))

    if not X_list:
        return (
            np.empty((0, lookback, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    song_ids = np.concatenate(sid_list, axis=0)
    return X_all, y_all, song_ids


# ---------------------------------------------------------------------------
# PyTorch model builders
# ---------------------------------------------------------------------------

def _get_device(verbose: bool = False):
    """Lightweight device detection."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if verbose:
            print(f"  [GPU] {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
    return dev


class _GlobalLSTM(nn.Module):
    """LSTM encoder → decoder for Mel-frame prediction."""
    def __init__(self, n_mels: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_mels)

    def forward(self, x):
        # x: (batch, lookback, n_mels)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # use last hidden state


class _GlobalTransformer(nn.Module):
    """Transformer encoder → mean-pool → MLP for Mel-frame prediction."""
    def __init__(self, n_mels: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, n_mels)

    def forward(self, x):
        # x: (batch, lookback, n_mels)
        x = self.input_proj(x)          # → (batch, lookback, d_model)
        enc = self.encoder(x)           # → (batch, lookback, d_model)
        pooled = enc.mean(dim=1)        # → (batch, d_model)
        return self.fc(pooled)          # → (batch, n_mels)


# ---------------------------------------------------------------------------
# Global LSTM
# ---------------------------------------------------------------------------

class GlobalLSTMPredictor:
    """LSTM trained on pooled Mel windows from multiple songs."""

    def __init__(self, n_mels: int = 128, lookback: int = 30,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2):
        self.n_mels = n_mels
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._model: Optional[nn.Module] = None
        self.device = None
        self.scaler = StandardScaler()

    def _build(self):
        self._model = _GlobalLSTM(
            n_mels=self.n_mels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train on pooled dataset.

        Parameters
        ----------
        X : ndarray, shape (N, lookback, n_mels)
        y : ndarray, shape (N, n_mels)
        epochs, lr, batch_size : training hyperparams.
        verbose : bool

        Returns
        -------
        losses : list of epoch losses.
        """
        if self._model is None:
            self._build()
        self.device = _get_device(verbose=verbose)
        model = self._model.to(self.device)

        # Scale: fit per mel-band scaler on pooled data
        N, L, M = X.shape
        X_flat = X.reshape(-1, M)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, L, M)
        y_scaled = self.scaler.transform(y)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, len(X)), shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        losses = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for bX, by in loader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(bX)
                    loss = criterion(pred, by)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / max(len(loader), 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  [Global LSTM] Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new Mel windows (X must be raw, auto-scaled internally)."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        model = self._model.to(self.device)
        model.eval()
        # Scale input
        N, L, M = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, M)).reshape(N, L, M)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_scaled = model(X_t).cpu().numpy()
        return self.scaler.inverse_transform(pred_scaled)

    def evaluate_per_song(
        self, song_mel_specs: Dict[str, np.ndarray],
        song_ids: np.ndarray,
    ) -> Dict:
        """
        Compute RMSE / MAE per song and globally.

        Returns
        -------
        dict with keys:
            per_song : {song_name: {"rmse": float, "mae": float, "n_windows": int}}
            global_rmse, global_mae
        """
        song_names = sorted(song_mel_specs.keys())
        per_song = {}

        all_preds, all_trues = [], []

        for sid, name in enumerate(song_names):
            mel = song_mel_specs[name]
            X_s, y_s = prepare_mel_windows_single(mel, self.lookback, 1)
            if len(X_s) == 0:
                per_song[name] = {"rmse": np.nan, "mae": np.nan, "n_windows": 0}
                continue

            preds = self.predict(X_s)  # returns inverse-transformed
            rmse = float(np.sqrt(np.mean((preds - y_s) ** 2)))
            mae = float(np.mean(np.abs(preds - y_s)))
            per_song[name] = {"rmse": rmse, "mae": mae, "n_windows": len(X_s)}

            all_preds.append(preds)
            all_trues.append(y_s)

        if all_preds:
            ap = np.concatenate(all_preds, axis=0)
            at = np.concatenate(all_trues, axis=0)
            global_rmse = float(np.sqrt(np.mean((ap - at) ** 2)))
            global_mae = float(np.mean(np.abs(ap - at)))
        else:
            global_rmse = global_mae = np.nan

        return {"per_song": per_song, "global_rmse": global_rmse, "global_mae": global_mae}


# ---------------------------------------------------------------------------
# Global Transformer
# ---------------------------------------------------------------------------

class GlobalTransformerPredictor:
    """Transformer trained on pooled Mel windows from multiple songs."""

    def __init__(self, n_mels: int = 128, lookback: int = 30,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        self.n_mels = n_mels
        self.lookback = lookback
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self._model: Optional[nn.Module] = None
        self.device = None
        self.scaler = StandardScaler()

    def _build(self):
        self._model = _GlobalTransformer(
            n_mels=self.n_mels, d_model=self.d_model,
            nhead=self.nhead, num_layers=self.num_layers,
            dropout=self.dropout,
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        epochs: int = 30, lr: float = 0.001,
        batch_size: int = 128, verbose: bool = True,
    ) -> List[float]:
        if self._model is None:
            self._build()
        self.device = _get_device(verbose=verbose)
        model = self._model.to(self.device)

        # Scale: fit per mel-band scaler on pooled data
        N, L, M = X.shape
        X_flat = X.reshape(-1, M)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, L, M)
        y_scaled = self.scaler.transform(y)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(batch_size, len(X)), shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        criterion = nn.MSELoss()
        use_amp = torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        losses = []
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for bX, by in loader:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(bX)
                    loss = criterion(pred, by)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)
            losses.append(avg_loss)
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  [Global Transformer] Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        model = self._model.to(self.device)
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_scaled = model(X_t).cpu().numpy()
        return self.scaler.inverse_transform(pred_scaled)

    def evaluate_per_song(
        self, song_mel_specs: Dict[str, np.ndarray],
        song_ids: np.ndarray,
    ) -> Dict:
        song_names = sorted(song_mel_specs.keys())
        per_song = {}
        all_preds, all_trues = [], []

        for sid, name in enumerate(song_names):
            mel = song_mel_specs[name]
            X_s, y_s = prepare_mel_windows_single(mel, self.lookback, 1)
            if len(X_s) == 0:
                per_song[name] = {"rmse": np.nan, "mae": np.nan, "n_windows": 0}
                continue

            preds = self.predict(X_s)
            rmse = float(np.sqrt(np.mean((preds - y_s) ** 2)))
            mae = float(np.mean(np.abs(preds - y_s)))
            per_song[name] = {"rmse": rmse, "mae": mae, "n_windows": len(X_s)}

            all_preds.append(preds)
            all_trues.append(y_s)

        if all_preds:
            ap = np.concatenate(all_preds, axis=0)
            at = np.concatenate(all_trues, axis=0)
            global_rmse = float(np.sqrt(np.mean((ap - at) ** 2)))
            global_mae = float(np.mean(np.abs(ap - at)))
        else:
            global_rmse = global_mae = np.nan

        return {"per_song": per_song, "global_rmse": global_rmse, "global_mae": global_mae}


# ---------------------------------------------------------------------------
# ARIMA Local Baseline (one ARIMA per song)
# ---------------------------------------------------------------------------

def train_arima_baselines(
    song_mel_specs: Dict[str, np.ndarray],
    forecast_horizon: int = 1,
    verbose: bool = True,
) -> Dict:
    """
    Fit one ARIMA per song on the per-frame mean Mel energy.
    This is the LOCAL baseline — no parameter sharing.

    Returns
    -------
    dict: {song_name: {"rmse": float, "mae": float, "aic": float, "order": str}}
    """
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    results = {}
    for name, mel in sorted(song_mel_specs.items()):
        # Average across Mel bands → 1-D series
        series = np.asarray(mel.mean(axis=0), dtype=np.float64).ravel()

        if len(series) < 20:
            results[name] = {"rmse": np.nan, "mae": np.nan, "aic": np.nan, "order": "N/A", "error": "too short"}
            continue

        train_size = int(len(series) * 0.8)
        train = series[:train_size]
        test = series[train_size:train_size + forecast_horizon]

        # Determine d
        try:
            adf = adfuller(train, maxlag=min(10, len(train) // 3))
            d = 0 if adf[1] < 0.05 else 1
        except Exception:
            d = 1

        best_aic = np.inf
        best_result = None
        best_order = (0, d, 0)

        for p in range(3):
            for q in range(3):
                if p == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train, order=(p, d, q))
                        fitted = model.fit(method_kwargs={"maxiter": 300})
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_result = fitted
                        best_order = (p, d, q)
                except Exception:
                    continue

        if best_result is not None:
            try:
                forecast = best_result.forecast(steps=forecast_horizon)
                forecast = np.atleast_1d(forecast)
                test_arr = np.atleast_1d(test)
                min_len = min(len(forecast), len(test_arr))
                if min_len > 0:
                    rmse = float(np.sqrt(np.mean((forecast[:min_len] - test_arr[:min_len]) ** 2)))
                    mae = float(np.mean(np.abs(forecast[:min_len] - test_arr[:min_len])))
                else:
                    rmse = mae = np.nan
            except Exception:
                rmse = mae = np.nan
        else:
            rmse = mae = np.nan
            best_aic = np.nan
            best_order = (0, 0, 0)

        results[name] = {
            "rmse": rmse, "mae": mae, "aic": float(best_aic),
            "order": f"({best_order[0]},{best_order[1]},{best_order[2]})",
        }

        if verbose:
            print(f"  [ARIMA] {name[:30]:<30s}  order={results[name]['order']}  "
                  f"rmse={rmse:.4f}" if not np.isnan(rmse) else f"  [ARIMA] {name[:30]:<30s}  failed")

    return results


# ---------------------------------------------------------------------------
# Joint HMM — trained on ALL songs' Mel features
# ---------------------------------------------------------------------------

def train_joint_hmm(
    song_mel_specs: Dict[str, np.ndarray],
    n_states: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Fit one HMM on Mel features pooled from ALL songs.

    The HMM discovers latent musical states shared across the genre.
    Each state represents a recurrent timbral/textural pattern.

    To keep dimensionality tractable, we use the per-frame mean across
    Mel bands as the observation (1-D Gaussian HMM).

    Returns
    -------
    dict with:
        n_states, state_means, state_stdevs, transition_matrix,
        state_fractions, per_song_state_sequences, log_likelihood
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        return {"error": "hmmlearn not installed", "n_states": 0}

    song_names = sorted(song_mel_specs.keys())

    # Pool all songs' mean-Mel series
    all_series = []
    song_boundaries = []  # (start_idx, end_idx) for each song
    offset = 0

    for name in song_names:
        mel = song_mel_specs[name]
        series = np.asarray(mel.mean(axis=0), dtype=np.float64).reshape(-1, 1)
        n = len(series)
        if n < 5:
            continue
        all_series.append(series)
        song_boundaries.append((offset, offset + n))
        offset += n

    if not all_series:
        return {"error": "no valid songs", "n_states": 0}

    pooled = np.concatenate(all_series, axis=0)
    n_total = len(pooled)
    actual_states = min(n_states, max(2, n_total // 20))

    # Fit with multiple restarts
    best_model = None
    best_score = -np.inf

    for attempt in range(5):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = hmm.GaussianHMM(
                    n_components=actual_states,
                    covariance_type="diag",
                    n_iter=200,
                    tol=1e-4,
                    random_state=random_state + attempt,
                )
                model.fit(pooled)
            score = model.score(pooled)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception:
            pass

    if best_model is None:
        return {"error": "HMM fit failed", "n_states": 0}

    # Global state sequence
    global_states = best_model.predict(pooled)

    # Per-song state sequences
    per_song_states = {}
    for sidx, (name, (start, end)) in enumerate(zip(song_names, song_boundaries)):
        per_song_states[name] = global_states[start:end].tolist()

    # State fractions (global)
    _, counts = np.unique(global_states, return_counts=True)
    fractions = (counts / counts.sum()).tolist()

    # Transition matrix
    transmat = best_model.transmat_.tolist()

    # State means
    means = best_model.means_.ravel().tolist()
    stdevs = np.sqrt(best_model.covars_.ravel()).tolist() if best_model.covars_.ndim == 2 \
        else [np.sqrt(float(c)) for c in best_model.covars_.ravel()]

    result = {
        "n_states": actual_states,
        "state_means": means,
        "state_stdevs": stdevs,
        "state_fractions": fractions,
        "transition_matrix": transmat,
        "per_song_state_sequences": per_song_states,
        "log_likelihood": float(best_score),
        "n_total_frames": n_total,
        "n_songs": len(song_names),
    }

    if verbose:
        print(f"  [Joint HMM] {actual_states} states discovered from "
              f"{len(song_names)} songs ({n_total} frames)")
        print(f"  [Joint HMM] State fractions: "
              f"{', '.join(f'S{i}: {f:.1%}' for i, f in enumerate(fractions))}")
        print(f"  [Joint HMM] Log-likelihood: {best_score:.1f}")

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_global_models(
    song_mel_specs: Dict[str, np.ndarray],
    lookback: int = 30,
    epochs: int = 30,
    n_hmm_states: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Run all Global ML training phases.

    Parameters
    ----------
    song_mel_specs : {song_name: mel_spec (n_mels, n_frames)}
    lookback : sliding window size
    epochs : training epochs for LSTM and Transformer
    n_hmm_states : number of HMM hidden states
    verbose : print progress

    Returns
    -------
    report : dict with keys:
        lstm, transformer, arima, hmm, summary
    """
    if len(song_mel_specs) < 1:
        return {"error": "No songs provided"}

    report: dict = {}
    n_mels = next(iter(song_mel_specs.values())).shape[0]
    t_start = time.time()

    # ── Prepare pooled dataset ──────────────────────────────────────────
    if verbose:
        print("\n[Global ML] Preparing sliding-window dataset...")
    X_all, y_all, song_ids = prepare_mel_windows_all_songs(
        song_mel_specs, lookback=lookback, forecast_horizon=1,
    )
    n_windows = len(X_all)
    if verbose:
        print(f"  Pooled windows: {n_windows} (lookback={lookback}, n_mels={n_mels})")

    if n_windows < 10:
        report["error"] = f"Too few windows ({n_windows}) — need at least 10"
        return report

    # ── Global LSTM ────────────────────────────────────────────────────
    if verbose:
        print("\n[Global ML] Training Global LSTM...")
    lstm = GlobalLSTMPredictor(
        n_mels=n_mels, lookback=lookback, hidden_size=128, num_layers=2, dropout=0.2,
    )
    lstm_losses = lstm.fit(X_all, y_all, epochs=epochs, verbose=verbose)
    lstm_eval = lstm.evaluate_per_song(song_mel_specs, song_ids)
    report["lstm"] = {
        "loss_history": lstm_losses,
        "final_loss": lstm_losses[-1] if lstm_losses else np.nan,
        "per_song": lstm_eval["per_song"],
        "global_rmse": lstm_eval["global_rmse"],
        "global_mae": lstm_eval["global_mae"],
    }
    if verbose:
        print(f"  [Global LSTM] Final loss: {report['lstm']['final_loss']:.6f}, "
              f"Global RMSE: {report['lstm']['global_rmse']:.4f}")

    # ── Global Transformer ─────────────────────────────────────────────
    if verbose:
        print("\n[Global ML] Training Global Transformer...")
    transformer = GlobalTransformerPredictor(
        n_mels=n_mels, lookback=lookback, d_model=128, nhead=4, num_layers=2, dropout=0.1,
    )
    tf_losses = transformer.fit(X_all, y_all, epochs=epochs, verbose=verbose)
    tf_eval = transformer.evaluate_per_song(song_mel_specs, song_ids)
    report["transformer"] = {
        "loss_history": tf_losses,
        "final_loss": tf_losses[-1] if tf_losses else np.nan,
        "per_song": tf_eval["per_song"],
        "global_rmse": tf_eval["global_rmse"],
        "global_mae": tf_eval["global_mae"],
    }
    if verbose:
        print(f"  [Global Transformer] Final loss: {report['transformer']['final_loss']:.6f}, "
              f"Global RMSE: {report['transformer']['global_rmse']:.4f}")

    # ── ARIMA Local Baselines ──────────────────────────────────────────
    if verbose:
        print("\n[Global ML] Training ARIMA baselines (per-song)...")
    report["arima"] = train_arima_baselines(
        song_mel_specs, forecast_horizon=1, verbose=verbose,
    )

    # ── Joint HMM ──────────────────────────────────────────────────────
    if verbose:
        print("\n[Global ML] Training Joint HMM...")
    report["hmm"] = train_joint_hmm(
        song_mel_specs, n_states=n_hmm_states, verbose=verbose,
    )

    # ── Summary ───────────────────────────────────────────────────────
    arima_rmses = [v["rmse"] for v in report["arima"].values()
                   if not np.isnan(v.get("rmse", np.nan))]
    lstm_rmses = [v["rmse"] for v in report["lstm"]["per_song"].values()
                  if not np.isnan(v.get("rmse", np.nan))]
    tf_rmses = [v["rmse"] for v in report["transformer"]["per_song"].values()
                if not np.isnan(v.get("rmse", np.nan))]

    report["summary"] = {
        "n_songs": len(song_mel_specs),
        "n_windows": n_windows,
        "lookback": lookback,
        "n_mels": n_mels,
        "epochs": epochs,
        "arima_mean_rmse": float(np.mean(arima_rmses)) if arima_rmses else np.nan,
        "lstm_global_rmse": report["lstm"]["global_rmse"],
        "transformer_global_rmse": report["transformer"]["global_rmse"],
        "hmm_n_states": report["hmm"].get("n_states", 0),
        "training_time_s": float(time.time() - t_start),
    }

    if verbose:
        print(f"\n[Global ML] Done in {report['summary']['training_time_s']:.1f}s")
        print(f"  ARIMA (Local)  mean RMSE: {report['summary']['arima_mean_rmse']:.4f}")
        print(f"  LSTM  (Global)       RMSE: {report['summary']['lstm_global_rmse']:.4f}")
        print(f"  Transformer (Global) RMSE: {report['summary']['transformer_global_rmse']:.4f}")

    return report
