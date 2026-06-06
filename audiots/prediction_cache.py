"""Prediction model cache — eliminates redundant deep-model training.

In the analysis pipeline the same trend / volatility / mel-band series can
be fed to LSTM or Transformer predictors multiple times from different code
paths (``predict_all_trends``, ``analyze_lstm_insights``,
``run_all_predictions``, ``analyze_band_predictability``).

This module provides a thread-safe in-memory cache that stores ``(forecast,
metrics)`` tuples keyed by a content hash of the input series together with
the model hyper-parameters.  When a cache hit occurs the expensive GPU
training is skipped entirely.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PredictionCache:
    """Thread-safe LRU-ish cache for deep-model prediction results.

    Keys are deterministic content hashes; identical inputs always map to the
    same key regardless of how many times or from which code path
    ``predict()`` is called.
    """

    def __init__(self, max_size: int = 30) -> None:
        self._cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self._max_size = max_size
        self._access_order: list = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        series: np.ndarray,
        model_type: str,
        lookback: int,
        forecast_horizon: int,
        epochs: int,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Return cached ``(forecast, metrics)`` or ``None`` on miss."""
        key = self._make_key(series, model_type, lookback, forecast_horizon, epochs)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)  # move to end (most recent)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
        return None

    def put(
        self,
        series: np.ndarray,
        model_type: str,
        lookback: int,
        forecast_horizon: int,
        epochs: int,
        result: Tuple[np.ndarray, Dict[str, Any]],
    ) -> None:
        """Store a prediction result in the cache."""
        key = self._make_key(series, model_type, lookback, forecast_horizon, epochs)
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
            self._cache[key] = result
            self._access_order.append(key)
            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                old_key = self._access_order.pop(0)
                del self._cache[old_key]

    def clear(self) -> None:
        """Empty the cache (useful between test runs)."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Return hit/miss counters."""
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "max_size": self._max_size,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_key(
        self,
        series: np.ndarray,
        model_type: str,
        lookback: int,
        forecast_horizon: int,
        epochs: int,
    ) -> str:
        """Build a stable cache key from input data and hyper-parameters."""
        # Round to float32 for stability — tiny float64 diffs shouldn't
        # invalidate the cache.
        series_bytes = np.asarray(series, dtype=np.float32).ravel().tobytes()
        h = hashlib.sha256(series_bytes).hexdigest()[:16]
        return f"{h}_{model_type}_lb{lookback}_fh{forecast_horizon}_ep{epochs}"


# ------------------------------------------------------------------
# Global singleton — import this everywhere
# ------------------------------------------------------------------

_prediction_cache = PredictionCache(max_size=30)


def get_prediction_cache() -> PredictionCache:
    """Return the module-level singleton PredictionCache."""
    return _prediction_cache
