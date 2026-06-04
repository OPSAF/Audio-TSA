"""
Volatility Layer — Audio Dynamics Volatility Analysis
======================================================

Computes **volatility (rolling variance)** on top of the four trend time
series produced by ``dynamics.extract_dynamics()``, and optionally fits
ARCH / GARCH models to capture conditional heteroskedasticity.

Layer position
--------------
::

    dynamics.extract_dynamics()         ←  Trend Layer (4 trend series)
            │
            ▼
    volatility.compute_volatility_layer()  ←  Volatility Layer (NEW)
            │
            ├──  Rolling volatility per trend
            ├──  GARCH(1,1) conditional volatility
            ├──  Volatility similarity (cross-audio)
            └──  Volatility prediction input

Integration points
------------------
1. **Prediction** — volatility series become prediction targets
   (``prediction.predict_volatility()``).
2. **Dual Audio Similarity** — volatility similarity feeds into the
   global similarity aggregation as an additional structural dimension.
3. **Trend Layer** — sits directly on top of the existing Trend Layer
   output from ``dynamics.extract_dynamics()``.

Output structure
----------------
::

    vol_layer = {
        "times":          np.ndarray,   # centre time of each window (s)
        "energy_vol":     np.ndarray,   # rolling std of energy trend
        "brightness_vol": np.ndarray,   # rolling std of brightness trend
        "complexity_vol": np.ndarray,   # rolling std of complexity trend
        "rhythm_vol":     np.ndarray,   # rolling std of rhythm trend
        "garch_models":   dict,         # GARCH fit per trend
        "params":         dict,         # window, garch_order, etc.
    }

Dependencies
------------
numpy, scipy (already in project).  statsmodels provides ARCH/GARCH; a
pure-scipy fallback is included so the module never fails on import.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# GARCH fitting — with automatic backend selection
# ---------------------------------------------------------------------------

def _fit_garch_statsmodels(returns: np.ndarray, p: int = 1, q: int = 1) -> Dict:
    """Fit GARCH(p,q) using statsmodels (preferred backend)."""
    try:
        from statsmodels.tsa.arch import arch_model
    except ImportError:
        try:
            from arch import arch_model
        except ImportError:
            return _fit_garch_manual(returns, p=p, q=q)

    try:
        # Scale returns for numerical stability
        scale = returns.std()
        if scale < 1e-12:
            return _make_constant_vol(returns)
        scaled = returns / scale

        model = arch_model(scaled, vol="GARCH", p=p, q=q, dist="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        omega = result.params["omega"] * (scale ** 2)
        alpha = result.params.get("alpha[1]", 0.0)
        beta = result.params.get("beta[1]", 0.0)
        persistence = alpha + beta

        cond_vol = np.sqrt(result.conditional_volatility) * scale

        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistence": float(persistence),
            "half_life": float(
                np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf
            ),
            "conditional_volatility": cond_vol.ravel(),
            "converged": True,
            "backend": "statsmodels",
            "aic": float(result.aic) if hasattr(result, "aic") else None,
            "bic": float(result.bic) if hasattr(result, "bic") else None,
            "log_likelihood": float(result.loglikelihood) if hasattr(result, "loglikelihood") else None,
        }
    except Exception:
        return _fit_garch_manual(returns, p=p, q=q)


def _fit_garch_manual(returns: np.ndarray, p: int = 1, q: int = 1,
                      max_iter: int = 1000) -> Dict:
    """
    Pure scipy GARCH(p,q) MLE fallback.

    GARCH(1,1):
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    Constraints:
        ω > 0,  α ≥ 0,  β ≥ 0,  α + β < 1  (covariance stationarity)
    """
    from scipy.optimize import minimize

    n = len(returns)
    if n < 10:
        return _make_constant_vol(returns)

    # Pre-compute squared returns
    r2 = returns ** 2

    def nll(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 1e-12 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e15

        sigma2 = np.empty(n)
        sigma2[0] = r2[0] + omega  # initialise with unconditional variance upper bound

        for t in range(1, n):
            sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]

        if np.any(sigma2 <= 1e-12):
            return 1e15

        return float(0.5 * np.sum(np.log(sigma2) + r2 / sigma2))

    # Smart initial guesses - try multiple starting points
    var_r = float(np.var(returns))
    initial_guesses = [
        # Standard starting point
        (var_r * 0.05, 0.10, min(0.80, 0.999 - 0.10 - 0.01)),
        # Lower alpha, higher beta
        (var_r * 0.01, 0.05, 0.90),
        # Higher alpha, lower beta
        (var_r * 0.10, 0.20, 0.70),
        # Very low omega
        (var_r * 0.001, 0.15, 0.75),
        # Equal alpha and beta
        (var_r * 0.02, 0.12, 0.12),
    ]

    for omega0, alpha0, beta0 in initial_guesses:
        try:
            result = minimize(
                nll,
                np.array([omega0, alpha0, beta0]),
                method="L-BFGS-B",
                bounds=[(1e-10, None), (0, 0.5), (0, 0.999)],
                options={"maxiter": max_iter, "disp": False},
            )
            
            if result.success and result.fun < 1e10:
                omega, alpha, beta = result.x
                persistence = alpha + beta

                # Ensure valid persistence
                if persistence >= 1.0:
                    continue

                # Compute conditional volatility
                sigma2 = np.empty(n)
                sigma2[0] = omega / (1 - persistence) if persistence < 1 else var_r / 2.0
                for t in range(1, n):
                    sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]

                return {
                    "omega": float(omega),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "persistence": float(persistence),
                    "half_life": float(
                        np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf
                    ),
                    "conditional_volatility": np.sqrt(np.maximum(sigma2, 1e-12)),
                    "converged": True,
                    "backend": "scipy",
                }
        except Exception:
            continue

    return _make_constant_vol(returns)


def _make_constant_vol(returns: np.ndarray) -> Dict:
    """Return constant-volatility result when GARCH fitting is impossible."""
    std = float(np.std(returns))
    n = len(returns)
    return {
        "omega": std ** 2,
        "alpha": 0.0,
        "beta": 0.0,
        "persistence": 0.0,
        "half_life": 0.0,
        "conditional_volatility": np.full(n, std),
        "converged": False,
        "backend": "constant",
    }


def _fit_garch(series: np.ndarray, p: int = 1, q: int = 1) -> Dict:
    """
    Auto-select GARCH backend.

    Parameters
    ----------
    series : ndarray    Trend time series (will be de-meaned for GARCH).
    p, q : int          GARCH order.

    Returns
    -------
    dict with keys: omega, alpha, beta, persistence, half_life,
         conditional_volatility, converged, backend.
    """
    # De-mean to get (approximate) zero-mean returns
    returns = series - np.mean(series)
    if len(returns) < 10:
        return _make_constant_vol(returns)

    # Try statsmodels first, fall back to scipy
    return _fit_garch_statsmodels(returns, p=p, q=q)


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------

def _rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation with edge handling."""
    n = len(series)
    if n < 2 or window < 2:
        return np.zeros(n)

    window = min(window, n)
    result = np.empty(n)

    # First window-1 values: expanding-window std
    for i in range(window - 1):
        result[i] = float(np.std(series[: i + 2]))

    # Rolling window
    for i in range(window - 1, n):
        result[i] = float(np.std(series[i - window + 1: i + 1]))

    result = np.nan_to_num(result, nan=0.0, posinf=0.0)
    return result


def compute_rolling_volatility(
    trend_series: np.ndarray,
    window: int = 10,
) -> np.ndarray:
    """
    Compute rolling volatility (standard deviation) of a trend series.

    Parameters
    ----------
    trend_series : ndarray    A single trend (e.g. energy, brightness).
    window : int              Number of windows for the rolling std.

    Returns
    -------
    vol_series : ndarray      Same length as input.  First ``window-1``
                               values use an expanding window.
    """
    return _rolling_std(np.asarray(trend_series, dtype=np.float64).ravel(), window)


# ---------------------------------------------------------------------------
# Volatility Layer — main API
# ---------------------------------------------------------------------------

def compute_volatility_layer(
    dynamics: Dict,
    rolling_window: int = 10,
    garch_order: Tuple[int, int] = (1, 1),
    fit_garch: bool = True,
) -> Dict:
    """
    Compute the Volatility Layer on top of Trend Layer output.

    Parameters
    ----------
    dynamics : dict          Output of ``dynamics.extract_dynamics()``.
    rolling_window : int     Window size (in number of trend windows) for
                              rolling volatility.
    garch_order : (p, q)     ARCH/GARCH order.
    fit_garch : bool         If True, fit GARCH model to each trend residual.

    Returns
    -------
    vol_layer : dict         See module docstring for canonical structure.
    """
    energy = np.asarray(dynamics["energy"], dtype=np.float64).ravel()
    brightness = np.asarray(dynamics["brightness"], dtype=np.float64).ravel()
    complexity = np.asarray(dynamics["complexity"], dtype=np.float64).ravel()
    rhythm = np.asarray(dynamics["rhythm"], dtype=np.float64).ravel()
    times = np.asarray(dynamics["times"], dtype=np.float64).ravel()

    n = len(energy)
    rw = min(rolling_window, max(2, n // 3))

    # Rolling volatility per trend
    energy_vol = compute_rolling_volatility(energy, window=rw)
    brightness_vol = compute_rolling_volatility(brightness, window=rw)
    complexity_vol = compute_rolling_volatility(complexity, window=rw)
    rhythm_vol = compute_rolling_volatility(rhythm, window=rw)

    # ---- GARCH fitting (on residuals) ----
    garch_models: Dict = {}
    if fit_garch and n >= 20:
        p, q = garch_order
        for key, arr in [
            ("energy", energy),
            ("brightness", brightness),
            ("complexity", complexity),
            ("rhythm", rhythm),
        ]:
            garch_models[key] = _fit_garch(arr, p=p, q=q)
    else:
        for key in ["energy", "brightness", "complexity", "rhythm"]:
            arr = dynamics[key]
            garch_models[key] = _make_constant_vol(
                np.asarray(arr, dtype=np.float64).ravel()
            )

    result: Dict = {
        "times": times,
        "energy_vol": energy_vol,
        "brightness_vol": brightness_vol,
        "complexity_vol": complexity_vol,
        "rhythm_vol": rhythm_vol,
        "energy_vol_norm": (energy_vol - energy_vol.mean())
        / (energy_vol.std() + 1e-12),
        "brightness_vol_norm": (brightness_vol - brightness_vol.mean())
        / (brightness_vol.std() + 1e-12),
        "complexity_vol_norm": (complexity_vol - complexity_vol.mean())
        / (complexity_vol.std() + 1e-12),
        "rhythm_vol_norm": (rhythm_vol - rhythm_vol.mean())
        / (rhythm_vol.std() + 1e-12),
        "garch_models": garch_models,
        "params": {
            "rolling_window": rw,
            "garch_order": garch_order,
            "fit_garch": fit_garch,
            "n_windows": n,
        },
    }

    return result


# ---------------------------------------------------------------------------
# Volatility statistics
# ---------------------------------------------------------------------------

def summarize_volatility(vol_layer: Dict) -> Dict:
    """
    Compute per-dimension volatility statistics.

    Parameters
    ----------
    vol_layer : dict    Output of ``compute_volatility_layer()``.

    Returns
    -------
    summary : dict
        Keys: energy, brightness, complexity, rhythm.
        Each value is a dict with: mean_vol, std_vol, max_vol,
        vol_of_vol, garch_persistence, garch_half_life, garch_converged,
        volatility_regime.
    """
    summary = {}

    for key in ["energy", "brightness", "complexity", "rhythm"]:
        vol_arr = vol_layer[f"{key}_vol"]
        n = len(vol_arr)

        if n < 2:
            val = float(vol_arr[0]) if n == 1 else 0.0
            summary[key] = {
                "mean_vol": val,
                "std_vol": 0.0,
                "max_vol": val,
                "vol_of_vol": 0.0,
                "garch_persistence": None,
                "garch_half_life": None,
                "garch_converged": False,
                "gateway_backend": None,
                "volatility_regime": "low",
            }
            continue

        garch = vol_layer.get("garch_models", {}).get(key, {})

        mean_v = float(vol_arr.mean())
        std_v = float(vol_arr.std())
        max_v = float(vol_arr.max())

        # Vol-of-vol: std of rolling vol relative to mean vol
        vov = float(std_v / (mean_v + 1e-12))

        # Regime classification
        if mean_v > 0.1:
            regime = "high"
        elif mean_v > 0.03:
            regime = "medium"
        else:
            regime = "low"

        summary[key] = {
            "mean_vol": mean_v,
            "std_vol": std_v,
            "max_vol": max_v,
            "vol_of_vol": vov,
            "garch_persistence": garch.get("persistence"),
            "garch_half_life": garch.get("half_life"),
            "garch_converged": garch.get("converged", False),
            "garch_backend": garch.get("backend"),
            "garch_omega": garch.get("omega"),
            "garch_alpha": garch.get("alpha"),
            "garch_beta": garch.get("beta"),
            "volatility_regime": regime,
            "n_windows": n,
        }

    return summary


# ---------------------------------------------------------------------------
# Volatility similarity
# ---------------------------------------------------------------------------

def _resample_1d(a: np.ndarray, b: np.ndarray,
                 target_len: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Resample two 1-D arrays to the same target length."""
    x_old_a = np.linspace(0, 1, len(a))
    x_old_b = np.linspace(0, 1, len(b))
    x_new = np.linspace(0, 1, target_len)
    a_rs = np.interp(x_new, x_old_a, a)
    b_rs = np.interp(x_new, x_old_b, b)
    return a_rs, b_rs


def compute_volatility_similarity(
    vol1: Dict,
    vol2: Dict,
) -> Dict:
    """
    Compute similarity of volatility structure between two audio files.

    Measures
    --------
    * **pearson_r**          — linear correlation of volatility profiles
    * **derivative_r**       — shape similarity of volatility curves
    * **distribution_sim**   — Wasserstein-based overlap of vol distributions
    * **garch_param_sim**    — similarity of GARCH(1,1) parameters
    * **structural_score**   — weighted aggregate

    Parameters
    ----------
    vol1, vol2 : dict    Outputs of ``compute_volatility_layer()``.

    Returns
    -------
    scores : dict with per_trend scores and global_volatility_similarity.
    """
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    per_trend = {}
    scores = []

    for key in trend_keys:
        a = np.asarray(vol1[f"{key}_vol"], dtype=np.float64).ravel()
        b = np.asarray(vol2[f"{key}_vol"], dtype=np.float64).ravel()

        target_len = max(50, min(300, (len(a) + len(b)) // 2))
        a_rs, b_rs = _resample_1d(a, b, target_len)

        # Pearson correlation
        if np.std(a_rs) > 0 and np.std(b_rs) > 0:
            pearson_r, _ = stats.pearsonr(a_rs, b_rs)
            pearson_r = max(0.0, float(pearson_r))
        else:
            pearson_r = 0.0

        # Derivative correlation (shape)
        da = np.gradient(a_rs)
        db = np.gradient(b_rs)
        if np.std(da) > 0 and np.std(db) > 0:
            deriv_r, _ = stats.pearsonr(da, db)
            deriv_r = max(0.0, float(deriv_r))
        else:
            deriv_r = pearson_r

        # Distribution overlap (Wasserstein via CDF)
        eps = 1e-12
        bins = min(30, max(10, target_len // 5))
        hist_a, edges = np.histogram(a_rs, bins=bins, density=True)
        hist_b, _ = np.histogram(b_rs, bins=edges, density=True)
        cdf_a = np.cumsum(hist_a) / (hist_a.sum() + eps)
        cdf_b = np.cumsum(hist_b) / (hist_b.sum() + eps)
        emd = float(np.sum(np.abs(cdf_a - cdf_b)) * (edges[1] - edges[0]))
        distribution_sim = float(np.exp(-emd * 2.0))

        # GARCH parameter similarity
        g1 = vol1.get("garch_models", {}).get(key, {})
        g2 = vol2.get("garch_models", {}).get(key, {})
        p1 = g1.get("persistence", 0.0) or 0.0
        p2 = g2.get("persistence", 0.0) or 0.0
        garch_param_sim = float(np.exp(-abs(p1 - p2) * 3.0))

        # Structural score (volatility shape > distribution > GARCH params)
        structural_score = float(
            0.30 * pearson_r
            + 0.35 * deriv_r
            + 0.20 * distribution_sim
            + 0.15 * garch_param_sim
        )
        structural_score = np.clip(structural_score, 0.0, 1.0)

        per_trend[key] = {
            "trend": key,
            "pearson_r": pearson_r,
            "derivative_r": deriv_r,
            "distribution_sim": distribution_sim,
            "garch_param_sim": garch_param_sim,
            "structural_score": structural_score,
        }
        scores.append(structural_score)

    # Global: energy-vol and rhythm-vol weighted higher
    weights = [0.35, 0.20, 0.15, 0.30]  # energy, brightness, complexity, rhythm
    global_score = float(np.dot(scores, weights))
    global_score_pct = float(np.clip(global_score, 0.0, 1.0) * 100.0)

    dominant_idx = int(np.argmax(scores))
    dominant_trend = trend_keys[dominant_idx]

    # Coherence: how consistent are the volatility similarities
    coherence = 1.0 - float(np.std(scores))

    return {
        "per_trend": per_trend,
        "global_volatility_similarity": global_score_pct,
        "dominant_trend": dominant_trend,
        "volatility_coherence": float(np.clip(coherence, 0.0, 1.0)),
        "trend_weights": dict(zip(trend_keys, weights)),
    }


# ---------------------------------------------------------------------------
# High-level: combined Trend + Volatility analysis
# ---------------------------------------------------------------------------

def analyze_audio_dynamics(
    y: np.ndarray,
    sr: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    rolling_window: int = 10,
    fit_garch: bool = True,
    garch_order: Tuple[int, int] = (1, 1),
) -> Dict:
    """
    Run full Trend Layer + Volatility Layer analysis on a single audio.

    This is the high-level entry point.  It calls
    ``dynamics.extract_dynamics()`` (Trend Layer) and then
    ``compute_volatility_layer()`` (Volatility Layer) in sequence.

    Parameters
    ----------
    y : ndarray           Raw audio waveform (mono).
    sr : int              Sample rate (Hz).
    window_size : float   Trend extraction window in seconds.
    hop_size : float      Hop size in seconds.
    rolling_window : int  Window size (in trend steps) for rolling vol.
    fit_garch : bool      Fit ARCH/GARCH models.
    garch_order : (p, q)  GARCH order.

    Returns
    -------
    result : dict
        Keys:
        - ``trend_layer`` : the dynamics dict from extract_dynamics()
        - ``volatility_layer`` : the vol_layer dict
        - ``trend_summary`` : from summarize_dynamics()
        - ``volatility_summary`` : from summarize_volatility()
        - ``params`` : consolidated parameters
    """
    from .dynamics import extract_dynamics, summarize_dynamics

    trend_layer = extract_dynamics(y, sr, window_size=window_size, hop_size=hop_size)
    volatility_layer = compute_volatility_layer(
        trend_layer,
        rolling_window=rolling_window,
        garch_order=garch_order,
        fit_garch=fit_garch,
    )
    trend_summary = summarize_dynamics(trend_layer)
    vol_summary = summarize_volatility(volatility_layer)

    return {
        "trend_layer": trend_layer,
        "volatility_layer": volatility_layer,
        "trend_summary": trend_summary,
        "volatility_summary": vol_summary,
        "params": {
            "window_size": window_size,
            "hop_size": hop_size,
            "rolling_window": rolling_window,
            "garch_order": garch_order,
            "fit_garch": fit_garch,
            "sr": sr,
            "n_windows": trend_layer["params"]["n_windows"],
        },
    }


# ---------------------------------------------------------------------------
# Volatility-based predictions bridge
# ---------------------------------------------------------------------------

def prepare_volatility_prediction_data(
    vol_layer: Dict,
    trend_key: str = "energy",
    lookback: int = 30,
    forecast_horizon: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sliding-window data for predicting volatility.

    Parameters
    ----------
    vol_layer : dict       Output of ``compute_volatility_layer()``.
    trend_key : str        Which volatility dimension to use.
    lookback : int         Lookback window length.
    forecast_horizon : int Forecast steps.

    Returns
    -------
    X : (n_samples, lookback) ndarray
    y : (n_samples, forecast_horizon) ndarray
    """
    series = vol_layer[f"{trend_key}_vol"]
    X, y_list = [], []
    n = len(series)
    for i in range(n - lookback - forecast_horizon + 1):
        X.append(series[i: i + lookback])
        y_list.append(series[i + lookback: i + lookback + forecast_horizon])
    if len(X) == 0:
        return np.zeros((0, lookback)), np.zeros((0, forecast_horizon))
    return np.array(X), np.array(y_list)


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_volatility_report(
    vol_layer: Dict,
    show_garch: bool = True,
):
    """Print a formatted volatility analysis report."""
    vol_summary = summarize_volatility(vol_layer)
    params = vol_layer["params"]
    n = params["n_windows"]

    print()
    print("=" * 70)
    print("  VOLATILITY LAYER — Rolling Volatility & GARCH Analysis")
    print("=" * 70)
    print(f"  Windows: {n}  |  Rolling window: {params['rolling_window']} steps  "
          f"|  GARCH{params['garch_order']}")
    print("-" * 70)

    # Volatility statistics table
    header = (
        f"  {'Trend':<14s} {'MeanVol':>9s} {'StdVol':>9s} "
        f"{'MaxVol':>9s} {'VoV':>7s} {'Regime':>8s}"
    )
    print(header)
    print("  " + "-" * len(header))
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = vol_summary[key]
        print(
            f"  {key:<14s} {s['mean_vol']:9.5f} {s['std_vol']:9.5f} "
            f"{s['max_vol']:9.5f} {s['vol_of_vol']:7.3f} "
            f"{s['volatility_regime']:>8s}"
        )
    print("-" * 70)

    # GARCH parameters
    if show_garch:
        any_converged = any(
            vol_summary[k].get("garch_converged") for k in ["energy", "brightness", "complexity", "rhythm"]
        )
        if any_converged:
            print()
            garch_header = (
                f"  {'Trend':<14s} {'ω':>10s} {'α':>8s} {'β':>8s} "
                f"{'α+β':>8s} {'HalfLife':>9s} {'Backend':>12s}"
            )
            print(garch_header)
            print("  " + "-" * len(garch_header))
            for key in ["energy", "brightness", "complexity", "rhythm"]:
                s = vol_summary[key]
                if s.get("garch_converged"):
                    omega = s.get("garch_omega", 0)
                    alpha = s.get("garch_alpha", 0)
                    beta = s.get("garch_beta", 0)
                    persistence = s.get("garch_persistence", 0) or 0
                    hl = s.get("garch_half_life", np.inf)
                    hl_str = f"{hl:.1f}" if hl != np.inf else "∞"
                    backend = s.get("garch_backend", "?")
                    print(
                        f"  {key:<14s} {omega:10.6f} {alpha:8.4f} {beta:8.4f} "
                        f"{persistence:8.4f} {hl_str:>9s} {backend:>12s}"
                    )

    print("=" * 70)
    print()


def print_volatility_similarity_report(sim_result: Dict):
    """Print a formatted volatility similarity report."""
    per_trend = sim_result["per_trend"]

    print()
    print("-" * 70)
    print("  VOLATILITY SIMILARITY (Structure-Aware)")
    print("-" * 70)
    header = (
        f"  {'Trend':<14s} {'Pearson':>8s} {'Deriv':>7s} "
        f"{'Dist':>7s} {'GARCH':>7s} {'Struct':>8s}"
    )
    print(header)
    print("  " + "-" * len(header))
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = per_trend[key]
        print(
            f"  {key:<14s} {s['pearson_r']:8.3f} {s['derivative_r']:7.3f} "
            f"{s['distribution_sim']:7.3f} {s['garch_param_sim']:7.3f} "
            f"{s['structural_score']:8.3f}"
        )
    print("  " + "-" * len(header))
    print(f"  GLOBAL VOLATILITY SIMILARITY: {sim_result['global_volatility_similarity']:.1f}%")
    print(f"  Dominant volatility trend: {sim_result['dominant_trend']}")
    print(f"  Volatility coherence: {sim_result['volatility_coherence']:.3f}")
    print("-" * 70)
    print()
