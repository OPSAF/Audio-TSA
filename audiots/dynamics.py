"""
Audio Dynamics Layer / Trend Analysis
======================================

Feature enhancement layer inserted between **Feature Extraction** and
**Time Series Analysis**.  Extracts four time-series-level audio dynamic
features via sliding-window + hop-based processing, then provides
structural segmentation and structure-aware dual-audio comparison.

All computations use numpy/scipy only — no deep learning frameworks.

Provides
--------
* ``extract_dynamics()``          — 4 core trend time series
* ``detect_structural_segments()`` — climax / calm / buildup / transition
* ``compute_trend_similarity()``   — per-trend structural similarity
* ``compute_dynamics_similarity()``— unified dynamics similarity score
* ``summarize_dynamics()``         — statistical summary per trend
* ``print_dynamics_report()``      — formatted console report

Integration points
------------------
1. **Time Series Analysis** — ACF/PACF can be computed on each trend.
2. **Prediction**           — Energy / Brightness trends replace or
   augment Mel-band prediction targets.
3. **Dual Audio Similarity**— trend-level similarity scores feed into
   the global similarity aggregation (structure-aware).

Output structure (always this shape)
-------------------------------------
::

    dynamics = {
        "times":        np.ndarray,   # centre time of each window (s)
        "energy":       np.ndarray,   # RMS energy per window
        "brightness":   np.ndarray,   # spectral centroid per window
        "complexity":   np.ndarray,   # spectral entropy per window
        "rhythm":       np.ndarray,   # onset density per window
        "energy_norm":  np.ndarray,   # z-score normalised
        "brightness_norm": np.ndarray,
        "complexity_norm": np.ndarray,
        "rhythm_norm":  np.ndarray,
        "params":       dict,         # window_size, hop_size, sr, n_windows
    }
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float64(y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y, dtype=np.float64)


def _fft_window(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of a short window."""
    n = len(y)
    if n < 2:
        return np.array([0]), np.array([0])
    Y = np.fft.rfft(y)
    mag = np.abs(Y) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, mag


def _onset_strength(y: np.ndarray) -> float:
    """Onset strength via spectral flux (energy difference)."""
    energy = y ** 2
    flux = np.diff(energy)
    return float(np.mean(np.abs(flux)))


def _spectral_centroid(freqs: np.ndarray, mag: np.ndarray) -> float:
    s = mag.sum()
    if s == 0:
        return 0.0
    return float(np.dot(freqs, mag) / s)


def _spectral_entropy(mag: np.ndarray) -> float:
    s = mag.sum()
    if s == 0:
        return 0.0
    p = mag / s
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


# ---------------------------------------------------------------------------
# 1. Core extraction
# ---------------------------------------------------------------------------

def extract_dynamics(
    y: np.ndarray,
    sr: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    normalize: bool = True,
) -> Dict:
    """
    Extract four trend time series from an audio signal.

    Parameters
    ----------
    y : ndarray           Raw audio waveform (mono).
    sr : int              Sample rate (Hz).
    window_size : float   Analysis window in seconds.
    hop_size : float      Hop between consecutive windows in seconds.
    normalize : bool      If True, also compute z-score normalised versions.

    Returns
    -------
    dynamics : dict       See module docstring for the canonical structure.
    """
    y = _as_float64(y).ravel()
    win_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    if win_samples > len(y):
        win_samples = len(y)
        hop_samples = max(1, win_samples // 4)

    n_windows = max(1, (len(y) - win_samples) // hop_samples + 1)

    times = np.zeros(n_windows)
    energy = np.zeros(n_windows)
    brightness = np.zeros(n_windows)
    complexity = np.zeros(n_windows)
    rhythm = np.zeros(n_windows)

    # Pre-emphasis (optional but helps onset detection)
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    for i in range(n_windows):
        start = i * hop_samples
        end = start + win_samples
        win = y[start:end]
        win_pre = y_pre[start:end]

        # --- Energy (RMS) ---
        rms = float(np.sqrt(np.mean(win ** 2)))
        energy[i] = rms

        # --- Brightness (spectral centroid) ---
        freqs, mag = _fft_window(win, sr)
        brightness[i] = _spectral_centroid(freqs, mag)

        # --- Complexity (spectral entropy) ---
        complexity[i] = _spectral_entropy(mag)

        # --- Rhythm (onset density) ---
        rhythm[i] = _onset_strength(win_pre)

        times[i] = (start + end) / 2.0 / sr

    result: Dict = {
        "times": times,
        "energy": energy,
        "brightness": brightness,
        "complexity": complexity,
        "rhythm": rhythm,
        "params": {
            "window_size": window_size,
            "hop_size": hop_size,
            "sr": sr,
            "n_windows": n_windows,
        },
    }

    if normalize:
        eps = 1e-12
        result["energy_norm"] = (energy - energy.mean()) / (energy.std() + eps)
        result["brightness_norm"] = (brightness - brightness.mean()) / (brightness.std() + eps)
        result["complexity_norm"] = (complexity - complexity.mean()) / (complexity.std() + eps)
        result["rhythm_norm"] = (rhythm - rhythm.mean()) / (rhythm.std() + eps)

    return result


# ---------------------------------------------------------------------------
# 2. Structural segment detection
# ---------------------------------------------------------------------------

# Labels used for structural segmentation
_SEGMENT_LABELS = {
    "climax":      "climax / high-energy peak",
    "calm":        "calm / low-energy valley",
    "buildup":     "buildup / rising intensity",
    "release":     "release / falling intensity",
    "transition":  "transition / high complexity",
    "sustained":   "sustained / stable",
}


def detect_structural_segments(
    dynamics: Dict,
    climax_percentile: float = 85.0,
    calm_percentile: float = 25.0,
    peak_distance: float = 2.0,
) -> Dict:
    """
    Classify each window into a structural role.

    Rules (priority order)
    ----------------------
    1. **climax**      — energy > *climax_percentile* AND rhythm >
                         *climax_percentile*
    2. **calm**         — energy < *calm_percentile* AND rhythm <
                         *calm_percentile*
    3. **buildup**      — energy derivative > +1 std (rising phase)
    4. **release**      — energy derivative < -1 std (falling phase)
    5. **transition**   — complexity > 80th percentile (unstable spectrum)
    6. **sustained**    — everything else

    Parameters
    ----------
    dynamics : dict          Output of ``extract_dynamics()``.
    climax_percentile : float  Percentile threshold for high energy.
    calm_percentile : float    Percentile threshold for low energy.
    peak_distance : float      Minimum distance (seconds) between
                               consecutive peaks of the same label.

    Returns
    -------
    segments : dict
        Keys: ``labels`` (str array, one per window),
        ``climax_indices``, ``calm_indices``, ``buildup_indices``,
        ``transition_indices``, ``energy_derivative``,
        ``energy_peaks``, ``energy_valleys``.
    """
    energy = dynamics["energy"]
    brightness = dynamics["brightness"]
    complexity = dynamics["complexity"]
    rhythm = dynamics["rhythm"]
    times = dynamics["times"]
    n = len(energy)

    if n < 3:
        return {
            "labels": np.full(n, "sustained"),
            "climax_indices": [],
            "calm_indices": [],
            "buildup_indices": [],
            "transition_indices": [],
            "energy_derivative": np.zeros(n),
            "energy_peaks": [],
            "energy_valleys": [],
        }

    # Thresholds
    e_hi = np.percentile(energy, climax_percentile)
    e_lo = np.percentile(energy, calm_percentile)
    r_hi = np.percentile(rhythm, climax_percentile)
    r_lo = np.percentile(rhythm, calm_percentile)
    c_hi = np.percentile(complexity, 80)

    # Energy derivative (smoothed)
    from scipy.ndimage import uniform_filter1d
    energy_smooth = uniform_filter1d(energy, size=max(1, n // 10))
    e_deriv = np.gradient(energy_smooth)
    e_deriv_std = np.std(e_deriv) + 1e-12
    e_deriv_norm = e_deriv / e_deriv_std

    # Classify
    labels = np.full(n, "sustained", dtype=object)
    for i in range(n):
        if energy[i] >= e_hi and rhythm[i] >= r_hi:
            labels[i] = "climax"
        elif energy[i] <= e_lo and rhythm[i] <= r_lo:
            labels[i] = "calm"
        elif e_deriv_norm[i] > 1.0:
            labels[i] = "buildup"
        elif e_deriv_norm[i] < -1.0:
            labels[i] = "release"
        elif complexity[i] >= c_hi:
            labels[i] = "transition"

    # Find energy peaks & valleys
    peak_samples = max(1, int(peak_distance / dynamics["params"]["hop_size"]))
    peaks, _ = find_peaks(energy, distance=peak_samples, prominence=energy.std() * 0.3)
    valleys, _ = find_peaks(-energy, distance=peak_samples, prominence=energy.std() * 0.3)

    climax_idx = [int(i) for i in np.where(labels == "climax")[0]]
    calm_idx = [int(i) for i in np.where(labels == "calm")[0]]
    buildup_idx = [int(i) for i in np.where(labels == "buildup")[0]]
    transition_idx = [int(i) for i in np.where(labels == "transition")[0]]

    return {
        "labels": labels,
        "climax_indices": climax_idx,
        "calm_indices": calm_idx,
        "buildup_indices": buildup_idx,
        "transition_indices": transition_idx,
        "energy_derivative": e_deriv,
        "energy_peaks": peaks.tolist(),
        "energy_valleys": valleys.tolist(),
        "label_descriptions": _SEGMENT_LABELS,
        "params": {
            "climax_percentile": climax_percentile,
            "calm_percentile": calm_percentile,
            "peak_distance": peak_distance,
        },
    }


# ---------------------------------------------------------------------------
# 3. Per-trend structural similarity
# ---------------------------------------------------------------------------

def _resample_to_match(a: np.ndarray, b: np.ndarray, target_len: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Resample both series to the same length via interpolation."""
    x_old_a = np.linspace(0, 1, len(a))
    x_old_b = np.linspace(0, 1, len(b))
    x_new = np.linspace(0, 1, target_len)
    a_resampled = np.interp(x_new, x_old_a, a)
    b_resampled = np.interp(x_new, x_old_b, b)
    return a_resampled, b_resampled


def compute_trend_similarity(
    dyn1: Dict,
    dyn2: Dict,
    trend_key: str = "energy",
) -> Dict:
    """
    Compute structural similarity between two dynamics for a single trend.

    Measures (all in [0, 1], higher = more similar)
    ------------------------------------------------
    * **pearson_r**      — linear correlation after length normalisation
    * **spearman_r**     — rank correlation (monotonic relationship)
    * **derivative_r**   — Pearson on first derivative (shape similarity)
    * **peak_overlap**   — Jaccard-like overlap of peak positions
    * **distribution_emd** — Earth Mover's Distance on histograms (inverted)

    The **structural_score** is a weighted average emphasising derivative
    correlation and peak overlap (shape > value).

    Parameters
    ----------
    dyn1, dyn2 : dict   Outputs of ``extract_dynamics()``.
    trend_key : str     One of "energy", "brightness", "complexity", "rhythm".

    Returns
    -------
    scores : dict with keys pearson_r, spearman_r, derivative_r,
             peak_overlap, distribution_sim, structural_score.
    """
    a = dyn1.get(trend_key, dyn1.get(f"{trend_key}_norm", dyn1["energy"]))
    b = dyn2.get(trend_key, dyn2.get(f"{trend_key}_norm", dyn2["energy"]))

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    # Resample to common length
    target_len = max(50, min(300, (len(a) + len(b)) // 2))
    a_rs, b_rs = _resample_to_match(a, b, target_len)

    # --- Pearson (linear) ---
    pearson_r, _ = stats.pearsonr(a_rs, b_rs) if np.std(a_rs) > 0 and np.std(b_rs) > 0 else (0.0, 1.0)
    pearson_r = max(0.0, float(pearson_r))

    # --- Spearman (rank / monotonic) ---
    spearman_r, _ = stats.spearmanr(a_rs, b_rs) if np.std(a_rs) > 0 and np.std(b_rs) > 0 else (0.0, 1.0)
    spearman_r = max(0.0, float(spearman_r))

    # --- Derivative correlation (shape) ---
    da = np.gradient(a_rs)
    db = np.gradient(b_rs)
    if np.std(da) > 0 and np.std(db) > 0:
        deriv_r, _ = stats.pearsonr(da, db)
        deriv_r = max(0.0, float(deriv_r))
    else:
        deriv_r = pearson_r

    # --- Peak overlap ---
    eps = 1e-12
    a_norm = (a_rs - a_rs.min()) / (a_rs.max() - a_rs.min() + eps)
    b_norm = (b_rs - b_rs.min()) / (b_rs.max() - b_rs.min() + eps)
    peaks_a = set(find_peaks(a_norm, height=0.6, distance=max(1, target_len // 20))[0].tolist())
    peaks_b = set(find_peaks(b_norm, height=0.6, distance=max(1, target_len // 20))[0].tolist())
    if peaks_a or peaks_b:
        peak_overlap = len(peaks_a & peaks_b) / max(1, len(peaks_a | peaks_b))
    else:
        peak_overlap = 1.0

    # --- Distribution similarity (Earth Mover's Distance via histograms) ---
    bins = min(30, max(10, target_len // 5))
    hist_a, edges = np.histogram(a_rs, bins=bins, density=True)
    hist_b, _ = np.histogram(b_rs, bins=edges, density=True)
    cdf_a = np.cumsum(hist_a) / (hist_a.sum() + eps)
    cdf_b = np.cumsum(hist_b) / (hist_b.sum() + eps)
    emd = np.sum(np.abs(cdf_a - cdf_b)) * (edges[1] - edges[0])
    distribution_sim = float(np.exp(-emd * 2.0))

    # --- Structural score (weighted: shape > value) ---
    structural_score = float(
        0.15 * pearson_r
        + 0.15 * spearman_r
        + 0.35 * deriv_r
        + 0.25 * peak_overlap
        + 0.10 * distribution_sim
    )
    structural_score = np.clip(structural_score, 0.0, 1.0)

    return {
        "trend": trend_key,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "derivative_r": deriv_r,
        "peak_overlap": peak_overlap,
        "distribution_sim": distribution_sim,
        "structural_score": structural_score,
    }


# ---------------------------------------------------------------------------
# 4. Unified dynamics similarity
# ---------------------------------------------------------------------------

def compute_dynamics_similarity(
    dyn1: Dict,
    dyn2: Dict,
) -> Dict:
    """
    Compute structure-aware similarity between two audio dynamics.

    Returns per-trend scores and a weighted aggregate.

    Parameters
    ----------
    dyn1, dyn2 : dict   Outputs of ``extract_dynamics()``.

    Returns
    -------
    result : dict
        ``per_trend`` : dict  trend_key → similarity dict
        ``global_dynamics_similarity`` : float 0–100 (%)
        ``dominant_trend`` : str   the trend with the strongest match
    """
    trend_keys = ["energy", "brightness", "complexity", "rhythm"]
    per_trend = {}
    scores = []

    for key in trend_keys:
        sim = compute_trend_similarity(dyn1, dyn2, trend_key=key)
        per_trend[key] = sim
        scores.append(sim["structural_score"])

    # Weighted: energy 35%, brightness 25%, rhythm 25%, complexity 15%
    weights = [0.35, 0.25, 0.15, 0.25]
    global_score = float(np.dot(scores, weights))
    global_score_pct = float(np.clip(global_score, 0.0, 1.0) * 100.0)

    dominant_idx = int(np.argmax(scores))
    dominant_trend = trend_keys[dominant_idx]

    # Also compute a "structural coherence" score:
    # how consistently the trends agree in their similarity patterns
    coherence = 1.0 - float(np.std(scores))  # lower variance → more coherent

    return {
        "per_trend": per_trend,
        "global_dynamics_similarity": global_score_pct,
        "dominant_trend": dominant_trend,
        "structural_coherence": float(np.clip(coherence, 0.0, 1.0)),
        "trend_weights": dict(zip(trend_keys, weights)),
    }


# ---------------------------------------------------------------------------
# 5. Statistical summary
# ---------------------------------------------------------------------------

def summarize_dynamics(dynamics: Dict) -> Dict:
    """
    Compute per-trend statistics: mean, std, min, max, peaks, valleys.

    Returns a compact dict suitable for JSON serialisation.
    """
    summary = {}
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        arr = dynamics[key]
        n = len(arr)
        if n < 2:
            summary[key] = {
                "mean": float(arr[0]) if n == 1 else 0.0,
                "std": 0.0,
                "min": float(arr[0]) if n == 1 else 0.0,
                "max": float(arr[0]) if n == 1 else 0.0,
                "n_peaks": 0,
                "trend_direction": "flat",
                "range_pct": 0.0,
            }
            continue

        peaks, _ = find_peaks(arr, distance=max(1, n // 10))
        # Simple trend direction via linear regression
        x = np.arange(n)
        slope, _, _, _, _ = stats.linregress(x, arr)
        if slope > arr.std() * 0.3 / n:
            direction = "rising"
        elif slope < -arr.std() * 0.3 / n:
            direction = "falling"
        else:
            direction = "flat"

        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_peaks": len(peaks),
            "trend_direction": direction,
            "range_pct": float((arr.max() - arr.min()) / (arr.max() + 1e-12)),
        }

    return summary


# ---------------------------------------------------------------------------
# 6. Report printing
# ---------------------------------------------------------------------------

def print_dynamics_report(dynamics: Dict, segments: Optional[Dict] = None):
    """Print a formatted dynamics analysis report to stdout."""
    summary = summarize_dynamics(dynamics)
    params = dynamics["params"]
    n = params["n_windows"]

    print()
    print("=" * 70)
    print("  AUDIO DYNAMICS / TREND ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Windows: {n}  |  Window: {params['window_size']}s  "
          f"|  Hop: {params['hop_size']}s  |  SR: {params['sr']} Hz")
    print("-" * 70)
    header = f"  {'Trend':<14s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s} {'Peaks':>6s} {'Direction':>10s}"
    print(header)
    print("  " + "-" * len(header))
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = summary[key]
        print(f"  {key:<14s} {s['mean']:10.4f} {s['std']:10.4f} "
              f"{s['min']:10.4f} {s['max']:10.4f} {s['n_peaks']:6d} "
              f"{s['trend_direction']:>10s}")
    print("-" * 70)

    if segments:
        n_climax = len(segments.get("climax_indices", []))
        n_calm = len(segments.get("calm_indices", []))
        n_buildup = len(segments.get("buildup_indices", []))
        n_transition = len(segments.get("transition_indices", []))
        print(f"  Structural segments: climax={n_climax}, calm={n_calm}, "
              f"buildup={n_buildup}, transition={n_transition}")
        if n_climax > 0:
            climax_times = dynamics["times"][segments["climax_indices"]]
            print(f"  Climax times (s): {np.round(climax_times, 2).tolist()}")
    print("=" * 70)
    print()


def print_dynamics_similarity_report(sim_result: Dict):
    """Print a formatted dynamics similarity report."""
    per_trend = sim_result["per_trend"]

    print()
    print("-" * 70)
    print("  DYNAMICS SIMILARITY (Structure-Aware)")
    print("-" * 70)
    header = (f"  {'Trend':<14s} {'Pearson':>8s} {'Spearman':>9s} "
              f"{'Deriv':>7s} {'Peaks':>7s} {'Dist':>7s} {'Struct':>8s}")
    print(header)
    print("  " + "-" * len(header))
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = per_trend[key]
        print(f"  {key:<14s} {s['pearson_r']:8.3f} {s['spearman_r']:9.3f} "
              f"{s['derivative_r']:7.3f} {s['peak_overlap']:7.3f} "
              f"{s['distribution_sim']:7.3f} {s['structural_score']:8.3f}")
    print("  " + "-" * len(header))
    print(f"  GLOBAL DYNAMICS SIMILARITY: {sim_result['global_dynamics_similarity']:.1f}%")
    print(f"  Dominant trend: {sim_result['dominant_trend']}")
    print(f"  Structural coherence: {sim_result['structural_coherence']:.3f}")
    print("-" * 70)
    print()


# ---------------------------------------------------------------------------
# 7. Trend Layer formalisation (lightweight wrapper)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field
from typing import Any as _Any


@dataclass
class TrendLayer:
    """
    Formalised Trend Layer — the four core audio dynamic trends.

    This is a typed wrapper around the dict returned by
    ``extract_dynamics()``.  It exists for documentation and for
    type-safe consumption by downstream modules (Volatility Layer,
    Prediction, Similarity).
    """
    times: np.ndarray                              # centre time of each window (s)
    energy: np.ndarray                             # RMS energy per window
    brightness: np.ndarray                         # spectral centroid per window
    complexity: np.ndarray                         # spectral entropy per window
    rhythm: np.ndarray                             # onset density per window
    energy_norm: np.ndarray                        # z-score normalised
    brightness_norm: np.ndarray
    complexity_norm: np.ndarray
    rhythm_norm: np.ndarray
    params: dict = dc_field(default_factory=dict)  # window_size, hop_size, sr, n_windows

    @classmethod
    def from_dynamics(cls, dynamics: Dict) -> "TrendLayer":
        """Construct from the dict returned by ``extract_dynamics()``."""
        return cls(
            times=dynamics["times"],
            energy=dynamics["energy"],
            brightness=dynamics["brightness"],
            complexity=dynamics["complexity"],
            rhythm=dynamics["rhythm"],
            energy_norm=dynamics.get("energy_norm", dynamics["energy"]),
            brightness_norm=dynamics.get("brightness_norm", dynamics["brightness"]),
            complexity_norm=dynamics.get("complexity_norm", dynamics["complexity"]),
            rhythm_norm=dynamics.get("rhythm_norm", dynamics["rhythm"]),
            params=dynamics.get("params", {}),
        )

    def to_dict(self) -> Dict:
        """Convert back to the canonical dict format."""
        return {
            "times": self.times,
            "energy": self.energy,
            "brightness": self.brightness,
            "complexity": self.complexity,
            "rhythm": self.rhythm,
            "energy_norm": self.energy_norm,
            "brightness_norm": self.brightness_norm,
            "complexity_norm": self.complexity_norm,
            "rhythm_norm": self.rhythm_norm,
            "params": self.params,
        }

    @property
    def n_windows(self) -> int:
        return self.params.get("n_windows", len(self.times))

    @property
    def window_size(self) -> float:
        return self.params.get("window_size", 0.5)

    @property
    def hop_size(self) -> float:
        return self.params.get("hop_size", 0.25)


def analyze_trend_layer(
    dynamics: Dict,
    detect_segments: bool = True,
    climax_percentile: float = 85.0,
    calm_percentile: float = 25.0,
) -> Dict:
    """
    Run the full Trend Layer analysis on a dynamics dict.

    This is the formal entry point for the Trend Layer.  It wraps
    ``extract_dynamics()`` (if a raw waveform is given), runs
    ``detect_structural_segments()`` and ``summarize_dynamics()``,
    and returns a consolidated result.

    Parameters
    ----------
    dynamics : dict          Output of ``extract_dynamics()`` (or raw waveform will
                              be processed if ``_y`` key present).
    detect_segments : bool   Whether to run structural segment detection.
    climax_percentile : float
    calm_percentile : float

    Returns
    -------
    trend_analysis : dict
        Keys: ``trend_layer`` (TrendLayer), ``segments``, ``summary``,
        ``n_climax``, ``n_calm``, ``n_buildup``, ``n_transition``.
    """
    summary = summarize_dynamics(dynamics)

    result: Dict[str, _Any] = {
        "trend_layer": TrendLayer.from_dynamics(dynamics),
        "summary": summary,
    }

    if detect_segments:
        segments = detect_structural_segments(
            dynamics,
            climax_percentile=climax_percentile,
            calm_percentile=calm_percentile,
        )
        result["segments"] = segments
        result["n_climax"] = len(segments.get("climax_indices", []))
        result["n_calm"] = len(segments.get("calm_indices", []))
        result["n_buildup"] = len(segments.get("buildup_indices", []))
        result["n_transition"] = len(segments.get("transition_indices", []))
    else:
        result["segments"] = None
        result["n_climax"] = 0
        result["n_calm"] = 0
        result["n_buildup"] = 0
        result["n_transition"] = 0

    return result
