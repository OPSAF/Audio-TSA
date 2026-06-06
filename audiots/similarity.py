"""
Audio Similarity Analysis Module
================================

Computes relative similarity between two audio files **without requiring
strict time alignment**.  The method is robust against:

* Different audio lengths
* Time shifts / segment reordering
* Tempo variations (within reason)

----
Method overview
~~~~~~~~~~~~~~

1. **Multi-scale window feature extraction** – each window gets a rich feature
   vector: rhythm (onset density, tempo ACF), energy (RMS, entropy, ZCR),
   timbre (spectral centroid/bandwidth/rolloff/flatness/entropy), harmony
   (chroma), time-series (ACF, PACF), and compact MFCC statistics.

2. **Feature-distribution matching** – we model each audio's feature space
   via kernel density estimation (KDE) and compute Wasserstein distances per
   dimension.  This is inherently *length-independent* and *order-agnostic*.

3. **Local similarity matrix** – pairwise cosine similarity between all
   windows of audio A and audio B.  High-similarity clusters reveal
   corresponding segments regardless of their temporal position.

4. **Hidden pattern discovery** – detect climax windows (peak energy +
   peak novelty), repeated segments (self-similarity lag analysis), and
   melodic motifs (chroma cross-correlation).

5. **Weighted aggregation** – each window is weighted by its information
   content (spectral entropy), then local similarities are aggregated via
   top-K matching and distribution overlap to produce a global similarity
   percentage.

6. **Interpretability** – every high-similarity segment is annotated with
   dominant rhythmic, energetic and timbral characteristics.

References
~~~~~~~~~~
* Foote, J. (2000). "Automatic audio segmentation using a measure of audio
  novelty." IEEE ICME.
* Logan, B. & Salomon, A. (2001). "A music similarity function based on
  signal analysis." IEEE ICME.
* Tzanetakis, G. & Cook, P. (2002). "Musical genre classification of audio
  signals." IEEE Trans. Speech and Audio Processing.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import cdist as scipy_cdist
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_len(x):
    try:
        return len(x)
    except TypeError:
        return 0


def _as_float64(y):
    """Ensure float64 contiguous array."""
    return np.ascontiguousarray(y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Low-level feature helpers
# ---------------------------------------------------------------------------

def _compute_fft_window(y: np.ndarray, sr: int):
    """Magnitude spectrum of a short window."""
    n = len(y)
    Y = np.fft.rfft(y)
    mag = np.abs(Y) / n
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return freqs, mag


def _spectral_rolloff(freqs: np.ndarray, mag: np.ndarray, roll: float = 0.85):
    cs = np.cumsum(mag)
    if cs[-1] == 0:
        return 0.0
    idx = np.searchsorted(cs, roll * cs[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def _chroma_vector(y: np.ndarray, sr: int, n_chroma: int = 12, ref_hz: float = 27.5):
    """12-bin chroma (pitch-class profile)."""
    freqs, mag = _compute_fft_window(y, sr)
    chroma = np.zeros(n_chroma)
    valid = freqs > 0
    if not np.any(valid):
        return chroma
    freqs_v, mag_v = freqs[valid], mag[valid]
    pitch = np.round(12.0 * np.log2(freqs_v / ref_hz)).astype(int) % 12
    np.add.at(chroma, pitch, mag_v)
    s = chroma.sum()
    return chroma / s if s > 0 else chroma


def _compute_acf_short(x: np.ndarray, nlags: int) -> np.ndarray:
    """ACF for a single window (efficient with correlate)."""
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return np.ones(nlags)
    acf = np.correlate(xc, xc, mode="full")[len(xc) - 1:]
    return acf[:nlags] / denom


def _compute_pacf_short(x: np.ndarray, nlags: int) -> np.ndarray:
    """PACF via Levinson-Durbin recursion (stable for short windows)."""
    acf = _compute_acf_short(x, nlags + 1)  # need lag 0 … nlags
    pacf = np.zeros(nlags)
    pacf[0] = 1.0
    if nlags <= 1:
        return pacf

    # Levinson-Durbin
    phi = np.zeros(nlags)
    phi[0] = acf[1]
    pacf[1] = phi[0]

    for k in range(1, nlags - 1):
        num = acf[k + 1]
        denom = 1.0
        for j in range(k):
            num -= phi[j] * acf[k - j]
            denom -= phi[j] * acf[j + 1]
        rho = num / denom if abs(denom) > 1e-10 else 0.0
        new_phi = phi.copy()
        new_phi[k] = rho
        for j in range(k):
            new_phi[j] = phi[j] - rho * phi[k - 1 - j]
        phi = new_phi
        pacf[k + 1] = rho

    return pacf


def _mfcc_compact(y: np.ndarray, sr: int, n_mfcc: int = 13):
    """Compact MFCC stats for a window (avoids full librosa if possible)."""
    try:
        import librosa
        n_fft = min(2048, len(y))
        hop = max(1, n_fft // 4)
        mfcc = librosa.feature.mfcc(
            y=y.astype(float), sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop
        )
        return mfcc.mean(axis=1), mfcc.std(axis=1)
    except ImportError:
        # Fallback: log-spaced band energies
        freqs, mag = _compute_fft_window(y, sr)
        if len(mag) <= n_mfcc:
            return np.zeros(n_mfcc), np.zeros(n_mfcc)
        valid = freqs > 0
        freqs_v, mag_v = freqs[valid], mag[valid]
        edges = np.logspace(np.log10(max(20, freqs_v[0])), np.log10(sr / 2), n_mfcc + 1)
        band_energy = np.zeros(n_mfcc)
        for b in range(n_mfcc):
            mask = (freqs_v >= edges[b]) & (freqs_v < edges[b + 1])
            if mask.any():
                band_energy[b] = np.mean(mag_v[mask])
        # DCT to approximate MFCC
        from scipy.fft import dct
        mfcc_approx = dct(band_energy + 1e-10, type=2, norm="ortho")
        return mfcc_approx, np.zeros_like(mfcc_approx)


# ---------------------------------------------------------------------------
# Window feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(
    y: np.ndarray,
    sr: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Extract a rich feature vector for every overlapping window.

    Parameters
    ----------
    y : ndarray   Raw audio samples.
    sr : int      Sample rate (Hz).
    window_size : float  Window duration in seconds.
    hop_size : float     Hop duration in seconds.

    Returns
    -------
    features : list of dict  One dict per window (see below for keys).
    times : ndarray         Centre time of each window (seconds).

    Feature keys per window
    -----------------------
    Rhythm
        onset_density, onset_mean, onset_std, tempo_acf_peak
    Energy
        rms_energy, energy_entropy, zero_crossing_rate, peak_to_rms
    Timbre
        spectral_centroid, spectral_bandwidth, spectral_rolloff,
        spectral_flatness, spectral_entropy
    Harmony
        chroma             (12-vector)
    Time-series
        acf                (5-vector, lags 1-5)
        pacf               (5-vector, lags 1-5)
    MFCC
        mfcc_mean, mfcc_std (n_mfcc-vectors each)
    """
    y = _as_float64(y)
    win_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)

    if win_samples > len(y):
        win_samples = len(y)
        hop_samples = max(1, win_samples // 4)

    n_windows = max(1, (len(y) - win_samples) // hop_samples + 1)
    features_list: List[Dict] = []
    window_times = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * hop_samples
        end = start + win_samples
        win = y[start:end]

        if len(win) < 64:
            continue

        f: Dict = {}

        # ---- Rhythm ----
        energy = win ** 2
        flux = np.diff(energy)
        flux_abs = np.abs(flux)
        f["onset_density"] = float(
            np.sum(flux_abs > flux_abs.mean() + flux_abs.std()) / max(len(win), 1)
        )
        f["onset_mean"] = float(flux_abs.mean())
        f["onset_std"] = float(flux_abs.std())

        # Tempo ACF (coarse)
        env = np.abs(win)
        env_ds = signal.resample(env, min(256, len(env)))
        env_acf = np.correlate(env_ds - env_ds.mean(), env_ds - env_ds.mean(), mode="full")
        env_acf = env_acf[len(env_acf) // 2:]
        env_acf = env_acf / (env_acf[0] + 1e-10)
        tempo_lo = max(1, int(0.3 * len(env_ds) / window_size))  # ~200 BPM
        tempo_hi = min(len(env_acf), int(1.0 * len(env_ds) / window_size))  # ~60 BPM
        if tempo_lo < tempo_hi:
            f["tempo_acf_peak"] = float(env_acf[tempo_lo:tempo_hi].max())
        else:
            f["tempo_acf_peak"] = 0.0

        # ---- Energy ----
        rms = float(np.sqrt(energy.mean()))
        f["rms_energy"] = rms
        f["energy_entropy"] = float(
            -np.sum(energy * np.log(energy + 1e-10)) / len(win)
        )
        f["zero_crossing_rate"] = float(
            np.sum(np.abs(np.diff(np.signbit(win)))) / (2 * len(win))
        )
        f["peak_to_rms"] = float(np.abs(win).max() / (rms + 1e-10))

        # ---- Timbre ----
        freqs, mag = _compute_fft_window(win, sr)
        mag_norm = mag / (mag.sum() + 1e-10)
        sc = float(np.dot(freqs, mag_norm))
        f["spectral_centroid"] = sc
        f["spectral_bandwidth"] = float(
            np.sqrt(np.dot((freqs - sc) ** 2, mag_norm))
        )
        f["spectral_rolloff"] = _spectral_rolloff(freqs, mag)
        f["spectral_flatness"] = float(
            np.exp(np.mean(np.log(mag + 1e-10))) / (mag.mean() + 1e-10)
        )
        f["spectral_entropy"] = float(
            -np.sum(mag_norm * np.log(mag_norm + 1e-10))
        )

        # ---- Harmony ----
        f["chroma"] = _chroma_vector(win, sr)

        # ---- Time-series ----
        f["acf"] = _compute_acf_short(win, 6)[1:]    # 5 values
        f["pacf"] = _compute_pacf_short(win, 6)[1:]  # 5 values

        # ---- MFCC ----
        mfcc_m, mfcc_s = _mfcc_compact(win, sr)
        f["mfcc_mean"] = mfcc_m
        f["mfcc_std"] = mfcc_s

        features_list.append(f)
        window_times[i] = (start + end) / 2.0 / sr

    return features_list, window_times


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

# Scalar keys (no vector sub-keys)
_SCALAR_KEYS = [
    "onset_density", "onset_mean", "onset_std", "tempo_acf_peak",
    "rms_energy", "energy_entropy", "zero_crossing_rate", "peak_to_rms",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_flatness", "spectral_entropy",
]

_VECTOR_KEYS = ["acf", "pacf", "chroma", "mfcc_mean", "mfcc_std"]

_FEATURE_DIM = (
    len(_SCALAR_KEYS) + 5 + 5 + 12 + 13 + 13
)  # total feature dimension


def features_to_matrix(features_list: List[Dict]) -> np.ndarray:
    """Flatten list-of-dicts into a (n_windows, n_features) matrix.

    The feature dimension is determined from the first window and must
    be consistent across all windows.
    """
    if not features_list:
        return np.zeros((0, 0))

    # Determine actual dimension from the first feature dict
    first = features_list[0]
    dim = len(_SCALAR_KEYS)
    for vk in _VECTOR_KEYS:
        v = np.asarray(first.get(vk, np.zeros(1)), dtype=np.float64).ravel()
        dim += len(v)

    n = len(features_list)
    mat = np.zeros((n, dim))
    for i, f in enumerate(features_list):
        idx = 0
        # Scalars
        arr_scalar = np.array([f.get(k, 0.0) for k in _SCALAR_KEYS], dtype=np.float64)
        mat[i, idx: idx + len(_SCALAR_KEYS)] = arr_scalar
        idx += len(_SCALAR_KEYS)
        # Vectors
        for vk in _VECTOR_KEYS:
            v = np.asarray(f.get(vk, np.zeros(1)), dtype=np.float64).ravel()
            L = len(v)
            mat[i, idx: idx + L] = v[:L]
            idx += L
    return mat


def feature_names() -> List[str]:
    """Human-readable names for every dimension in the feature matrix."""
    names = list(_SCALAR_KEYS)
    for vk in _VECTOR_KEYS:
        if vk == "acf":
            names += [f"acf_lag{i}" for i in range(1, 6)]
        elif vk == "pacf":
            names += [f"pacf_lag{i}" for i in range(1, 6)]
        elif vk == "chroma":
            names += [f"chroma_{i}" for i in range(12)]
        elif vk == "mfcc_mean":
            names += [f"mfcc_mean_{i}" for i in range(13)]
        elif vk == "mfcc_std":
            names += [f"mfcc_std_{i}" for i in range(13)]
    return names


# ---------------------------------------------------------------------------
# Distribution matching (length-independent)
# ---------------------------------------------------------------------------

def compute_feature_distribution_distance(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    method: str = "wasserstein",
) -> Tuple[float, np.ndarray]:
    """
    Compare feature *distributions* of two audio files.

    Returns an aggregate distance (scalar) and a per-dimension distance
    array.  0 = identical distributions.
    """
    n_dims = feats_a.shape[1]
    per_dim = np.zeros(n_dims)

    for d in range(n_dims):
        col_a = feats_a[:, d]
        col_b = feats_b[:, d]
        std_a, std_b = np.std(col_a), np.std(col_b)
        if std_a < 1e-12 and std_b < 1e-12:
            per_dim[d] = 0.0
            continue

        try:
            if method == "wasserstein":
                # Use histogram-based Wasserstein (robust, fast)
                bins = max(10, min(50, int(np.sqrt(min(len(col_a), len(col_b))))))
                hist_a, edges = np.histogram(col_a, bins=bins, density=True)
                hist_b, _ = np.histogram(col_b, bins=edges, density=True)
                cdf_a = np.cumsum(hist_a) * (edges[1] - edges[0])
                cdf_b = np.cumsum(hist_b) * (edges[1] - edges[0])
                per_dim[d] = float(np.sum(np.abs(cdf_a - cdf_b)) * (edges[1] - edges[0]))
            elif method == "js":
                # Jensen-Shannon via KDE
                kde_a = gaussian_kde(col_a)
                kde_b = gaussian_kde(col_b)
                x_vals = np.linspace(
                    min(col_a.min(), col_b.min()),
                    max(col_a.max(), col_b.max()),
                    200,
                )
                pa = kde_a(x_vals) + 1e-12
                pb = kde_b(x_vals) + 1e-12
                pa /= pa.sum()
                pb /= pb.sum()
                m = 0.5 * (pa + pb)
                js = 0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))
                per_dim[d] = float(js)
            else:
                # Fallback: normalised mean difference
                per_dim[d] = float(np.abs(col_a.mean() - col_b.mean()) / (std_a + std_b + 1e-10))
        except Exception:
            per_dim[d] = float(np.abs(col_a.mean() - col_b.mean()) / (std_a + std_b + 1e-10))

    # Aggregate: median across dimensions (robust to outliers)
    valid_dims = per_dim[~np.isnan(per_dim)]
    aggregate = float(np.median(valid_dims)) if len(valid_dims) > 0 else 0.0
    # Replace any remaining NaN with 0
    per_dim = np.nan_to_num(per_dim, nan=0.0)
    return aggregate, per_dim


# ---------------------------------------------------------------------------
# Local similarity matrix
# ---------------------------------------------------------------------------

def compute_local_similarity_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    metric: str = "cosine",
    whiten: bool = True,
    pca_components: float = 0.95,
) -> np.ndarray:
    """
    Pairwise similarity between all windows of A and B.

    Uses joint PCA whitening to decorrelate features and emphasise
    dimensions that carry real variance *across* the two signals.
    A Gaussian kernel is then applied to amplify discriminative power.

    Returns (n_windows_A, n_windows_B) matrix where entry [i,j] is
    similarity in [0, 1] (1 = identical).
    """
    if whiten and min(feats_a.shape[0], feats_b.shape[0]) > 5:
        # Joint standardisation + PCA
        combined = np.vstack([feats_a, feats_b])
        # Remove mean, scale to unit variance
        mean = combined.mean(axis=0)
        std = combined.std(axis=0)
        std[std < 1e-10] = 1.0
        combined_std = (combined - mean) / std

        # PCA
        cov = np.cov(combined_std, rowvar=False)
        try:
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            # Sort descending
            order = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            # Keep components explaining pca_components variance
            cumvar = np.cumsum(eigenvals) / eigenvals.sum()
            n_keep = max(3, int(np.searchsorted(cumvar, pca_components)) + 1)
            n_keep = min(n_keep, len(eigenvals), combined.shape[0] - 1)
            # Project
            W = eigenvecs[:, :n_keep] / np.sqrt(eigenvals[:n_keep] + 1e-10)
            a_proj = (feats_a - mean) / std @ W
            b_proj = (feats_b - mean) / std @ W
        except np.linalg.LinAlgError:
            a_proj = feats_a
            b_proj = feats_b
    else:
        a_proj = feats_a
        b_proj = feats_b

    # Cosine similarity in whitened space
    norms_a = np.linalg.norm(a_proj, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b_proj, axis=1, keepdims=True)
    a_norm = a_proj / (norms_a + 1e-12)
    b_norm = b_proj / (norms_b + 1e-12)
    cos_sim = np.dot(a_norm, b_norm.T)

    # Gaussian kernel to amplify differences: sim = exp(-gamma * (1 - cos)^2)
    # gamma controls sharpness — higher → more discriminative
    gamma = 3.0
    sim = np.exp(-gamma * (1.0 - cos_sim) ** 2)
    return np.clip(sim, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hidden pattern discovery
# ---------------------------------------------------------------------------

def discover_patterns(
    feats: np.ndarray,
    times: np.ndarray,
    y: np.ndarray,
    sr: int,
    feat_list: Optional[List[Dict]] = None,
) -> Dict:
    """
    Discover structural patterns: climax windows, repeated segments,
    and novelty boundaries.

    Returns a dict with keys:
        climax_indices, repetition_pairs, novelty_curve, segment_boundaries
    """

    # ---- Climax: windows with top-5 % energy + top-20 % onset density ----
    if feat_list is not None:
        rms = np.array([f["rms_energy"] for f in feat_list])
        onset = np.array([f["onset_density"] for f in feat_list])
    else:
        # Approximate from feature matrix
        rms_idx = _SCALAR_KEYS.index("rms_energy")
        onset_idx = _SCALAR_KEYS.index("onset_density")
        rms = feats[:, rms_idx]
        onset = feats[:, onset_idx]

    n = len(rms)
    if n < 3:
        return {
            "climax_indices": [],
            "repetition_pairs": [],
            "novelty_curve": np.zeros(n),
            "segment_boundaries": [],
        }

    rms_thresh = np.percentile(rms, 85)
    onset_thresh = np.percentile(onset, 80)
    climax = np.where((rms >= rms_thresh) & (onset >= onset_thresh))[0].tolist()

    # ---- Repetition: self-similarity lag analysis ----
    n_feat = min(feats.shape[0], 200)  # downsample for speed
    idx_sub = np.linspace(0, feats.shape[0] - 1, n_feat, dtype=int)
    feats_sub = feats[idx_sub]
    norms = np.linalg.norm(feats_sub, axis=1, keepdims=True)
    feats_n = feats_sub / (norms + 1e-12)
    self_sim = np.dot(feats_n, feats_n.T)

    repetition_pairs = []
    min_lag = max(2, int(0.5 * sr / (0.25 * sr)))  # at least 0.5 s apart
    for lag in range(min_lag, n_feat):
        diag = np.diag(self_sim, k=lag)
        peaks = signal.find_peaks(diag, height=0.85, distance=3)[0]
        for p in peaks:
            if len(repetition_pairs) < 50:
                repetition_pairs.append(
                    {
                        "window_a": int(idx_sub[p]),
                        "window_b": int(idx_sub[p + lag]),
                        "similarity": float(diag[p]),
                        "time_a": float(times[int(idx_sub[p])]),
                        "time_b": float(times[int(idx_sub[p + lag])]),
                    }
                )

    # ---- Novelty curve (checkerboard kernel) ----
    kernel_size = max(3, n_feat // 20)
    kernel = np.ones((kernel_size, kernel_size))
    half = kernel_size // 2
    kernel[:half, half:] = -1
    kernel[half:, :half] = -1
    novelty = np.zeros(n_feat)
    for i in range(kernel_size, n_feat - kernel_size):
        patch = self_sim[i - kernel_size: i, i: i + kernel_size]
        if patch.shape == kernel.shape:
            novelty[i] = np.sum(patch * kernel)
    novelty = np.maximum(novelty, 0)
    novelty = novelty / (novelty.max() + 1e-10)

    # Interpolate back to original length
    novelty_full = np.interp(np.linspace(0, 1, feats.shape[0]),
                             np.linspace(0, 1, n_feat), novelty)

    # Segment boundaries (novelty peaks)
    boundaries: List[int] = []
    if n_feat > kernel_size * 2:
        peaks_n, props = signal.find_peaks(novelty_full, height=0.3, distance=kernel_size)
        boundaries = peaks_n.tolist()

    return {
        "climax_indices": climax,
        "repetition_pairs": repetition_pairs,
        "novelty_curve": novelty_full,
        "segment_boundaries": boundaries,
    }


# ---------------------------------------------------------------------------
# Similarity aggregation
# ---------------------------------------------------------------------------

def aggregate_similarity(
    sim_matrix: np.ndarray,
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    feat_list_a: Optional[List[Dict]] = None,
    feat_list_b: Optional[List[Dict]] = None,
    distribution_distance: float = 0.0,
) -> Dict:
    """
    Aggregate a local similarity matrix into global scores.

    Strategies
    ----------
    1. **Top-K matching** – for each window in A, take the max similarity
       with any window in B.  Mean over A and B independently, then average.
    2. **Entropy-weighted** – weight each window by its spectral entropy
       (higher entropy = more information = higher weight).
    3. **Distribution bonus** – convert distribution distance to a
       similarity bonus (0-1), then blend with the matching score.

    Returns
    -------
    dict with keys:
        global_similarity         : float 0-100 (%)
        per_window_a_similarity   : ndarray (mean max-sim per A window)
        per_window_b_similarity   : ndarray
        top_k_match_score         : float
        entropy_weighted_score    : float
        distribution_similarity   : float
        window_weights_a          : ndarray
        window_weights_b          : ndarray
        matched_pairs             : list of (i, j, sim)
    """

    n_a, n_b = sim_matrix.shape

    # Entropy weights
    if feat_list_a is not None:
        ent_a = np.array([f["spectral_entropy"] for f in feat_list_a])
    else:
        ent_idx = _SCALAR_KEYS.index("spectral_entropy")
        ent_a = feats_a[:, ent_idx]
    if feat_list_b is not None:
        ent_b = np.array([f["spectral_entropy"] for f in feat_list_b])
    else:
        ent_idx = _SCALAR_KEYS.index("spectral_entropy")
        ent_b = feats_b[:, ent_idx]

    w_a = ent_a / (ent_a.sum() + 1e-12)
    w_b = ent_b / (ent_b.sum() + 1e-12)

    # --- Top-K matching ---
    # For each A window, best match in B (and vice versa)
    best_for_a = sim_matrix.max(axis=1)  # (n_a,)
    best_for_b = sim_matrix.max(axis=0)  # (n_b,)

    # Also find matched pairs (for interpretability)
    matched_pairs = []
    for i in range(n_a):
        j = int(np.argmax(sim_matrix[i]))
        if sim_matrix[i, j] > 0.7:
            matched_pairs.append((i, j, float(sim_matrix[i, j])))

    # Simple mean
    score_simple = 0.5 * best_for_a.mean() + 0.5 * best_for_b.mean()

    # Entropy-weighted mean
    score_weighted_a = np.dot(w_a, best_for_a)
    score_weighted_b = np.dot(w_b, best_for_b)
    score_weighted = 0.5 * (score_weighted_a + score_weighted_b)

    # --- Distribution similarity ---
    # Convert distance → similarity (exponential decay)
    if np.isnan(distribution_distance) or np.isinf(distribution_distance):
        dist_similarity = 0.5  # neutral
    else:
        dist_similarity = float(np.exp(-distribution_distance * 3.0))

    # --- Global blend ---
    # 60 % matching, 25 % entropy-weighted, 15 % distribution
    global_sim = 0.60 * score_simple + 0.25 * score_weighted + 0.15 * dist_similarity
    if np.isnan(global_sim):
        global_sim = score_simple  # fallback

    return {
        "global_similarity": float(np.clip(global_sim, 0.0, 1.0) * 100.0),
        "per_window_a_similarity": best_for_a,
        "per_window_b_similarity": best_for_b,
        "top_k_match_score": float(score_simple),
        "entropy_weighted_score": float(score_weighted),
        "distribution_similarity": float(dist_similarity),
        "window_weights_a": w_a,
        "window_weights_b": w_b,
        "matched_pairs": matched_pairs,
    }


# ---------------------------------------------------------------------------
# Interpretability annotations
# ---------------------------------------------------------------------------

def annotate_similar_segments(
    matched_pairs: List[Tuple[int, int, float]],
    feat_list_a: List[Dict],
    feat_list_b: List[Dict],
    times_a: np.ndarray,
    times_b: np.ndarray,
    top_n: int = 10,
) -> List[Dict]:
    """
    Annotate the top-N high-similarity segment pairs with interpretable
    labels: rhythmic character, energy level, timbral quality.
    """
    sorted_pairs = sorted(matched_pairs, key=lambda x: x[2], reverse=True)[:top_n]

    annotations = []
    for i, j, sim in sorted_pairs:
        fa = feat_list_a[i]
        fb = feat_list_b[j]

        # Rhythm label
        onset_a, onset_b = fa["onset_density"], fb["onset_density"]
        mean_onset = (onset_a + onset_b) / 2.0
        if mean_onset > 0.3:
            rhythm = "高密度打击 / dense percussive"
        elif mean_onset > 0.15:
            rhythm = "中等节奏 / moderate rhythmic"
        else:
            rhythm = "稀疏/持续音 / sparse / sustained"

        # Energy label
        rms_a, rms_b = fa["rms_energy"], fb["rms_energy"]
        mean_rms = (rms_a + rms_b) / 2.0
        if mean_rms > 0.5:
            energy = "高能量 / high energy"
        elif mean_rms > 0.15:
            energy = "中等能量 / medium energy"
        else:
            energy = "低能量/柔和 / low energy / soft"

        # Timbre label
        sc_a, sc_b = fa["spectral_centroid"], fb["spectral_centroid"]
        mean_sc = (sc_a + sc_b) / 2.0
        sf_a, sf_b = fa["spectral_flatness"], fb["spectral_flatness"]
        mean_sf = (sf_a + sf_b) / 2.0
        if mean_sf > 0.5:
            timbre = "噪声明亮 / noisy-bright"
        elif mean_sc > 2000:
            timbre = "明亮 / bright"
        elif mean_sc > 800:
            timbre = "温暖 / warm"
        else:
            timbre = "暗沉 / dark"

        # Dominant chroma (pitch class)
        chroma_a = fa["chroma"]
        chroma_b = fb["chroma"]
        chroma_mean = (chroma_a + chroma_b) / 2.0
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        dominant_pitch = pitch_names[int(np.argmax(chroma_mean))]

        annotations.append(
            {
                "window_a": int(i),
                "window_b": int(j),
                "time_a": float(times_a[i]),
                "time_b": float(times_b[j]),
                "similarity": float(sim),
                "rhythm": rhythm,
                "energy": energy,
                "timbre": timbre,
                "dominant_pitch": dominant_pitch,
                "rms_a": float(rms_a),
                "rms_b": float(rms_b),
                "onset_density_a": float(onset_a),
                "onset_density_b": float(onset_b),
                "spectral_centroid_a": float(sc_a),
                "spectral_centroid_b": float(sc_b),
            }
        )

    return annotations


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyze_similarity(
    y1: np.ndarray,
    sr1: int,
    y2: np.ndarray,
    sr2: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    dist_method: str = "wasserstein",
    verbose: bool = True,
) -> Dict:
    """
    Full audio similarity analysis pipeline.

    Parameters
    ----------
    y1, y2 : ndarray   Raw audio waveforms.
    sr1, sr2 : int     Sample rates (will be aligned to sr1 if different).
    window_size : float  Analysis window in seconds.
    hop_size : float     Hop between windows in seconds.
    dist_method : str    "wasserstein" or "js" for distribution matching.
    verbose : bool       Print progress information.

    Returns
    -------
    results : dict
        Keys:
        - feature_matrix_1, feature_matrix_2
        - feature_list_1, feature_list_2
        - window_times_1, window_times_2
        - local_similarity_matrix   (n_a × n_b)
        - distribution_distance
        - per_dim_distance
        - aggregation              (dict from aggregate_similarity)
        - patterns_1, patterns_2   (dict from discover_patterns)
        - annotations              (list of annotated segment pairs)
        - feature_names            (list of human-readable feature names)
        - params                   (dict of parameters used)
    """
    # Resample if needed
    if sr1 != sr2:
        if verbose:
            print(f"  [Resample] Audio 2 from {sr2} Hz → {sr1} Hz")
        try:
            import librosa
            y2 = librosa.resample(y2.astype(float), orig_sr=sr2, target_sr=sr1)
        except ImportError:
            from scipy.signal import resample
            y2 = resample(y2, int(len(y2) * sr1 / sr2))
        sr2 = sr1

    # Align to mono
    y1 = _as_float64(y1).ravel()
    y2 = _as_float64(y2).ravel()

    if verbose:
        t1, t2 = len(y1) / sr1, len(y2) / sr1
        print(f"  Audio 1: {t1:.2f}s, Audio 2: {t2:.2f}s ({sr1} Hz)")

    # ---- Step 1: Extract window features ----
    if verbose:
        print("  [1/6] Extracting window features for Audio 1 ...")
    feat_list_1, times_1 = extract_window_features(y1, sr1, window_size, hop_size)

    if verbose:
        print("  [1/6] Extracting window features for Audio 2 ...")
    feat_list_2, times_2 = extract_window_features(y2, sr1, window_size, hop_size)

    n_w1 = len(feat_list_1)
    n_w2 = len(feat_list_2)
    if verbose:
        print(f"         Audio 1: {n_w1} windows, Audio 2: {n_w2} windows")

    feat_mat_1 = features_to_matrix(feat_list_1)
    feat_mat_2 = features_to_matrix(feat_list_2)

    # ---- Step 2: Distribution matching ----
    if verbose:
        print(f"  [2/6] Computing feature distribution distance ({dist_method}) ...")
    dist_agg, dist_per_dim = compute_feature_distribution_distance(
        feat_mat_1, feat_mat_2, method=dist_method
    )
    if verbose:
        print(f"         Aggregate distribution distance: {dist_agg:.4f}")

    # ---- Step 3: Local similarity matrix ----
    if verbose:
        print("  [3/6] Computing local similarity matrix ...")
    sim_matrix = compute_local_similarity_matrix(feat_mat_1, feat_mat_2)
    if verbose:
        high_sim_frac = (sim_matrix > 0.7).mean()
        print(f"         {sim_matrix.shape[0]}x{sim_matrix.shape[1]} matrix, "
              f"{high_sim_frac*100:.1f}% entries > 0.7")

    # ---- Step 4: Aggregation ----
    if verbose:
        print("  [4/6] Aggregating to global similarity score ...")
    aggregation = aggregate_similarity(
        sim_matrix, feat_mat_1, feat_mat_2,
        feat_list_a=feat_list_1, feat_list_b=feat_list_2,
        distribution_distance=dist_agg,
    )
    if verbose:
        print(f"         Global similarity: {aggregation['global_similarity']:.1f}%")

    # ---- Step 5: Pattern discovery ----
    if verbose:
        print("  [5/6] Discovering hidden patterns ...")
    patterns_1 = discover_patterns(feat_mat_1, times_1, y1, sr1, feat_list_1)
    patterns_2 = discover_patterns(feat_mat_2, times_2, y2, sr1, feat_list_2)
    if verbose:
        nc1 = len(patterns_1["climax_indices"])
        nc2 = len(patterns_2["climax_indices"])
        nr1 = len(patterns_1["repetition_pairs"])
        nr2 = len(patterns_2["repetition_pairs"])
        print(f"         Audio 1: {nc1} climax windows, {nr1} repetition pairs")
        print(f"         Audio 2: {nc2} climax windows, {nr2} repetition pairs")

    # ---- Step 6: Annotate ----
    if verbose:
        print("  [6/6] Annotating high-similarity segments ...")
    annotations = annotate_similar_segments(
        aggregation["matched_pairs"],
        feat_list_1, feat_list_2, times_1, times_2,
        top_n=15,
    )
    if verbose:
        print(f"         {len(annotations)} annotated segment pairs")

    return {
        "feature_matrix_1": feat_mat_1,
        "feature_matrix_2": feat_mat_2,
        "feature_list_1": feat_list_1,
        "feature_list_2": feat_list_2,
        "window_times_1": times_1,
        "window_times_2": times_2,
        "local_similarity_matrix": sim_matrix,
        "distribution_distance": dist_agg,
        "per_dim_distance": dist_per_dim,
        "aggregation": aggregation,
        "patterns_1": patterns_1,
        "patterns_2": patterns_2,
        "annotations": annotations,
        "feature_names": feature_names(),
        "params": {
            "window_size": window_size,
            "hop_size": hop_size,
            "dist_method": dist_method,
            "sr": sr1,
            "n_windows_1": n_w1,
            "n_windows_2": n_w2,
            "duration_1": float(len(y1) / sr1),
            "duration_2": float(len(y2) / sr1),
        },
    }


# ---------------------------------------------------------------------------
# Convenience: load + analyse
# ---------------------------------------------------------------------------

def analyze_similarity_from_files(
    filepath1: str,
    filepath2: str,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    target_sr: int = 16000,
    dist_method: str = "wasserstein",
    verbose: bool = True,
) -> Dict:
    """
    Load two audio files and run the full similarity pipeline.

    Parameters
    ----------
    filepath1, filepath2 : str   Paths to WAV/MP3 files.
    (other parameters as in analyze_similarity)

    Returns
    -------
    results : dict  Same structure as analyze_similarity, plus:
        - file1, file2 : file paths
        - audio_info_1, audio_info_2 : dicts with duration, sr, samples
    """
    from .loader import load_audio

    print(f"Loading Audio 1: {filepath1}")
    y1, sr1 = load_audio(filepath1, target_sr=target_sr)
    print(f"Loading Audio 2: {filepath2}")
    y2, sr2 = load_audio(filepath2, target_sr=target_sr)

    results = analyze_similarity(
        y1, sr1, y2, sr2,
        window_size=window_size,
        hop_size=hop_size,
        dist_method=dist_method,
        verbose=verbose,
    )

    results["file1"] = filepath1
    results["file2"] = filepath2
    results["audio_info_1"] = {
        "duration": len(y1) / sr1,
        "sample_rate": sr1,
        "samples": len(y1),
    }
    results["audio_info_2"] = {
        "duration": len(y2) / sr2,
        "sample_rate": sr2,
        "samples": len(y2),
    }

    return results


def print_similarity_report(results: Dict):
    """Pretty-print a similarity analysis report."""
    agg = results["aggregation"]
    params = results["params"]

    print()
    print("=" * 70)
    print("  AUDIO SIMILARITY ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Audio 1: {params['duration_1']:.2f}s  ({params['n_windows_1']} windows)")
    print(f"  Audio 2: {params['duration_2']:.2f}s  ({params['n_windows_2']} windows)")
    print(f"  Window: {params['window_size']}s, Hop: {params['hop_size']}s")
    print("-" * 70)
    print(f"  GLOBAL SIMILARITY:  {agg['global_similarity']:.1f}%")
    print("-" * 70)
    print(f"  Top-K matching score:       {agg['top_k_match_score']*100:.1f}%")
    print(f"  Entropy-weighted score:      {agg['entropy_weighted_score']*100:.1f}%")
    print(f"  Distribution similarity:     {agg['distribution_similarity']*100:.1f}%")
    print(f"  Feature distribution dist:   {results['distribution_distance']:.4f}")
    print("-" * 70)

    # Pattern summary
    p1 = results["patterns_1"]
    p2 = results["patterns_2"]
    print(f"  Climax windows:   Audio1={len(p1['climax_indices']):3d}  "
          f"Audio2={len(p2['climax_indices']):3d}")
    print(f"  Repetition pairs: Audio1={len(p1['repetition_pairs']):3d}  "
          f"Audio2={len(p2['repetition_pairs']):3d}")
    print(f"  Segment boundaries: Audio1={len(p1['segment_boundaries']):3d}  "
          f"Audio2={len(p2['segment_boundaries']):3d}")
    print("-" * 70)

    # Top annotations
    annotations = results.get("annotations", [])
    if annotations:
        print(f"  Top high-similarity segment pairs ({min(5, len(annotations))} shown):")
        print(f"  {'Time A':>8s}  {'Time B':>8s}  {'Sim':>6s}  {'Rhythm':<30s}  {'Energy':<20s}")
        print("  " + "-" * 64)
        for ann in annotations[:5]:
            print(f"  {ann['time_a']:7.2f}s  {ann['time_b']:7.2f}s  "
                  f"{ann['similarity']:.3f}  {ann['rhythm']:<30s}  {ann['energy']:<20s}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# 7. Volatility similarity (bridge to volatility.py)
# ---------------------------------------------------------------------------

def compute_volatility_similarity(
    vol1: Dict,
    vol2: Dict,
) -> Dict:
    """
    Compute similarity of volatility structure between two audio files.

    This is a bridge to ``volatility.compute_volatility_similarity()``.
    It accepts volatility layer dicts and returns a score suitable for
    blending into the global similarity aggregation.

    Parameters
    ----------
    vol1, vol2 : dict    Outputs of ``volatility.compute_volatility_layer()``.

    Returns
    -------
    scores : dict
        ``global_volatility_similarity`` : float 0–100 (%)
        ``per_trend`` : dict  trend_key → similarity sub-scores
    """
    from .volatility import compute_volatility_similarity as _vol_sim
    return _vol_sim(vol1, vol2)


def blend_volatility_into_similarity(
    base_similarity: float,       # 0–100, from aggregate_similarity
    volatility_similarity: float, # 0–100, from compute_volatility_similarity
    vol_weight: float = 0.15,
) -> float:
    """
    Blend volatility similarity into the base global similarity score.

    Parameters
    ----------
    base_similarity : float        Global similarity % from the main pipeline.
    volatility_similarity : float  Volatility similarity %.
    vol_weight : float             Weight for the volatility component
                                   (default 0.15 means 85 % base, 15 % vol).

    Returns
    -------
    blended : float  0–100 (%).
    """
    return float(np.clip(
        (1.0 - vol_weight) * base_similarity + vol_weight * volatility_similarity,
        0.0, 100.0,
    ))
