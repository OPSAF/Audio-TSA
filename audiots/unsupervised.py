"""
Unsupervised Audio Pattern Discovery
=====================================

Discovers intrinsic structure in audio **without labels, without
forecasting targets**.  Replaces the supervised prediction paradigm
with genuine unsupervised exploration.

Capabilities
------------
* **Change-point detection** — automatically segment audio into
  structurally distinct sections.
* **Motif discovery** — find recurring temporal patterns (simplified
  matrix-profile approach).
* **Spectral decomposition (NMF)** — decompose the mel spectrogram
  into a small set of "spectral building blocks" and their activations.
* **Recurrence quantification (RQA)** — measure determinism,
  laminarity, and dynamical complexity of the audio feature trajectory.
* **Anomaly detection** — flag statistically unusual moments.

Every discovery is backed by time ranges and quantitative evidence.
No training data required — purely unsupervised.

Dependencies
------------
numpy, scipy, sklearn (already used elsewhere in the project).
No PyTorch / TensorFlow.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A structurally homogeneous audio segment."""
    start: float           # seconds
    end: float             # seconds
    label: str             # human-readable label
    character: str         # description of this segment's character
    dominant_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class Motif:
    """A recurring temporal pattern."""
    pattern_index: int
    length_seconds: float
    occurrences: List[float]          # start times (seconds)
    significance: float               # 0-1, how strong is this motif?
    description: str


@dataclass
class SpectralComponent:
    """One spectral building block from NMF decomposition."""
    index: int
    activation_peaks: List[float]      # times when most active
    character: str                     # e.g. "低频持续音" / "高频打击"
    weight: float                      # how much of the total energy it explains


@dataclass
class Anomaly:
    """A statistically unusual moment."""
    time: float
    duration: float
    anomaly_score: float
    description: str


@dataclass
class UnsupervisedReport:
    """Complete unsupervised exploration report."""
    # Segmentation
    change_points: List[float] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)

    # Motifs
    motifs: List[Motif] = field(default_factory=list)
    discords: List[Anomaly] = field(default_factory=list)

    # Decomposition
    n_components: int = 4
    spectral_components: List[SpectralComponent] = field(default_factory=list)
    reconstruction_error: float = 0.0

    # Recurrence
    recurrence_rate: float = 0.0
    determinism: float = 0.0
    laminarity: float = 0.0
    mean_diag_length: float = 0.0
    recurrence_interpretation: str = ""

    # Clustering
    n_clusters: int = 0
    cluster_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_profiles: List[Dict] = field(default_factory=list)
    silhouette_score: float = 0.0

    # Overview
    overview: str = ""

    # Metadata
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float64(y):
    return np.ascontiguousarray(y, dtype=np.float64)


def _zscore(arr: np.ndarray) -> np.ndarray:
    s = arr.std()
    return (arr - arr.mean()) / (s + 1e-12)


# ---------------------------------------------------------------------------
# 1. Change-point detection & segmentation
# ---------------------------------------------------------------------------

def detect_change_points(
    feats: np.ndarray,
    times: np.ndarray,
    min_segment_length: float = 1.0,
    n_bkps_max: int = 8,
    penalty_factor: float = 1.0,
) -> Tuple[List[float], np.ndarray]:
    """
    Detect structural change points in multi-dimensional feature time series.

    Uses a kernel-based approach: at each candidate point, compares the
    feature distribution before vs. after using the Maximum Mean Discrepancy
    (MMD) approximation via a simple two-sample energy statistic.

    Parameters
    ----------
    feats : (n_windows, n_features) ndarray
    times : (n_windows,) ndarray   Centre time of each window.
    min_segment_length : float     Minimum segment length in seconds.
    n_bkps_max : int               Maximum number of change points.
    penalty_factor : float         Higher = fewer change points (stricter).

    Returns
    -------
    change_points : list of float  Times (seconds) of detected changes.
    change_score : ndarray         Change likelihood at each window.
    """
    n = feats.shape[0]
    if n < 6:
        return [], np.zeros(n)

    # Normalise features
    f = _zscore(feats.T).T  # column-wise z-score

    # Minimum distance between change points in samples
    hop = (times[1] - times[0]) if len(times) > 1 else 0.25
    min_dist = max(3, int(min_segment_length / hop))

    # Compute change score at each interior point
    change_score = np.zeros(n)
    half_win = max(3, min_dist // 2)

    for i in range(half_win, n - half_win):
        left = f[max(0, i - half_win):i]
        right = f[i:min(n, i + half_win)]
        if len(left) < 2 or len(right) < 2:
            continue
        # Energy statistic: 2 * mean_cross_dist - mean_left_dist - mean_right_dist
        mean_left = left.mean(axis=0)
        mean_right = right.mean(axis=0)
        # Simple MMD proxy: squared distance between means
        score = np.sum((mean_left - mean_right) ** 2)
        # Penalise by within-segment variance
        within_var = np.sum(np.var(left, axis=0)) + np.sum(np.var(right, axis=0)) + 1e-12
        change_score[i] = score / within_var

    # Normalise
    change_score = change_score / (change_score.max() + 1e-12)

    # Find peaks
    peaks, props = signal.find_peaks(
        change_score,
        distance=min_dist,
        height=max(0.15, np.percentile(change_score, 70) * penalty_factor),
    )

    # Limit number
    if len(peaks) > n_bkps_max:
        peak_heights = change_score[peaks]
        top_idx = np.argsort(peak_heights)[-n_bkps_max:]
        peaks = peaks[top_idx]

    change_points = [float(times[p]) for p in sorted(peaks)]
    return change_points, change_score


def characterise_segment(
    dyn_segment: Dict[str, np.ndarray],
    start_time: float,
    end_time: float,
) -> Segment:
    """Describe the character of one audio segment."""
    energy = dyn_segment.get("energy", np.array([0]))
    brightness = dyn_segment.get("brightness", np.array([0]))
    rhythm = dyn_segment.get("rhythm", np.array([0]))
    complexity = dyn_segment.get("complexity", np.array([0]))

    e_mean = float(energy.mean())
    b_mean = float(brightness.mean())
    r_mean = float(rhythm.mean())
    c_mean = float(complexity.mean())

    # Energy
    if e_mean > 0.4:
        e_desc = "高能量"
    elif e_mean > 0.15:
        e_desc = "中等能量"
    else:
        e_desc = "低能量"

    # Energy trend
    if len(energy) > 2:
        slope, _, _, _, _ = stats.linregress(np.arange(len(energy)), energy)
        if slope > energy.std() * 0.3 / len(energy):
            e_desc += "、上升趋势"
        elif slope < -energy.std() * 0.3 / len(energy):
            e_desc += "、下降趋势"

    # Brightness
    if b_mean > 3000:
        b_desc = "明亮"
    elif b_mean > 1500:
        b_desc = "中等亮度"
    else:
        b_desc = "偏暗/温暖"

    # Rhythm
    if r_mean > 0.02:
        r_desc = "密集节奏"
    elif r_mean > 0.01:
        r_desc = "中等节奏"
    else:
        r_desc = "稀疏/持续音"

    character = f"{e_desc}，{b_desc}，{r_desc}"

    return Segment(
        start=start_time,
        end=end_time,
        label="segment",
        character=character,
        dominant_features={
            "energy_mean": e_mean,
            "brightness_mean": b_mean,
            "rhythm_mean": r_mean,
            "complexity_mean": c_mean,
        },
    )


# ---------------------------------------------------------------------------
# 2. Motif discovery (simplified matrix profile)
# ---------------------------------------------------------------------------

def discover_motifs(
    feats: np.ndarray,
    times: np.ndarray,
    min_length: float = 0.5,
    max_length: float = 3.0,
    top_k: int = 5,
) -> Tuple[List[Motif], List[Anomaly]]:
    """
    Find recurring temporal patterns (motifs) and unusual subsequences
    (discords) using a simplified matrix-profile approach.

    Parameters
    ----------
    feats : (n, d) ndarray   Feature matrix (e.g. dynamics z-scored).
    times : (n,) ndarray     Window centre times.
    min_length, max_length : float  Subsequence length range.
    top_k : int              Number of top motifs to report.

    Returns
    -------
    motifs : list of Motif
    discords : list of Anomaly
    """
    n, d = feats.shape
    if n < 8:
        return [], []

    hop = (times[1] - times[0]) if len(times) > 1 else 0.25
    sub_len = max(3, min(n // 3, int(max_length / hop)))
    sub_len = max(sub_len, int(min_length / hop))

    if sub_len >= n // 2:
        sub_len = n // 3
    if sub_len < 2:
        return [], []

    # Build subsequence matrix: each row is a flattened subsequence
    n_subs = n - sub_len + 1
    # Downsample for efficiency if needed
    max_subs = 300
    if n_subs > max_subs:
        stride = max(1, n_subs // max_subs)
        indices = np.arange(0, n_subs, stride)
    else:
        indices = np.arange(n_subs)

    # Compute pairwise distances between subsequences
    subs = np.zeros((len(indices), sub_len * d))
    for k, idx in enumerate(indices):
        subs[k] = feats[idx:idx + sub_len].ravel()

    # Z-normalize each subsequence row
    subs_mean = subs.mean(axis=1, keepdims=True)
    subs_std = subs.std(axis=1, keepdims=True) + 1e-12
    subs_norm = (subs - subs_mean) / subs_std

    # Compute distance matrix (use dot product for efficiency)
    dist_mat = 1.0 - np.dot(subs_norm, subs_norm.T)  # cosine-like
    np.fill_diagonal(dist_mat, np.inf)

    # Matrix profile: nearest neighbor distance for each subsequence
    mp = dist_mat.min(axis=1)
    mp_idx = dist_mat.argmin(axis=1)

    # ---- Motifs: subsequences with very low nearest-neighbor distance ----
    motif_threshold = np.percentile(mp, 15)
    motif_candidates = np.where(mp < motif_threshold)[0]

    # Group overlapping matches into motif sets
    used = set()
    motifs = []
    motif_idx = 0

    for ci in motif_candidates:
        if ci in used:
            continue
        # Find all subsequences that match this one
        matches = np.where(dist_mat[ci] < motif_threshold * 1.5)[0]
        match_indices = [indices[ci]]
        for m in matches:
            if m != ci and m not in used:
                # Check if match is close enough
                if dist_mat[ci, m] < motif_threshold * 1.5:
                    match_indices.append(indices[m])
                    used.add(int(m))
        used.add(int(ci))

        if len(match_indices) >= 2:
            start_times = [float(times[idx]) for idx in sorted(match_indices)]
            significance = float(1.0 - mp[ci])
            motifs.append(Motif(
                pattern_index=motif_idx,
                length_seconds=float(sub_len * hop),
                occurrences=start_times,
                significance=significance,
                description=(
                    f"重复 {len(start_times)} 次的模式 "
                    f"（长度 {sub_len*hop:.1f}s，显著性 {significance:.2f}）"
                ),
            ))
            motif_idx += 1
            if len(motifs) >= top_k:
                break

    # ---- Discords: subsequences with HIGHEST nearest-neighbor distance ----
    discord_threshold = np.percentile(mp, 90)
    discord_candidates = np.where(mp > discord_threshold)[0]
    discords = []
    for ci in discord_candidates[:3]:
        t = float(times[indices[ci]])
        discords.append(Anomaly(
            time=t,
            duration=float(sub_len * hop),
            anomaly_score=float(mp[ci]),
            description=f"最不寻常的 {sub_len*hop:.1f}s 片段（孤立度 {mp[ci]:.2f}）",
        ))

    return motifs, discords


# ---------------------------------------------------------------------------
# 3. Spectral decomposition (NMF)
# ---------------------------------------------------------------------------

def decompose_spectrogram(
    mel_spec: np.ndarray,
    mel_times: np.ndarray,
    n_components: int = 4,
    random_state: int = 42,
) -> Dict:
    """
    Decompose mel spectrogram into spectral building blocks via NMF.

    Each component is a "spectral template" — a frequency profile that
    activates at specific times.  Together they reconstruct the original
    spectrogram.  This reveals the *latent structure*: what spectral
    patterns combine to form the audio.

    Parameters
    ----------
    mel_spec : (n_mels, n_frames) ndarray   Mel spectrogram (dB).
    mel_times : (n_frames,) ndarray         Frame times.
    n_components : int                      Number of templates.
    random_state : int                      For reproducibility.

    Returns
    -------
    dict with:
        components : (n_components, n_mels) spectral templates
        activations : (n_components, n_frames) when each template is active
        reconstruction_error : float
        component_descriptions : list of str
    """
    from sklearn.decomposition import NMF

    # Ensure non-negative (shift if dB values are negative)
    spec_shifted = mel_spec - mel_spec.min()
    spec_shifted = np.maximum(spec_shifted, 0)

    n_mels, n_frames = spec_shifted.shape
    actual_n = min(n_components, n_mels, n_frames // 2)
    actual_n = max(2, actual_n)

    # NMF expects (n_samples, n_features); treat frames as samples, mel-bands as features
    X = spec_shifted.T  # (n_frames, n_mels)

    model = NMF(
        n_components=actual_n,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
        alpha_W=0.01,
        alpha_H=0.01,
    )
    W = model.fit_transform(X)        # (n_frames, n_components)
    H = model.components_             # (n_components, n_mels)

    # Reconstruction error
    recon = np.dot(W, H)  # (n_frames, n_mels)
    error = float(np.mean((X - recon) ** 2) / (np.mean(X ** 2) + 1e-12))

    # Normalise each component
    components = np.zeros((actual_n, n_mels))
    activations = np.zeros((actual_n, n_frames))
    for i in range(actual_n):
        scale = H[i].max() + 1e-12
        components[i] = H[i] / scale
        activations[i] = W[:, i] * scale  # W[:, i] is (n_frames,), the activation time series

    # Describe each component
    mel_freqs = np.linspace(0, 8000, n_mels)  # approximate
    descriptions = []
    for i in range(actual_n):
        comp = components[i]
        centroid = np.dot(mel_freqs, comp) / (comp.sum() + 1e-12)
        # Where is the energy concentrated?
        low_energy = comp[:n_mels//3].sum() / (comp.sum() + 1e-12)
        high_energy = comp[2*n_mels//3:].sum() / (comp.sum() + 1e-12)

        if low_energy > 0.5:
            char = "低频主导"
        elif high_energy > 0.4:
            char = "高频主导"
        elif centroid > 2000:
            char = "中高频"
        else:
            char = "全频带"

        weight = float(activations[i].sum() / activations.sum())

        descriptions.append({
            "index": i,
            "character": f"{char}（质心 ≈{centroid:.0f} Hz）",
            "weight": weight,
            "activation_peaks": _find_activation_peaks(activations[i], mel_times),
        })

    return {
        "components": components,
        "activations": activations,
        "reconstruction_error": error,
        "n_components": actual_n,
        "component_descriptions": descriptions,
    }


def _find_activation_peaks(activation: np.ndarray, times: np.ndarray,
                            top_n: int = 5) -> List[float]:
    """Find top activation peak times for one NMF component."""
    peaks, props = signal.find_peaks(activation, distance=max(1, len(activation)//10))
    if len(peaks) == 0:
        return []
    heights = activation[peaks]
    top_idx = peaks[np.argsort(heights)[-top_n:]]
    return [float(times[i]) for i in sorted(top_idx)]


# ---------------------------------------------------------------------------
# 4. Recurrence quantification analysis (RQA)
# ---------------------------------------------------------------------------

def recurrence_analysis(
    feats: np.ndarray,
    times: np.ndarray,
    embedding_dim: int = 3,
    embedding_delay: int = 2,
    recurrence_threshold: float = 0.15,
) -> Dict:
    """
    Quantify the dynamical structure of the audio feature trajectory.

    Builds a recurrence plot from the multi-dimensional feature time series,
    then computes standard RQA measures.

    Parameters
    ----------
    feats : (n, d) ndarray   Feature time series (e.g. dynamics).
    times : (n,) ndarray     Time stamps.
    embedding_dim, embedding_delay : int   Phase-space embedding params.
    recurrence_threshold : float   Fraction of max distance for threshold.

    Returns
    -------
    dict with RQA measures and interpretation.
    """
    n = feats.shape[0]
    if n < 10:
        return {
            "recurrence_rate": 0, "determinism": 0, "laminarity": 0,
            "mean_diag_length": 0, "trapping_time": 0,
            "entropy_diag": 0,
            "interpretation": "序列太短，无法分析",
        }

    # Normalise
    f = _zscore(feats.T).T

    # Phase-space embedding (time-delay)
    if embedding_dim > 1 and n > embedding_dim * embedding_delay:
        embedded = np.zeros((n - (embedding_dim - 1) * embedding_delay,
                             embedding_dim * f.shape[1]))
        for d in range(embedding_dim):
            start = d * embedding_delay
            end = start + embedded.shape[0]
            embedded[:, d * f.shape[1]:(d + 1) * f.shape[1]] = f[start:end, :]
        f = embedded

    # Recurrence matrix
    n = f.shape[0]
    if n > 300:
        # Downsample for efficiency
        stride = max(1, n // 300)
        idx = np.arange(0, n, stride)
        f = f[idx]
        n = f.shape[0]

    dist_mat = cdist(f, f, metric="euclidean")
    threshold = recurrence_threshold * dist_mat.max()
    R = (dist_mat < threshold).astype(int)
    np.fill_diagonal(R, 0)

    # ---- RQA measures ----
    recurrence_rate = float(R.sum()) / (n * n)

    # Diagonal line histogram
    diag_hist = _count_lines(R, diag=True)
    vert_hist = _count_lines(R, diag=False)

    # Determinism: fraction of recurrence points in diagonal lines >= 2
    diag_points = sum(k * v for k, v in diag_hist.items() if k >= 2)
    total_points = max(R.sum(), 1)
    determinism = float(diag_points) / total_points

    # Laminarity: fraction in vertical lines >= 2
    vert_points = sum(k * v for k, v in vert_hist.items() if k >= 2)
    laminarity = float(vert_points) / total_points

    # Mean diagonal line length
    diag_total = sum(v for k, v in diag_hist.items() if k >= 2)
    if diag_total > 0:
        mean_diag = sum(k * v for k, v in diag_hist.items() if k >= 2) / diag_total
    else:
        mean_diag = 1.0

    # Entropy of diagonal line lengths
    diag_probs = np.array([v for k, v in diag_hist.items() if k >= 2])
    diag_probs = diag_probs / (diag_probs.sum() + 1e-12)
    entropy_diag = float(-np.sum(diag_probs * np.log(diag_probs + 1e-12)))

    # Trapping time (mean vertical line length)
    vert_total = sum(v for k, v in vert_hist.items() if k >= 2)
    if vert_total > 0:
        trapping_time = sum(k * v for k, v in vert_hist.items() if k >= 2) / vert_total
    else:
        trapping_time = 1.0

    # Interpretation
    if determinism > 0.8:
        interp = "高度确定性动态，表明音频具有强烈的周期性或规则结构（如稳定节拍、持续音调）"
    elif determinism > 0.5:
        interp = "中等确定性，音频有一定规律但夹杂变化（如带有即兴变化的旋律）"
    elif determinism > 0.2:
        interp = "低确定性，音频动态较为随机（如噪声、环境音、自由节奏）"
    else:
        interp = "接近随机动态，缺乏可预测的结构"

    if laminarity > 0.6:
        interp += "；高 laminarity 表明存在持续/停滞状态"

    return {
        "recurrence_rate": recurrence_rate,
        "determinism": determinism,
        "laminarity": laminarity,
        "mean_diag_length": mean_diag,
        "trapping_time": trapping_time,
        "entropy_diag": entropy_diag,
        "interpretation": interp,
        "threshold": threshold,
    }


def _count_lines(R: np.ndarray, diag: bool = True) -> Dict[int, int]:
    """Count the histogram of diagonal or vertical line lengths in R."""
    n = R.shape[0]
    hist: Dict[int, int] = {}

    if diag:
        # Diagonal lines
        for offset in range(-n + 1, n):
            d = np.diag(R, k=offset)
            _accumulate_line_lengths(d, hist)
    else:
        # Vertical lines
        for col in range(n):
            _accumulate_line_lengths(R[:, col], hist)

    return hist


def _accumulate_line_lengths(arr: np.ndarray, hist: Dict[int, int]):
    """Find consecutive runs of 1s and accumulate their lengths."""
    length = 0
    for val in arr:
        if val == 1:
            length += 1
        else:
            if length >= 1:
                hist[length] = hist.get(length, 0) + 1
            length = 0
    if length >= 1:
        hist[length] = hist.get(length, 0) + 1


# ---------------------------------------------------------------------------
# 5. Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    feats: np.ndarray,
    times: np.ndarray,
    contamination: float = 0.05,
) -> List[Anomaly]:
    """
    Flag statistically unusual time windows using isolation-based scoring.

    Uses a simple Mahalanobis-distance approach (fast, no training needed).
    """
    n = feats.shape[0]
    if n < 5:
        return []

    # Mahalanobis distance from centroid
    mean = feats.mean(axis=0)
    try:
        cov = np.cov(feats.T)
        cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        return []

    scores = np.zeros(n)
    for i in range(n):
        diff = feats[i] - mean
        scores[i] = float(np.sqrt(diff @ cov_inv @ diff))

    threshold = np.percentile(scores, 100 * (1 - contamination))
    anomaly_idx = np.where(scores > threshold)[0]

    hop = (times[1] - times[0]) if len(times) > 1 else 0.25
    anomalies = []
    for idx in anomaly_idx[:10]:
        anomalies.append(Anomaly(
            time=float(times[idx]),
            duration=float(hop),
            anomaly_score=float(scores[idx]),
            description=(
                f"统计异常点（Mahalanobis 距离 {scores[idx]:.2f}，"
                f"超出阈值 {threshold:.2f}）"
            ),
        ))

    return anomalies


# ---------------------------------------------------------------------------
# 6. Clustering
# ---------------------------------------------------------------------------

def cluster_segments(
    feats: np.ndarray,
    times: np.ndarray,
    n_clusters: int = 4,
) -> Dict:
    """
    Cluster audio windows into acoustically similar groups.
    Uses K-Means with silhouette score auto-evaluation.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = feats.shape[0]
    if n < n_clusters * 2:
        n_clusters = max(2, n // 2)

    # Standardise
    f = _zscore(feats.T).T

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(f)

    # Silhouette score
    try:
        sil = float(silhouette_score(f, labels)) if n_clusters > 1 else 0.0
    except Exception:
        sil = 0.0

    # Characterise each cluster
    profiles = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        cluster_feats = feats[mask]
        cluster_times = times[mask]

        profiles.append({
            "cluster": c,
            "size": int(mask.sum()),
            "time_coverage": f"{cluster_times.min():.1f}s – {cluster_times.max():.1f}s",
            "mean_vector": cluster_feats.mean(axis=0).tolist(),
            "fraction": float(mask.sum()) / n,
        })

    return {
        "n_clusters": n_clusters,
        "labels": labels,
        "profiles": profiles,
        "silhouette_score": sil,
    }


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def explore_unsupervised(
    y: np.ndarray,
    sr: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    n_components: int = 4,
    n_clusters: int = 4,
    verbose: bool = True,
) -> UnsupervisedReport:
    """
    Full unsupervised exploration of a single audio file.

    Discovers structure without any labels or forecasting targets.

    Parameters
    ----------
    y : ndarray      Raw audio waveform.
    sr : int         Sample rate.
    window_size, hop_size : float   Dynamics extraction params.
    n_components : int    Number of NMF spectral components.
    n_clusters : int      Number of clusters for segmentation.

    Returns
    -------
    UnsupervisedReport with segmentation, motifs, decomposition, RQA, anomalies.
    """
    from .dynamics import extract_dynamics, detect_structural_segments
    from .features import compute_mel_spectrogram

    y = _as_float64(y).ravel()
    duration = len(y) / sr

    if verbose:
        print(f"  Audio: {duration:.2f}s ({sr} Hz)")

    # ---- Extract dynamics features ----
    if verbose:
        print("  [1/5] Extracting dynamics features ...")
    dyn = extract_dynamics(y, sr, window_size=window_size, hop_size=hop_size)
    times = dyn["times"]
    seg_raw = detect_structural_segments(dyn)

    # Build feature matrix from dynamics
    feats = np.column_stack([
        dyn["energy_norm"],
        dyn["brightness_norm"],
        dyn["complexity_norm"],
        dyn["rhythm_norm"],
    ])

    # ---- Change point detection ----
    if verbose:
        print("  [2/5] Detecting change points & segmenting ...")
    change_points, change_score = detect_change_points(
        feats, times, min_segment_length=1.0,
    )

    # Build segments from change points + structural labels
    segments = _build_segments(dyn, times, change_points, seg_raw, duration)

    # ---- Motif discovery ----
    if verbose:
        print("  [3/5] Discovering recurring motifs ...")
    motifs, discords = discover_motifs(feats, times, top_k=5)

    # ---- Spectral decomposition (NMF) ----
    if verbose:
        print("  [4/5] Decomposing spectrogram (NMF) ...")
    try:
        _, mel_times, mel_spec = compute_mel_spectrogram(y, sr, n_mels=64)
        decomp = decompose_spectrogram(mel_spec, mel_times, n_components=n_components)
    except Exception as e:
        if verbose:
            print(f"         NMF skipped: {e}")
        decomp = {"components": np.array([]), "activations": np.array([]),
                  "reconstruction_error": 0, "n_components": 0,
                  "component_descriptions": []}

    # ---- Recurrence quantification ----
    if verbose:
        print("  [5/5] Recurrence quantification analysis ...")
    rqa = recurrence_analysis(feats, times)

    # ---- Anomaly detection ----
    anomalies = detect_anomalies(feats, times)

    # ---- Clustering ----
    clustering = cluster_segments(feats, times, n_clusters=n_clusters)

    # ---- Spectral components ----
    spectral_components = []
    for desc in decomp.get("component_descriptions", []):
        spectral_components.append(SpectralComponent(
            index=desc["index"],
            activation_peaks=desc.get("activation_peaks", []),
            character=desc["character"],
            weight=desc["weight"],
        ))

    # ---- Compose overview ----
    overview_parts = []
    overview_parts.append(f"音频长度 {duration:.1f}s，")

    if change_points:
        overview_parts.append(f"检测到 {len(change_points)} 个结构变化点，"
                              f"可分为 {len(segments)} 个段落。")
    else:
        overview_parts.append(f"结构较为统一，未检测到显著变化点。")

    if motifs:
        m = motifs[0]
        overview_parts.append(f"发现 {m.occurrences} 处重复 motif"
                              f"（显著性 {m.significance:.2f}）。")

    overview_parts.append(
        f"动态特征：{rqa['interpretation']}。"
    )

    if decomp.get("reconstruction_error", 1.0) < 0.5:
        overview_parts.append(
            f"频谱可分解为 {decomp['n_components']} 个独立成分"
            f"（重建误差 {decomp['reconstruction_error']:.1%}）。"
        )

    overview_parts.append(
        "以上分析全部基于无监督方法，"
        "旨在发现音频的内在结构规律。"
    )

    report = UnsupervisedReport(
        change_points=change_points,
        segments=segments,
        motifs=motifs,
        discords=discords,
        n_components=decomp.get("n_components", 0),
        spectral_components=spectral_components,
        reconstruction_error=decomp.get("reconstruction_error", 0),
        recurrence_rate=rqa["recurrence_rate"],
        determinism=rqa["determinism"],
        laminarity=rqa["laminarity"],
        mean_diag_length=rqa["mean_diag_length"],
        recurrence_interpretation=rqa["interpretation"],
        n_clusters=clustering["n_clusters"],
        cluster_labels=clustering.get("labels", np.array([])),
        cluster_profiles=clustering["profiles"],
        silhouette_score=clustering["silhouette_score"],
        overview="".join(overview_parts),
        params={
            "window_size": window_size,
            "hop_size": hop_size,
            "sr": sr,
            "duration": duration,
            "n_components": n_components,
            "n_clusters": n_clusters,
        },
    )

    if verbose:
        n_m = len(motifs)
        n_cp = len(change_points)
        n_anom = len(anomalies)
        print(f"  [Done]  {n_cp} change points, {len(segments)} segments, "
              f"{n_m} motifs, {n_anom} anomalies found.")
        print()

    return report


def _build_segments(
    dyn: Dict,
    times: np.ndarray,
    change_points: List[float],
    seg_raw: Dict,
    duration: float,
) -> List[Segment]:
    """Build Segment objects from change points and structural labels."""
    segments = []
    boundaries = sorted(set([0.0] + change_points + [duration]))

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]
        if t_end - t_start < 0.3:
            continue

        # Extract dynamics for this time range
        mask = (times >= t_start) & (times <= t_end)
        if mask.sum() < 2:
            continue

        seg_dyn = {
            "energy": dyn["energy"][mask],
            "brightness": dyn["brightness"][mask],
            "complexity": dyn["complexity"][mask],
            "rhythm": dyn["rhythm"][mask],
        }
        seg = characterise_segment(seg_dyn, t_start, t_end)

        # Override label if this segment contains structural info
        mid = (t_start + t_end) / 2
        for ci in seg_raw.get("climax_indices", []):
            if ci < len(times) and abs(times[ci] - mid) < (t_end - t_start):
                seg.label = "高潮段 (climax)"
                break
        for ci in seg_raw.get("calm_indices", []):
            if ci < len(times) and abs(times[ci] - mid) < (t_end - t_start):
                seg.label = "平静段 (calm)"
                break
        for ci in seg_raw.get("buildup_indices", []):
            if ci < len(times) and abs(times[ci] - mid) < (t_end - t_start):
                seg.label = "积蓄段 (buildup)"
                break

        if seg.label == "segment":
            seg.label = f"段落 {i+1}"

        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# 8. Report printing
# ---------------------------------------------------------------------------

def print_unsupervised_report(report: UnsupervisedReport):
    """Pretty-print the unsupervised exploration report."""
    print()
    print("=" * 72)
    print("  UNSUPERVISED AUDIO PATTERN DISCOVERY")
    print("=" * 72)

    # Segmentation
    print()
    print("--- Structure & Segmentation ---")
    if report.change_points:
        print(f"  Change points at: {[f'{cp:.1f}s' for cp in report.change_points]}")
    else:
        print("  No significant structural change points detected.")
    if report.segments:
        print(f"  {len(report.segments)} segments identified:")
        for seg in report.segments:
            print(f"    [{seg.start:.1f}s – {seg.end:.1f}s] {seg.label}")
            print(f"      {seg.character}")

    # Motifs
    print()
    print("--- Recurring Motifs ---")
    if report.motifs:
        for m in report.motifs:
            print(f"  Motif [{m.pattern_index}]: {m.description}")
            print(f"    出现时刻: {[f'{t:.1f}s' for t in m.occurrences]}")
    else:
        print("  No significant recurring motifs found.")
    if report.discords:
        print("  Discords (unusual moments):")
        for d in report.discords[:3]:
            print(f"    {d.time:.1f}s: {d.description}")

    # Spectral decomposition
    print()
    print("--- Spectral Decomposition (NMF) ---")
    if report.spectral_components:
        print(f"  {report.n_components} spectral components "
              f"(reconstruction error: {report.reconstruction_error:.1%}):")
        for sc in report.spectral_components:
            print(f"    Component {sc.index}: {sc.character} "
                  f"(weight: {sc.weight:.1%})")
            if sc.activation_peaks:
                print(f"      活跃于: {[f'{t:.1f}s' for t in sc.activation_peaks]}")
    else:
        print("  NMF decomposition not available.")

    # Recurrence quantification
    print()
    print("--- Recurrence Quantification (RQA) ---")
    print(f"  Recurrence rate:  {report.recurrence_rate:.3f}")
    print(f"  Determinism:      {report.determinism:.3f}")
    print(f"  Laminarity:       {report.laminarity:.3f}")
    print(f"  Mean diag length: {report.mean_diag_length:.2f}")
    print(f"  Interpretation:   {report.recurrence_interpretation}")

    # Clustering
    print()
    print("--- Clustering ---")
    if report.cluster_profiles:
        print(f"  {report.n_clusters} clusters (silhouette: {report.silhouette_score:.3f}):")
        for cp in report.cluster_profiles:
            print(f"    Cluster {cp['cluster']}: {cp['size']} windows "
                  f"({cp['fraction']:.0%}), {cp['time_coverage']}")

    # Overview
    print()
    print("--- Overview ---")
    overview = report.overview
    width = 68
    for i in range(0, len(overview), width):
        print(f"  {overview[i:i+width]}")

    print()
    print("=" * 72)
    print()
