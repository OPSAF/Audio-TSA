"""
Audio Discovery Engine
======================

Explores and discovers relationships within and between audio files.
**Does NOT output a single similarity score.** Instead, it surfaces
multi-dimensional findings, each backed by concrete evidence.

Philosophy
----------
Similarity is subjective.  This module does not judge — it *discovers*:

* Which segments correspond across two audios, and **why**.
* What rhythmic / energetic / timbral patterns they share.
* How they differ, and which features discriminate them most.
* Repeated motifs and structural landmarks within a single audio.

Every finding includes:
  - time ranges (where)
  - evidence method + confidence (why)
  - human-readable summary (what it means)

Architecture
------------
::

    explore(y1, sr1, y2=None, sr2=None)
        |
        +-- self_discovery(y, sr)          # per-audio analysis
        |       +-- structural segments
        |       +-- repeated motifs
        |       +-- audio characterisation
        |
        +-- cross_discover(dyn1, dyn2)     # cross-audio exploration
        |       +-- per-dimension segment mapping
        |       +-- shared pattern discovery
        |
        +-- analyze_contrast(dyn1, dyn2)   # difference analysis
        |
        +-- compose_overview(discovery)    # natural-language summary

Dependencies
------------
numpy, scipy only.  Reuses ``dynamics.py`` for trend extraction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Concrete evidence backing a finding."""
    method: str          # e.g. "derivative_correlation", "peak_alignment"
    confidence: float    # 0–1
    detail: str          # human-readable explanation


@dataclass
class TimeRange:
    """A time interval in seconds."""
    start: float
    end: float


@dataclass
class SegmentMatch:
    """A pair of corresponding segments in two audios."""
    a: TimeRange
    b: TimeRange
    evidence: Evidence
    dimension: str = ""


@dataclass
class Discovery:
    """A single discovery (finding)."""
    title: str                         # short headline
    dimension: str                     # "energy" | "brightness" | "complexity" | "rhythm" | "structure"
    discovery_type: str                # "segment_correspondence" | "pattern_match" | "motif" | "contrast"
    summary: str                       # one-paragraph human-readable summary
    segment_matches: List[SegmentMatch] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CharacterProfile:
    """What defines an audio's identity?"""
    rhythm_signature: str     # e.g. "稳定 ~120 BPM，节奏密度中等"
    energy_profile: str       # e.g. "能量逐步上升，在 3.2s 达到峰值"
    timbre_quality: str       # e.g. "明亮高频主导，音色温暖"
    standout_features: List[Tuple[str, str]] = field(default_factory=list)
    # (feature_name, description) e.g. ("spectral_centroid", "异常高 (4500 Hz)，远高于典型语音")


@dataclass
class DiscoveryReport:
    """Complete discovery report."""
    # Self-discovery (per audio)
    audio_a_profile: Optional[CharacterProfile] = None
    audio_b_profile: Optional[CharacterProfile] = None
    audio_a_segments: Dict[str, Any] = field(default_factory=dict)
    audio_b_segments: Dict[str, Any] = field(default_factory=dict)
    audio_a_motifs: List[Dict] = field(default_factory=list)
    audio_b_motifs: List[Dict] = field(default_factory=list)

    # Cross-discovery
    discoveries: List[Discovery] = field(default_factory=list)

    # Contrast analysis
    contrasts: List[Discovery] = field(default_factory=list)

    # Overview
    overview: str = ""

    # Metadata
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float64(y):
    return np.ascontiguousarray(y, dtype=np.float64)


def _describe_rhythm(onset_density: float, n_peaks: int, duration: float) -> str:
    bpm_est = n_peaks / max(duration, 0.1) * 60
    if onset_density > 0.03:
        density = "high rhythm density"
    elif onset_density > 0.015:
        density = "medium rhythm density"
    else:
        density = "sparse, sustained tones dominant"

    if bpm_est > 160:
        tempo = "fast"
    elif bpm_est > 100:
        tempo = "moderate"
    elif bpm_est > 50:
        tempo = "slow"
    else:
        tempo = "very slow"

    return f"{tempo}, {density} (~{bpm_est:.0f} peaks/min)"


def _describe_energy(energy: np.ndarray, times: np.ndarray) -> str:
    n = len(energy)
    if n < 3:
        return "stable energy"
    x = np.arange(n)
    slope, _, _, _, _ = stats.linregress(x, energy)
    peak_idx = int(np.argmax(energy))
    peak_time = times[peak_idx]

    if slope > energy.std() * 0.3 / n:
        trend = "rising energy"
    elif slope < -energy.std() * 0.3 / n:
        trend = "falling energy"
    else:
        trend = "stable energy"

    return f"{trend}, peak at {peak_time:.1f}s"


def _describe_timbre(centroid_mean: float, flatness: float) -> str:
    if centroid_mean > 3000:
        bright = "bright"
    elif centroid_mean > 1500:
        bright = "medium brightness"
    else:
        bright = "dark/warm"

    if flatness > 0.7:
        noise = ", noisy"
    elif flatness > 0.4:
        noise = ", mixed timbre"
    else:
        noise = ", pure tone"

    return f"{bright}{noise} (centroid {centroid_mean:.0f} Hz)"


def _find_peaks_with_prominence(arr: np.ndarray, distance: int = 3,
                                 min_prominence: float = 0.15) -> List[int]:
    """Find peaks with configurable prominence threshold."""
    if len(arr) < distance * 2:
        return []
    prominence = float(np.std(arr) * min_prominence)
    peaks, props = signal.find_peaks(arr, distance=distance,
                                     prominence=max(prominence, 1e-10))
    return peaks.tolist()


# ---------------------------------------------------------------------------
# 1. Self-discovery (single audio)
# ---------------------------------------------------------------------------

def self_discovery(
    y: np.ndarray,
    sr: int,
    window_size: float = 0.5,
    hop_size: float = 0.25,
) -> Dict:
    """
    Explore a single audio: find structural segments, repeated motifs,
    and characterise its sonic identity.

    Returns a dict with keys:
        segments, motifs, character, dynamics, summary_stats
    """
    from .dynamics import extract_dynamics, detect_structural_segments, summarize_dynamics

    y = _as_float64(y).ravel()
    dyn = extract_dynamics(y, sr, window_size=window_size, hop_size=hop_size)
    seg = detect_structural_segments(dyn)
    stats_summary = summarize_dynamics(dyn)

    duration = len(y) / sr

    # ---- Character profile ----
    onset_mean = float(dyn["rhythm"].mean())
    energy = dyn["energy"]
    times = dyn["times"]
    n_energy_peaks = len(_find_peaks_with_prominence(energy, distance=max(1, len(energy)//10)))

    centroid_mean = float(dyn["brightness"].mean())
    # Estimate spectral flatness from complexity trend
    complexity_mean = float(dyn["complexity"].mean())
    # Normalize complexity to 0-1 range for flatness proxy
    flatness_proxy = np.clip(complexity_mean / 15.0, 0.0, 1.0)

    character = CharacterProfile(
        rhythm_signature=_describe_rhythm(onset_mean, n_energy_peaks, duration),
        energy_profile=_describe_energy(energy, times),
        timbre_quality=_describe_timbre(centroid_mean, flatness_proxy),
        standout_features=[
            ("rms_energy_mean", f"{energy.mean():.4f}"),
            ("spectral_centroid_mean", f"{centroid_mean:.0f} Hz"),
            ("spectral_entropy_mean", f"{complexity_mean:.2f}"),
            ("onset_density_mean", f"{onset_mean:.4f}"),
        ],
    )

    # ---- Repeated motifs (self-similarity) ----
    motifs = _discover_self_motifs(dyn, y, sr)

    # ---- Assemble structural segments as time-ranged findings ----
    segment_findings = {
        "climax": [
            {"time": float(times[i]), "energy": float(energy[i])}
            for i in seg.get("climax_indices", []) if i < len(times)
        ],
        "calm": [
            {"time": float(times[i]), "energy": float(energy[i])}
            for i in seg.get("calm_indices", []) if i < len(times)
        ],
        "buildup": [
            {"time": float(times[i]), "energy": float(energy[i])}
            for i in seg.get("buildup_indices", []) if i < len(times)
        ],
        "transition": [
            {"time": float(times[i]), "complexity": float(dyn["complexity"][i])}
            for i in seg.get("transition_indices", []) if i < len(times)
        ],
    }

    return {
        "segments": segment_findings,
        "motifs": motifs,
        "character": character,
        "dynamics": dyn,
        "segments_raw": seg,
        "summary_stats": stats_summary,
        "duration": duration,
    }


def _discover_self_motifs(dyn: Dict, y: np.ndarray, sr: int) -> List[Dict]:
    """
    Find repeated patterns within a single audio using self-similarity
    on the multi-trend feature vector.
    """
    energy_n = dyn.get("energy_norm", dyn["energy"])
    brightness_n = dyn.get("brightness_norm", dyn["brightness"])
    rhythm_n = dyn.get("rhythm_norm", dyn["rhythm"])
    times = dyn["times"]
    n = len(energy_n)

    if n < 10:
        return []

    # Build compact multi-trend feature
    features = np.column_stack([
        energy_n, brightness_n, rhythm_n,
        dyn.get("complexity_norm", np.zeros(n)),
    ])
    # Normalize rows
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    feats_n = features / (norms + 1e-12)

    # Self-similarity matrix
    self_sim = np.dot(feats_n, feats_n.T)

    motifs = []
    min_lag = max(3, n // 10)
    for lag in range(min_lag, n - 1):
        diag = np.diag(self_sim, k=lag)
        if len(diag) < 3:
            continue
        peaks, props = signal.find_peaks(diag, height=0.85, distance=3)
        for p in peaks:
            if len(motifs) >= 20:
                break
            motifs.append({
                "start_a": float(times[p]),
                "start_b": float(times[p + lag]),
                "lag_seconds": float(times[p + lag] - times[p]),
                "similarity": float(diag[p]),
            })

    return sorted(motifs, key=lambda m: m["similarity"], reverse=True)[:20]


# ---------------------------------------------------------------------------
# 2. Cross-discovery (two audios)
# ---------------------------------------------------------------------------

def cross_discover(
    dyn1: Dict,
    dyn2: Dict,
    feat_list_1: Optional[List[Dict]] = None,
    feat_list_2: Optional[List[Dict]] = None,
    extra_dims: Optional[Dict] = None,
) -> List[Discovery]:
    """
    Explore relationships between two audio dynamics.

    For each dimension (energy, brightness, complexity, rhythm, plus
    optional tempo and timbre), finds corresponding segments and shared
    patterns. Each finding is reported independently with evidence.
    """
    discoveries: List[Discovery] = []

    _DIM_NAMES = {
        "energy":      ("Energy Trend", "Energy Trend"),
        "brightness":  ("Brightness Trend", "Brightness Trend"),
        "complexity":  ("Complexity Trend", "Complexity Trend"),
        "rhythm":      ("Rhythm Density", "Rhythm Density"),
    }

    for dim_key, (dim_name_cn, dim_name_en) in _DIM_NAMES.items():
        a = dyn1[dim_key]
        b = dyn2[dim_key]
        times_a = dyn1["times"]
        times_b = dyn2["times"]

        matches = _find_corresponding_segments(a, b, times_a, times_b, dim_key)

        if matches:
            evidence_list = [
                Evidence(method=m["evidence_method"], confidence=m["confidence"],
                         detail=m["evidence_detail"])
                for m in matches
            ]
            segment_matches = [
                SegmentMatch(
                    a=TimeRange(start=m["time_a_start"], end=m["time_a_end"]),
                    b=TimeRange(start=m["time_b_start"], end=m["time_b_end"]),
                    evidence=Evidence(method=m["evidence_method"],
                                      confidence=m["confidence"],
                                      detail=m["evidence_detail"]),
                    dimension=dim_key,
                ) for m in matches
            ]

            discoveries.append(Discovery(
                title=dim_name_cn, dimension=dim_key,
                discovery_type="segment_correspondence",
                summary=_compose_dimension_summary(dim_key, dim_name_cn, matches),
                segment_matches=segment_matches, evidence=evidence_list,
                meta={"n_matches": len(matches),
                      "avg_confidence": float(np.mean([m["confidence"] for m in matches]))},
            ))
        else:
            discoveries.append(Discovery(
                title=dim_name_cn, dimension=dim_key,
                discovery_type="segment_correspondence",
                summary=f"No significant corresponding segments found in {dim_name_en}. "
                        f"The two audios differ in this dimension.",
                segment_matches=[],
                evidence=[Evidence(method="segment_search", confidence=0.0,
                                   detail="No matches with confidence > 0.6 found")],
                meta={"n_matches": 0, "avg_confidence": 0.0},
            ))

    # ── Extra dimensions: tempo + timbre ────────────────────────────────
    if extra_dims:
        # Tempo comparison
        t1 = extra_dims.get("tempo_a", {})
        t2 = extra_dims.get("tempo_b", {})
        if t1.get("bpm") and t2.get("bpm"):
            bpm_diff = abs(t1["bpm"] - t2["bpm"])
            bpm_sim = np.clip(1.0 - bpm_diff / 40.0, 0.0, 1.0)  # 40 BPM diff → 0 similarity
            conf = bpm_sim * min(t1.get("confidence", 0.5), t2.get("confidence", 0.5))
            summary = (
                f"Tempo comparison: A≈{t1['bpm']:.0f} BPM (conf={t1.get('confidence',0):.2f}), "
                f"B≈{t2['bpm']:.0f} BPM (conf={t2.get('confidence',0):.2f}). "
                f"BPM difference={bpm_diff:.0f}. "
                f"{'Very similar tempo — likely compatible for mixing.' if bpm_diff < 5 else 'Moderate tempo difference.' if bpm_diff < 15 else 'Significantly different tempos — different rhythmic feels.'}"
            )
            discoveries.append(Discovery(
                title="Tempo", dimension="tempo",
                discovery_type="statistical_comparison",
                summary=summary,
                segment_matches=[],
                evidence=[Evidence(method="onset_acf", confidence=float(conf),
                                   detail=f"BPM A={t1['bpm']:.0f}, B={t2['bpm']:.0f}, diff={bpm_diff:.0f}")],
                meta={"n_matches": 0, "avg_confidence": float(conf),
                      "bpm_a": t1["bpm"], "bpm_b": t2["bpm"], "bpm_diff": bpm_diff},
            ))
        else:
            discoveries.append(Discovery(
                title="Tempo", dimension="tempo",
                discovery_type="statistical_comparison",
                summary="Could not reliably estimate tempo for one or both audios.",
                segment_matches=[],
                evidence=[Evidence(method="onset_acf", confidence=0.0,
                                   detail="Tempo estimation failed")],
                meta={"n_matches": 0, "avg_confidence": 0.0},
            ))

        # Timbre (MFCC) comparison
        mfcc_sim = extra_dims.get("mfcc_similarity", {})
        if mfcc_sim:
            ts = mfcc_sim.get("similarity", 0.5)
            discoveries.append(Discovery(
                title="Timbre (MFCC)", dimension="timbre",
                discovery_type="statistical_comparison",
                summary=mfcc_sim.get("detail", f"MFCC timbre similarity={ts:.2f}"),
                segment_matches=[],
                evidence=[Evidence(method="mfcc_statistics", confidence=float(ts),
                                   detail=mfcc_sim.get("detail", ""))],
                meta={"n_matches": 0, "avg_confidence": float(ts)},
            ))

    return discoveries


def _find_corresponding_segments(
    a: np.ndarray,
    b: np.ndarray,
    times_a: np.ndarray,
    times_b: np.ndarray,
    dimension: str,
    min_confidence: float = 0.55,
) -> List[Dict]:
    """
    Find segments in B that correspond to peaks / interesting regions in A.

    Strategy:
      1. Compute a self-similarity baseline (A vs time-shifted A) as the
         "perfect match" ceiling.
      2. For each interesting peak in A, find the best derivative-correlation
         match in B.
      3. Normalise cross-match confidence against the self-baseline so that
         the score reflects *genuine* similarity rather than just "both
         signals have some shape".

    This fixes the problem where two different-genre songs with similar
    generic dynamic shapes (e.g. crescendo→decrescendo) got high scores.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    n_a, n_b = len(a), len(b)
    if n_a < 5 or n_b < 5:
        return []

    # ── Baselines for normalisation ──────────────────────────────────────
    # Self-baseline (ceiling): compare A against a time-shifted version of
    # itself.  Smooth signals are inherently self-similar, so this ceiling
    # may be high (>0.8).  That's fine — it just means the raw trend shapes
    # are easy to match.  The *relative* drop when comparing different songs
    # is what matters.
    shift = max(1, n_a // 4)
    a_shifted = np.roll(a, shift)
    self_bl = _compute_match_strength(a, a_shifted, times_a, times_a,
                                       min_confidence=0.3)
    self_strength = max(self_bl.get("avg_confidence", 0.55), 0.40)

    # The 4 trend dimensions primarily measure DYNAMIC PHRASING similarity
    # (not genre). That's intentional — tempo + timbre dimensions handle
    # genre discrimination.  So we keep the raw confidence and add a
    # self-baseline annotation for transparency.
    norm_range = max(self_strength, 0.50)

    # ── Cross-match ──────────────────────────────────────────────────────
    # Find interesting points in A (peaks and valleys)
    dist_a = max(1, n_a // 8)
    peaks_a = _find_peaks_with_prominence(a, distance=dist_a, min_prominence=0.2)
    valleys_a = _find_peaks_with_prominence(-a, distance=dist_a, min_prominence=0.2)
    interesting = sorted(set(peaks_a + valleys_a))

    # Cap number of query points to avoid spurious matches on long signals
    max_interesting = max(6, min(20, n_a // 6))
    if len(interesting) > max_interesting:
        # Keep the most prominent ones
        prominences = [abs(float(a[i])) for i in interesting]
        idx_sorted = np.argsort(prominences)[::-1][:max_interesting]
        interesting = sorted([interesting[i] for i in idx_sorted])

    matches = []
    half_win = max(2, min(n_a, n_b) // 20)

    for idx_a in interesting:
        lo_a = max(0, idx_a - half_win)
        hi_a = min(n_a, idx_a + half_win)
        window_a = a[lo_a:hi_a]
        if len(window_a) < 3:
            continue

        deriv_a = np.gradient(window_a)
        deriv_a_norm = deriv_a / (np.linalg.norm(deriv_a) + 1e-12)

        best_corr = -1.0
        best_idx_b = 0
        win_len = len(window_a)
        for j in range(n_b - win_len):
            window_b = b[j:j + win_len]
            deriv_b = np.gradient(window_b)
            deriv_b_norm = deriv_b / (np.linalg.norm(deriv_b) + 1e-12)
            corr = float(np.dot(deriv_a_norm, deriv_b_norm))
            if corr > best_corr:
                best_corr = corr
                best_idx_b = j

        if best_corr > min_confidence:
            window_b_matched = b[best_idx_b:best_idx_b + win_len]
            raw_r, _ = stats.pearsonr(window_a, window_b_matched)

            # ── Confidence with self-baseline annotation ─────────────────
            # The raw confidence captures shape+value correlation.
            # We divide by self_strength so that:
            #   1.0 = matches as well as the signal matches itself
            #   0.5 = half as well
            # This is NOT a genre classifier — it measures dynamic phrasing
            # similarity.  Tempo + timbre dimensions (added separately)
            # provide genre-discriminative evidence.
            raw_confidence = float(0.6 * best_corr + 0.4 * max(0, raw_r))
            normalized = raw_confidence / norm_range
            normalized = np.clip(normalized, 0.0, 1.0)

            # Small bonus for matching a longer target signal
            if n_b > n_a * 1.5:
                normalized = min(1.0, normalized * 1.10)

            confidence = float(normalized)

            if confidence >= min_confidence:
                hop_a = (times_a[1] - times_a[0]) if len(times_a) > 1 else 0.25
                hop_b = (times_b[1] - times_b[0]) if len(times_b) > 1 else 0.25

                # ── Musical interpretation ──────────────────────────────
                evidence_method = "shape_correlation_normalized"
                if confidence > 0.85:
                    evidence_detail = (
                        f"Strong structural correspondence (norm. conf={confidence:.2f}). "
                        f"The shape of {_dim_label(dimension)} changes similarly in both audios "
                        f"(derivative r={best_corr:.2f}, value r={raw_r:.2f}). "
                        f"This suggests shared dynamic phrasing."
                    )
                elif confidence > 0.70:
                    evidence_detail = (
                        f"Moderate correspondence (norm. conf={confidence:.2f}). "
                        f"Similar {_dim_label(dimension)} contour detected "
                        f"(derivative r={best_corr:.2f}), but differences exist in absolute levels."
                    )
                elif confidence > 0.55:
                    evidence_detail = (
                        f"Weak correspondence (norm. conf={confidence:.2f}). "
                        f"The {_dim_label(dimension)} shapes are loosely aligned "
                        f"but may reflect generic signal structure rather than musical similarity."
                    )
                else:
                    evidence_detail = (
                        f"Marginal match (norm. conf={confidence:.2f}) — "
                        f"likely coincidental shape alignment, not musically meaningful."
                    )

                matches.append({
                    "time_a_start": float(times_a[lo_a]),
                    "time_a_end": float(times_a[min(hi_a, len(times_a) - 1)]),
                    "time_b_start": float(times_b[best_idx_b]),
                    "time_b_end": float(times_b[min(best_idx_b + win_len, len(times_b) - 1)]),
                    "evidence_method": evidence_method,
                    "confidence": confidence,
                    "evidence_detail": evidence_detail,
                    "derivative_corr": float(best_corr),
                    "pearson_r": float(raw_r),
                    "normalized": True,
                    "self_baseline": float(self_strength),
                })

    # Deduplicate
    matches = sorted(matches, key=lambda m: m["confidence"], reverse=True)
    kept = []
    used_b_ranges = []
    for m in matches:
        b_start, b_end = m["time_b_start"], m["time_b_end"]
        overlap = False
        for ub_start, ub_end in used_b_ranges:
            if not (b_end <= ub_start or b_start >= ub_end):
                overlap = True
                break
        if not overlap:
            kept.append(m)
            used_b_ranges.append((b_start, b_end))
            if len(kept) >= 10:
                break

    return sorted(kept, key=lambda m: m["time_a_start"])


def _dim_label(dim_key: str) -> str:
    """Human-readable dimension name for evidence descriptions."""
    return {
        "energy": "energy level",
        "brightness": "spectral brightness",
        "complexity": "spectral complexity",
        "rhythm": "rhythm density",
    }.get(dim_key, dim_key)


def _compute_match_strength(
    a: np.ndarray, b: np.ndarray,
    times_a: np.ndarray, times_b: np.ndarray,
    min_confidence: float = 0.3,
) -> Dict:
    """
    Compute a lightweight match-strength summary between two signals.
    Used for self-baseline estimation. Returns avg_confidence and n_matches.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 5 or len(b) < 5:
        return {"avg_confidence": 0.4, "n_matches": 0}

    n_samples = min(8, len(a) // 5)
    if n_samples < 2:
        return {"avg_confidence": 0.5, "n_matches": 1}

    step = max(1, len(a) // (n_samples + 1))
    confs = []
    win_len = max(3, len(a) // 15)

    for i in range(1, n_samples + 1):
        center = i * step
        lo = max(0, center - win_len // 2)
        hi = min(len(a), center + win_len // 2)
        wa = a[lo:hi]
        if len(wa) < 3:
            continue
        da = np.gradient(wa)
        da_n = da / (np.linalg.norm(da) + 1e-12)

        best = -1.0
        for j in range(len(b) - len(wa)):
            wb = b[j:j + len(wa)]
            db = np.gradient(wb)
            db_n = db / (np.linalg.norm(db) + 1e-12)
            c = float(np.dot(da_n, db_n))
            if c > best:
                best = c
        if best > min_confidence:
            confs.append(best)

    avg = float(np.mean(confs)) if confs else 0.4
    return {"avg_confidence": avg, "n_matches": len(confs)}


def _estimate_tempo(onset_density: np.ndarray, hop_s: float = 0.25) -> Dict:
    """
    Estimate tempo from onset-density autocorrelation.

    Returns dict with:
      - bpm: estimated beats per minute (dominant period)
      - confidence: 0-1 how clear the tempo is
      - periodicity_score: ACF peak strength
    """
    od = np.asarray(onset_density, dtype=np.float64).ravel()
    if len(od) < 8:
        return {"bpm": None, "confidence": 0.0, "periodicity_score": 0.0}

    # Normalise
    od = od - od.mean()
    od = od / (od.std() + 1e-12)

    # Autocorrelation
    acf = np.correlate(od, od, mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]  # normalise to lag-0

    # Search for peaks in the plausible tempo range: 40-220 BPM
    # At hop_s=0.25s, 40 BPM = 60/40*4 = 6.0s period = 24 lags
    #               220 BPM = 60/220*4 = 1.09s period = 4 lags
    min_lag = max(2, int(60.0 / 220.0 / hop_s))
    max_lag = min(len(acf) - 1, int(60.0 / 40.0 / hop_s))
    if max_lag <= min_lag:
        return {"bpm": None, "confidence": 0.0, "periodicity_score": 0.0}

    search_range = acf[min_lag:max_lag + 1]
    if len(search_range) < 2:
        return {"bpm": None, "confidence": 0.0, "periodicity_score": 0.0}

    peak_idx = int(np.argmax(search_range))
    peak_val = float(search_range[peak_idx])
    lag = min_lag + peak_idx
    period_s = lag * hop_s
    bpm = 60.0 / period_s if period_s > 0 else None

    # Confidence: how strong is the ACF peak relative to surrounding noise?
    if bpm is not None and 40 <= bpm <= 220:
        bg = float(np.median(search_range))
        conf = np.clip((peak_val - bg) / max(peak_val, 1e-12), 0.0, 1.0)
    else:
        conf = 0.0

    return {"bpm": bpm, "confidence": conf, "periodicity_score": peak_val}


def _compute_mfcc_similarity(
    y1: np.ndarray, sr1: int,
    y2: np.ndarray, sr2: int,
) -> Dict:
    """
    Compute timbre similarity via MFCC statistics.

    Compares the distribution of MFCC coefficients (mean + std per coefficient)
    rather than the time series directly. MFCCs capture timbre (instrumentation,
    vocal quality, production style) — which IS genre-discriminative.
    """
    from .features import compute_mfcc
    try:
        mfcc1, _ = compute_mfcc(y1, sr1, n_mfcc=13)
        mfcc2, _ = compute_mfcc(y2, sr2, n_mfcc=13)
    except Exception:
        return {"similarity": 0.5, "detail": "MFCC computation failed"}

    # Per-coefficient distribution similarity
    sims = []
    for i in range(min(13, mfcc1.shape[0], mfcc2.shape[0])):
        c1 = mfcc1[i, :].ravel()
        c2 = mfcc2[i, :].ravel()
        # Compare both mean AND variance
        mean_sim = 1.0 - np.clip(abs(float(c1.mean() - c2.mean())) /
                                   max(float(c1.std() + c2.std()), 1e-8), 0, 1)
        # Use Pearson r to compare the temporal patterns
        if len(c1) > 5 and len(c2) > 5:
            # Resample to same length for correlation
            target_len = min(len(c1), len(c2), 100)
            x_old = np.linspace(0, 1, len(c1))
            x_new = np.linspace(0, 1, target_len)
            c1_rs = np.interp(x_new, x_old, c1)
            x_old2 = np.linspace(0, 1, len(c2))
            c2_rs = np.interp(x_new, x_old2, c2)
            r, _ = stats.pearsonr(c1_rs, c2_rs)
            r = max(0, r)
        else:
            r = 0.5
        sims.append(0.5 * mean_sim + 0.5 * r)

    overall = float(np.mean(sims)) if sims else 0.5
    detail = (f"MFCC timbre similarity={overall:.2f} "
              f"({'similar instrument/texture' if overall > 0.6 else 'different timbre' if overall < 0.4 else 'mixed timbre'})")

    return {"similarity": overall, "detail": detail}


def _compose_dimension_summary(dim_key: str, dim_name: str,
                                matches: List[Dict]) -> str:
    """Compose a natural language summary for a dimension's findings."""
    if not matches:
        return f"No significant corresponding segments found in {dim_name} dimension."

    n = len(matches)
    confs = [m["confidence"] for m in matches]
    avg_conf = np.mean(confs)
    max_conf = max(confs)

    time_ranges_a = [(m["time_a_start"], m["time_a_end"]) for m in matches]
    time_ranges_b = [(m["time_b_start"], m["time_b_end"]) for m in matches]

    if avg_conf > 0.8:
        quality = "high correspondence"
    elif avg_conf > 0.65:
        quality = "moderate correspondence"
    else:
        quality = "weak correspondence"

    return (
        f"Found {n} {quality} segments in {dim_name} dimension. "
        f"Avg confidence {avg_conf:.2f}, max {max_conf:.2f}. "
        f"Audio A segments: {_format_time_ranges(time_ranges_a)}; "
        f"Audio B segments: {_format_time_ranges(time_ranges_b)}."
    )


def _format_time_ranges(ranges: List[Tuple[float, float]]) -> str:
    if not ranges:
        return "none"
    parts = [f"{s:.1f}s-{e:.1f}s" for s, e in ranges[:3]]
    if len(ranges) > 3:
        parts.append(f"+{len(ranges)-3} more")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# 3. Contrast analysis
# ---------------------------------------------------------------------------

def analyze_contrast(dyn1: Dict, dyn2: Dict) -> List[Discovery]:
    """
    Analyse how two audios DIFFER.  Differences are discoveries too.

    For each dimension, compare distributions and characterise the divergence.
    """
    contrasts: List[Discovery] = []

    _CONTRAST_DIM_NAMES = {
        "energy":      ("Energy", "Energy"),
        "brightness":  ("Brightness", "Brightness"),
        "complexity":  ("Complexity", "Complexity"),
        "rhythm":      ("Rhythm", "Rhythm"),
    }

    for dim_key, (dim_name_cn, dim_name_en) in _CONTRAST_DIM_NAMES.items():
        a = dyn1[dim_key]
        b = dyn2[dim_key]

        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        std_a, std_b = float(np.std(a)), float(np.std(b))

        # Normalised difference
        pooled_std = np.sqrt(std_a**2 + std_b**2) + 1e-12
        diff_ratio = abs(mean_a - mean_b) / pooled_std

        # Distribution overlap (Wasserstein via histograms)
        bins = max(10, min(30, int(np.sqrt(min(len(a), len(b))))))
        hist_a, edges = np.histogram(a, bins=bins, density=True)
        hist_b, _ = np.histogram(b, bins=edges, density=True)
        cdf_a = np.cumsum(hist_a) / (hist_a.sum() + 1e-12)
        cdf_b = np.cumsum(hist_b) / (hist_b.sum() + 1e-12)
        emd = float(np.sum(np.abs(cdf_a - cdf_b)) * (edges[1] - edges[0]))
        overlap = float(np.exp(-emd * 2.0))

        # Compose finding
        if diff_ratio > 1.0:
            if mean_a > mean_b:
                direction = f"Audio A => {dim_name_cn}({dim_name_en}) significantly higher than Audio B"
                ratio_str = f"A is {mean_a/(mean_b+1e-12):.1f}x of B"
            else:
                direction = f"Audio B => {dim_name_cn}({dim_name_en}) significantly higher than Audio A"
                ratio_str = f"B is {mean_b/(mean_a+1e-12):.1f}x of A"

            contrasts.append(Discovery(
                title=f"{dim_name_cn} Contrast",
                dimension=dim_key,
                discovery_type="contrast",
                summary=f"{direction}. {ratio_str}. Distribution overlap only {overlap:.1%}.",
                evidence=[Evidence(
                    method="distribution_comparison",
                    confidence=float(np.clip(diff_ratio / 3.0, 0.0, 1.0)),
                    detail=f"A: mean={mean_a:.4f} std={std_a:.4f} | "
                           f"B: mean={mean_b:.4f} std={std_b:.4f} | "
                           f"normalized diff ratio: {diff_ratio:.2f}",
                )],
                meta={
                    "mean_a": mean_a, "mean_b": mean_b,
                    "diff_ratio": diff_ratio,
                    "distribution_overlap": float(overlap),
                },
            ))
        else:
            contrasts.append(Discovery(
                title=f"{dim_name_cn} Contrast",
                dimension=dim_key,
                discovery_type="contrast",
                summary=f"Both audios similar in {dim_name_en} dimension "
                        f"(normalized diff {diff_ratio:.2f}, distribution overlap {overlap:.1%}).",
                evidence=[Evidence(
                    method="distribution_comparison",
                    confidence=float(1.0 - np.clip(diff_ratio / 2.0, 0.0, 1.0)),
                    detail=f"A: mean={mean_a:.4f} | B: mean={mean_b:.4f} | "
                           f"small difference",
                )],
                meta={
                    "mean_a": mean_a, "mean_b": mean_b,
                    "diff_ratio": diff_ratio,
                    "distribution_overlap": float(overlap),
                },
            ))

    return contrasts


# ---------------------------------------------------------------------------
# 4. Overview composer
# ---------------------------------------------------------------------------

def compose_overview(
    discoveries: List[Discovery],
    contrasts: List[Discovery],
    duration_a: float,
    duration_b: float,
) -> str:
    """
    Compose an interpretable overview that separates two kinds of similarity:

    1. **Dynamic phrasing similarity** — how similar are the energy/brightness/
       complexity/rhythm contours?  Two songs from completely different genres
       can share phrasing patterns (e.g. both have a loud climax, both fade out).
       High scores here mean "structurally similar", NOT "same genre".

    2. **Genre-relevant similarity** — tempo and timbre (MFCC).  These ARE
       genre-discriminative.  High scores here mean "likely same genre".
    """
    parts = []

    dur_ratio = max(duration_a, duration_b) / max(min(duration_a, duration_b), 0.1)
    parts.append(f"Audio A ({duration_a:.1f}s) vs Audio B ({duration_b:.1f}s)")

    # ── Separate phrasing vs genre dimensions ───────────────────────────
    phrasing_dims = {"energy", "brightness", "complexity", "rhythm"}
    phrasing_discs = [d for d in discoveries if d.dimension in phrasing_dims]
    genre_discs = [d for d in discoveries if d.dimension in ("tempo", "timbre")]

    # Phrasing similarity
    phrasing_confs = [d.meta.get("avg_confidence", 0) for d in phrasing_discs
                      if d.meta.get("n_matches", 0) > 0]
    phrasing_avg = float(np.mean(phrasing_confs)) if phrasing_confs else 0.0
    phrasing_total = sum(d.meta.get("n_matches", 0) for d in phrasing_discs)

    # Genre similarity
    genre_confs = [d.meta.get("avg_confidence", 0) for d in genre_discs]
    genre_avg = float(np.mean(genre_confs)) if genre_confs else 0.0

    # ── Phrasing similarity section ─────────────────────────────────────
    if phrasing_avg > 0.60:
        parts.append(
            f"Dynamic phrasing: STRONG similarity (avg {phrasing_avg:.2f}, "
            f"{phrasing_total} matched segments). "
            f"The two audios share similar energy/brightness/complexity/rhythm contours — "
            f"they have comparable structural phrasing. "
            f"(Note: this does NOT imply same genre — different genres can share phrasing patterns.)"
        )
    elif phrasing_avg > 0.40:
        parts.append(
            f"Dynamic phrasing: MODERATE similarity (avg {phrasing_avg:.2f}, "
            f"{phrasing_total} matched segments). "
            f"Some shared dynamic patterns detected."
        )
    else:
        parts.append(
            f"Dynamic phrasing: LOW similarity (avg {phrasing_avg:.2f}). "
            f"The two audios have fundamentally different dynamic structures."
        )

    # ── Genre-relevant section ──────────────────────────────────────────
    if genre_avg > 0.60:
        parts.append(
            f"Genre indicators: SIMILAR tempo+timbre (avg {genre_avg:.2f}) — "
            f"these two audios likely share genre, instrumentation, or production style."
        )
    elif genre_avg > 0.35:
        parts.append(
            f"Genre indicators: MIXED (avg {genre_avg:.2f}) — "
            f"some genre-relevant features match, some differ. "
            f"Check tempo and timbre details below."
        )
    else:
        parts.append(
            f"Genre indicators: DIFFERENT (avg {genre_avg:.2f}) — "
            f"tempo and/or timbre differ significantly. "
            f"The two audios are likely from different genres or production styles."
        )

    # ── Duration context ────────────────────────────────────────────────
    if dur_ratio > 2.0:
        parts.append(
            f"Note: large duration difference ({duration_a:.0f}s vs {duration_b:.0f}s, "
            f"ratio {dur_ratio:.1f}:1) makes comparison less reliable."
        )

    # ── Contrasts ───────────────────────────────────────────────────────
    strong_contrasts = [c for c in contrasts if c.meta.get("diff_ratio", 0) > 1.0]
    if strong_contrasts:
        dims = [c.dimension for c in strong_contrasts]
        parts.append(f"Significant differences in {', '.join(dims)} dimensions.")

    parts.append(
        "Note: phrasing similarity measures structural dynamics, not genre. "
        "Tempo + timbre provide genre-discriminative evidence. "
        "Confidence values are normalised against self-similarity baseline."
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def explore(
    y1: np.ndarray,
    sr1: int,
    y2: Optional[np.ndarray] = None,
    sr2: Optional[int] = None,
    window_size: float = 0.5,
    hop_size: float = 0.25,
    verbose: bool = True,
) -> DiscoveryReport:
    """
    Explore one or two audio files.  Find structure, motifs, correspondences,
    and contrasts.  Does NOT output a single similarity number.

    Parameters
    ----------
    y1 : ndarray       Raw waveform of primary audio.
    sr1 : int          Sample rate (Hz).
    y2 : ndarray, optional  Second audio for cross-discovery.
    sr2 : int, optional     Sample rate of second audio.
    window_size, hop_size : float  Dynamics extraction parameters.
    verbose : bool     Print progress.

    Returns
    -------
    DiscoveryReport with fields:
        - audio_a_profile, audio_b_profile
        - audio_a_segments, audio_b_segments
        - audio_a_motifs, audio_b_motifs
        - discoveries (cross-audio)
        - contrasts
        - overview (natural language)
    """
    from .dynamics import extract_dynamics

    y1 = _as_float64(y1).ravel()

    if verbose:
        duration1 = len(y1) / sr1
        print(f"  Audio A: {duration1:.2f}s ({sr1} Hz)")

    # ---- Self-discovery: Audio A ----
    if verbose:
        print("  [Discover] Analysing Audio A structure, motifs, character ...")
    self_a = self_discovery(y1, sr1, window_size=window_size, hop_size=hop_size)

    dyn1 = self_a["dynamics"]

    report = DiscoveryReport(
        audio_a_profile=self_a["character"],
        audio_b_profile=None,
        audio_a_segments=self_a["segments"],
        audio_b_segments={},
        audio_a_motifs=self_a["motifs"],
        audio_b_motifs=[],
        discoveries=[],
        contrasts=[],
        overview="",
        params={
            "window_size": window_size,
            "hop_size": hop_size,
            "sr": sr1,
            "duration_a": self_a["duration"],
        },
    )

    if y2 is not None:
        y2 = _as_float64(y2).ravel()
        if sr2 is None:
            sr2 = sr1
        if sr2 != sr1:
            try:
                import librosa
                y2 = librosa.resample(y2.astype(float), orig_sr=sr2, target_sr=sr1)
            except ImportError:
                from scipy.signal import resample
                y2 = resample(y2, int(len(y2) * sr1 / sr2))
            sr2 = sr1

        duration2 = len(y2) / sr1
        report.params["duration_b"] = duration2
        if verbose:
            print(f"  Audio B: {duration2:.2f}s")

        # ---- Self-discovery: Audio B ----
        if verbose:
            print("  [Discover] Analysing Audio B structure, motifs, character ...")
        self_b = self_discovery(y2, sr1, window_size=window_size, hop_size=hop_size)
        dyn2_obj = self_b["dynamics"]

        report.audio_b_profile = self_b["character"]
        report.audio_b_segments = self_b["segments"]
        report.audio_b_motifs = self_b["motifs"]

        # ---- Compute tempo + timbre features for cross-discovery ----
        extra_dims = {}
        try:
            tempo_a = _estimate_tempo(dyn1["rhythm"], hop_s=0.25)
            tempo_b = _estimate_tempo(dyn2_obj["rhythm"], hop_s=0.25)
            extra_dims["tempo_a"] = tempo_a
            extra_dims["tempo_b"] = tempo_b
        except Exception:
            pass
        try:
            extra_dims["mfcc_similarity"] = _compute_mfcc_similarity(
                y1, sr1, y2, sr1)
        except Exception:
            pass

        # ---- Cross-discovery ----
        if verbose:
            print("  [Discover] Cross-audio segment mapping ...")
        discoveries = cross_discover(dyn1, dyn2_obj, extra_dims=extra_dims)
        report.discoveries = discoveries

        # ---- Contrast analysis ----
        if verbose:
            print("  [Discover] Contrast / difference analysis ...")
        contrasts = analyze_contrast(dyn1, dyn2_obj)
        report.contrasts = contrasts

        # ---- Compose overview ----
        report.overview = compose_overview(
            discoveries, contrasts, self_a["duration"], duration2,
        )

    else:
        # Single audio: just self-discovery
        report.overview = (
            f"Structural exploration of Audio A ({self_a['duration']:.1f}s) completed. "
            f"Found {len(self_a['segments']['climax'])} climax segments, "
            f"{len(self_a['segments']['calm'])} calm segments, "
            f"{len(self_a['motifs'])} repeated motifs."
        )

    if verbose:
        n_disc = len(report.discoveries)
        n_cont = len(report.contrasts)
        print(f"  [Done]  {n_disc} discoveries, {n_cont} contrasts found.")
        print()

    return report


# ---------------------------------------------------------------------------
# 6. Report printing
# ---------------------------------------------------------------------------

def print_discovery_report(report: DiscoveryReport):
    """Pretty-print the full discovery report."""
    print()
    print("=" * 72)
    print("  AUDIO DISCOVERY REPORT")
    print("=" * 72)

    # Self-discovery A
    p = report.audio_a_profile
    if p:
        print()
        print("--- Audio A: Character Profile ---")
        print(f"  Rhythm:    {p.rhythm_signature}")
        print(f"  Energy:    {p.energy_profile}")
        print(f"  Timbre:    {p.timbre_quality}")

    seg_a = report.audio_a_segments
    if seg_a:
        n_c = len(seg_a.get("climax", []))
        n_calm = len(seg_a.get("calm", []))
        n_b = len(seg_a.get("buildup", []))
        print(f"  Structure: {n_c} climax, {n_calm} calm, {n_b} buildup segments")
        motifs_a = report.audio_a_motifs
        if motifs_a:
            print(f"  Motifs:    {len(motifs_a)} repeated patterns found")

    # Self-discovery B
    p = report.audio_b_profile
    if p:
        print()
        print("--- Audio B: Character Profile ---")
        print(f"  Rhythm:    {p.rhythm_signature}")
        print(f"  Energy:    {p.energy_profile}")
        print(f"  Timbre:    {p.timbre_quality}")

    seg_b = report.audio_b_segments
    if seg_b:
        n_c = len(seg_b.get("climax", []))
        n_calm = len(seg_b.get("calm", []))
        n_b = len(seg_b.get("buildup", []))
        print(f"  Structure: {n_c} climax, {n_calm} calm, {n_b} buildup segments")

    # Cross discoveries
    discoveries = report.discoveries
    if discoveries:
        print()
        print("--- Cross-Audio Discoveries ---")
        for i, disc in enumerate(discoveries):
            print(f"\n  [{i+1}] {disc.title}")
            print(f"  {disc.summary}")
            if disc.segment_matches:
                print(f"  Matched segments ({len(disc.segment_matches)}):")
                for sm in disc.segment_matches[:3]:
                    print(f"    A: {sm.a.start:.1f}s–{sm.a.end:.1f}s  <-->  "
                          f"B: {sm.b.start:.1f}s–{sm.b.end:.1f}s")
                    print(f"      evidence: {sm.evidence.method} "
                          f"(confidence={sm.evidence.confidence:.2f})")
                if len(disc.segment_matches) > 3:
                    print(f"    ... and {len(disc.segment_matches) - 3} more")

    # Contrasts
    contrasts = report.contrasts
    if contrasts:
        print()
        print("--- Contrasts / Differences ---")
        for i, c in enumerate(contrasts):
            print(f"  [{i+1}] {c.title}")
            print(f"  {c.summary}")

    # Overview
    if report.overview:
        print()
        print("--- Overview ---")
        # Word-wrap the overview
        overview = report.overview
        width = 68
        for i in range(0, len(overview), width):
            print(f"  {overview[i:i+width]}")

    print()
    print("=" * 72)
    print()
