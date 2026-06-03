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
        density = "高节奏密度"
    elif onset_density > 0.015:
        density = "中等节奏密度"
    else:
        density = "稀疏、持续音为主"

    if bpm_est > 160:
        tempo = "快速"
    elif bpm_est > 100:
        tempo = "中速"
    elif bpm_est > 50:
        tempo = "慢速"
    else:
        tempo = "非常缓慢"

    return f"{tempo}，{density}（~{bpm_est:.0f} peaks/min）"


def _describe_energy(energy: np.ndarray, times: np.ndarray) -> str:
    n = len(energy)
    if n < 3:
        return "能量稳定"
    x = np.arange(n)
    slope, _, _, _, _ = stats.linregress(x, energy)
    peak_idx = int(np.argmax(energy))
    peak_time = times[peak_idx]

    if slope > energy.std() * 0.3 / n:
        trend = "能量整体上升"
    elif slope < -energy.std() * 0.3 / n:
        trend = "能量整体下降"
    else:
        trend = "能量整体平稳"

    return f"{trend}，峰值在 {peak_time:.1f}s"


def _describe_timbre(centroid_mean: float, flatness: float) -> str:
    if centroid_mean > 3000:
        bright = "明亮"
    elif centroid_mean > 1500:
        bright = "中等亮度"
    else:
        bright = "偏暗/温暖"

    if flatness > 0.7:
        noise = "，噪声成分多"
    elif flatness > 0.4:
        noise = "，混合音色"
    else:
        noise = "，音调纯净"

    return bright + noise + f"（质心 {centroid_mean:.0f} Hz）"


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
) -> List[Discovery]:
    """
    Explore relationships between two audio dynamics.

    For each dimension (energy, brightness, complexity, rhythm),
    finds corresponding segments and shared patterns.
    Each finding is reported independently with evidence.
    """
    discoveries: List[Discovery] = []

    dimensions = [
        ("energy", "能量趋势", "Energy Trend"),
        ("brightness", "亮度趋势", "Brightness Trend"),
        ("complexity", "复杂度趋势", "Complexity Trend"),
        ("rhythm", "节奏密度", "Rhythm Density"),
    ]

    for dim_key, dim_name_cn, dim_name_en in dimensions:
        a = dyn1[dim_key]
        b = dyn2[dim_key]
        times_a = dyn1["times"]
        times_b = dyn2["times"]

        # ---- 2a. Find corresponding segments ----
        matches = _find_corresponding_segments(a, b, times_a, times_b, dim_key)

        if matches:
            evidence_list = [
                Evidence(
                    method=m["evidence_method"],
                    confidence=m["confidence"],
                    detail=m["evidence_detail"],
                )
                for m in matches
            ]
            segment_matches = []
            for m in matches:
                segment_matches.append(SegmentMatch(
                    a=TimeRange(start=m["time_a_start"], end=m["time_a_end"]),
                    b=TimeRange(start=m["time_b_start"], end=m["time_b_end"]),
                    evidence=Evidence(
                        method=m["evidence_method"],
                        confidence=m["confidence"],
                        detail=m["evidence_detail"],
                    ),
                    dimension=dim_key,
                ))

            discovery = Discovery(
                title=f"{dim_name_cn} ({dim_name_en})",
                dimension=dim_key,
                discovery_type="segment_correspondence",
                summary=_compose_dimension_summary(dim_key, dim_name_cn, matches),
                segment_matches=segment_matches,
                evidence=evidence_list,
                meta={
                    "n_matches": len(matches),
                    "avg_confidence": float(np.mean([m["confidence"] for m in matches])),
                },
            )
            discoveries.append(discovery)
        else:
            # Report no match as a finding too
            discoveries.append(Discovery(
                title=f"{dim_name_cn} ({dim_name_en})",
                dimension=dim_key,
                discovery_type="segment_correspondence",
                summary=f"在{dim_name_cn}维度上未发现显著对应的段落。两段音频在该维度表现不同。",
                segment_matches=[],
                evidence=[Evidence(
                    method="segment_search",
                    confidence=0.0,
                    detail="未找到 confidence > 0.6 的匹配段落",
                )],
                meta={"n_matches": 0, "avg_confidence": 0.0},
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

    Strategy: for each prominent peak in A, scan B for the best-matching
    window via derivative correlation.  Only keep matches above threshold.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if len(a) < 5 or len(b) < 5:
        return []

    # Find interesting points in A (peaks and valleys)
    dist_a = max(1, len(a) // 8)
    peaks_a = _find_peaks_with_prominence(a, distance=dist_a, min_prominence=0.2)
    valleys_a = _find_peaks_with_prominence(-a, distance=dist_a, min_prominence=0.2)
    interesting = sorted(set(peaks_a + valleys_a))

    matches = []
    half_win = max(2, min(len(a), len(b)) // 20)

    for idx_a in interesting:
        # Extract a context window around the interesting point in A
        lo_a = max(0, idx_a - half_win)
        hi_a = min(len(a), idx_a + half_win)
        window_a = a[lo_a:hi_a]
        if len(window_a) < 3:
            continue

        # Derivative of window A (the "shape" we're looking for)
        deriv_a = np.gradient(window_a)
        deriv_a_norm = deriv_a / (np.linalg.norm(deriv_a) + 1e-12)

        # Slide over B, find best derivative correlation
        best_corr = -1.0
        best_idx_b = 0
        win_len = len(window_a)
        for j in range(len(b) - win_len):
            window_b = b[j:j + win_len]
            deriv_b = np.gradient(window_b)
            deriv_b_norm = deriv_b / (np.linalg.norm(deriv_b) + 1e-12)
            corr = float(np.dot(deriv_a_norm, deriv_b_norm))
            if corr > best_corr:
                best_corr = corr
                best_idx_b = j

        # Also check absolute value correlation for overall match
        if best_corr > min_confidence:
            # Compute additional evidence
            window_b_matched = b[best_idx_b:best_idx_b + win_len]
            # Pearson on raw values
            raw_r, _ = stats.pearsonr(window_a, window_b_matched)

            confidence = float(0.6 * best_corr + 0.4 * max(0, raw_r))

            if confidence >= min_confidence:
                hop_a = (times_a[1] - times_a[0]) if len(times_a) > 1 else 0.25
                hop_b = (times_b[1] - times_b[0]) if len(times_b) > 1 else 0.25

                evidence_method = "derivative_correlation"
                if best_corr > 0.85:
                    evidence_detail = (
                        f"形状高度匹配（导函数相关 r={best_corr:.3f}），"
                        f"原始值相关 r={raw_r:.3f}"
                    )
                elif best_corr > 0.7:
                    evidence_detail = (
                        f"形状较为匹配（导函数相关 r={best_corr:.3f}），"
                        f"原始值相关 r={raw_r:.3f}"
                    )
                else:
                    evidence_detail = (
                        f"存在弱对应关系（导函数相关 r={best_corr:.3f}），"
                        f"可能为偶然匹配"
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
                })

    # Deduplicate: keep highest-confidence matches, ensure no overlapping B segments
    matches = sorted(matches, key=lambda m: m["confidence"], reverse=True)
    kept = []
    used_b_ranges = []
    for m in matches:
        b_start, b_end = m["time_b_start"], m["time_b_end"]
        # Check overlap with already-kept matches
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


def _compose_dimension_summary(dim_key: str, dim_name: str,
                                matches: List[Dict]) -> str:
    """Compose a natural language summary for a dimension's findings."""
    if not matches:
        return f"在{dim_name}维度上未发现显著对应的段落。"

    n = len(matches)
    confs = [m["confidence"] for m in matches]
    avg_conf = np.mean(confs)
    max_conf = max(confs)

    time_ranges_a = [(m["time_a_start"], m["time_a_end"]) for m in matches]
    time_ranges_b = [(m["time_b_start"], m["time_b_end"]) for m in matches]

    if avg_conf > 0.8:
        quality = "高度对应"
    elif avg_conf > 0.65:
        quality = "较为对应"
    else:
        quality = "存在弱对应"

    return (
        f"在{dim_name}维度发现 {n} 处{quality}的段落。"
        f"平均置信度 {avg_conf:.2f}，最高 {max_conf:.2f}。"
        f"Audio A 匹配段: {_format_time_ranges(time_ranges_a)}；"
        f"Audio B 匹配段: {_format_time_ranges(time_ranges_b)}。"
    )


def _format_time_ranges(ranges: List[Tuple[float, float]]) -> str:
    if not ranges:
        return "无"
    parts = [f"{s:.1f}s–{e:.1f}s" for s, e in ranges[:3]]
    if len(ranges) > 3:
        parts.append(f"等{len(ranges)}段")
    return "，".join(parts)


# ---------------------------------------------------------------------------
# 3. Contrast analysis
# ---------------------------------------------------------------------------

def analyze_contrast(dyn1: Dict, dyn2: Dict) -> List[Discovery]:
    """
    Analyse how two audios DIFFER.  Differences are discoveries too.

    For each dimension, compare distributions and characterise the divergence.
    """
    contrasts: List[Discovery] = []

    dimensions = [
        ("energy", "能量", "Energy"),
        ("brightness", "亮度", "Brightness"),
        ("complexity", "复杂度", "Complexity"),
        ("rhythm", "节奏密度", "Rhythm"),
    ]

    for dim_key, dim_name_cn, dim_name_en in dimensions:
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
                direction = f"Audio A 的{dim_name_cn}显著高于 Audio B"
                ratio_str = f"A 是 B 的 {mean_a/(mean_b+1e-12):.1f} 倍"
            else:
                direction = f"Audio B 的{dim_name_cn}显著高于 Audio A"
                ratio_str = f"B 是 A 的 {mean_b/(mean_a+1e-12):.1f} 倍"

            contrasts.append(Discovery(
                title=f"{dim_name_cn}对比 ({dim_name_en} Contrast)",
                dimension=dim_key,
                discovery_type="contrast",
                summary=f"{direction}。{ratio_str}。分布重叠度仅 {overlap:.1%}。",
                evidence=[Evidence(
                    method="distribution_comparison",
                    confidence=float(np.clip(diff_ratio / 3.0, 0.0, 1.0)),
                    detail=f"A: mean={mean_a:.4f} std={std_a:.4f} | "
                           f"B: mean={mean_b:.4f} std={std_b:.4f} | "
                           f"标准化差异比: {diff_ratio:.2f}",
                )],
                meta={
                    "mean_a": mean_a, "mean_b": mean_b,
                    "diff_ratio": diff_ratio,
                    "distribution_overlap": float(overlap),
                },
            ))
        else:
            contrasts.append(Discovery(
                title=f"{dim_name_cn}对比 ({dim_name_en} Contrast)",
                dimension=dim_key,
                discovery_type="contrast",
                summary=f"两段音频在{dim_name_cn}维度上表现相近"
                        f"（标准化差异 {diff_ratio:.2f}，分布重叠 {overlap:.1%}）。",
                evidence=[Evidence(
                    method="distribution_comparison",
                    confidence=float(1.0 - np.clip(diff_ratio / 2.0, 0.0, 1.0)),
                    detail=f"A: mean={mean_a:.4f} | B: mean={mean_b:.4f} | "
                           f"差异较小",
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
    """Compose a natural-language overview paragraph from discoveries."""
    parts = []

    parts.append(f"Audio A（{duration_a:.1f}s）与 Audio B（{duration_b:.1f}s）")

    # Summarise discoveries
    n_matches_total = sum(d.meta.get("n_matches", 0) for d in discoveries)
    high_conf_dims = [
        d.title for d in discoveries
        if d.meta.get("avg_confidence", 0) > 0.7 and d.meta.get("n_matches", 0) > 0
    ]

    if high_conf_dims:
        parts.append(
            f"在 {', '.join(high_conf_dims)} 维度上发现了较为对应的段落"
            f"（共 {n_matches_total} 处匹配）。"
        )
    elif n_matches_total > 0:
        parts.append(
            f"共发现 {n_matches_total} 处可能的对应段落，但置信度普遍不高，"
            f"两段音频的结构关联性较弱。"
        )
    else:
        parts.append("未发现显著的段落对应关系。")

    # Summarise contrasts
    strong_contrasts = [
        c for c in contrasts
        if c.meta.get("diff_ratio", 0) > 1.0
    ]
    if strong_contrasts:
        dims = [c.dimension for c in strong_contrasts]
        parts.append(f"两段音频在 {', '.join(dims)} 上存在明显差异。")

    similar_dims = [
        c for c in contrasts
        if c.meta.get("diff_ratio", 0) <= 1.0 and c.meta.get("diff_ratio", 0) > 0
    ]
    if similar_dims:
        dims = [c.dimension for c in similar_dims]
        parts.append(f"在 {', '.join(dims)} 上表现相近。")

    parts.append(
        "以上发现仅供参考——音频相似性本质上是主观的，"
        "这些多维度证据帮助你自行判断两段音频之间的关系。"
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

        # ---- Cross-discovery ----
        if verbose:
            print("  [Discover] Cross-audio segment mapping ...")
        discoveries = cross_discover(dyn1, dyn2_obj)
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
            f"对 Audio A（{self_a['duration']:.1f}s）进行了结构探索，"
            f"发现了 {len(self_a['segments']['climax'])} 处高潮段、"
            f"{len(self_a['segments']['calm'])} 处平静段、"
            f"{len(self_a['motifs'])} 处重复 motif。"
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
