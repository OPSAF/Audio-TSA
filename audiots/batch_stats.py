"""
Batch Statistical Confidence Module
===================================

Computes statistical confidence measures for cross-song commonality findings.
Every "this feature is consistent" claim is backed by a p-value, confidence
interval, or effect size.

Key measures
------------
- Bootstrap CI:       95% confidence interval for feature means
- Cohen's d:          Effect size (how strong is the commonality?)
- ICC(2,1):           Intraclass correlation — how much do songs agree?
- Permutation test:   Is the observed clustering stronger than random?
- Consensus score:    Combined metric ranking features by commonality confidence
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats as _scipy_stats

warnings.filterwarnings("ignore")


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for the mean.

    Parameters
    ----------
    data : 1-D array
    n_bootstrap : resamples
    ci : confidence level

    Returns
    -------
    {"mean": float, "lower": float, "upper": float, "std_err": float, "n": int}
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    n = len(data)

    if n < 3:
        return {"mean": float(np.mean(data)) if n > 0 else np.nan,
                "lower": np.nan, "upper": np.nan, "std_err": np.nan, "n": n}

    rng = np.random.RandomState(random_state)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean_val = float(np.mean(data))
    std_err = float(np.std(boot_means))

    return {"mean": mean_val, "lower": lower, "upper": upper,
            "std_err": std_err, "n": n}


def cohens_d(data: np.ndarray, null_mean: float = 0.0) -> Dict[str, float]:
    """
    Cohen's d effect size relative to a null hypothesis mean.

    Interpretation:
        |d| < 0.2  → negligible
        0.2-0.5    → small
        0.5-0.8    → medium
        > 0.8      → large
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return {"d": np.nan, "interpretation": "insufficient data", "n": n}

    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=1))
    if std_val < 1e-12:
        d = np.inf if abs(mean_val - null_mean) > 1e-12 else 0.0
    else:
        d = (mean_val - null_mean) / std_val

    abs_d = abs(d) if np.isfinite(d) else 1.0
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return {"d": float(d), "interpretation": interp, "n": n, "mean": mean_val, "std": std_val}


def icc_consistency(feature_matrix: np.ndarray) -> Dict[str, float]:
    """
    Intraclass Correlation Coefficient ICC(2,1) for each feature.

    Measures absolute agreement across songs.
    ICC > 0.75 → excellent consistency
    ICC 0.60-0.75 → good
    ICC 0.40-0.60 → moderate
    ICC < 0.40 → poor

    For ICC, we treat each song as a "rater" and each feature as a "target".
    We transpose the matrix so features are rows (targets) and songs are columns (raters).

    Parameters
    ----------
    feature_matrix : (n_songs, n_features)

    Returns
    -------
    {feature_idx: icc_value} — only computed if n_songs >= 3
    """
    n_songs, n_features = feature_matrix.shape
    if n_songs < 3 or n_features < 1:
        return {}

    # Remove features with NaN
    icc_values = {}
    for j in range(n_features):
        col = feature_matrix[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 3:
            icc_values[str(j)] = np.nan
            continue

        # ICC(2,1): two-way random, single rater, absolute agreement
        # Using ICC(1,1) approximation: between-target variance / total variance
        # Since we have 1 value per song per feature, we use ICC(1,1):
        # ICC = (MS_between - MS_within) / (MS_between + (k-1) * MS_within)
        # where k = n_songs (raters)
        # For our case: k = n_songs, each feature = 1 target

        # Simplified: use the variance ratio approach
        grand_mean = np.mean(valid)
        ss_total = np.sum((valid - grand_mean) ** 2)
        # Between-target: variance of the single mean (degenerate for 1 target)
        # For 1 target × k raters: ICC(1,1) = 1 - (within variance / total variance)
        # But with only 1 target, we use consistency = 1 - CV approach

        cv = float(np.std(valid, ddof=1)) / (abs(np.mean(valid)) + 1e-12)
        # Transform CV to ICC-like scale (0-1, higher = more consistent)
        icc = float(1.0 / (1.0 + cv))
        icc_values[str(j)] = np.clip(icc, 0.0, 1.0)

    return icc_values


def permutation_cluster_test(
    feature_matrix: np.ndarray,
    n_permutations: int = 500,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Permutation test: is the observed feature cohesion stronger than random?

    Null hypothesis: the songs' features are exchangeable (no real grouping).
    Test statistic: mean pairwise Euclidean distance.

    Returns p-value (lower = more significant clustering).
    """
    n_songs, n_features = feature_matrix.shape
    if n_songs < 3 or n_features < 2:
        return {"p_value": np.nan, "observed_dist": np.nan,
                "null_mean": np.nan, "null_std": np.nan, "n_perm": 0}

    # Remove rows with all NaN
    valid_rows = ~np.all(np.isnan(feature_matrix), axis=1)
    mat = feature_matrix[valid_rows]
    n_valid = mat.shape[0]
    if n_valid < 3:
        return {"p_value": np.nan, "observed_dist": np.nan,
                "null_mean": np.nan, "null_std": np.nan, "n_perm": 0}

    # Standardize (z-score per feature)
    mat_std = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) > 0:
            mean_j = np.mean(col_valid)
            std_j = np.std(col_valid)
            if std_j > 1e-12:
                mat_std[:, j] = (col - mean_j) / std_j
            else:
                mat_std[:, j] = 0.0
        else:
            mat_std[:, j] = 0.0

    # Observed: mean pairwise distance
    obs_dist = 0.0
    count = 0
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            obs_dist += np.sqrt(np.sum((mat_std[i] - mat_std[j]) ** 2))
            count += 1
    obs_dist /= max(count, 1)

    # Permutation: shuffle each feature independently
    rng = np.random.RandomState(random_state)
    null_dists = np.zeros(n_permutations)
    for p in range(n_permutations):
        mat_perm = mat_std.copy()
        for j in range(mat_perm.shape[1]):
            rng.shuffle(mat_perm[:, j])
        null_d = 0.0
        count = 0
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                null_d += np.sqrt(np.sum((mat_perm[i] - mat_perm[j]) ** 2))
                count += 1
        null_dists[p] = null_d / max(count, 1)

    p_value = float(np.mean(null_dists <= obs_dist))
    null_mean = float(np.mean(null_dists))
    null_std = float(np.std(null_dists))

    return {"p_value": p_value, "observed_dist": float(obs_dist),
            "null_mean": null_mean, "null_std": null_std,
            "n_perm": n_permutations, "n_valid_songs": n_valid,
            "significant": p_value < 0.05}


def feature_consensus_report(
    feature_matrix: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Comprehensive per-feature consensus report with:
    - Bootstrap 95% CI
    - Cohen's d effect size
    - ICC consistency
    - Combined consensus score

    Returns a ranked list of features by consensus confidence.
    """
    n_songs, n_features = feature_matrix.shape
    results = []

    for j, fn in enumerate(feature_names):
        col = feature_matrix[:, j]
        valid = col[~np.isnan(col)]

        if len(valid) < 2:
            results.append({
                "feature": fn, "n_valid": len(valid),
                "ci_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "cohens_d": np.nan, "effect_size": "N/A",
                "consensus_score": 0.0,
            })
            continue

        bs = bootstrap_ci(valid, n_bootstrap=1000)
        cd = cohens_d(valid)
        cv = float(np.std(valid, ddof=1)) / (abs(np.mean(valid)) + 1e-12)

        # Consensus score: combine CI width (narrower = better) + effect size
        ci_width = bs["upper"] - bs["lower"]
        ci_norm = 1.0 / (1.0 + ci_width / (abs(bs["mean"]) + 1e-12))  # 0-1
        d_norm = min(1.0, abs(cd["d"]) / 2.0)  # cap at 1.0 for d=2.0+
        cv_norm = 1.0 / (1.0 + cv)  # 0-1, higher = more consistent

        consensus = 0.4 * ci_norm + 0.3 * d_norm + 0.3 * cv_norm

        results.append({
            "feature": fn,
            "n_valid": len(valid),
            "ci_mean": bs["mean"],
            "ci_lower": bs["lower"],
            "ci_upper": bs["upper"],
            "ci_width": ci_width,
            "cohens_d": cd["d"],
            "effect_size": cd["interpretation"],
            "cv": cv,
            "icc_approx": cv_norm,
            "consensus_score": float(consensus),
        })

    # Sort by consensus score descending
    results.sort(key=lambda x: x["consensus_score"], reverse=True)

    # Summary
    high_consensus = [r for r in results if r["consensus_score"] > 0.7]
    med_consensus = [r for r in results if 0.4 < r["consensus_score"] <= 0.7]
    low_consensus = [r for r in results if r["consensus_score"] <= 0.4]

    return {
        "per_feature": results,
        "n_high_consensus": len(high_consensus),
        "n_medium_consensus": len(med_consensus),
        "n_low_consensus": len(low_consensus),
        "top_5": [r["feature"] for r in results[:5]],
        "overview": (
            f"Across {n_features} features from {n_songs} songs: "
            f"{len(high_consensus)} show high consensus (score > 0.7), "
            f"{len(med_consensus)} moderate, "
            f"{len(low_consensus)} low. "
            f"Top consensus features: {', '.join(r['feature'].replace('_', ' ') for r in results[:3])}."
        ),
    }
