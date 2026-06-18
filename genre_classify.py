#!/usr/bin/env python3
"""
Genre Classification via Time-Series Feature Engineering
========================================================

Extracts comprehensive time-series features from audio, then runs
unsupervised exploration + multi-classifier comparison.

No deep generative models (LSTM/Transformer/HMM) — instead: ACF/PACF,
trend dynamics, volatility, periodicity, white-noise tests, ARIMA
characteristics, spectral features.

Classifiers: SVM, Logistic Regression, Random Forest, XGBoost, MLP

Usage
-----
    python genre_classify.py                          # defaults
    python genre_classify.py --n-train 100 --n-test 20 --fast
    python genre_classify.py --n-train 500 --n-test 200 --output results/genre_exp/
"""

import os, sys, json, time, warnings, argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Font ────────────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
for name in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    if name in {f.name for f in fm.fontManager.ttflist}:
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                             "figure.figsize": (12, 8)})

from audiots import loader, features, dynamics, volatility, analysis as _analysis
from audiots import model_analysis as _ma

# ── Constants ───────────────────────────────────────────────────────────────
GTZAN_DIR = os.path.join(PROJECT_ROOT, "Data", "genres_original")
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
SR = 16000
N_MELS = 128


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_ts_features(filepath: str, verbose: bool = False) -> dict:
    """
    Extract a rich set of time-series features from one audio file.
    Returns a flat dict of scalar features (all JSON-serialisable).
    """
    y, sr = loader.load_audio(filepath, target_sr=SR)
    feats: dict = {}

    # ── 1. ACF & PACF (lags 1-20) ──────────────────────────────────────
    y_down = y[:min(len(y), sr * 3)]  # first 3 seconds for ACF
    lags, acf_vals, ci = _analysis.compute_acf(y_down, nlags=20)
    _, pacf_vals, _ = _analysis.compute_pacf(y_down, nlags=20)
    for i in range(1, 21):
        feats[f"acf_lag{i}"] = float(acf_vals[i]) if i < len(acf_vals) else np.nan
        feats[f"pacf_lag{i}"] = float(pacf_vals[i]) if i < len(pacf_vals) else np.nan

    # ── 2. Dynamics trends (energy / brightness / complexity / rhythm) ──
    dyn = dynamics.extract_dynamics(y, sr, window_size=0.5, hop_size=0.25)
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        series = np.asarray(dyn[key], dtype=np.float64).ravel()
        feats[f"{key}_mean"] = float(np.mean(series))
        feats[f"{key}_std"] = float(np.std(series))
        feats[f"{key}_min"] = float(np.min(series))
        feats[f"{key}_max"] = float(np.max(series))
        feats[f"{key}_n_peaks"] = float(len(dynamics.detect_structural_segments(dyn).get("energy_peaks", [])))
        # Trend slope (linear regression)
        if len(series) > 1:
            slope = np.polyfit(np.arange(len(series)), series, 1)[0]
            feats[f"{key}_slope"] = float(slope)
        else:
            feats[f"{key}_slope"] = 0.0

    # ── 3. Volatility ──────────────────────────────────────────────────
    vol = volatility.compute_volatility_layer(dyn, rolling_window=10, fit_garch=True)
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        vk = f"{key}_vol"
        if vk in vol:
            vs = vol[vk]
            feats[f"{key}_vol_mean"] = float(np.mean(vs))
            feats[f"{key}_vol_std"] = float(np.std(vs))
        else:
            feats[f"{key}_vol_mean"] = 0.0
            feats[f"{key}_vol_std"] = 0.0
        # GARCH persistence
        garch_models = vol.get("garch_models", {})
        if key in garch_models:
            gm = garch_models[key]
            if gm is not None and hasattr(gm, "params"):
                try:
                    p = gm.params
                    alpha = float(p.get("alpha[1]", 0.1))
                    beta = float(p.get("beta[1]", 0.8))
                except Exception:
                    alpha, beta = 0.1, 0.8
            else:
                alpha, beta = 0.1, 0.8
        else:
            alpha, beta = 0.1, 0.8
        feats[f"{key}_garch_alpha"] = alpha
        feats[f"{key}_garch_beta"] = beta
        feats[f"{key}_garch_persist"] = alpha + beta

    # ── 4. Periodicity ─────────────────────────────────────────────────
    per = _analysis.analyze_periodicity(y, sr)
    feats["dominant_freq"] = float(per.get("dominant_frequency", 0))
    feats["dominant_period"] = float(per.get("dominant_period", 0))
    feats["n_period_peaks"] = float(per.get("n_peaks", 0))

    # ── 5. Complexity ──────────────────────────────────────────────────
    cpx = _analysis.analyze_complexity(y)
    feats["zero_crossing_rate"] = float(cpx.get("zero_crossing_rate", 0))
    feats["sample_entropy"] = float(cpx.get("sample_entropy", 0))

    # ── 6. Spectral flatness ───────────────────────────────────────────
    _, mag_fft = features.compute_fft(y, sr)
    feats["spectral_flatness"] = float(_analysis.compute_spectral_flatness(mag_fft))

    # ── 7. White noise tests ───────────────────────────────────────────
    y_wn = y[:min(len(y), sr * 2)]
    white = _analysis.test_white_noise(y_wn)
    if isinstance(white, dict):
        for test_name in ["ljung_box", "box_pierce", "jarque_bera",
                          "acf_test", "variance_stationarity", "runs_test"]:
            td = white.get(test_name, {})
            if isinstance(td, dict):
                feats[f"wn_{test_name}_pval"] = float(td.get("p_value", 1.0))
            else:
                feats[f"wn_{test_name}_pval"] = 1.0
        overall = white.get("overall", {})
        if isinstance(overall, dict):
            feats["wn_tests_passed"] = float(overall.get("tests_passed", 0))
        else:
            feats["wn_tests_passed"] = 0.0
    else:
        for tn in ["ljung_box", "box_pierce", "jarque_bera",
                    "acf_test", "variance_stationarity", "runs_test"]:
            feats[f"wn_{tn}_pval"] = 1.0
        feats["wn_tests_passed"] = 0.0

    # ── 8. ARIMA characteristics ───────────────────────────────────────
    try:
        arima_insights = _ma.analyze_arima_insights(dyn)
        for key in ["energy", "brightness", "complexity", "rhythm"]:
            ai = arima_insights.per_trend.get(key)
            if ai:
                feats[f"{key}_arima_p"] = float(ai.best_order[0])
                feats[f"{key}_arima_d"] = float(ai.best_order[1])
                feats[f"{key}_arima_q"] = float(ai.best_order[2])
                feats[f"{key}_arima_aic"] = float(ai.aic)
                feats[f"{key}_is_stationary"] = float(ai.is_stationary)
                feats[f"{key}_is_white_noise"] = float(ai.is_white_noise)
    except Exception:
        for key in ["energy", "brightness", "complexity", "rhythm"]:
            for suf in ["p", "d", "q", "aic"]:
                feats[f"{key}_arima_{suf}"] = 0.0
            feats[f"{key}_is_stationary"] = 0.0
            feats[f"{key}_is_white_noise"] = 0.0

    # ── 9. Mel spectral stats ──────────────────────────────────────────
    _, _, mel_spec = features.compute_mel_spectrogram(y, sr, n_mels=N_MELS)
    mel_mean = mel_spec.mean(axis=1)  # mean over time → (n_mels,)
    feats["mel_centroid"] = float(np.average(np.arange(N_MELS), weights=mel_mean + 1e-12))
    feats["mel_spread"] = float(np.sqrt(np.average((np.arange(N_MELS) - feats["mel_centroid"])**2,
                                                    weights=mel_mean + 1e-12)))
    feats["mel_skewness"] = float(_scipy_skew(mel_mean))

    return feats


def _scipy_skew(x):
    from scipy.stats import skew
    return skew(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(n_train_per_genre: int = 10, n_test_per_genre: int = 2,
                  fast: bool = True, verbose: bool = True,
                  cache_dir: Optional[str] = None,
                  train_dir: Optional[str] = None,
                  test_dir: Optional[str] = None) -> dict:
    """
    Build dataset from audio files.

    If train_dir / test_dir are provided, use pre-split folders directly.
    Otherwise fall back to auto-splitting from GTZAN_DIR.
    """
    # ── Level 2 cache: full dataset dump ────────────────────────────────
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        if train_dir and test_dir:
            ds_key = f"ds_train_{os.path.basename(train_dir)}_test_{os.path.basename(test_dir)}"
        else:
            ds_key = f"ds_t{n_train_per_genre}_v{n_test_per_genre}"
        ds_path = os.path.join(cache_dir, f"{ds_key}.npz")
        if os.path.exists(ds_path):
            if verbose:
                print(f"  [CACHE] Loading full dataset from {ds_path}")
            cached = np.load(ds_path, allow_pickle=True)
            return {
                "X_train": cached["X_train"],
                "y_train": cached["y_train"],
                "X_test": cached["X_test"],
                "y_test": cached["y_test"],
                "feature_names": list(cached["feature_names"]),
                "genre_names": GENRES,
                "class_labels": {i: g for i, g in enumerate(GENRES)},
            }

        # ── Level 1 cache: per-song checkpoints ─────────────────────────
        feat_cache = os.path.join(cache_dir, "feat_cache")
        os.makedirs(feat_cache, exist_ok=True)

    def _song_cache_path(filepath: str) -> str:
        """Deterministic cache path for a song's features."""
        import hashlib
        h = hashlib.md5(filepath.encode()).hexdigest()[:12]
        return os.path.join(feat_cache, f"{h}.npz")

    def _load_or_extract(filepath: str) -> dict:
        """Extract features, using per-song checkpoint if available."""
        if cache_dir:
            sp = _song_cache_path(filepath)
            if os.path.exists(sp):
                try:
                    cached = np.load(sp, allow_pickle=True)
                    return dict(cached["feats"].item())
                except Exception:
                    pass  # corrupt cache — re-extract

        # Extract
        feats = extract_ts_features(filepath)

        # Save checkpoint
        if cache_dir:
            sp = _song_cache_path(filepath)
            try:
                np.savez_compressed(sp, feats=np.array([feats], dtype=object))
            except Exception:
                pass

        return feats

    X_train, y_train = [], []
    X_test, y_test = [], []
    feature_names = None

    use_pre_split = (train_dir is not None and test_dir is not None)

    for label, genre in enumerate(GENRES):
        if use_pre_split:
            # Use pre-split train/test folders directly
            train_genre_dir = os.path.join(train_dir, genre)
            test_genre_dir = os.path.join(test_dir, genre)
            train_files = sorted([f for f in os.listdir(train_genre_dir) if f.endswith(".wav")]) \
                if os.path.isdir(train_genre_dir) else []
            test_files = sorted([f for f in os.listdir(test_genre_dir) if f.endswith(".wav")]) \
                if os.path.isdir(test_genre_dir) else []
            n_train = len(train_files)
            n_test = len(test_files)
        else:
            # Auto-split from GTZAN_DIR
            genre_dir = os.path.join(GTZAN_DIR, genre)
            if not os.path.isdir(genre_dir):
                continue
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
            n_avail = len(files)
            n_train = min(n_train_per_genre, max(1, n_avail - n_test_per_genre))
            n_test = min(n_test_per_genre, n_avail - n_train)
            train_files = files[:n_train]
            test_files = files[n_train:n_train + n_test]

        if verbose:
            print(f"  {genre:12s}: {n_train} train + {n_test} test")

        # ── Train ───────────────────────────────────────────────────────
        for i, fn in enumerate(train_files):
            fp = os.path.join(train_genre_dir if use_pre_split else os.path.join(GTZAN_DIR, genre), fn)
            try:
                feats = _load_or_extract(fp)
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                row = [feats.get(k, np.nan) for k in feature_names]
                X_train.append(row)
                y_train.append(label)
                if verbose and (i + 1) % 3 == 0:
                    print(f"    [{i+1}/{n_train}] {fn}")
            except Exception as e:
                if verbose:
                    print(f"    [SKIP] {fn}: {e}")

        # ── Test ────────────────────────────────────────────────────────
        for i, fn in enumerate(test_files):
            fp = os.path.join(test_genre_dir if use_pre_split else os.path.join(GTZAN_DIR, genre), fn)
            try:
                feats = _load_or_extract(fp)
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                row = [feats.get(k, np.nan) for k in feature_names]
                X_test.append(row)
                y_test.append(label)
            except Exception as e:
                if verbose:
                    print(f"    [SKIP] {fn}: {e}")

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Handle empty test set
    if X_test.size == 0:
        X_test = np.empty((0, X_train.shape[1]), dtype=np.float64)
        y_test = np.empty((0,), dtype=np.int32)

    # Fill NaN/Inf with column median (train) or 0 (fallback)
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        mask = np.isnan(col) | np.isinf(col)
        if mask.any():
            med = np.nanmedian(col)
            X_train[mask, j] = med if np.isfinite(med) else 0.0
    if X_test.shape[0] > 0:
        for j in range(X_test.shape[1]):
            col = X_test[:, j]
            mask = np.isnan(col) | np.isinf(col)
            if mask.any():
                med = np.nanmedian(X_train[:, j])
                X_test[mask, j] = med if np.isfinite(med) else 0.0

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names or [],
        "genre_names": GENRES,
        "class_labels": {i: g for i, g in enumerate(GENRES)},
    }

    # ── Save Level-2 cache (full dataset dump) ─────────────────────────
    if cache_dir:
        ds_path = os.path.join(cache_dir,
            f"ds_t{n_train_per_genre}_v{n_test_per_genre}.npz")
        np.savez_compressed(ds_path,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_names=np.array(feature_names, dtype=object),
        )
        if verbose:
            print(f"  [CACHE] Full dataset saved to {ds_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Unsupervised exploration
# ═══════════════════════════════════════════════════════════════════════════════

def run_unsupervised(data: dict, output_dir: str) -> List[str]:
    """PCA + t-SNE with label coloring."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    saved = []

    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])

    # Fill any remaining NaN with column median
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col)
            X[mask, j] = med if np.isfinite(med) else 0.0

    # Also replace inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    genres = data["genre_names"]
    colors = plt.cm.tab10.colors

    # ── PCA ────────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(14, 10))
    for i, g in enumerate(genres):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=40, alpha=0.7,
                   color=colors[i % len(colors)], label=g, edgecolors="gray", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — Time-Series Feature Space (colored by genre)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.2)
    p1 = os.path.join(output_dir, "U01_pca_projection.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    saved.append(os.path.basename(p1))

    # ── t-SNE ──────────────────────────────────────────────────────────
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        X_tsne = tsne.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(14, 10))
        for i, g in enumerate(genres):
            mask = y == i
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=40, alpha=0.7,
                       color=colors[i % len(colors)], label=g, edgecolors="gray", linewidth=0.3)
        ax.set_title("t-SNE — Time-Series Feature Space (colored by genre)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        ax.grid(True, alpha=0.2)
        p2 = os.path.join(output_dir, "U02_tsne_projection.png")
        fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
        saved.append(os.path.basename(p2))
    except Exception as e:
        print(f"  [WARN] t-SNE failed: {e}")

    # ── Feature correlation heatmap (top features only) ────────────────
    feat_names = data["feature_names"]
    if len(feat_names) > 40:
        # Keep most-variant features
        vars = np.var(X_scaled, axis=0)
        top_idx = np.argsort(vars)[-40:]
        X_sub = X_scaled[:, top_idx]
        fn_sub = [feat_names[i] for i in top_idx]
    else:
        X_sub = X_scaled
        fn_sub = feat_names

    corr = np.corrcoef(X_sub.T)
    corr = np.nan_to_num(corr)

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fn_sub))); ax.set_xticklabels(fn_sub, rotation=90, fontsize=6)
    ax.set_yticks(range(len(fn_sub))); ax.set_yticklabels(fn_sub, fontsize=6)
    ax.set_title("Feature Correlation Matrix (top 40 by variance)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    p3 = os.path.join(output_dir, "U03_feature_correlation.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    saved.append(os.path.basename(p3))

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(data: dict, output_dir: str) -> dict:
    """Train 5 classifiers, evaluate on test set, return results."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, f1_score)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    genres = data["genre_names"]

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "SVM (RBF kernel)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        "MLP (Deep NN)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

    results = {}
    saved_plots = []

    for name, clf in classifiers.items():
        print(f"\n  Training {name}...")
        t0 = time.time()
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "train_time_s": float(elapsed),
            "confusion_matrix": cm.tolist(),
        }

        print(f"    Accuracy: {acc:.3f}  |  F1 (macro): {f1:.3f}  |  Time: {elapsed:.1f}s")

        # ── Confusion matrix plot ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(genres))); ax.set_yticklabels(genres, fontsize=8)
        for i in range(len(genres)):
            for j in range(len(genres)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}\nAccuracy={acc:.3f}  F1={f1:.3f}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
        fn = f"C01_cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        fig.savefig(os.path.join(output_dir, fn), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_plots.append(fn)

    # ── Model comparison bar chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = list(results.keys())
    accs = [results[m]["accuracy"] for m in model_names]
    f1s = [results[m]["f1_macro"] for m in model_names]
    x = np.arange(len(model_names))
    width = 0.3
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, f1s, width, label="F1 (macro)", color="#55A868", alpha=0.85)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_title("Classifier Comparison — Time-Series Features", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 1.15)
    fn_comp = "C02_model_comparison.png"
    fig.savefig(os.path.join(output_dir, fn_comp), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    saved_plots.append(fn_comp)

    # ── Feature importance (from Random Forest) ────────────────────────
    rf = classifiers["Random Forest"]
    if hasattr(rf, "feature_importances_"):
        importances = rf.feature_importances_
        feat_names = data["feature_names"]
        top_n = min(20, len(feat_names))
        top_idx = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(top_n), importances[top_idx],
                color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)), edgecolor="gray")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Time-Series Features (Random Forest Importance)", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")
        fn_fi = "C03_feature_importance.png"
        fig.savefig(os.path.join(output_dir, fn_fi), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        saved_plots.append(fn_fi)
        results["_top_features"] = [(feat_names[i], float(importances[i])) for i in top_idx[::-1]]

    results["_plots"] = saved_plots
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ACF pattern comparison across genres
# ═══════════════════════════════════════════════════════════════════════════════

def plot_acf_by_genre(data: dict, output_dir: str) -> Optional[str]:
    """Average ACF pattern per genre — shows time-series character of each genre."""
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    feat_names = data["feature_names"]
    genres = data["genre_names"]

    # Find ACF feature indices
    acf_indices = [i for i, fn in enumerate(feat_names) if fn.startswith("acf_lag")]
    if not acf_indices:
        return None

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()
    lags = list(range(1, len(acf_indices) + 1))
    colors = plt.cm.tab10.colors

    for gi, genre in enumerate(genres):
        ax = axes[gi]
        mask = y == gi
        if mask.sum() == 0:
            continue
        genre_acfs = X[mask][:, acf_indices]
        mean_acf = genre_acfs.mean(axis=0)
        std_acf = genre_acfs.std(axis=0)
        ax.plot(lags[:len(mean_acf)], mean_acf, "k-", linewidth=2, color=colors[gi])
        ax.fill_between(lags[:len(mean_acf)], mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.2, color=colors[gi])
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"{genre} (n={mask.sum()})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Average ACF Pattern by Genre — Time-Series Signature", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(output_dir, "U04_acf_by_genre.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    return os.path.basename(fp)


# ═══════════════════════════════════════════════════════════════════════════════
# Additional interpretable visualizations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_boxplots_by_genre(data: dict, output_dir: str) -> Optional[str]:
    """
    Boxplots of key time-series features across genres.
    Shows which features best separate genres visually.
    """
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    feat_names = data["feature_names"]
    genres = data["genre_names"]

    # Pick representative features from each category
    candidates = [
        ("acf_lag1", "ACF Lag-1"),
        ("pacf_lag1", "PACF Lag-1"),
        ("energy_mean", "Energy Mean"),
        ("rhythm_slope", "Rhythm Slope"),
        ("energy_garch_persist", "GARCH Persist (Energy)"),
        ("zero_crossing_rate", "Zero-Crossing Rate"),
        ("sample_entropy", "Sample Entropy"),
        ("dominant_freq", "Dominant Frequency"),
        ("spectral_flatness", "Spectral Flatness"),
        ("wn_ljung_box_pval", "Ljung-Box p-value"),
    ]
    available = [(fn, label) for fn, label in candidates if fn in feat_names]
    if len(available) < 3:
        return None

    n_cols = min(5, len(available))
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    colors = plt.cm.tab10.colors

    for idx, (fn, label) in enumerate(available):
        ax = axes[idx // n_cols][idx % n_cols]
        fi = feat_names.index(fn)
        data_by_genre = [X[y == gi, fi] for gi in range(len(genres))]
        bp = ax.boxplot(data_by_genre, labels=genres, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors[:len(genres)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(len(available), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Key Time-Series Features by Genre — Distribution Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(output_dir, "U05_feature_boxplots_by_genre.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    return os.path.basename(fp)


def plot_acf_decay_by_genre(data: dict, output_dir: str) -> Optional[str]:
    """
    ACF decay speed comparison: lag-1 value and time-to-zero-crossing per genre.
    This directly shows time-series memory differences between genres.
    """
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    feat_names = data["feature_names"]
    genres = data["genre_names"]
    colors = plt.cm.tab10.colors

    acf_indices = [i for i, fn in enumerate(feat_names) if fn.startswith("acf_lag")]
    if not acf_indices:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: ACF lag-1 bar chart
    lag1_idx = feat_names.index("acf_lag1") if "acf_lag1" in feat_names else acf_indices[0]
    lag1_means, lag1_stds = [], []
    for gi in range(len(genres)):
        vals = X[y == gi, lag1_idx]
        lag1_means.append(np.mean(vals))
        lag1_stds.append(np.std(vals))
    ax1.bar(range(len(genres)), lag1_means, yerr=lag1_stds,
            color=colors[:len(genres)], edgecolor="gray", capsize=4, alpha=0.8)
    ax1.set_xticks(range(len(genres))); ax1.set_xticklabels(genres, rotation=45, ha="right")
    ax1.set_ylabel("ACF at Lag-1")
    ax1.set_title("Short-Term Memory (ACF Lag-1) — higher = stronger serial dependence",
                  fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    # Right: ACF half-life (lag where ACF drops below 0.5*lag1)
    half_lives = []
    for gi in range(len(genres)):
        genre_acfs = X[y == gi][:, acf_indices]
        mean_acf = genre_acfs.mean(axis=0)
        lag1_val = mean_acf[0]
        threshold = abs(lag1_val) * 0.5
        hl = len(mean_acf)
        for k, v in enumerate(mean_acf):
            if abs(v) < threshold:
                hl = k + 1
                break
        half_lives.append(hl)
    ax2.bar(range(len(genres)), half_lives,
            color=colors[:len(genres)], edgecolor="gray", alpha=0.8)
    ax2.set_xticks(range(len(genres))); ax2.set_xticklabels(genres, rotation=45, ha="right")
    ax2.set_ylabel("ACF Half-Life (lags)")
    ax2.set_title("Memory Length (ACF Half-Life) — higher = longer temporal dependence",
                  fontsize=11, fontweight="bold")
    # Annotation
    for i, (lag1, hl) in enumerate(zip(lag1_means, half_lives)):
        ax2.text(i, hl + 0.2, f"lag1={lag1:.2f}", ha="center", fontsize=7)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Time-Series Memory by Genre — ACF Decay Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(output_dir, "U06_acf_decay_by_genre.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    return os.path.basename(fp)


def plot_genre_separability(data: dict, output_dir: str) -> Optional[str]:
    """
    Silhouette analysis + inter/intra-class distance ratio.
    Shows how separable genres are in the time-series feature space.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_samples
    from scipy.spatial.distance import cdist

    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    genres = data["genre_names"]
    colors = plt.cm.tab10.colors

    # Silhouette scores
    try:
        sil_vals = silhouette_samples(X_s, y)
        sil_per_genre = [sil_vals[y == gi].mean() for gi in range(len(genres))]
        sil_global = np.mean(sil_vals)
    except Exception:
        sil_per_genre = [0] * len(genres)
        sil_global = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Silhouette per genre
    bars = ax1.bar(range(len(genres)), sil_per_genre,
                   color=colors[:len(genres)], edgecolor="gray", alpha=0.8)
    ax1.axhline(y=sil_global, color="black", linestyle="--", alpha=0.6,
                label=f"Global mean = {sil_global:.3f}")
    ax1.set_xticks(range(len(genres))); ax1.set_xticklabels(genres, rotation=45, ha="right")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Genre Separability (Silhouette)\nhigher = more distinct", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.2, axis="y")
    # Color bars by quality
    for bar, s in zip(bars, sil_per_genre):
        if s < 0:
            bar.set_facecolor("#C44E52")
        elif s < sil_global:
            bar.set_facecolor("#F9A65A")

    # Right: Inter/intra class distance ratio per genre
    intra_dists, inter_dists = [], []
    for gi in range(len(genres)):
        mask_i = y == gi
        Xi = X_s[mask_i]
        if Xi.shape[0] > 1:
            intra = np.mean(cdist(Xi, Xi))
        else:
            intra = 0
        Xo = X_s[~mask_i]
        inter = np.mean(cdist(np.atleast_2d(Xi.mean(axis=0)), Xo)) if Xo.shape[0] > 0 else 0
        intra_dists.append(intra)
        inter_dists.append(inter)

    x = np.arange(len(genres))
    width = 0.3
    ax2.bar(x - width / 2, intra_dists, width, label="Intra-class (compactness)",
            color="#4C72B0", alpha=0.8)
    ax2.bar(x + width / 2, inter_dists, width, label="Inter-class (separation)",
            color="#C44E52", alpha=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(genres, rotation=45, ha="right")
    ax2.set_ylabel("Euclidean Distance")
    ax2.set_title("Compactness vs Separation\nlower intra + higher inter = better", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Genre Separability in Time-Series Feature Space", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(output_dir, "U07_genre_separability.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    return os.path.basename(fp)


def plot_whitenoise_stationarity_by_genre(data: dict, output_dir: str) -> Optional[str]:
    """
    White-noise score and stationarity per genre.
    Shows which genres have more structured (non-random) signals.
    """
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    feat_names = data["feature_names"]
    genres = data["genre_names"]
    colors = plt.cm.tab10.colors

    # White noise: tests_passed (higher = closer to white noise)
    wn_idx = feat_names.index("wn_tests_passed") if "wn_tests_passed" in feat_names else None
    # Stationarity: count stationary trends
    stat_cols = [i for i, fn in enumerate(feat_names) if fn.endswith("_is_stationary")]
    # GARCH persistence average
    garch_cols = [i for i, fn in enumerate(feat_names) if fn.endswith("_garch_persist")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. White noise score
    ax = axes[0]
    if wn_idx is not None:
        wn_means, wn_stds = [], []
        for gi in range(len(genres)):
            vals = X[y == gi, wn_idx]
            wn_means.append(np.mean(vals))
            wn_stds.append(np.std(vals))
        ax.bar(range(len(genres)), wn_means, yerr=wn_stds,
               color=colors[:len(genres)], edgecolor="gray", capsize=4, alpha=0.8)
    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_ylabel("White-Noise Tests Passed (/5)")
    ax.set_title("Signal Randomness\n(higher = more noise-like)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # 2. Stationarity score
    ax = axes[1]
    if stat_cols:
        stat_means, stat_stds = [], []
        for gi in range(len(genres)):
            vals = X[y == gi][:, stat_cols].mean(axis=1)
            stat_means.append(np.mean(vals))
            stat_stds.append(np.std(vals))
        ax.bar(range(len(genres)), stat_means, yerr=stat_stds,
               color=colors[:len(genres)], edgecolor="gray", capsize=4, alpha=0.8)
    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_ylabel("Fraction of Stationary Trends")
    ax.set_title("Trend Stationarity\n(higher = more stable dynamics)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # 3. GARCH persistence
    ax = axes[2]
    if garch_cols:
        garch_means, garch_stds = [], []
        for gi in range(len(genres)):
            vals = X[y == gi][:, garch_cols].mean(axis=1)
            vals = vals[np.isfinite(vals)]
            garch_means.append(np.mean(vals) if len(vals) > 0 else 0)
            garch_stds.append(np.std(vals) if len(vals) > 0 else 0)
        ax.bar(range(len(genres)), garch_means, yerr=garch_stds,
               color=colors[:len(genres)], edgecolor="gray", capsize=4, alpha=0.8)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.4, label="α+β=1 (unit root)")
    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_ylabel("Mean GARCH Persistence (α+β)")
    ax.set_title("Volatility Memory\n(higher = longer vol clustering)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Signal Structure by Genre — Randomness, Stationarity & Volatility Memory",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fp = os.path.join(output_dir, "U08_signal_structure_by_genre.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    return os.path.basename(fp)


def plot_aggregated_classifier_analysis(results: dict, data: dict,
                                         output_dir: str) -> List[str]:
    """
    Post-classification analysis:
    - Aggregated confusion matrix (sum across classifiers)
    - Per-classifier per-genre F1 heatmap
    - Feature category importance breakdown
    """
    saved = []
    genres = data["genre_names"]
    feat_names = data["feature_names"]
    colors = plt.cm.tab10.colors

    # ── C04: Aggregated confusion matrix ───────────────────────────────
    all_cms = []
    for name, r in results.items():
        if name.startswith("_"):
            continue
        cm = r.get("confusion_matrix", [])
        if cm:
            all_cms.append(np.array(cm))
    if all_cms:
        agg_cm = sum(all_cms)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(agg_cm, cmap="YlOrRd", aspect="auto")
        for i in range(len(genres)):
            for j in range(len(genres)):
                ax.text(j, i, str(agg_cm[i, j]), ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white" if agg_cm[i, j] > agg_cm.max() / 2 else "black")
        ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(genres))); ax.set_yticklabels(genres, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Aggregated Confusion Matrix ({len(all_cms)} classifiers combined)\n"
                     f"Overall accuracy: {np.trace(agg_cm) / agg_cm.sum():.3f}",
                     fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.85)
        fn = "C04_aggregated_confusion.png"
        fig.savefig(os.path.join(output_dir, fn), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(); saved.append(fn)

    # ── C05: Per-classifier per-genre F1 heatmap ───────────────────────
    classifier_names = [k for k in results.keys() if not k.startswith("_")]
    if classifier_names and all_cms:
        f1_matrix = np.zeros((len(classifier_names), len(genres)))
        has_data = False
        for ci, name in enumerate(classifier_names):
            cm = np.array(results[name].get("confusion_matrix", []))
            if cm.size == 0:
                continue
            for gi in range(len(genres)):
                tp = cm[gi, gi]
                fp = cm[:, gi].sum() - tp
                fn = cm[gi, :].sum() - tp
                f1_matrix[ci, gi] = 2 * tp / (2 * tp + fp + fn + 1e-12)
                has_data = True
        if has_data:
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(f1_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
            ax.set_yticks(range(len(classifier_names))); ax.set_yticklabels(classifier_names)
            for i in range(len(classifier_names)):
                for j in range(len(genres)):
                    ax.text(j, i, f"{f1_matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if f1_matrix[i, j] < 0.5 else "black")
            ax.set_xlabel("Genre"); ax.set_title("Per-Classifier Per-Genre F1 Score", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax, shrink=0.85, label="F1")
            fn = "C05_per_classifier_f1_heatmap.png"
            fig.savefig(os.path.join(output_dir, fn), dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(); saved.append(fn)

    # ── C06: Feature category importance ───────────────────────────────
    rf = None
    for name, r in results.items():
        if name.startswith("_"):
            continue
    top_feats = results.get("_top_features", [])
    if not top_feats:
        return saved

    # Categorize top features
    categories = {
        "ACF/PACF": ["acf_lag", "pacf_lag"],
        "Dynamics": ["energy_", "brightness_", "complexity_", "rhythm_"],
        "Volatility+GARCH": ["_vol_", "_garch_"],
        "ARIMA": ["_arima_", "_is_stationary", "_is_white_noise"],
        "Complexity": ["zero_crossing", "sample_entropy"],
        "Spectral": ["spectral_flatness", "mel_", "dominant_"],
        "WhiteNoise": ["wn_"],
    }

    cat_scores = {k: 0.0 for k in categories}
    cat_counts = {k: 0 for k in categories}
    for fn, imp in top_feats:
        for cat, keywords in categories.items():
            if any(kw in fn for kw in keywords):
                cat_scores[cat] += imp
                cat_counts[cat] += 1
                break

    # Normalize by count
    for cat in cat_scores:
        if cat_counts[cat] > 0:
            cat_scores[cat] /= cat_counts[cat]

    fig, ax = plt.subplots(figsize=(10, 6))
    cat_names = [k for k in categories if cat_counts[k] > 0]
    cat_vals = [cat_scores[k] for k in cat_names]
    cat_colors = plt.cm.Set2.colors[:len(cat_names)]
    bars = ax.barh(range(len(cat_names)), cat_vals, color=cat_colors, edgecolor="gray", alpha=0.85)
    for bar, val, cnt in zip(bars, cat_vals, [cat_counts[k] for k in cat_names]):
        ax.text(bar.get_width() + max(cat_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} ({cnt} features)", va="center", fontsize=9)
    ax.set_yticks(range(len(cat_names))); ax.set_yticklabels(cat_names)
    ax.set_xlabel("Mean Importance (Random Forest)")
    ax.set_title("Feature Category Importance — Which TS Feature Types Matter Most?",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    fn = "C06_feature_category_importance.png"
    fig.savefig(os.path.join(output_dir, fn), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); saved.append(fn)

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genre Classification via Time-Series Features")
    parser.add_argument("--n-train", type=int, default=10,
                        help="Training samples per genre (default: 10, total=100)")
    parser.add_argument("--n-test", type=int, default=2,
                        help="Test samples per genre (default: 2, total=20)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: results/genre_classify_TIMESTAMP)")
    parser.add_argument("--fast", action="store_true", help="Skip slow feature extraction steps")
    parser.add_argument("--train-dir", default=None,
                        help="Pre-split training directory (e.g. ./train). "
                             "If set with --test-dir, uses these folders directly.")
    parser.add_argument("--test-dir", default=None,
                        help="Pre-split test directory (e.g. ./test). "
                             "If set with --train-dir, uses these folders directly.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(PROJECT_ROOT, "results", f"genre_classify_{ts}")
    cache_dir = os.path.join(PROJECT_ROOT, "results", ".genre_cache")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve train/test dirs relative to project root if given
    train_dir = os.path.join(PROJECT_ROOT, args.train_dir) if args.train_dir else None
    test_dir = os.path.join(PROJECT_ROOT, args.test_dir) if args.test_dir else None

    print("=" * 70)
    print("  GENRE CLASSIFICATION via TIME-SERIES FEATURES")
    print("=" * 70)
    if train_dir and test_dir:
        print(f"  Train dir: {train_dir}")
        print(f"  Test  dir: {test_dir}")
    else:
        print(f"  Train: {args.n_train} × 10 genres = {args.n_train * 10} songs")
        print(f"  Test:  {args.n_test} × 10 genres = {args.n_test * 10} songs")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ── Step 1: Build dataset ──────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  STEP 1: Feature Extraction")
    print(f"{'='*50}")
    t0 = time.time()
    data = build_dataset(n_train_per_genre=args.n_train, n_test_per_genre=args.n_test,
                         cache_dir=cache_dir, train_dir=train_dir, test_dir=test_dir)
    print(f"\n  Dataset: {data['X_train'].shape[0]} train × {data['X_train'].shape[1]} features")
    print(f"           {data['X_test'].shape[0]} test")
    print(f"  Feature extraction time: {time.time() - t0:.1f}s")

    # ── Step 2: Unsupervised exploration ───────────────────────────────
    print(f"\n{'='*50}")
    print("  STEP 2: Unsupervised Exploration")
    print(f"{'='*50}")
    unsup_plots = run_unsupervised(data, output_dir)
    print(f"  Plots: {', '.join(unsup_plots)}")

    # ── Step 3: ACF by genre ──────────────────────────────────────────
    acf_plot = plot_acf_by_genre(data, output_dir)
    if acf_plot:
        unsup_plots.append(acf_plot)

    # ── Additional unsupervised plots ──────────────────────────────────
    for plot_fn, label in [
        (plot_feature_boxplots_by_genre, "Feature boxplots by genre"),
        (plot_acf_decay_by_genre, "ACF decay analysis"),
        (plot_genre_separability, "Genre separability"),
        (plot_whitenoise_stationarity_by_genre, "Signal structure by genre"),
    ]:
        try:
            p = plot_fn(data, output_dir)
            if p: unsup_plots.append(p)
        except Exception as e:
            print(f"  [WARN] {label}: {e}")

    # ── Step 4: Train classifiers ──────────────────────────────────────
    print(f"\n{'='*50}")
    print("  STEP 3: Multi-Classifier Training")
    print(f"{'='*50}")
    results = train_and_evaluate(data, output_dir)

    # ── Post-classification plots ──────────────────────────────────────
    post_plots = plot_aggregated_classifier_analysis(results, data, output_dir)
    if post_plots:
        print(f"  Post-classifier plots: {', '.join(post_plots)}")

    # ── Step 5: Summary report ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"\n  {'Model':<25s} {'Accuracy':>10s} {'F1 Macro':>10s} {'Time':>8s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*8}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<25s} {r['accuracy']:10.3f} {r['f1_macro']:10.3f} {r['train_time_s']:7.1f}s")

    # Best model
    valid = {k: v for k, v in results.items() if not k.startswith("_")}
    best = max(valid, key=lambda k: valid[k]["accuracy"])
    print(f"\n  Best model: {best} (Accuracy={valid[best]['accuracy']:.3f})")

    # Top features
    top_feats = results.get("_top_features", [])
    if top_feats:
        print(f"\n  Top 5 time-series features (by RF importance):")
        for fn, imp in top_feats[:5]:
            print(f"    {fn:<50s} {imp:.4f}")

    # ── Save report ────────────────────────────────────────────────────
    report = {
        "config": {"n_train_per_genre": args.n_train, "n_test_per_genre": args.n_test},
        "dataset_shape": {"train": list(data["X_train"].shape), "test": list(data["X_test"].shape)},
        "n_features": len(data["feature_names"]),
        "classifiers": {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                        for k, v in results.items() if not k.startswith("_")},
        "top_features": top_feats[:10],
        "unsup_plots": unsup_plots,
        "classifier_plots": results.get("_plots", []),
        "post_classifier_plots": post_plots,
    }
    rp = os.path.join(output_dir, "summary_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, default=str, ensure_ascii=False, indent=2)

    print(f"\n  Report saved → {rp}")
    print(f"  All outputs → {output_dir}/")
    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
