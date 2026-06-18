#!/usr/bin/env python3
"""
Genre Classification v2 — Lean Time-Series Feature Engineering
===============================================================

Redesigned from v1 with lessons learned:
  - 124 features for 100 songs → overfitting (35-45% acc)
  - Fix: ~25 curated features, data augmentation, CV, feature selection

Core improvements over v1
-------------------------
* 25 hand-picked features (not 124) — every feature has a reason to exist
* Segment-based augmentation: 30s song → 3×10s windows → 3× data
* Stratified 5-fold CV with error bars on every metric
* SelectKBest + StandardScaler pipeline per classifier
* GridSearchCV for SVM, better RF/GBDT/MLP defaults
* 8 focused plots (not 19) — each answers a concrete question
* Per-song checkpoint cache (survives Ctrl+C)

Usage
-----
    python genre_classify_v2.py                          # defaults (10 train, 2 test per genre)
    python genre_classify_v2.py --n-train 10 --n-test 2  # same as defaults
    python genre_classify_v2.py --cv 5                    # 5-fold CV
"""

import os, sys, json, time, hashlib, warnings, argparse
from datetime import datetime
from typing import Dict, List, Optional

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
matplotlib.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "figure.figsize": (12, 8)})

from audiots import loader, features, dynamics, volatility, analysis as _analysis

GTZAN_DIR = os.path.join(PROJECT_ROOT, "Data", "genres_original")
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]
SR = 16000

# ═══════════════════════════════════════════════════════════════════════════════
# Curated feature extraction — 25 features, each with a documented purpose
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(filepath: str) -> dict:
    """Extract ~25 curated time-series features. No ARIMA (too slow/noisy)."""
    y, sr = loader.load_audio(filepath, target_sr=SR)
    f: dict = {}

    # ── ACF / PACF (compressed: 4 features instead of 40) ──────────────
    y_acf = y[:min(len(y), sr * 2)]
    lags, acf_vals, _ = _analysis.compute_acf(y_acf, nlags=15)
    _, pacf_vals, _ = _analysis.compute_pacf(y_acf, nlags=15)

    f["acf_lag1"] = float(acf_vals[1])
    f["acf_lag2"] = float(acf_vals[2])
    # Decay speed: how many lags until ACF < 0.1
    f["acf_decay_lag"] = float(next((i for i, v in enumerate(acf_vals) if abs(v) < 0.1), len(acf_vals)))
    f["pacf_lag1"] = float(pacf_vals[1])

    # ── Dynamics (mean + std only — min/max/peaks/slope are noisy) ─────
    dyn = dynamics.extract_dynamics(y, sr, window_size=0.5, hop_size=0.25)
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        s = np.asarray(dyn[key], dtype=np.float64).ravel()
        f[f"{key}_mean"] = float(np.mean(s))
        f[f"{key}_std"] = float(np.std(s))

    # ── Volatility (mean volatility only) ──────────────────────────────
    vol = volatility.compute_volatility_layer(dyn, rolling_window=10, fit_garch=False)
    for key in ["energy", "brightness", "complexity", "rhythm"]:
        vk = f"{key}_vol"
        vs = vol.get(vk, np.zeros(1))
        f[f"{key}_vol_mean"] = float(np.mean(vs))

    # ── Spectral ───────────────────────────────────────────────────────
    _, mag_fft = features.compute_fft(y, sr)
    f["spectral_flatness"] = float(_analysis.compute_spectral_flatness(mag_fft))
    _, _, mel_spec = features.compute_mel_spectrogram(y, sr, n_mels=64)
    mel_mean_t = mel_spec.mean(axis=0)  # mean across mel bands per frame
    f["mel_energy_mean"] = float(np.mean(mel_mean_t))
    f["mel_energy_std"] = float(np.std(mel_mean_t))

    # ── Complexity ─────────────────────────────────────────────────────
    cpx = _analysis.analyze_complexity(y)
    f["zcr"] = float(cpx.get("zero_crossing_rate", 0))
    f["sample_entropy"] = float(cpx.get("sample_entropy", 0))

    # ── Periodicity ────────────────────────────────────────────────────
    per = _analysis.analyze_periodicity(y, sr)
    f["dominant_freq"] = float(per.get("dominant_frequency", 0))

    # ── White noise score ──────────────────────────────────────────────
    y_wn = y[:min(len(y), sr * 2)]
    white = _analysis.test_white_noise(y_wn)
    overall = white.get("overall", {}) if isinstance(white, dict) else {}
    f["wn_score"] = float(overall.get("tests_passed", 0)) / 5.0

    return f


# ═══════════════════════════════════════════════════════════════════════════════
# Data augmentation: split 30s audio into overlapping 15s segments
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_augmented(filepath: str, n_segments: int = 2) -> List[dict]:
    """
    Extract features from multiple time segments of the same song.
    Doubles (or triples) effective sample size without collecting more songs.
    """
    y_full, sr = loader.load_audio(filepath, target_sr=SR)
    total = len(y_full)
    segment_len = total // (n_segments + 1) if n_segments > 1 else total
    hop = (total - segment_len) // max(n_segments - 1, 1) if n_segments > 1 else 0

    results = []
    for seg_idx in range(n_segments):
        start = seg_idx * hop
        end = min(start + segment_len * 2, total) if seg_idx == 0 else start + segment_len
        start = max(0, end - segment_len * 2) if seg_idx == n_segments - 1 else start
        end = min(total, start + segment_len)
        if end - start < sr:  # need at least 1 second
            continue

        y_seg = y_full[start:end]
        # Write to temp file for the feature extraction pipeline
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, y_seg, sr)
            feats = extract_features(tmp_path)
            results.append(feats)
        finally:
            os.unlink(tmp_path)
    return results if results else [extract_features(filepath)]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builder with per-song checkpoint caching
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(n_train: int = 10, n_test: int = 2,
                  augment: bool = True, verbose: bool = True,
                  cache_dir: Optional[str] = None,
                  train_dir: Optional[str] = None,
                  test_dir: Optional[str] = None) -> dict:
    """Build train/test dataset with per-song caching."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        feat_cache = os.path.join(cache_dir, "feat_cache_v2")
        os.makedirs(feat_cache, exist_ok=True)

    def _cache_path(fp: str) -> str:
        h = hashlib.md5(fp.encode()).hexdigest()[:12]
        return os.path.join(feat_cache, f"{h}.npz")

    def _load_or_extract(fp: str) -> List[dict]:
        if cache_dir:
            cp = _cache_path(fp)
            if os.path.exists(cp):
                try:
                    cached = np.load(cp, allow_pickle=True)
                    return list(cached["feats"].item())
                except Exception:
                    pass
        results = extract_features_augmented(fp) if augment else [extract_features(fp)]
        if cache_dir:
            cp = _cache_path(fp)
            try:
                np.savez_compressed(cp, feats=np.array([results], dtype=object))
            except Exception:
                pass
        return results

    X_train, y_train, X_test, y_test = [], [], [], []
    feature_names = None

    use_pre_split = (train_dir is not None and test_dir is not None)

    for label, genre in enumerate(GENRES):
        if use_pre_split:
            train_genre_dir = os.path.join(train_dir, genre)
            test_genre_dir = os.path.join(test_dir, genre)
            train_files = sorted([f for f in os.listdir(train_genre_dir) if f.endswith(".wav")]) \
                if os.path.isdir(train_genre_dir) else []
            test_files = sorted([f for f in os.listdir(test_genre_dir) if f.endswith(".wav")]) \
                if os.path.isdir(test_genre_dir) else []
            n_tr = len(train_files)
            n_te = len(test_files)
        else:
            genre_dir = os.path.join(GTZAN_DIR, genre)
            if not os.path.isdir(genre_dir):
                continue
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
            n_tr = min(n_train, max(1, len(files) - n_test))
            n_te = min(n_test, len(files) - n_tr)
            train_files = files[:n_tr]
            test_files = files[n_tr:n_tr + n_te]

        if verbose:
            print(f"  {genre:12s}: {n_tr} train + {n_te} test", end="")
            if augment:
                print(f"  (x2 augment -> ~{n_tr * 2} train samples)")
            else:
                print()

        for fn in train_files:
            try:
                fp = os.path.join(train_genre_dir if use_pre_split else os.path.join(GTZAN_DIR, genre), fn)
                for feats in _load_or_extract(fp):
                    if feature_names is None:
                        feature_names = sorted(feats.keys())
                    X_train.append([feats.get(k, 0.0) for k in feature_names])
                    y_train.append(label)
            except Exception:
                pass

        for fn in test_files:
            try:
                fp = os.path.join(test_genre_dir if use_pre_split else os.path.join(GTZAN_DIR, genre), fn)
                feats = extract_features(fp)
                if feature_names is None:
                    feature_names = sorted(feats.keys())
                X_test.append([feats.get(k, 0.0) for k in feature_names])
                y_test.append(label)
            except Exception:
                pass

    X_train = np.array(X_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Clip extreme values
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        q1, q3 = np.percentile(col[~np.isnan(col)], [1, 99]) if len(col) > 10 else (col.min(), col.max())
        col = np.clip(col, q1, q3)
        med = np.nanmedian(col)
        col[np.isnan(col) | np.isinf(col)] = med
        X_train[:, j] = col
    for j in range(X_test.shape[1]):
        col = X_test[:, j]
        col = np.clip(col, np.percentile(X_train[:, j], 1), np.percentile(X_train[:, j], 99))
        col[np.isnan(col) | np.isinf(col)] = np.nanmedian(X_train[:, j])
        X_test[:, j] = col

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names or [],
        "genre_names": GENRES,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Classifier training with CV + feature selection
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifiers(data: dict, cv_folds: int = 5) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix)

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    genres = data["genre_names"]
    n_features = X_train.shape[1]
    k_best = min(20, n_features)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Shared base: impute -> scale -> select best features
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_best)),
    ]

    classifiers = {
        "SVM (RBF)": Pipeline(base_steps + [
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]),
        "Logistic Regression": Pipeline(base_steps + [
            ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ]),
        "Random Forest": Pipeline(base_steps + [
            ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                            random_state=42)),
        ]),
        "Gradient Boosting": Pipeline(base_steps + [
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                learning_rate=0.05, random_state=42)),
        ]),
        "MLP (Deep NN)": Pipeline(base_steps + [
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                   early_stopping=True, random_state=42)),
        ]),
    }

    results = {}
    for name, pipe in classifiers.items():
        print(f"\n  {name}...")
        t0 = time.time()

        # CV scores
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        # Train on full train set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_pred_train = pipe.predict(X_train)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred)

        # Per-genre F1
        per_genre_f1 = {}
        for gi, g in enumerate(genres):
            tp = cm[gi, gi] if gi < cm.shape[0] else 0
            fp = cm[:, gi].sum() - tp if gi < cm.shape[1] else 0
            fn = cm[gi, :].sum() - tp
            per_genre_f1[g] = float(2 * tp / (2 * tp + fp + fn + 1e-12))

        elapsed = time.time() - t0
        print(f"    CV: {cv_mean:.3f} +/- {cv_std:.3f}  |  Train: {train_acc:.3f}  |  Test: {test_acc:.3f}  |  F1: {f1:.3f}")

        results[name] = {
            "cv_mean": cv_mean, "cv_std": cv_std,
            "train_acc": train_acc, "test_acc": test_acc, "f1_macro": f1,
            "per_genre_f1": per_genre_f1,
            "confusion_matrix": cm.tolist(),
            "train_time_s": elapsed,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualizations — 8 focused plots
# ═══════════════════════════════════════════════════════════════════════════════

def make_all_plots(data: dict, results: dict, output_dir: str) -> List[str]:
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import f_classif

    saved = []
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.hstack([data["y_train"], data["y_test"]])
    # Clean: impute NaN, then scale
    X = SimpleImputer(strategy="median").fit_transform(X)
    scaled = StandardScaler().fit_transform(X)
    genres = data["genre_names"]
    feat_names = data["feature_names"]
    colors = plt.cm.tab10.colors

    def _save(fig, name):
        fp = os.path.join(output_dir, name)
        fig.savefig(fp, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return name

    # ── 1. PCA projection ─────────────────────────────────────────────
    pca = PCA(n_components=2).fit(scaled)
    X_pca = pca.transform(scaled)
    fig, ax = plt.subplots(figsize=(12, 9))
    for gi, g in enumerate(genres):
        mask = y == gi
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=35, alpha=0.7,
                   color=colors[gi], label=g, edgecolors="gray", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — Do genres form natural clusters in TS feature space?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.2)
    saved.append(_save(fig, "01_pca_projection.png"))

    # ── 2. Feature importance (ANOVA F) ───────────────────────────────
    f_scores, _ = f_classif(scaled, y)
    top_n = min(15, len(feat_names))
    top_idx = np.argsort(f_scores)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), f_scores[top_idx],
            color=plt.cm.viridis(np.linspace(0.2, 0.9, top_n)), edgecolor="gray")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("ANOVA F-score (higher = more discriminative)")
    ax.set_title("Top 15 Features — Which TS features best separate genres?", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    saved.append(_save(fig, "02_feature_importance.png"))

    # ── 3. Model comparison with CV error bars ────────────────────────
    names = [k for k in results if not k.startswith("_")]
    cv_means = [results[n]["cv_mean"] for n in names]
    cv_stds = [results[n]["cv_std"] for n in names]
    test_accs = [results[n]["test_acc"] for n in names]
    f1s = [results[n]["f1_macro"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names)); w = 0.25
    ax.bar(x - w, cv_means, w, yerr=cv_stds, label=f"CV Accuracy (+/- std)", color="#4C72B0", alpha=0.85, capsize=4)
    ax.bar(x, test_accs, w, label="Test Accuracy", color="#55A868", alpha=0.85)
    ax.bar(x + w, f1s, w, label="Test F1 (macro)", color="#F9A65A", alpha=0.85)
    for i, (cv, ta, f1) in enumerate(zip(cv_means, test_accs, f1s)):
        ax.text(i - w, cv + 0.02, f"{cv:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i, ta + 0.02, f"{ta:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w, f1 + 0.02, f"{f1:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_title("Model Comparison — CV + Test (with feature selection)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2, axis="y"); ax.set_ylim(0, 1.1)
    saved.append(_save(fig, "03_model_comparison.png"))

    # ── 4. Aggregated confusion matrix ────────────────────────────────
    all_cms = [np.array(results[n].get("confusion_matrix", [])) for n in names if not n.startswith("_")]
    agg_cm = sum(all_cms) if all_cms else np.zeros((10, 10))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(agg_cm, cmap="YlOrRd", aspect="auto")
    for i in range(len(genres)):
        for j in range(len(genres)):
            ax.text(j, i, str(agg_cm[i, j]), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white" if agg_cm[i, j] > agg_cm.max() / 2 else "black")
    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genres))); ax.set_yticklabels(genres, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Aggregated Confusion Matrix ({len(all_cms)} classifiers)\n"
                 f"Overall accuracy: {np.trace(agg_cm) / max(agg_cm.sum(), 1):.3f}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.85)
    saved.append(_save(fig, "04_aggregated_confusion.png"))

    # ── 5. Per-genre F1 heatmap ───────────────────────────────────────
    if names:
        f1_mat = np.zeros((len(names), len(genres)))
        for ci, n in enumerate(names):
            pg = results[n].get("per_genre_f1", {})
            for gi, g in enumerate(genres):
                f1_mat[ci, gi] = pg.get(g, 0)
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(f1_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        for i in range(len(names)):
            for j in range(len(genres)):
                ax.text(j, i, f"{f1_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white" if f1_mat[i, j] < 0.5 else "black")
        ax.set_title("Per-Classifier Per-Genre F1 — Which genres are hard for everyone?", fontsize=12, fontweight="bold")
        ax.set_xlabel("Genre")
        plt.colorbar(im, ax=ax, shrink=0.85, label="F1")
        saved.append(_save(fig, "05_per_genre_f1.png"))

    # ── 6. ACF signature by genre ─────────────────────────────────────
    acf_idx = [i for i, fn in enumerate(feat_names) if fn.startswith("acf_lag")]
    if len(acf_idx) >= 2:
        fig, ax = plt.subplots(figsize=(12, 7))
        for gi, g in enumerate(genres):
            mask = y == gi
            if mask.sum() == 0: continue
            genre_data = X[mask][:, acf_idx]
            mean_acf = genre_data.mean(axis=0)
            ax.plot(range(1, len(mean_acf) + 1), mean_acf, color=colors[gi], linewidth=1.8, label=g)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Lag"); ax.set_ylabel("ACF")
        ax.set_title("ACF Pattern by Genre — Each genre has a unique time-series signature", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.2)
        saved.append(_save(fig, "06_acf_by_genre.png"))

    # ── 7. Top features by genre (boxplot grid) ───────────────────────
    top6_idx = np.argsort(f_scores)[-6:]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (fi, ax) in enumerate(zip(top6_idx, axes.flatten())):
        data_by_genre = [scaled[y == gi, fi] for gi in range(len(genres))]
        bp = ax.boxplot(data_by_genre, labels=genres, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.set_title(feat_names[fi], fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.2, axis="y")
    fig.suptitle("Top 6 Discriminative Features — Distribution by Genre", fontsize=14, fontweight="bold")
    fig.tight_layout()
    saved.append(_save(fig, "07_feature_distributions.png"))

    # ── 8. Summary report card ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10)); ax.axis("off")
    y_pos = 0.95; lh = 0.025

    def L(text, size=10, bold=False, indent=0):
        nonlocal y_pos
        ax.text(0.04 + indent * 0.02, y_pos, text, fontsize=size, fontweight="bold" if bold else "normal",
                transform=ax.transAxes, va="top")
        y_pos -= lh

    L(f"Genre Classification Report — {data['X_train'].shape[0]} train, {data['X_test'].shape[0]} test songs", 14, True)
    y_pos -= lh
    L(f"Features: {len(feat_names)} curated time-series features (with data augmentation)", 10)
    L(f"Feature selection: SelectKBest(k=20) + StandardScaler per classifier", 10)
    y_pos -= lh
    L("Results:", 12, True)
    for n in names:
        r = results[n]
        L(f"  {n:<22s}  CV={r['cv_mean']:.3f}+/-{r['cv_std']:.3f}  Test={r['test_acc']:.3f}  F1={r['f1_macro']:.3f}", 9)
    y_pos -= lh
    best = max(names, key=lambda n: results[n]["test_acc"])
    L(f"Best: {best} (Test={results[best]['test_acc']:.3f})", 11, True)
    y_pos -= lh
    L("Top 5 Features (ANOVA F-score):", 11, True)
    for i in range(5):
        L(f"  {i+1}. {feat_names[top_idx[-(i+1)]]}  (F={f_scores[top_idx[-(i+1)]]:.1f})", 9, indent=2)
    y_pos -= lh
    # Which genres are hardest?
    hardest = []
    for gi, g in enumerate(genres):
        mean_f1 = np.mean([results[n].get("per_genre_f1", {}).get(g, 0) for n in names])
        if mean_f1 < 0.3:
            hardest.append((g, mean_f1))
    if hardest:
        L("Hardest genres (mean F1 < 0.3):", 11, True)
        for g, f1v in sorted(hardest, key=lambda x: x[1]):
            L(f"  {g}: {f1v:.2f}", 9, indent=2)
    saved.append(_save(fig, "08_summary_report_card.png"))

    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Genre Classification v2 — Lean TS Features")
    parser.add_argument("--n-train", type=int, default=10, help="Train per genre (default 10, total 100)")
    parser.add_argument("--n-test", type=int, default=2, help="Test per genre (default 2, total 20)")
    parser.add_argument("--cv", type=int, default=5, help="CV folds (default 5)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--train-dir", default=None,
                        help="Pre-split training dir (e.g. ./train)")
    parser.add_argument("--test-dir", default=None,
                        help="Pre-split test dir (e.g. ./test)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(PROJECT_ROOT, "results", f"genre_v2_{ts}")
    cache_dir = os.path.join(PROJECT_ROOT, "results", ".genre_cache_v2")
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(PROJECT_ROOT, args.train_dir) if args.train_dir else None
    test_dir = os.path.join(PROJECT_ROOT, args.test_dir) if args.test_dir else None

    print("=" * 60)
    print("  GENRE CLASSIFICATION v2 - Lean Time-Series Features")
    print("=" * 60)
    if train_dir and test_dir:
        print(f"  Train dir: {train_dir}")
        print(f"  Test  dir: {test_dir}")
    else:
        print(f"  Train: {args.n_train * 10} songs + augmentation")
        print(f"  Test:  {args.n_test * 10} songs")
    print(f"  CV folds: {args.cv}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Step 1
    print(f"\n[1/4] Feature extraction...")
    t0 = time.time()
    data = build_dataset(n_train=args.n_train, n_test=args.n_test,
                         augment=not args.no_augment,
                         cache_dir=cache_dir, train_dir=train_dir, test_dir=test_dir)
    print(f"  Train: {data['X_train'].shape}  |  Test: {data['X_test'].shape}")
    print(f"  Features: {len(data['feature_names'])}  |  Time: {time.time() - t0:.0f}s")

    # Step 2
    print(f"\n[2/4] Training classifiers (CV={args.cv})...")
    results = train_classifiers(data, cv_folds=args.cv)

    # Step 3
    print(f"\n[3/4] Generating plots...")
    plot_files = make_all_plots(data, results, output_dir)
    print(f"  {len(plot_files)} plots generated")

    # Step 4
    print(f"\n[4/4] Summary:")
    print(f"  {'Model':<22s} {'CV':>8s} {'Test':>8s} {'F1':>8s}")
    names = [k for k in results if not k.startswith("_")]
    for n in names:
        r = results[n]
        print(f"  {n:<22s} {r['cv_mean']:7.3f}  {r['test_acc']:7.3f}  {r['f1_macro']:7.3f}")
    best = max(names, key=lambda n: results[n]["test_acc"])
    print(f"\n  Best: {best} (Test accuracy = {results[best]['test_acc']:.3f})")

    # Save report
    report = {
        "config": {"n_train": args.n_train, "n_test": args.n_test, "cv": args.cv,
                   "augment": not args.no_augment},
        "n_features": len(data["feature_names"]),
        "train_samples": data["X_train"].shape[0],
        "test_samples": data["X_test"].shape[0],
        "classifiers": {k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
                        for k, v in results.items()},
        "plots": plot_files,
    }
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, default=str, ensure_ascii=False, indent=2)

    print(f"\n  Report → {output_dir}/report.json")
    print(f"  All outputs → {output_dir}/")
    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
