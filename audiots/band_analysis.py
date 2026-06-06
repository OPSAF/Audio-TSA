"""Frequency band analysis module with optimizations."""

import numpy as np
import multiprocessing as mp
from functools import partial

from . import config as _cfg


def downsample_series(series, target_length=200):
    """Downsample series to target length using interpolation."""
    if len(series) <= target_length:
        return series
    indices = np.linspace(0, len(series) - 1, target_length, dtype=np.int32)
    return series[indices]


def analyze_single_band(band_data, band_info, forecast_horizon=20, epochs=None):
    """Analyze a single frequency band with all models."""
    if epochs is None:
        epochs = _cfg.DEFAULT_BAND_EPOCHS

    from .prediction import predict_arima, predict_hmm, LSTMPredictor, TransformerPredictor, compute_metrics

    # Use same approach as run_all_predictions - no downsampling
    band_mean = np.mean(band_data, axis=0)
    band_mean = np.asarray(band_mean, dtype=np.float64).ravel()

    # Ensure minimum length for meaningful prediction
    if len(band_mean) < forecast_horizon + 10:
        pad_size = forecast_horizon + 10 - len(band_mean)
        band_mean = np.pad(band_mean, (0, pad_size), mode="edge")

    train_size = int(len(band_mean) * 0.8)
    train_series = band_mean[:train_size]
    true_values = band_mean[train_size:train_size + forecast_horizon]

    band_results = {}

    # ARIMA - fast, always run
    try:
        arima_forecast, arima_metrics = predict_arima(train_series, forecast_horizon)
        band_results["ARIMA"] = {
            "forecast": arima_forecast, "metrics": arima_metrics, "true": true_values,
        }
    except Exception as e:
        band_results["ARIMA"] = {
            "forecast": np.zeros(forecast_horizon),
            "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": str(e)},
            "true": true_values,
        }

    # HMM - fast, always run
    try:
        hmm_forecast, hmm_metrics = predict_hmm(train_series, forecast_horizon)
        band_results["HMM"] = {
            "forecast": hmm_forecast, "metrics": hmm_metrics, "true": true_values,
        }
    except Exception as e:
        band_results["HMM"] = {
            "forecast": np.zeros(forecast_horizon),
            "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": str(e)},
            "true": true_values,
        }

    # LSTM / Transformer — only if enough data
    sufficient = len(train_series) >= _cfg.MIN_WINDOWS_DEEP_MODEL

    if sufficient:
        try:
            lstm = LSTMPredictor(lookback=30)
            lstm_forecast, lstm_metrics = lstm.predict(
                train_series, forecast_horizon, epochs=epochs
            )
            band_results["LSTM"] = {
                "forecast": lstm_forecast, "metrics": lstm_metrics, "true": true_values,
            }
        except Exception as e:
            band_results["LSTM"] = {
                "forecast": np.zeros(forecast_horizon),
                "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": str(e)},
                "true": true_values,
            }

        try:
            transformer = TransformerPredictor(lookback=30)
            tf_forecast, tf_metrics = transformer.predict(
                train_series, forecast_horizon, epochs=epochs
            )
            band_results["Transformer"] = {
                "forecast": tf_forecast, "metrics": tf_metrics, "true": true_values,
            }
        except Exception as e:
            band_results["Transformer"] = {
                "forecast": np.zeros(forecast_horizon),
                "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": str(e)},
                "true": true_values,
            }
    else:
        # Skip deep models — return NaN placeholders
        skip_err = f"Series too short for deep models ({len(train_series)} < {_cfg.MIN_WINDOWS_DEEP_MODEL})"
        band_results["LSTM"] = {
            "forecast": np.zeros(forecast_horizon),
            "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": skip_err},
            "true": true_values,
        }
        band_results["Transformer"] = {
            "forecast": np.zeros(forecast_horizon),
            "metrics": {"RMSE": np.nan, "MAE": np.nan, "MSE": np.nan, "error": skip_err},
            "true": true_values,
        }

    return {"info": band_info, "predictions": band_results}


def analyze_band_predictability(mel_spec, forecast_horizon=20, parallel=False, epochs=None):
    """
    Analyze predictability across different frequency bands.

    Parameters
    ----------
    mel_spec : ndarray       Mel spectrogram (n_mels, n_frames).
    forecast_horizon : int   Forecast steps.
    parallel : bool          Use multiprocessing.  **Off by default** because
                              multiprocessing + CUDA causes hangs on Windows.
    epochs : int or None     Epochs for deep models.  Falls back to config default.
    """
    if epochs is None:
        epochs = _cfg.DEFAULT_BAND_EPOCHS

    n_mels = mel_spec.shape[0]
    third = n_mels // 3

    bands = {
        "low":  {"data": mel_spec[:third, :],          "name": "Low Band",  "range": f"0-{third}"},
        "mid":  {"data": mel_spec[third:2 * third, :], "name": "Mid Band",  "range": f"{third}-{2 * third}"},
        "high": {"data": mel_spec[2 * third:, :],      "name": "High Band", "range": f"{2 * third}-{n_mels}"},
    }

    if parallel and len(bands) > 1:
        num_workers = min(len(bands), mp.cpu_count() - 1 or 1)
        print(f"  [Parallel] Using {num_workers} workers for band analysis...")

        with mp.Pool(processes=num_workers) as pool:
            analyze_func = partial(
                analyze_single_band, forecast_horizon=forecast_horizon, epochs=epochs
            )
            band_items = list(bands.items())
            band_data_list = [item[1]["data"] for item in band_items]
            band_info_list = [item[1] for item in band_items]
            results_list = pool.starmap(analyze_func, zip(band_data_list, band_info_list))
            results = {}
            for i, (band_key, _) in enumerate(band_items):
                results[band_key] = results_list[i]
    else:
        results = {}
        for band_key, band_info in bands.items():
            print(f"  Processing {band_info['name']}...")
            results[band_key] = analyze_single_band(
                band_info["data"], band_info,
                forecast_horizon=forecast_horizon, epochs=epochs,
            )

    return results


def compute_band_error_summary(band_results):
    """Compute summary statistics of prediction errors across bands."""
    summary = {}

    for band_key, band_data in band_results.items():
        rmse_values = []
        mae_values = []

        for model_name, model_data in band_data['predictions'].items():
            metrics = model_data['metrics']
            if 'RMSE' in metrics and not np.isnan(metrics['RMSE']):
                rmse_values.append(metrics['RMSE'])
                mae_values.append(metrics.get('MAE', np.nan))

        summary[band_key] = {
            'name': band_data['info']['name'],
            'avg_rmse': np.mean(rmse_values) if rmse_values else np.nan,
            'avg_mae': np.mean(mae_values) if mae_values else np.nan,
            'min_rmse': np.min(rmse_values) if rmse_values else np.nan,
            'best_model': None
        }

        if rmse_values:
            models = list(band_data['predictions'].keys())
            valid_models = [m for m in models if 'RMSE' in band_data['predictions'][m]['metrics'] and not np.isnan(band_data['predictions'][m]['metrics']['RMSE'])]
            if valid_models:
                best_model = min(valid_models, key=lambda m: band_data['predictions'][m]['metrics']['RMSE'])
                summary[band_key]['best_model'] = best_model

    return summary


def get_predictability_rank(summary):
    """Rank bands by predictability (lower error = more predictable)."""
    bands_with_error = [(k, v['avg_rmse']) for k, v in summary.items() if not np.isnan(v['avg_rmse'])]
    sorted_bands = sorted(bands_with_error, key=lambda x: x[1])

    ranks = []
    for i, (band_key, error) in enumerate(sorted_bands):
        ranks.append({
            'rank': i + 1,
            'band': summary[band_key]['name'],
            'avg_rmse': error,
            'best_model': summary[band_key]['best_model']
        })

    return ranks


def print_band_summary(summary):
    """Print a formatted summary of band prediction results."""
    print("\n[Band Analysis Summary]")
    print("-" * 50)
    print(f"  {'Band':<12} {'Avg RMSE':<12} {'Avg MAE':<12} {'Best Model':<12}")
    print("-" * 50)
    
    for band_key, data in summary.items():
        rmse_str = f"{data['avg_rmse']:.4f}" if not np.isnan(data['avg_rmse']) else "N/A"
        mae_str = f"{data['avg_mae']:.4f}" if not np.isnan(data['avg_mae']) else "N/A"
        best_model = data['best_model'] if data['best_model'] else "N/A"
        print(f"  {data['name']:<12} {rmse_str:<12} {mae_str:<12} {best_model:<12}")
    
    print("-" * 50)
    
    ranks = get_predictability_rank(summary)
    if ranks:
        print("\n[Predictability Ranking]")
        for rank in ranks:
            print(f"  #{rank['rank']} {rank['band']} (RMSE: {rank['avg_rmse']:.4f})")


def print_detailed_band_results(band_results):
    """Print detailed results for each model in each band."""
    print("\n[Detailed Band Analysis Results]")
    print("=" * 70)
    
    for band_key, band_data in band_results.items():
        band_name = band_data['info']['name']
        print(f"\n  {band_name} (Mel bins: {band_data['info']['range']})")
        print("  " + "-" * 50)
        print(f"  {'Model':<12} {'RMSE':<12} {'MAE':<12} {'MSE':<12}")
        print("  " + "-" * 50)
        
        for model_name, model_data in band_data['predictions'].items():
            metrics = model_data['metrics']
            rmse = f"{metrics.get('RMSE', np.nan):.4f}" if 'RMSE' in metrics and not np.isnan(metrics['RMSE']) else "N/A"
            mae = f"{metrics.get('MAE', np.nan):.4f}" if 'MAE' in metrics and not np.isnan(metrics['MAE']) else "N/A"
            mse = f"{metrics.get('MSE', np.nan):.4f}" if 'MSE' in metrics and not np.isnan(metrics['MSE']) else "N/A"
            print(f"  {model_name:<12} {rmse:<12} {mae:<12} {mse:<12}")
        
        if 'error' in metrics:
            print(f"  Error: {metrics['error']}")
    
    print("\n" + "=" * 70)