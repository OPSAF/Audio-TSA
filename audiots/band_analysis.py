"""Frequency band analysis module."""

import numpy as np


def analyze_band_predictability(mel_spec, forecast_horizon=20):
    """
    Analyze predictability across different frequency bands.
    Returns dict with band-specific prediction results.
    """
    n_mels = mel_spec.shape[0]
    third = n_mels // 3

    bands = {
        'low': {'data': mel_spec[:third, :], 'name': '低频带', 'range': f'0-{third}'},
        'mid': {'data': mel_spec[third:2 * third, :], 'name': '中频带', 'range': f'{third}-{2 * third}'},
        'high': {'data': mel_spec[2 * third:, :], 'name': '高频带', 'range': f'{2 * third}-{n_mels}'}
    }

    results = {}

    for band_key, band_info in bands.items():
        band_data = band_info['data']
        band_mean = np.mean(band_data, axis=0)

        train_size = int(len(band_mean) * 0.8)
        train_series = band_mean[:train_size]
        true_values = band_mean[train_size:train_size + forecast_horizon]

        from .prediction import predict_arima, predict_hmm, LSTMPredictor, TransformerPredictor, compute_metrics

        band_results = {}

        arima_forecast, arima_metrics = predict_arima(train_series, forecast_horizon)
        band_results['ARIMA'] = {
            'forecast': arima_forecast,
            'metrics': arima_metrics,
            'true': true_values
        }

        hmm_forecast, hmm_metrics = predict_hmm(train_series, forecast_horizon)
        band_results['HMM'] = {
            'forecast': hmm_forecast,
            'metrics': hmm_metrics,
            'true': true_values
        }

        lstm = LSTMPredictor(lookback=30)
        lstm_forecast, lstm_metrics = lstm.predict(train_series, forecast_horizon, epochs=40)
        band_results['LSTM'] = {
            'forecast': lstm_forecast,
            'metrics': lstm_metrics,
            'true': true_values
        }

        transformer = TransformerPredictor(lookback=30)
        tf_forecast, tf_metrics = transformer.predict(train_series, forecast_horizon, epochs=40)
        band_results['Transformer'] = {
            'forecast': tf_forecast,
            'metrics': tf_metrics,
            'true': true_values
        }

        results[band_key] = {
            'info': band_info,
            'predictions': band_results
        }

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