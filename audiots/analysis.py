"""Time series analysis module."""

import numpy as np
from scipy import signal
from scipy.stats import chi2, norm, jarque_bera


def compute_acf(x, nlags=40):
    """Compute autocorrelation function (ACF)."""
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)

    lags = np.arange(nlags)
    acf_vals = np.zeros(nlags)
    for lag in lags:
        if lag == 0:
            acf_vals[lag] = 1.0
        else:
            acf_vals[lag] = np.corrcoef(x_centered[lag:], x_centered[:-lag])[0, 1] if var > 0 else 0

    ci = 1.96 / np.sqrt(n)
    return lags, acf_vals, ci


def compute_pacf(x, nlags=40):
    """Compute partial autocorrelation function (PACF) using Yule-Walker."""
    n = len(x)
    lags = np.arange(nlags)
    pacf_vals = np.zeros(nlags)
    acf_raw = np.zeros(nlags + 1)

    x_centered = x - np.mean(x)
    var = np.var(x)
    for lag in range(nlags + 1):
        if lag == 0:
            acf_raw[0] = 1.0
        else:
            acf_raw[lag] = np.corrcoef(x_centered[lag:], x_centered[:-lag])[0, 1] if var > 0 else 0

    pacf_vals[0] = 1.0
    phi = np.zeros((nlags, nlags))
    phi[0, 0] = acf_raw[1]
    pacf_vals[1] = phi[0, 0]

    for k in range(2, nlags):
        numerator = acf_raw[k]
        for j in range(1, k):
            numerator -= phi[k - 2, j - 1] * acf_raw[k - j]
        denominator = 1.0
        for j in range(1, k):
            denominator -= phi[k - 2, j - 1] * acf_raw[j]
        if abs(denominator) < 1e-10:
            phi[k - 1, k - 1] = 0
        else:
            phi[k - 1, k - 1] = numerator / denominator

        for j in range(1, k):
            phi[k - 1, j - 1] = phi[k - 2, j - 1] - phi[k - 1, k - 1] * phi[k - 2, k - j - 1]

        pacf_vals[k] = phi[k - 1, k - 1]

    ci = 1.96 / np.sqrt(n)
    return lags, pacf_vals, ci


def fft_analysis(y, sr):
    """Perform FFT analysis on waveform."""
    from scipy.fft import fft, fftfreq
    n = len(y)
    Y = fft(y)
    mag = np.abs(Y[:n // 2]) / n
    phase = np.angle(Y[:n // 2])
    freqs = fftfreq(n, 1 / sr)[:n // 2]
    return freqs, mag, phase


def stft_analysis(y, sr, n_fft=2048, hop_length=512):
    """Perform STFT analysis."""
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
                             nfft=n_fft, window='hann')
    return f, t, np.abs(Zxx)


def analyze_periodicity(y, sr, nlags=40):
    """Analyze periodicity of a time series."""
    freqs, mag, _ = fft_analysis(y, sr)
    dominant_idx = np.argmax(mag[1:]) + 1
    dominant_freq = freqs[dominant_idx]
    period = 1.0 / dominant_freq if dominant_freq > 0 else np.inf

    lags, acf_vals, ci = compute_acf(y[:min(len(y), sr * 2)], nlags)

    peaks = []
    for i in range(2, len(acf_vals) - 1):
        if acf_vals[i] > ci and acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1]:
            peaks.append((i, acf_vals[i]))

    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:3]

    return {
        'dominant_frequency': dominant_freq,
        'dominant_period_seconds': period,
        'dominant_period_samples': int(period * sr) if period < np.inf else -1,
        'acf_peaks': peaks,
        'confidence_interval': ci
    }


def compute_spectral_flatness(mag_spec):
    """Compute spectral flatness (Wiener entropy)."""
    if np.max(mag_spec) < 1e-10:
        return 1.0
    geometric_mean = np.exp(np.mean(np.log(mag_spec + 1e-10)))
    arithmetic_mean = np.mean(mag_spec)
    return geometric_mean / (arithmetic_mean + 1e-10)


def analyze_complexity(y):
    """Analyze signal complexity using sample entropy approximation."""
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)

    m = 2
    r_scale = 0.2 * np.std(y)

    def _count_matches(data, m_val, r_val):
        count = 0
        n = len(data)
        for i in range(n - m_val):
            template = data[i:i + m_val]
            for j in range(i + 1, n - m_val):
                if np.max(np.abs(template - data[j:j + m_val])) <= r_val:
                    count += 1
        return count

    n = min(len(y), 500)
    downsampled = y[::max(1, len(y) // n)][:n]
    downsampled = (downsampled - np.mean(downsampled)) / (np.std(downsampled) + 1e-10)

    a = _count_matches(downsampled, m + 1, r_scale)
    b = _count_matches(downsampled, m, r_scale)
    sampen = -np.log((a + 1) / (b + 1)) if b > 0 else float('inf')

    return {
        'zero_crossing_rate': zero_crossings,
        'sample_entropy': sampen,
        'is_tonal': sampen < 0.5
    }


# ============================================================
# White Noise Testing Functions
# ============================================================

def ljung_box_test(x, lags=20):
    """
    Perform Ljung-Box test for white noise.
    
    The Ljung-Box test checks if the first 'lags' autocorrelation coefficients
    are all zero (white noise hypothesis).
    
    Returns:
        dict: Test results with statistic and p-value
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    
    if var < 1e-10:
        return {'statistic': 0.0, 'p_value': 1.0, 'lags': lags, 'is_white_noise': True}
    
    # Compute ACF values
    acf_vals = np.zeros(lags)
    for k in range(1, lags + 1):
        if k < n:
            acf_vals[k - 1] = np.corrcoef(x_centered[k:], x_centered[:-k])[0, 1]
        else:
            acf_vals[k - 1] = 0
    
    # Ljung-Box statistic
    q_stat = n * (n + 2) * np.sum([acf_vals[k] ** 2 / (n - k - 1) for k in range(lags)])
    
    # Degrees of freedom
    df = lags
    
    # P-value from chi-square distribution
    p_value = 1 - chi2.cdf(q_stat, df)
    
    return {
        'statistic': float(q_stat),
        'p_value': float(p_value),
        'lags': lags,
        'df': df,
        'is_white_noise': p_value > 0.05,
        'interpretation': 'Fail to reject H0: Data appears to be white noise' if p_value > 0.05 else 'Reject H0: Data is not white noise'
    }


def box_pierce_test(x, lags=20):
    """
    Perform Box-Pierce test for white noise.
    
    Simpler version of Ljung-Box test.
    
    Returns:
        dict: Test results with statistic and p-value
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    
    if var < 1e-10:
        return {'statistic': 0.0, 'p_value': 1.0, 'lags': lags, 'is_white_noise': True}
    
    # Compute ACF values
    acf_vals = np.zeros(lags)
    for k in range(1, lags + 1):
        if k < n:
            acf_vals[k - 1] = np.corrcoef(x_centered[k:], x_centered[:-k])[0, 1]
        else:
            acf_vals[k - 1] = 0
    
    # Box-Pierce statistic
    q_stat = n * np.sum(acf_vals ** 2)
    
    # Degrees of freedom
    df = lags
    
    # P-value from chi-square distribution
    p_value = 1 - chi2.cdf(q_stat, df)
    
    return {
        'statistic': float(q_stat),
        'p_value': float(p_value),
        'lags': lags,
        'df': df,
        'is_white_noise': p_value > 0.05,
        'interpretation': 'Fail to reject H0: Data appears to be white noise' if p_value > 0.05 else 'Reject H0: Data is not white noise'
    }


def jarque_bera_test(x):
    """
    Perform Jarque-Bera test for normality.
    
    White noise is typically assumed to be normally distributed.
    
    Returns:
        dict: Test results with statistic and p-value
    """
    try:
        jb_stat, p_value = jarque_bera(x)
    except Exception:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False, 'interpretation': 'Test failed'}
    
    return {
        'statistic': float(jb_stat),
        'p_value': float(p_value),
        'is_normal': p_value > 0.05,
        'interpretation': 'Fail to reject H0: Data appears to be normally distributed' if p_value > 0.05 else 'Reject H0: Data is not normally distributed'
    }


def acf_white_noise_test(x, nlags=40):
    """
    Test for white noise based on ACF values.
    
    In white noise, approximately 95% of ACF values should fall within 
    the confidence interval ±1.96/sqrt(n).
    
    Returns:
        dict: Test results with proportion of ACF values outside confidence interval
    """
    n = len(x)
    lags, acf_vals, ci = compute_acf(x, nlags)
    
    # Count ACF values outside confidence interval (excluding lag 0)
    outside_count = np.sum(np.abs(acf_vals[1:]) > ci)
    outside_ratio = outside_count / (nlags - 1)
    
    # Expected proportion for white noise is ~5%
    is_white_noise = outside_ratio < 0.10  # Allow up to 10% outside
    
    return {
        'outside_count': int(outside_count),
        'outside_ratio': float(outside_ratio),
        'expected_ratio': 0.05,
        'confidence_interval': float(ci),
        'nlags': nlags,
        'is_white_noise': bool(is_white_noise),
        'interpretation': f'{outside_count}/{nlags-1} ACF values outside 95% CI ({(outside_ratio*100):.1f}%)'
    }


def variance_stationarity_test(x, window_size=100):
    """
    Test for variance stationarity by comparing variance in different windows.
    
    White noise should have constant variance over time.
    
    Returns:
        dict: Test results with variance statistics
    """
    n = len(x)
    if n < 2 * window_size:
        return {'variance_ratio': np.nan, 'is_stationary': True, 'interpretation': 'Insufficient data'}
    
    # Split data into windows
    num_windows = n // window_size
    variances = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_var = np.var(x[start:end])
        variances.append(window_var)
    
    variances = np.array(variances)
    max_var = np.max(variances)
    min_var = np.min(variances)
    
    # Ratio of max to min variance
    variance_ratio = max_var / (min_var + 1e-10)
    
    # If variance ratio is too high, data may not be stationary
    is_stationary = variance_ratio < 10.0
    
    return {
        'variance_ratio': float(variance_ratio),
        'max_variance': float(max_var),
        'min_variance': float(min_var),
        'num_windows': int(num_windows),
        'window_size': int(window_size),
        'is_stationary': bool(is_stationary),
        'interpretation': f'Variance ratio: {variance_ratio:.2f} (threshold: 10.0)'
    }


def run_test(x):
    """
    Perform runs test for randomness.
    
    Tests whether the sequence of positive and negative deviations from mean
    follows a random pattern.
    
    Returns:
        dict: Test results with statistic and p-value
    """
    n = len(x)
    if n < 10:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_random': True, 'interpretation': 'Insufficient data'}
    
    mean_val = np.mean(x)
    deviations = x - mean_val
    
    # Create sequence of signs
    signs = np.where(deviations > 0, 1, -1)
    signs[deviations == 0] = 1  # Handle zeros
    
    # Count runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1
    
    # Expected number of runs
    n_pos = np.sum(signs > 0)
    n_neg = np.sum(signs < 0)
    
    if n_pos == 0 or n_neg == 0:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_random': True, 'interpretation': 'All values equal'}
    
    expected_runs = (2 * n_pos * n_neg) / n + 1
    variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))
    
    if variance_runs < 1e-10:
        return {'statistic': np.nan, 'p_value': np.nan, 'is_random': True, 'interpretation': 'Cannot compute variance'}
    
    # Z-score
    z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
    
    return {
        'statistic': float(z_stat),
        'p_value': float(p_value),
        'runs': int(runs),
        'expected_runs': float(expected_runs),
        'is_random': p_value > 0.05,
        'interpretation': 'Fail to reject H0: Sequence appears random' if p_value > 0.05 else 'Reject H0: Sequence is not random'
    }


def test_white_noise(x, lags=20):
    """
    Comprehensive white noise testing suite.
    
    Combines multiple tests to determine if a time series is white noise.
    
    Returns:
        dict: Combined test results with overall assessment
    """
    results = {
        'ljung_box': ljung_box_test(x, lags),
        'box_pierce': box_pierce_test(x, lags),
        'jarque_bera': jarque_bera_test(x),
        'acf_test': acf_white_noise_test(x, lags),
        'variance_stationarity': variance_stationarity_test(x),
        'runs_test': run_test(x)
    }
    
    # Overall assessment based on majority vote
    tests_passed = [
        results['ljung_box']['is_white_noise'],
        results['box_pierce']['is_white_noise'],
        results['acf_test']['is_white_noise'],
        results['variance_stationarity']['is_stationary'],
        results['runs_test']['is_random']
    ]
    
    # For normality, we consider it separately as white noise doesn't strictly require normality
    normality_ok = results['jarque_bera']['is_normal'] or np.isnan(results['jarque_bera']['p_value'])
    
    # Majority vote
    passed_count = sum(tests_passed)
    overall_assessment = passed_count >= 3  # At least 3 out of 5 tests pass
    
    results['overall'] = {
        'is_white_noise': overall_assessment,
        'tests_passed': passed_count,
        'total_tests': len(tests_passed),
        'normality_ok': normality_ok,
        'assessment': 'The data appears to be white noise' if overall_assessment else 'The data is NOT white noise'
    }
    
    return results


def print_white_noise_report(results):
    """Print a comprehensive white noise test report."""
    print("\n" + "=" * 70)
    print("  WHITE NOISE TEST REPORT")
    print("=" * 70)
    
    print("\n[Ljung-Box Test]")
    lb = results['ljung_box']
    print(f"  Statistic: {lb['statistic']:.4f}, p-value: {lb['p_value']:.4f}")
    print(f"  {lb['interpretation']}")
    
    print("\n[Box-Pierce Test]")
    bp = results['box_pierce']
    print(f"  Statistic: {bp['statistic']:.4f}, p-value: {bp['p_value']:.4f}")
    print(f"  {bp['interpretation']}")
    
    print("\n[Jarque-Bera Normality Test]")
    jb = results['jarque_bera']
    print(f"  Statistic: {jb['statistic']:.4f}, p-value: {jb['p_value']:.4f}")
    print(f"  {jb['interpretation']}")
    
    print("\n[ACF Test]")
    acf = results['acf_test']
    print(f"  {acf['interpretation']}")
    print(f"  Confidence interval: ±{acf['confidence_interval']:.4f}")
    
    print("\n[Variance Stationarity Test]")
    vs = results['variance_stationarity']
    print(f"  {vs['interpretation']}")
    print(f"  Max variance: {vs['max_variance']:.6f}, Min variance: {vs['min_variance']:.6f}")
    
    print("\n[Runs Test]")
    rt = results['runs_test']
    print(f"  Z-statistic: {rt['statistic']:.4f}, p-value: {rt['p_value']:.4f}")
    print(f"  Runs: {rt['runs']}, Expected: {rt['expected_runs']:.2f}")
    print(f"  {rt['interpretation']}")
    
    print("\n" + "-" * 70)
    overall = results['overall']
    status = "✓" if overall['is_white_noise'] else "✗"
    print(f"  {status} OVERALL ASSESSMENT: {overall['assessment']}")
    print(f"  Tests passed: {overall['tests_passed']}/{overall['total_tests']}")
    print(f"  Normality assumption: {'✓' if overall['normality_ok'] else '✗'}")
    print("-" * 70)