"""
简化测试脚本 - 验证项目基本功能
"""
import os
os.environ['NUMBA_DISABLE_CACHE'] = '1'

import sys
sys.path.insert(0, '.')

print("=" * 50)
print("测试音频时间序列分析项目")
print("=" * 50)

# 测试导入
print("\n[1/5] 测试模块导入...")
try:
    from audio_loader import generate_sample_audio
    print("      audio_loader: OK")
except Exception as e:
    print(f"      audio_loader: FAILED - {e}")

try:
    from feature_extraction import compute_mel_spectrogram, compute_mfcc
    print("      feature_extraction: OK")
except Exception as e:
    print(f"      feature_extraction: FAILED - {e}")

try:
    from time_series_analysis import compute_acf, analyze_periodicity
    print("      time_series_analysis: OK")
except Exception as e:
    print(f"      time_series_analysis: FAILED - {e}")

try:
    from prediction_models import run_all_predictions
    print("      prediction_models: OK")
except Exception as e:
    print(f"      prediction_models: FAILED - {e}")

try:
    from visualization import plot_waveform, plot_mel_spectrogram
    print("      visualization: OK")
except Exception as e:
    print(f"      visualization: FAILED - {e}")

# 测试音频生成
print("\n[2/5] 测试音频生成...")
try:
    y, sr = generate_sample_audio(duration=1.0)
    print(f"      音频生成成功: {len(y)} samples, {sr} Hz")
except Exception as e:
    print(f"      音频生成失败: {e}")

# 测试特征提取
print("\n[3/5] 测试特征提取...")
try:
    mel_freqs, mel_times, mel_spec = compute_mel_spectrogram(y, sr, n_mels=64)
    print(f"      Mel频谱: {mel_spec.shape}")
    
    mfcc, mfcc_times = compute_mfcc(y, sr, n_mfcc=13)
    print(f"      MFCC: {mfcc.shape}")
except Exception as e:
    print(f"      特征提取失败: {e}")

# 测试时间序列分析
print("\n[4/5] 测试时间序列分析...")
try:
    lags, acf_vals, ci = compute_acf(y[:sr], nlags=20)
    print(f"      ACF计算: {len(acf_vals)} 个滞后值")
    
    period_info = analyze_periodicity(y, sr)
    print(f"      周期分析: 主频 {period_info['dominant_frequency']:.1f} Hz")
except Exception as e:
    print(f"      时间序列分析失败: {e}")

# 测试可视化
print("\n[5/5] 测试可视化...")
import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
    import numpy as np
    t = np.linspace(0, len(y)/sr, len(y))
    fig = plt.figure(figsize=(8, 2))
    plt.plot(t, y)
    plt.title("Test Waveform")
    plt.savefig("test_waveform.png")
    plt.close()
    print("      波形图保存成功")
except Exception as e:
    print(f"      可视化失败: {e}")

print("\n" + "=" * 50)
print("测试完成!")
print("=" * 50)
