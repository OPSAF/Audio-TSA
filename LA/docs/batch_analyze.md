# 批量分析工具 (Batch Analyzer)

> 对**同流派多首音频**进行综合统计分析，自动挖掘流派共性，训练全局机器学习模型。

---

## 快速开始

```bash
# 基础用法：分析一个文件夹内所有音频
python batch_analyze.py data/rock/ -o results/rock/

# 快速模式（减少深度学习 epoch，适合测试）
python batch_analyze.py data/rock/ -o results/rock/ --fast

# 限制文件数量
python batch_analyze.py data/rock/ -o results/rock/ --max-files 20

# 跳过全局机器学习（只看统计分析）
python batch_analyze.py data/rock/ -o results/rock/ --no-global-ml

# 完整训练（30 epochs）
python batch_analyze.py data/rock/ -o results/rock/ --ml-epochs 30 --hmm-states 5
```

---

## 分析架构

```
Phase A: 单曲分析（复用共享 Pipeline）
  └── 每首歌曲 → 16 张详细图表（波形、FFT、STFT、Mel、MFCC、
      ACF/PACF、周期性、动态趋势、波动率/GARCH、预测对比、频带分析）
      → per_file/{song_name}/ 目录

Phase B: 跨曲共性分析
  └── 特征矩阵构建（N首歌 × D个特征维度）
  └── CV 变异系数排名 → 流派共性 vs 个体差异
  └── 分布统计（ARIMA类型、HMM状态数、最佳频带）
  └── 离群检测（|z| > 2.0）
      → commonality_report.json

Phase C: 全局机器学习训练
  └── Mel Spectrogram 滑动窗口 → 所有歌曲窗口汇总
  └── Global LSTM → 共享参数，学习流派时频规律
  └── Global Transformer → 共享参数
  └── ARIMA 每首独立建模（Local Baseline）
  └── Joint HMM → 所有歌曲特征共同训练
      → global_ml_report.json

Phase D: 批量可视化
  └── 10 张大尺寸跨曲对比仪表盘
      → figures/ 目录

Phase E: 综合输出
  └── aggregate_summary.json + batch_report.csv
```

---

## 核心设计理念

### 1. "同窗口"并排比较

不做两两配对的相似度计算，而是把**所有歌曲的同一指标放在同一个窗口里并排看**：

```
        Energy趋势   GARCH持久性   HMM状态   最佳频带   ...
Song 1  [0.5, ...]   0.85          3状态      低频       ...
Song 2  [0.6, ...]   0.92          3状态      低频       ...
Song 3  [0.4, ...]   0.78          2状态      中频       ...
...

        → 统计分析 → 共性发现 → 置信度评估
```

### 2. Global Model vs Local Model

| 模型 | 训练方式 | 学什么 |
|------|---------|--------|
| **Global LSTM** | 所有歌曲的 Mel 窗口汇总 → 一个共享模型 | 流派共享的时频变化规律 |
| **Global Transformer** | 同上 | 流派级注意力模式 |
| **Joint HMM** | 所有歌曲的 Mel 特征共同训练 | 跨曲共享的隐藏音乐状态 |
| **Local ARIMA** | 每首歌曲单独建模 | 单曲基线，用于对比 |

> ARIMA 作为 Local Model 基线：如果 Global Model 的 RMSE 接近或优于 ARIMA，
> 说明流派确实存在可学习的共享模式。

### 3. Mel Spectrogram 滑动窗口

```
每首歌曲 Mel Spectrogram (n_mels × n_frames)
         ↓  独立滑动窗口（不拼接歌曲）
  过去 N 帧 → 预测下一帧（全 Mel 频带）
         ↓
  所有歌曲的窗口汇总到同一个训练集
         ↓
  Global LSTM / Global Transformer（共享参数）
```

- 不拼接成一条时间序列：每首歌有独立结构，拼接会引入虚假依赖
- 保留全 Mel 频带：让模型学习频带间的相互关系
- ARIMA 用 1D（Mel 均值）作为 Local 基线

---

## 输出文件

### Per-file 详细图表（每首 16 张）

与 Web/CLI 单曲分析完全一致的输出：

| 图表 | 说明 |
|------|------|
| `waveform.png` | 音频波形 |
| `fft_spectrum.png` | FFT 频谱 |
| `stft_spectrogram.png` | STFT 时频谱 |
| `mel_spectrogram.png` | Mel 频谱图 |
| `mfcc_heatmap.png` | MFCC 系数 |
| `acf_pacf.png` | 自相关 / 偏自相关 |
| `periodicity.png` | 周期性分析 |
| `prediction_comparison.png` | 4 模型预测对比 |
| `error_comparison.png` | 模型误差对比 |
| `band_errors.png` / `band_error_bars.png` | 频带可预测性 |
| `dynamics/trends.png` | 动态趋势（能量/亮度/复杂度/节奏） |
| `dynamics/summary.png` | 动态趋势摘要 |
| `volatility/volatility_layer.png` | 滚动波动率层 |
| `volatility/garch_energy.png` | GARCH(1,1) 诊断 |
| `volatility/dynamics_analysis_summary.png` | 趋势+波动率综合摘要 |

### 批量仪表盘（10 张）

| # | 仪表盘 | 回答的音乐问题 |
|---|--------|---------------|
| **B01** | Spectral Dashboard | 流派的频谱指纹长什么样？不同频段能量有多一致？ |
| **B02** | Dynamics & Volatility | 四维动态趋势和波动率在歌曲间如何变化？GARCH 揭示了什么？ |
| **B03** | Model Ensemble | ARIMA/HMM/LSTM/Transformer 结构分析揭示了哪些共同模式？ |
| **B04** | Global ML Performance | 全局模型 vs 局部 ARIMA：流派级学习效果如何？ |
| **B05** | Statistical Summary | 哪些特征定义了流派共性？哪些是个体差异？（箱线图+分布） |
| **B06** | Report Card | 一页大白话总结所有关键发现 |
| **B07** | Genre Structure Decoder | ARIMA 趋势类型矩阵 + GARCH 持久性热力图 + 结构段组成 + 趋势方向 |
| **B08** | Audio Signal Deep Dive | 白噪声 6 项检验 + 周期性对比 + 复杂度空间 + 无监督模式指标 |
| **B09** | Unsupervised Discovery | PCA 投影 + 层次聚类 + NMF 共享频谱分量 — 自动发现潜在共性 |
| **B10** | Statistical Confidence | Bootstrap 95% CI + Cohen's d + 排列检验 p 值 — 共性有多可信？ |

---

## B09：无监督共性发现

不依赖人工指定特征，使用无监督算法**自动发现**歌曲间的潜在共性：

| 算法 | 作用 |
|------|------|
| **PCA 降维投影** | 将歌曲的特征向量投影到 2D，带 95% 置信椭圆。距离越近越相似 |
| **层次聚类** | Ward 链接 + 欧氏距离，树状图直观显示相似度层级 |
| **NMF 频谱分解** | 对所有歌曲的平均 Mel 频谱进行非负矩阵分解，找到共享频谱分量 |

---

## B10：统计学置信度

为每个共性发现提供严格的统计学支撑：

| 指标 | 含义 | 解读 |
|------|------|------|
| **Bootstrap 95% CI** | 特征均值的置信区间 | 区间越窄，估计越精确 |
| **Cohen's d** | 效应量 | \|d\| > 0.8：强烈流派特征；< 0.2：可忽略 |
| **排列检验 p 值** | 聚类是否显著于随机？ | p < 0.05：统计显著 |
| **共识评分** | 综合 CI + d + CV | > 0.7：高共识；< 0.4：低共识 |

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | (必需) | 输入音频文件夹 |
| `-o, --output` | `{input}_results` | 输出文件夹 |
| `--max-files` | 0 (不限制) | 最大处理文件数 |
| `--fast` | False | 减少深度学习 epoch |
| `--no-resume` | False | 不跳过已分析的文件 |
| `--no-global-ml` | False | 跳过 Phase C（全局 ML 训练） |
| `--lookback` | 30 | 滑动窗口大小（帧） |
| `--ml-epochs` | 30 | Global LSTM/Transformer 训练轮数 |
| `--hmm-states` | 5 | Joint HMM 隐藏状态数 |
| `--forecast-horizon` | 100 | 预测视野 |
| `--n-mels` | 128 | Mel 频带数 |
| `--sample-rate` | 16000 | 目标采样率 |

---

## 与 app/main 的关系

| 功能 | app.py | main.py | batch_analyze.py |
|------|--------|---------|------------------|
| 单曲详细分析 | ✅ | ✅ | ✅（Phase A，复用相同 Pipeline） |
| 双曲对比 | ✅ | ✅ | ❌（改为跨曲共性挖掘） |
| 跨曲统计分析 | ❌ | ❌ | ✅ |
| 无监督共性发现 | ❌ | ❌ | ✅（PCA + 聚类 + NMF） |
| 统计学置信度 | ❌ | ❌ | ✅（Bootstrap + Cohen's d + 排列检验） |
| Global ML 训练 | ❌ | ❌ | ✅（多曲共享模型参数） |
| Joint HMM | ❌ | ❌ | ✅（所有歌曲共同训练） |
| 趋势分析 (Trend) | 单曲 | 单曲 | 单曲 + 跨曲矩阵对比 |
| GARCH 波动率 | 单曲 | 单曲 | 单曲 + 跨曲持久性热力图 |
| 白噪声检验 | 单曲 | 单曲 | 单曲 + 跨曲 6 检验矩阵 |
| 结构段分析 | 单曲 | 单曲 | 单曲 + 跨曲段组成堆叠图 |

---

## 使用建议

1. **快速探索**：`--fast --no-global-ml` → 快速获得统计分析和共性报告
2. **深入分析**：默认参数 → 完整的 Global ML 对比
3. **大批量**：`--fast --max-files 50` → 50 首歌快速概览流派特征
4. **精细调参**：`--ml-epochs 50 --lookback 40 --hmm-states 7`

### 推荐阅读顺序

1. **B06 Report Card** → 一页了解全貌
2. **B09 Unsupervised Discovery** → 自动发现的结构
3. **B10 Statistical Confidence** → 哪些发现可信
4. B01-B05, B07-B08 → 按兴趣深入各维度
