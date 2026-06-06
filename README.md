# Audio Lab — 音频时间序列分析系统

面向音频信号的**多层时间序列分析平台**。从原始波形出发，逐层提取趋势（Trend）、波动率（Volatility）、隐状态（Hidden States）、记忆特征（Memory）和注意力模式（Attention），最终给出人类可读的结构化报告。

## 设计哲学

```
原始波形  ──→  特征层  ──→  趋势层  ──→  波动率层  ──→  模型结构层
(Waveform)   (Features)  (Trends)   (Volatility)   (Model Ensemble)
```

每一层建立在前一层之上，**不做重复计算**。四个预测模型（ARIMA / HMM / LSTM / Transformer）不再是"谁误差更小"的竞赛选手，而是四个**不同角度的结构侦探**，各自回答一个不同的问题：

| 模型 | 它回答的问题 | 提取的洞察 |
|------|-------------|-----------|
| **ARIMA** | 每个维度的变化有线性规律吗？ | 最优阶数 (p,d,q)、白噪声检验、趋势类型、周期性 |
| **HMM** | 音频可以分成几种内在状态？ | 无监督隐状态发现、状态画像、转移矩阵、段落结构 |
| **LSTM** | 预测这个音频需要多长的记忆？ | 最优 lookback、逐维可学性、记忆体制（短/中/长） |
| **Transformer** | 音频的哪些时间点之间存在自注意？ | 主导 lag、注意力头多样性、多尺度时间结构 |

## 功能模块

### 核心管线

| 模块 | 文件 | 功能 |
|------|------|------|
| **Loader** | `audiots/loader.py` | 音频加载（WAV/MP3/FLAC）、重采样、合成音频生成 |
| **Features** | `audiots/features.py` | 波形、FFT、STFT、Mel 频谱图、MFCC |
| **Dynamics** | `audiots/dynamics.py` | **Trend Layer** — 4 个动态趋势（energy, brightness, complexity, rhythm）+ 结构分段（climax/calm/buildup）+ 趋势相似度 |
| **Volatility** | `audiots/volatility.py` | **Volatility Layer** — 滚动波动率 + ARCH/GARCH(1,1) 条件异方差建模 + 波动率相似度 |
| **Analysis** | `audiots/analysis.py` | ACF、PACF、FFT、STFT、周期性检测、复杂度分析 |
| **Prediction** | `audiots/prediction.py` | ARIMA / HMM / LSTM / Transformer 预测引擎 + 趋势/波动率预测接口 |
| **Band Analysis** | `audiots/band_analysis.py` | 低频/中频/高频带可预测性并行分析 |
| **Unsupervised** | `audiots/unsupervised.py` | 变点检测、Motif 发现、NMF 频谱分解、RQA 递归量化、异常检测、聚类 |

### 对比与探索

| 模块 | 文件 | 功能 |
|------|------|------|
| **Model Analysis** | `audiots/model_analysis.py` | **模型结构分析** — 四个模型作为结构侦探，输出 `ModelEnsembleReport` |
| **Discovery** | `audiots/discovery.py` | 双音频探索 — 自发现、跨发现、对比分析、角色画像 |
| **Similarity** | `audiots/similarity.py` | 长度无关的音频相似度（窗口特征 + 分布匹配 + 局部相似矩阵） |
| **Visualization** | `audiots/visualization.py` | 全模块可视化 — 趋势、波动率、GARCH 诊断、模型分析仪表板 |
| **Similarity Viz** | `audiots/similarity_viz.py` | 相似度分析可视化 |
| **Discovery Viz** | `audiots/discovery_viz.py` | 探索分析可视化 |

### 数据流

```
loader.load_audio()
        │
        ▼
features.compute_*()           ← FFT, STFT, Mel, MFCC
        │
        ▼
dynamics.extract_dynamics()    ← 4 个趋势时间序列（只做一次滑动窗口 FFT）
        │
        ├──→ dynamics.detect_structural_segments()   ← 结构分段
        ├──→ volatility.compute_volatility_layer()   ← 滚动波动率 + GARCH
        │       │
        │       ├──→ volatility.compute_volatility_similarity()  ← 波动率相似度
        │       └──→ prediction.predict_all_trends/volatilities() ← 趋势/波动率预测
        │
        ├──→ model_analysis.analyze_model_ensemble()  ← ARIMA/HMM/LSTM/Transformer 结构侦探
        │       │
        │       ├──→ ARIMA: 线性结构 + 白噪声检验
        │       ├──→ HMM: 隐状态发现 + 转移模式
        │       ├──→ LSTM: 记忆分析 + 可学性排名
        │       └──→ Transformer: 注意力模式 + 时间尺度
        │
        ├──→ discovery.explore()          ← 双音频跨维度探索
        ├──→ similarity.analyze_similarity()  ← 全局相似度
        └──→ unsupervised.explore_unsupervised()  ← 无监督模式发现
```

## 安装

```bash
git clone https://github.com/OPSAF/Audio-TSA.git
cd Audio-TSA
pip install -r requirements.txt
```

### 依赖

| 包 | 用途 |
|----|------|
| `numpy`, `scipy` | 核心数值计算 |
| `librosa` | 音频处理与特征提取 |
| `matplotlib` | 可视化 |
| `scikit-learn` | 标准化、NMF、聚类 |
| `statsmodels` | ARIMA、ARCH/GARCH、统计检验 |
| `torch` | LSTM / Transformer 深度学习模型 |
| `hmmlearn` | HMM 隐马尔可夫模型 |
| `soundfile` | 音频文件读写 |
| `flask` | Web 界面 |

## 使用方法

### Web 界面（推荐）

```bash
python app.py
# 访问 http://localhost:5000
```

支持：
- 上传单/双音频文件
- 勾选分析模块
- 实时进度流（SSE）
- 交互式结果查看

### 终端

```bash
# 默认全分析（合成音频）
python main.py

# 分析单个音频
python main.py --audio1 song.wav

# 双音频对比
python main.py --audio1 a.wav --audio2 b.wav

# 指定分析模块
python main.py --audio1 audio.wav --analysis features,dynamics,model_analysis

# 不保存图片
python main.py --audio1 audio.wav --no-save
```

### 分析选项

| 选项 | 说明 |
|------|------|
| `features` | 特征提取（波形、FFT、STFT、Mel、MFCC） |
| `dynamics` | 动态趋势分析（能量、亮度、复杂度、节奏 + 结构分段） |
| `dynamics_analysis` | **音频动态分析**（Trend Layer + Volatility Layer + ARCH/GARCH + 趋势/波动率预测） |
| `model_analysis` | **模型结构分析**（ARIMA/HMM/LSTM/Transformer 四种结构侦探） |
| `timeseries` | 时序分析（ACF、PACF、周期性、复杂度） |
| `unsupervised` | 无监督模式发现（变点、motif、NMF、RQA、异常检测） |
| `prediction` | 机器学习预测（基于 Mel 频谱） |
| `band` | 频带可预测性分析（低/中/高频） |
| `comparison` | 双音频对比（探索 + 动态相似度 + 波动率相似度） |
| `visualization` | 生成所有分析图表 |

## 输出示例

### 控制台输出

```
======================================================================
  MODEL ENSEMBLE STRUCTURAL ANALYSIS
======================================================================

--- ARIMA: 线性结构分析 ---
  有线性结构的维度: energy, brightness, complexity, rhythm

  Trend               Order  Stationary  WhiteNoise             Type
  ------------------------------------------------------------------
  energy          (0, 0, 1)        True        True   moving-average
  brightness      (0, 0, 2)        True        True   moving-average
  complexity      (1, 0, 0)        True        True   mean-reverting
  rhythm          (0, 1, 1)       False        True      random-walk

  Energy: MA(1) 冲击驱动——短期扰动后缓慢恢复
  Complexity: AR(1) 均值回复——变化有惯性，检测到约 3 窗口的周期

--- HMM: 隐状态发现 ---
  HMM 发现 3 种内在状态: 安静段 (Calm), 活跃段 (Active), 高潮段 (Climax)
  转移模式: 状态0→状态1(75%) → 状态1→状态2(90%) → 状态2→状态0(40%)
  状态区分度: 82%（状态边界清晰）。结构呈循环性。

--- LSTM: 记忆与可学性分析 ---
  最优记忆长度: 12 窗口 (~3.0s)。中等记忆（1-3s）。
  各维度可学性: energy=0.012  brightness=0.045  rhythm=0.089  complexity=0.210
  最容易学习: energy，最难学习: complexity（接近随机）

--- Transformer: 注意力模式分析 ---
  Head 0: lag=2 (0.5s) — 局部相邻依赖
  Head 1: lag=16 (4.0s) — 长程结构层
  Head 2: lag=7 (1.8s) — 短程节奏层
  检测到 3 个时间尺度——多层次结构

--- 综合概述 ---
  ARIMA 发现 energy、brightness 维度具有线性结构；
  HMM 自动将音频划分为 3 种内在状态，状态区分度 82%；
  LSTM 表明最优记忆约 3.0s，energy 最可学；
  Transformer 检测到 3 个时间尺度的自相似结构。
```

### 可视化输出

```
outputs/
├── waveform.png                    # 波形
├── fft_spectrum.png                # FFT 频谱
├── mel_spectrogram.png             # Mel 频谱图
├── acf_pacf.png                    # ACF/PACF
├── prediction_comparison.png       # 预测对比
├── dynamics/
│   ├── trends.png                  # 4 趋势 + 结构分段
│   └── summary.png                 # 趋势统计仪表板
├── volatility/
│   ├── volatility_layer.png        # 波动率叠加带
│   ├── garch_energy.png           # GARCH 条件波动率诊断
│   ├── dynamics_analysis_summary.png  # Trend + Vol 综合仪表板
│   └── volatility_comparison.png  # 双音频波动率对比
├── discovery/
│   ├── self_discovery_a.png       # 自发现
│   ├── segment_mapping.png        # 跨音频段落映射
│   └── discovery_summary.png      # 探索仪表板
└── similarity/
    ├── similarity_matrix.png       # 局部相似矩阵
    └── similarity_summary.png      # 相似度仪表板
```

## 项目结构

```
Audio Lab/
├── audiots/                       # 核心分析包
│   ├── __init__.py                # v2.3.0
│   ├── loader.py                  # 音频加载
│   ├── features.py                # 特征提取
│   ├── dynamics.py                # Trend Layer（趋势 + 结构分段 + 相似度）
│   ├── volatility.py              # Volatility Layer（滚动波动率 + GARCH）
│   ├── model_analysis.py          # 模型结构分析（4 模型结构侦探）
│   ├── analysis.py                # 时序分析（ACF/PACF/FFT/STFT）
│   ├── prediction.py              # 预测引擎（ARIMA/HMM/LSTM/Transformer）
│   ├── band_analysis.py           # 频带可预测性
│   ├── unsupervised.py            # 无监督模式发现
│   ├── discovery.py               # 双音频探索引擎
│   ├── similarity.py              # 音频相似度
│   ├── visualization.py           # 可视化
│   ├── similarity_viz.py          # 相似度可视化
│   └── discovery_viz.py           # 探索可视化
├── templates/                     # Web 模板
├── static/                        # 静态资源
├── main.py                        # CLI 入口
├── app.py                         # Web 入口（Flask + SSE）
├── requirements.txt               # 依赖
└── README.md                      # 本文档
```

## 许可证

MIT License

---

## 📚 详细文档

Web 应用中可通过 `/docs/<page>` 访问，也可直接阅读 markdown 文件：

| 文档 | 内容 |
|------|------|
| [**使用指南**](docs/guide.md) | Web 界面操作说明、各分析选项详解 |
| [**理论基础**](docs/theory.md) | 从声波振动到数字信号处理的教学讲解 |
| [**分析模块详解**](docs/analysis.md) | 每个分析模块的算法、输入输出、原理 |
| [**结果解读**](docs/results.md) | 如何看懂图表和分析输出 |
