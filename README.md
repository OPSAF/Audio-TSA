# Audio Lab - 音频时间序列分析系统

音频时间序列分析与预测平台，支持特征提取、时序分析、机器学习预测和双音频对比分析。

## 功能特性

- **特征提取**: 波形、FFT频谱、STFT、Mel频谱图、MFCC系数
- **动态趋势分析**: 能量、亮度、复杂度、节奏分析
- **时序分析**: ACF/PACF、周期性检测、复杂度分析
- **无监督模式发现**: 变点检测、motif发现、聚类分析
- **机器学习预测**: ARIMA、HMM、LSTM、Transformer四种模型
- **频带分析**: 低/中/高频带可预测性评估
- **双音频对比**: 相似度分析、多维探索
- **可视化报告**: 交互式图表展示

## 安装

```bash
# 克隆仓库
git clone https://github.com/OPSAF/Audio-TSA.git
cd Audio-TSA

# 安装依赖
pip install -r requirements.txt
```

### 依赖列表

- numpy
- scipy
- librosa
- soundfile
- matplotlib
- flask
- statsmodels
- scikit-learn

- torch (可选，用于LSTM/Transformer预测)

## 使用方法

### 终端版本

```bash
# 默认全分析（使用合成音频）
python main.py

# 分析单个音频文件（默认全分析）
python main.py --audio1 path/to/audio.wav

# 分析两个音频文件（启用对比分析）
python main.py --audio1 audio1.wav --audio2 audio2.wav

# 指定分析模块
python main.py --audio1 audio.wav --analysis features,timeseries,prediction

# 不保存图片
python main.py --audio1 audio.wav --no-save

# 指定输出目录
python main.py --audio1 audio.wav --output ./my_output
```

### 分析选项

| 选项 | 说明 |
|------|------|
| `features` | 特征提取（波形、FFT、STFT、Mel、MFCC） |
| `dynamics` | 动态趋势分析（能量、亮度、复杂度、节奏） |
| `timeseries` | 时序分析（ACF、PACF、周期性、复杂度） |
| `unsupervised` | 无监督模式发现（聚类、motif检测） |
| `prediction` | 机器学习预测（ARIMA、HMM、LSTM、Transformer） |
| `band` | 频带分析（频率带可预测性评估） |
| `comparison` | 双音频对比（相似度分析）- 需要两个音频 |
| `visualization` | 可视化（生成分析图表） |

**注意**: 
- 不指定 `--analysis` 时默认执行所有分析
- `comparison` 选项仅在提供两个音频文件时有效
- `prediction` 包含所有四种机器学习模型

### Web界面

```bash
# 启动Web服务
python app.py

# 访问 http://localhost:5000
```

Web界面功能：
- 上传主音频文件（支持 WAV、MP3、FLAC）
- 可选上传副音频文件（用于对比分析）
- 勾选需要的分析模块
- 设置预测步长和Mel频带数
- 查看分析结果和可视化图表

## 项目结构

```
Audio Lab/
├── audiots/                 # 核心分析模块
│   ├── loader.py           # 音频加载
│   ├── features.py         # 特征提取
│   ├── dynamics.py         # 动态趋势分析
│   ├── analysis.py         # 时序分析
│   ├── prediction.py       # 机器学习预测
│   ├── band_analysis.py    # 频带分析
│   ├── unsupervised.py     # 无监督学习
│   ├── discovery.py        # 双音频探索
│   ├── similarity.py       # 相似度分析
│   └── visualization.py    # 可视化
├── templates/              # Web模板
│   ├── index.html         # 上传页面
│   └── results.html       # 结果页面
├── main.py                 # 终端入口
├── app.py                  # Web应用入口
├── requirements.txt        # 依赖列表
└── README.md              # 说明文档
```

## 输出说明

分析结果保存在 `outputs/` 目录（或指定目录）：

- `01_waveform.png` - 波形图
- `02_fft_spectrum.png` - FFT频谱
- `03_stft_spectrogram.png` - STFT频谱图
- `04_mel_spectrogram.png` - Mel频谱图
- `05_mfcc.png` - MFCC系数
- `06_acf_pacf.png` - ACF/PACF分析
- `07_periodicity.png` - 周期性分析
- `08_prediction_comparison.png` - 预测对比
- `09_model_error_bars.png` - 模型误差
- `10_band_error_heatmap.png` - 频带误差热图
- `11_band_error_bars.png` - 频带误差条形图

- `discovery/` - 双音频探索可视化
- `dynamics/` - 动态趋势可视化

## 示例

```bash
# 快速分析：仅特征提取和预测
python main.py --audio1 song.wav --analysis features,prediction,visualization

# 完整分析：双音频对比
python main.py --audio1 original.wav --audio2 processed.wav

# 轻量分析：不生成图片
python main.py --audio1 audio.wav --analysis features,timeseries --no-save
```

## 许可证

MIT License
