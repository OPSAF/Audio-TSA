# Batch Analyze 批量分析工具

> 纯终端运行的批量音频时间序列分析工具。与 Web 版共享全部分析模块，不依赖 Flask，适合大规模数据分析。

---

## 一、快速开始

```bash
# 分析一个文件夹里所有音频
python batch_analyze.py data/rock/ -o results/rock/

# 快速模式（仅此模式可一轮跑完 100 首）
python batch_analyze.py data/rock/ --fast --max-files 50

# 断点续跑（跳过已完成的文件，放心中断重启）
python batch_analyze.py data/rock/ -o results/rock/
```

---

## 二、命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | *必填* | 输入文件夹路径，递归扫描所有 WAV/MP3/FLAC |
| `-o` `--output` | `{input}_results` | 输出文件夹路径 |
| `--fast` | `False` | 快速模式，深度学习 epochs 降为 5-10 |
| `--max-files` | `0` (不限) | 最多分析多少个文件 |
| `--max-pairs` | `5` | 最多多少对双音频对比 |
| `--no-resume` | `False` | 不跳过已完成的文件（默认会跳过） |
| `--no-compare` | `False` | 跳过双音频对比 |
| `--forecast-horizon` | `100` | 预测视野（步数） |
| `--n-mels` | `128` | Mel 滤波器数量 |
| `--sample-rate` | `16000` | 重采样目标采样率 |

---

## 三、输出结构

```
results/rock/
├── per_file/                          # 每首歌的完整分析结果
│   ├── song_001_results.json
│   ├── song_002_results.json
│   └── ...
├── comparisons/                       # 双音频对比结果
│   ├── song_001__vs__song_005.json
│   └── ...
├── aggregate_summary.json             # 聚合统计 JSON
└── aggregate_report.csv               # 聚合表格 CSV（可用 Excel 打开）
```

### 3.1 单文件结果 JSON

每首歌的分析结果 JSON 包含以下字段（与 Web 版 `results.json` 完全一致）：

| 字段 | 内容 |
|------|------|
| `audio_info` | 时长、采样率、样本数 |
| `waveform` | 时域波形坐标 |
| `fft` | FFT 幅度谱 |
| `stft` | STFT 语谱图 |
| `mel` | Mel 语谱图 |
| `dynamics_serializable` | 四维动态趋势统计 + 高潮/安静段数量 |
| `volatility_serializable` | GARCH 波动率建模结果 |
| `model_analysis` | ARIMA/HMM/LSTM/Transformer 结构分析 |
| `acf_pacf` | 自相关/偏自相关函数 |
| `periodicity` | 主频率、主周期 |
| `complexity` | 零交叉率、样本熵 |
| `spectral_flatness` | 频谱平坦度 |
| `unsupervised_serializable` | 变点数、段落数、NMF 分解 |
| `predictions` | ARIMA/HMM/LSTM/Transformer 预测结果和 RMSE |
| `band_summary` | 各频带可预测性统计 |
| `predictability_rank` | 频带可预测性排名 |
| `band_results` | 各频带各模型详细预测指标 |

### 3.2 聚合 JSON

跨所有文件的统计聚合：

```json
{
  "n_files": 50,
  "audio": { "total_duration_s": 1500, "mean_duration_s": 30.0 },
  "dynamics_energy": { "mean_of_means": 0.142, "std_of_means": 0.035, ... },
  "dynamics_brightness": { ... },
  "dynamics_complexity": { ... },
  "dynamics_rhythm": { ... },
  "complexity": { "mean_zero_crossing_rate": 0.22, "mean_sample_entropy": 3.1 },
  "spectral_flatness": { "mean": 0.38, "std": 0.08 },
  "pred_ARIMA": { "mean_rmse": 4.2, "std_rmse": 1.3, ... },
  "pred_LSTM": { ... },
  "pred_HMM": { ... },
  "pred_Transformer": { ... },
  "hmm": { "mean_n_states": 3.0 },
  "best_band_distribution": { "Low Band": 35, "Mid Band": 10, "High Band": 5 },
  "pairwise": { "n_pairs": 5, "mean_volatility_similarity": 35.2 },
  "batch_meta": { "n_files_total": 50, "fast_mode": true, ... }
}
```

### 3.3 聚合 CSV

可以 Excel 直接打开的表格，一行一首歌，包含：

| 列 | 说明 |
|----|------|
| `file` | 文件名 |
| `duration_s` | 时长 |
| `energy_mean` / `energy_trend` | 能量均值 / 趋势方向 |
| `brightness_mean` / `brightness_trend` | 亮度均值 / 趋势 |
| `complexity_mean` / `complexity_trend` | 复杂度均值 / 趋势 |
| `rhythm_mean` / `rhythm_trend` | 节奏密度 / 趋势 |
| `n_climax` / `n_calm` | 高潮/安静段数量 |
| `zcr` / `sample_entropy` / `spectral_flatness` | 频谱特征 |
| `hmm_n_states` / `lstm_most_learnable` | 模型结构特征 |
| `ARIMA_rmse` / `LSTM_rmse` / ... | 各模型预测 RMSE |
| `best_band` / `best_band_rmse` | 最可预测频带 |

---

## 四、两种模式对比

| | 全量模式 `python batch_analyze.py ...` | 快速模式 `python batch_analyze.py ... --fast` |
|---|---|---|
| LSTM/Transformer epochs | 20-30 | 5-10 |
| 单首耗时 | ~200s | ~165s |
| 100首耗时 | ~5.5 小时 | ~4.6 小时 |
| 适合场景 | 最终报告、发表级精度 | 快速迭代、初步探索 |

---

## 五、典型工作流

### 5.1 先小规模试跑

```bash
# 每类拿 2 首歌试水，快速模式
python batch_analyze.py data/rock/ --fast --max-files 2 -o results/test_rock/
python batch_analyze.py data/pop/  --fast --max-files 2 -o results/test_pop/
```

检查输出结构和数据质量，确认没问题。

### 5.2 全量跑

```bash
# 每个风格单独跑（可并行开多个终端）
python batch_analyze.py data/GTZAN/rock/  -o results/rock/  --fast --max-files 50
python batch_analyze.py data/GTZAN/pop/   -o results/pop/   --fast --max-files 50
python batch_analyze.py data/GTZAN/metal/ -o results/metal/ --fast --max-files 50
# ... 10 类风格分别跑
```

### 5.3 断点续跑

```bash
# 跑一半崩了/关机了，直接重新跑同样命令
# --resume 默认开启，会自动跳过 per_file/ 下已有结果的文件
python batch_analyze.py data/rock/ -o results/rock/ --fast --max-files 50
```

### 5.4 收集报告数据

所有结果在 `results/{genre}/` 下：
- `aggregate_report.csv` → 直接拖进 Excel 画图
- `aggregate_summary.json` → 程序化读取
- `per_file/*.json` → 单首深入分析
- `comparisons/*.json` → 双音频对比数据

---

## 六、数据集准备建议

### 推荐目录结构

```
data/
├── rock/
│   ├── rock.00000.wav
│   ├── rock.00001.wav
│   └── ...
├── pop/
│   └── ...
├── jazz/
│   └── ...
└── ...  (10 个风格文件夹)
```

### 如果只有一个文件夹、文件全混在一起

```bash
# 直接传入，工具会分析里面所有音频
python batch_analyze.py data/all_songs/ -o results/mixed/
```

---

## 七、常见问题

### Q: 跑太慢了怎么办？

1. 用 `--fast` 模式
2. 减少 `--max-files`
3. 关闭双音频对比 `--no-compare`
4. 调低预测精度：修改 `audiots/config.py` 中的 `DEFAULT_PREDICTION_EPOCHS`

### Q: 某个文件失败了怎么办？

- 单个文件失败不会影响其他文件
- 失败的文件会打印在终端
- 修复后重新跑，`--resume` 会自动跳过已成功的文件

### Q: 磁盘空间够吗？

- 每首歌的 JSON 约 500KB-5MB（取决于数组大小）
- 100 首歌约 500MB
- 如需节省空间，可以后续脚本删掉 `waveform`/`fft`/`stft`/`mel` 等大数组字段

### Q: 双音频对比用在哪里？

- 同风格内随机配对 → 衡量风格一致性（intra-genre consistency）
- 不同风格间配对 → 衡量风格相似度
- `pairwise.mean_volatility_similarity` 越高说明该类风格内部越一致

---

返回 [文档首页](/docs) | [命令行工具指南](/docs/guide)
