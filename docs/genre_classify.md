# 流派分类实验 (Genre Classification)

> 基于**时间序列特征工程** + **无监督探索** + **多分类器对比**的音乐流派分类实验。

不使用 LSTM/Transformer/HMM 等深度生成模型，改用 ACF/PACF、趋势分析、波动率、
周期性、白噪声检验等经典时间序列方法提取特征，再用 SVM / 逻辑回归 / 随机森林 /
梯度提升 / MLP 进行分类。

---

## 快速开始

```bash
# 默认配置：每流派 10 首训练 + 2 首测试（共 100 + 20）
python genre_classify.py

# 小规模快速测试
python genre_classify.py --n-train 3 --n-test 1

# 大规模实验
python genre_classify.py --n-train 50 --n-test 10

# 指定输出目录
python genre_classify.py --output results/my_genre_exp/
```

---

## 断点续传

脚本支持**两级缓存**，中断后再次运行自动接续：

| 缓存级别 | 粒度 | 位置 | 说明 |
|---------|------|------|------|
| **Level 1** | 每首歌 | `results/.genre_cache/feat_cache/*.npz` | 每首歌提取完后立即保存，中断后已处理的不重做 |
| **Level 2** | 完整数据集 | `results/.genre_cache/ds_t*_v*.npz` | 全部歌曲处理完后打包，参数不变时秒级加载 |

```bash
# 第一次运行 — 逐首提取特征（每首约 15-20s）
python genre_classify.py --n-train 10 --n-test 2
# Ctrl+C 中断后...

# 第二次运行 — 已完成的歌曲从 Level-1 缓存恢复，继续剩余
python genre_classify.py --n-train 10 --n-test 2

# 第三次运行 — Level-2 完整缓存已存在，秒级启动
python genre_classify.py --n-train 10 --n-test 2
```

> 换 `--n-train` / `--n-test` 参数会触发重新提取（因为训练/测试划分可能不同）。

---

## 特征工程（124 维时间序列特征）

| 类别 | 维度 | 提取内容 | 时间序列意义 |
|------|------|---------|-------------|
| **ACF** | 20 | lag 1-20 自相关系数 | 信号在不同滞后下的自相似程度 |
| **PACF** | 20 | lag 1-20 偏自相关系数 | 剔除中间滞后影响的直接依赖 |
| **动态趋势** | 28 | energy/brightness/complexity/rhythm 的 mean/std/min/max/n_peaks/slope | 音频动态特征的统计分布和趋势方向 |
| **波动率 + GARCH** | 16 | 4 维趋势的 mean_vol / std_vol / GARCH α / GARCH β / α+β | 波动率聚集效应和持久性 |
| **周期性** | 3 | 主导频率 / 主导周期 / 峰值数 | 信号的周期性结构 |
| **复杂度** | 2 | 零交叉率 / 样本熵 | 信号的随机性和复杂度 |
| **频谱** | 4 | spectral flatness + Mel centroid / spread / skewness | 频谱形状特征 |
| **白噪声检验** | 7 | Ljung-Box / Box-Pierce / Jarque-Bera / ACF / 方差平稳性 / Runs Test p 值 + 通过数 | 信号偏离白噪声的程度 |
| **ARIMA** | 24 | 4 维趋势的 (p, d, q) 阶数 + AIC + 平稳性 + 白噪声标志 | 每维趋势的线性结构类型 |

---

## 分类器

| 模型 | 类型 | 说明 |
|------|------|------|
| **SVM (RBF kernel)** | 核方法 | 非线性决策边界，小样本友好 |
| **Logistic Regression** | 线性模型 | 线性可分的基线 |
| **Random Forest** | 树模型 | 100 棵树，max_depth=10，提供特征重要性 |
| **Gradient Boosting** | 集成学习 | 100 轮，max_depth=4 |
| **MLP (Deep NN)** | 深度学习 | 2 隐藏层 (128, 64)，ReLU 激活 |

---

## 输出文件

```
results/genre_classify_YYYYMMDD_HHMMSS/
├── summary_report.json              # 综合报告
│
├── U01_pca_projection.png           # PCA 2D 投影（按流派着色）
├── U02_tsne_projection.png          # t-SNE 投影
├── U03_feature_correlation.png      # Top 40 特征相关性矩阵
├── U04_acf_by_genre.png            # 每个流派的平均 ACF 曲线
├── U05_feature_boxplots_by_genre.png # 关键 TS 特征按流派箱线图
├── U06_acf_decay_by_genre.png       # ACF 衰减速度对比
├── U07_genre_separability.png       # 流派可分离性（Silhouette）
├── U08_signal_structure_by_genre.png # 信号结构（白噪声+平稳性+GARCH）
│
├── C01_cm_*.png                     # 每个分类器的混淆矩阵（5 张）
├── C02_model_comparison.png         # 模型准确率/F1 对比柱状图
├── C03_feature_importance.png       # 时间序列特征重要性 Top 20
├── C04_aggregated_confusion.png     # 聚合混淆矩阵（所有分类器求和）
├── C05_per_classifier_f1_heatmap.png # 每分类器每流派 F1 热力图
└── C06_feature_category_importance.png # 特征类别重要性分组
```

---

## 结果解读

### U01 PCA / U02 t-SNE

不同流派的歌曲在时间序列特征空间中是否有**自然的聚类**？
如果同类歌曲在 2D 投影中聚在一起，说明时间序列特征确实捕捉到了流派差异。

### U04 ACF by Genre

每个流派的**平均自相关曲线**展示了不同流派的时间依赖模式。
例如：古典乐可能 ACF 衰减慢（长记忆），摇滚可能衰减快（短记忆）。

### U05 Feature Boxplots by Genre

关键时间序列特征（ACF lag-1、能量均值、GARCH 持久性、零交叉率等）
在每个流派中的**分布箱线图**。中位数、四分位距、离群值一目了然。

### U06 ACF Decay by Genre

左图：ACF lag-1 值（短期记忆强度）。右图：ACF 半衰期（记忆持续多少滞后）。
**这是最直接的时间序列特征对比**——不同流派的信号"记忆力"差异。

### U07 Genre Separability

左图：每个流派的 Silhouette 分数（越高 = 越独立）。
右图：类内紧凑度 vs 类间分离度。红条高 + 蓝条低 = 好的分类前景。

### U08 Signal Structure by Genre

三面板：白噪声得分 / 趋势平稳性 / GARCH 波动率持久性。
展示了不同流派在**信号结构层面的根本差异**。

### C01 混淆矩阵

看哪些流派容易混淆（如 blues ↔ rock，disco ↔ pop），
这些混淆对反映了时间序列特征的局限性。

### C03 特征重要性

哪些时间序列特征对流派判别最有贡献？
如果 ACF/PACF 特征排名靠前，说明自相关结构是流派的关键区分因素。

### C04 聚合混淆矩阵

所有分类器的混淆矩阵求和，展示**跨模型一致的混淆模式**。
如果多个分类器都把 blues 错分成 rock，说明这两个流派在时间序列特征上确实接近。

### C05 Per-Classifier Per-Genre F1

每个分类器在每个流派上的 F1 分数热力图。
一眼看出哪个模型擅长哪些流派，哪些流派对所有模型都困难。

### C06 特征类别重要性

按特征来源分组（ACF/PACF、动态趋势、波动率+GARCH、ARIMA、复杂度、频谱、白噪声），
看**哪类时间序列特征对流派判别贡献最大**。

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-train` | 10 | 每流派训练样本数（× 10 流派 = 总训练数） |
| `--n-test` | 2 | 每流派测试样本数（× 10 流派 = 总测试数） |
| `--output` | `results/genre_classify_TIMESTAMP` | 输出目录 |
| `--fast` | False | 跳过部分慢速提取步骤 |

---

## 设计理念

### 为什么不用 LSTM / Transformer？

这个实验的目标是**验证时间序列特征工程的有效性**。
如果仅用 ACF、趋势、波动率等经典 TS 特征 + 简单分类器就能达到不错的准确率，
说明时间序列结构本身就携带了足够的流派信息 — 不需要黑箱深度学习来"发现"。

### 时间序列特征 vs 原始频谱特征

| 特征类型 | 例子 | 优点 | 缺点 |
|---------|------|------|------|
| **时间序列特征** | ACF, PACF, ARIMA 阶数, GARCH 参数 | 可解释，维度低，有统计意义 | 依赖窗口参数 |
| **原始频谱特征** | Mel spectrogram, MFCC | 信息完整 | 维度高，难解释 |

> 特征重要性排名中，如果 ACF/PACF/趋势特征排名靠前，
> 就证明了时间序列分析在音频流派识别中的价值。

### 与 batch_analyze 的区别

| | batch_analyze | genre_classify |
|---|---|---|
| 目标 | 同流派内挖掘共性 | 跨流派判别差异 |
| 方法 | 统计 + Global ML | 特征工程 + 分类器 |
| 模型 | LSTM / Transformer / HMM | SVM / RF / GBDT / MLP |
| 输出 | 10 张分析仪表盘 | 混淆矩阵 + 特征重要性 + 投影 |
| 时间序列特色 | 趋势分析 + GARCH | ACF/PACF + ARIMA + 白噪声检验 |
