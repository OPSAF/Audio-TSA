实验方案：音频时间序列分析

▎ 基于项目 Audio Lab 的实际功能设计 | 日期：2026-06-06

***

1. 项目功能概览

项目类型：CLI + Web 的音频时间序列多层分析平台
技术栈：Python, NumPy/SciPy, statsmodels, PyTorch, hmmlearn, librosa, Flask
入口点：

- CLI：python main.py --audio1 <file> \[--analysis <options>]
- Web：python app.py → <http://localhost:5000>
- Batch：python batch\_analyze.py \<input\_folder>

模块地图

模块: audiots/analysis.py
用途: 时序分析：ACF/PACF/FFT/周期性/白噪声检验
关键导出: compute\_acf(), compute\_pacf(), ljung\_box\_test(), test\_white\_noise(), analyze\_periodicity()
代码量: \~500行
────────────────────────────────────────
模块: audiots/dynamics.py
用途: Trend Layer：4个动态趋势 + 结构分段 + 趋势相似度
关键导出: extract\_dynamics(), detect\_structural\_segments(), compute\_trend\_similarity()
代码量: \~650行
────────────────────────────────────────
模块: audiots/volatility.py
用途: Volatility Layer：滚动波动率 + GARCH(1,1)
关键导出: compute\_volatility\_layer(), \_fit\_garch\_statsmodels()
代码量: \~400行
────────────────────────────────────────
模块: audiots/prediction.py
用途: 预测引擎：ARIMA/HMM/LSTM/Transformer
关键导出: predict\_arima(), predict\_hmm(), predict\_lstm(), predict\_transformer()
代码量: \~800行
────────────────────────────────────────
模块: audiots/model\_analysis.py
用途: 模型结构分析：四种结构侦探
关键导出: analyze\_arima\_insights(), analyze\_hmm\_insights(), analyze\_lstm\_insights(), analyze\_transformer\_insights(),
analyze\_model\_ensemble()
代码量: \~1250行
────────────────────────────────────────
模块: audiots/band\_analysis.py
用途: 频带可预测性并行分析
关键导出: analyze\_band\_predictability(), compute\_band\_error\_summary()
代码量: \~250行
────────────────────────────────────────
模块: audiots/unsupervised.py
用途: 无监督模式发现
关键导出: 变点检测、Motif发现、NMF分解、RQA、异常检测、聚类
代码量: \~800行
────────────────────────────────────────
模块: audiots/discovery.py
用途: 双音频跨维度探索
关键导出: self\_discovery(), cross\_discover(), explore()
代码量: \~1200行
────────────────────────────────────────
模块: audiots/similarity.py
用途: 长度无关音频相似度
关键导出: analyze\_similarity()
代码量: \~500行
────────────────────────────────────────
模块: audiots/visualization.py
用途: 全模块可视化
关键导出: 趋势图、波动率图、GARCH诊断、模型仪表板
代码量: \~1000行

可用能力

1. 时序特征提取 — 从原始音频波形中提取 4 个趋势时间序列（energy/brightness/complexity/rhythm），每段窗口约 0.5s，跳跃
   0.25s。入口点：dynamics.extract\_dynamics() at audiots/dynamics.py:106
2. 白噪声/平稳性综合检验 — 5 种检验（Ljung-Box、Box-Pierce、Jarque-Bera、ACF 置信区间、方差平稳性 +
   游程检验）的投票式综合判定。入口点：analysis.test\_white\_noise() at audiots/analysis.py:394
3. ARIMA 自动建模 — 自动 ADF 平稳性检验 + AIC/BIC 定阶 + 多阶候选搜索。入口点：prediction.predict\_arima() at
   audiots/prediction.py:97
4. HMM 隐状态发现 —
   无监督发现音频内在状态，输出状态画像、转移矩阵、段落结构。入口点：model\_analysis.analyze\_hmm\_insights() at
   audiots/model\_analysis.py:401
5. LSTM 记忆分析 — 自动搜索最优
   lookback，逐维度可学性排名，记忆体制分类（短/中/长）。入口点：model\_analysis.analyze\_lstm\_insights() at
   audiots/model\_analysis.py:573
6. Transformer 注意力分析 — 多头自注意力模式提取，主导 lag
   发现，多尺度时间结构检测。入口点：model\_analysis.analyze\_transformer\_insights() at audiots/model\_analysis.py:717
7. GARCH 波动率建模 — 滚动波动率 + GARCH(1,1)
   条件异方差，波动率持久性（α+β）分析。入口点：volatility.compute\_volatility\_layer() at audiots/volatility.py
8. 无监督模式发现 — 变点检测、Motif 发现、NMF 频谱分解、RQA 递归量化、异常检测。入口点：unsupervised 模块 at
   audiots/unsupervised.py
9. 频带可预测性分析 — 低/中/高频带独立预测 + 可预测性排名。入口点：band\_analysis.analyze\_band\_predictability() at
   audiots/band\_analysis.py:116
10. 双音频对比探索 — 自发现、跨发现、对比分析、角色画像。入口点：discovery.explore() at audiots/discovery.py:1028

***

1. 主题分析

主题：时间序列分析
核心问题：音频信号中隐含的时间结构——趋势性、周期性、波动聚集性、记忆长度和自注意力——能否被系统性地发现、建模和解释？

子问题分解

\#: 1
问题: 趋势结构问题：音频的四维趋势时间序列是否各自具有可识别的线性结构（均值回复、随机游走、趋势型或振荡型）？
假设: 不同维度的趋势具有不同类型的时间结构：energy 呈均值回复型（MA 特征），brightness 呈振荡型，complexity 接近白噪声
变量关系 (IV → DV): IV: 趋势维度(energy/brightness/complexity/rhythm) → DV:
ARIMA最优阶数(p,d,q)、白噪声检验p值、趋势类型分类
────────────────────────────────────────
\#: 2
问题: 波动聚集问题：音频趋势序列的波动率是否呈现时变聚集特征（GARCH效应）？不同维度的波动率持久性是否不同？
假设: rhythm 维度的波动率持久性（α+β）最高（击鼓节奏变化持续），complexity 维度的波动率持久性最低
变量关系 (IV → DV): IV: 趋势维度 → DV: GARCH(1,1)参数(α, β)、持久性(α+β)、条件波动率序列
────────────────────────────────────────
\#: 3
问题: 记忆长度问题：不同频带（低/中/高）的时间序列所需的最优预测记忆长度是否不同？
假设: 低频带需要更长的 lookback（慢变包络），高频带只需短 lookback（快速变化）
变量关系 (IV → DV): IV: 频率带(低频/中频/高频) → DV: LSTM最优lookback、预测RMSE、可学性分数
────────────────────────────────────────
\#: 4
问题: 内在状态问题：音频能否被自动划分为有限个内在状态？状态转移是否具有方向性？
假设: HMM 可发现 3-4 个内在状态，转移模式呈现方向性（如 calm→active→climax→calm 循环）
变量关系 (IV → DV): IV: HMM 状态数(2/3/4/5) → DV: 状态区分度、转移矩阵特征、BIC分数
────────────────────────────────────────
\#: 5
问题: 自注意力结构问题：音频时间序列是否具有多尺度自相似结构？不同注意力头是否捕捉到不同时间尺度的依赖？
假设: Transformer 多头注意力中至少有一个头捕捉短程依赖(<1s)、一个捕捉长程结构(>2s)，不同头之间的注意力模式不同
变量关系 (IV → DV): IV: 注意力头编号(0/1/2/3) → DV: 主导lag、注意力熵、头间余弦相似度
────────────────────────────────────────
\#: 6
问题: 变化点检测问题：无监督变点检测能否发现音频的结构边界（如音乐的主歌→副歌过渡）？
假设: 变点检测能发现 energy 和 rhythm 的突变位置，这些位置与人工标注的结构边界一致
变量关系 (IV → DV): IV: 检测算法(PELT/BinarySeg/阈值法) → DV: 检测到的变点数、变点位置的合理性

***

1. 功能匹配

┌──────────┬──────────┬───────────────────────────────────────────────────────────────────────────────┬──────────┐
│ 子问题 │ 支持度 │ 项目能力 │ 缺口说明 │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #1 │ ✅ │ model\_analysis.analyze\_arima\_insights() at audiots/model\_analysis.py:334 — 对 │ — │
│ 趋势结构 │ 直接支持 │ 4 个趋势维度逐一拟合 ARIMA 并分类 │ │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #2 │ ✅ │ volatility.compute\_volatility\_layer() at audiots/volatility.py — 滚动波动率 + │ — │
│ 波动聚集 │ 直接支持 │ GARCH(1,1) 拟合 │ │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #3 │ ✅ │ band\_analysis.analyze\_band\_predictability() at audiots/band\_analysis.py:116 + │ — │
│ 记忆长度 │ 直接支持 │ model\_analysis.analyze\_lstm\_insights() at audiots/model\_analysis.py:573 │ │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #4 │ ✅ │ model\_analysis.analyze\_hmm\_insights() at audiots/model\_analysis.py:401 — HMM │ — │
│ 内在状态 │ 直接支持 │ 隐状态发现 + 状态画像 │ │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #5 │ ✅ │ model\_analysis.analyze\_transformer\_insights() at │ — │
│ 自注意力 │ 直接支持 │ audiots/model\_analysis.py:717 — 注意力模式提取 + 多尺度分析 │ │
├──────────┼──────────┼───────────────────────────────────────────────────────────────────────────────┼──────────┤
│ #6 │ ✅ │ unsupervised 模块 at audiots/unsupervised.py — 变点检测 + 段落标注 │ — │
│ 变化点 │ 直接支持 │ │ │
└──────────┴──────────┴───────────────────────────────────────────────────────────────────────────────┴──────────┘

结论：全部 6 个子问题均有直接的项目功能支持，无需额外开发。

***

1. 实验设计

***

实验 1：趋势线性结构分类实验

子问题：#1 — 四维趋势的线性结构类型

假设：不同维度的趋势具有可区分的 ARIMA 结构特征：energy 呈现 MA 型（冲击驱动），complexity 接近白噪声，rhythm
可能是非平稳随机游走。

自变量 (IV)：趋势维度 — energy, brightness, complexity, rhythm（4 个水平）
因变量 (DV)：

- ARIMA 最优阶数 (p, d, q)
- Ljung-Box 白噪声检验 p 值（>0.05 为白噪声）
- 趋势类型分类（mean-reverting / random-walk / trending / oscillating）
  控制变量：
- 音频文件：使用项目内置合成音频 loader.generate\_sample\_audio() at audiots/loader.py:36
- 采样率：16000 Hz
- 窗口参数：window\_size=0.5s, hop\_size=0.25s（默认值）

项目功能使用：

- dynamics.extract\_dynamics() at audiots/dynamics.py:106 — 提取 4 个趋势序列
- model\_analysis.analyze\_arima\_insights() at audiots/model\_analysis.py:334 — ARIMA 结构分析

实验步骤：

1. 生成合成测试音频：
   python -c "from audiots.loader import generate\_sample\_audio; y, sr = generate\_sample\_audio(duration=5.0, sr=16000);
   import numpy as np; np.save('experiments/synthetic.npy', {'y': y, 'sr': sr})"
2. 提取动态趋势：
   from audiots import loader, dynamics, model\_analysis
   y, sr = loader.generate\_sample\_audio(duration=5.0, sr=16000)
   dyn = dynamics.extract\_dynamics(y, sr, window\_size=0.5, hop\_size=0.25)
3. 运行 ARIMA 结构分析：
   arima\_insights = model\_analysis.analyze\_arima\_insights(dyn)
   for name, insight in arima\_insights.per\_trend.items():
   print(f"{name}: order={insight.best\_order}, stationary={insight.is\_stationary}, "
   f"white\_noise={insight.is\_white\_noise}, type={insight.trend\_type}, "
   f"p\_value={insight.ljung\_box\_pvalue:.4f}")
4. 使用多个真实音频重复（推荐 ≥3 个不同风格音频）：
   for audio in speech.wav music\_pop.wav ambient.wav; do
   python main.py --audio1 "data/$audio" --analysis model\_analysis --save-json --output "results/exp1\_$(basename
   $audio .wav)"
   done
5. 汇总对比，验证假设

数据需求：

- 输入格式：WAV/MP3/FLAC 音频文件
- 最小样本量：≥3 个不同风格音频，每个 ≥3 秒
- 数据来源：内置合成音频 + 用户提供的真实音频

预期结果：

- 若假设正确：energy 为 MA 型 (p=0, q≥1)、complexity 白噪声检验 p>0.05、rhythm 的 d=1（非平稳）
- 若假设错误：所有维度均呈现相同的 ARIMA 结构类型，则说明四维趋势在时间结构上没有本质区别

评估指标：

┌──────────────────┬──────────────────────────┬───────────────────────────┬────────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├──────────────────┼──────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 最优阶数 (p,d,q) │ AIC 最小的 ARIMA 阶数 │ p∈\[0,4], d∈\[0,1], q∈\[0,2] │ ArimaTrendInsight.best\_order │
├──────────────────┼──────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 白噪声 p 值 │ Ljung-Box 检验的 p 值 │ 0\~1, >0.05 为白噪声 │ ArimaTrendInsight.ljung\_box\_pvalue │
├──────────────────┼──────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ 趋势类型 │ ARIMA 结构对应的类型标签 │ 4 类之一 │ ArimaTrendInsight.trend\_type │
├──────────────────┼──────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ AIC 值 │ 模型拟合优度 │ 越小越好 │ ArimaTrendInsight.aic │
└──────────────────┴──────────────────────────┴───────────────────────────┴────────────────────────────────────┘

可重现性：

- 随机种子：不适用（ARIMA 是确定性算法）
- 环境：Python 3.8+, statsmodels≥0.13
- 预计运行时间：每次分析约 30 秒

***

实验 2：波动率时变聚集性实验

子问题：#2 — 波动率 GARCH 效应

假设：音频趋势序列存在波动率聚集现象（大波动后面跟着大波动）。rhythm
维度的波动率持久性（α+β）最高（鼓点节拍变化的持续性），complexity 最低（频谱复杂度波动更随机）。

自变量 (IV)：趋势维度 — energy, brightness, complexity, rhythm
因变量 (DV)：

- GARCH(1,1) 参数：ω（长期均值）, α（冲击系数）, β（持续性系数）
- 波动率持久性：α + β（越接近 1 = 波动率聚集越强）
- 条件波动率序列的标准差
  控制变量：音频文件、采样率、窗口参数（同实验1）

项目功能使用：

- dynamics.extract\_dynamics() at audiots/dynamics.py:106 — 趋势层输入
- volatility.compute\_volatility\_layer() at audiots/volatility.py — 滚动波动率 + GARCH 拟合

实验步骤：

1. 提取趋势 + 波动率层：
   from audiots import loader, dynamics, volatility

y, sr = loader.generate\_sample\_audio(duration=10.0, sr=16000) # 更长音频确保足够窗口
dyn = dynamics.extract\_dynamics(y, sr, window\_size=0.5, hop\_size=0.25)
vol = volatility.compute\_volatility\_layer(dyn, rolling\_window=10, fit\_garch=True)

for dim in \['energy', 'brightness', 'complexity', 'rhythm']:
g = vol\['garch\_models'].get(dim, {})
print(f"{dim}: ω={g.get('omega', 'N/A'):.6f}, α={g.get('alpha', 'N/A'):.4f}, "
f"β={g.get('beta', 'N/A'):.4f}, persistence={g.get('persistence', 'N/A'):.4f}")
2\. CLI 运行（含完整可视化）：
python main.py --audio1 data/song.wav --analysis dynamics\_analysis,visualization --output results/exp2\_volatility/
3\. 检查可视化输出 volatility/garch\_energy.png 等 GARCH 诊断图

数据需求：

- 输入格式：音频文件
- 最小样本量：≥3 个不同风格音频，≥5 秒（确保足够窗口进行 GARCH 拟合）
- 数据来源：内置合成 + 真实音频

预期结果：

- 若假设正确：rhythm 的 α+β > 0.85（强持久性），complexity 的 α+β < 0.5（弱持久性）
- 若假设错误：所有维度 α+β 接近，无显著差异 → 波动率聚集是整体现象，不分维度

评估指标：

┌────────────────┬──────────────────────────────┬──────────────────┬─────────────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├────────────────┼──────────────────────────────┼──────────────────┼─────────────────────────────────────────┤
│ α (ARCH) │ 前期冲击对当前波动率的影响 │ 0\~1 │ vol\['garch\_models']\[dim]\['alpha'] │
├────────────────┼──────────────────────────────┼──────────────────┼─────────────────────────────────────────┤
│ β (GARCH) │ 前期波动率对当前波动率的持续 │ 0\~1 │ vol\['garch\_models']\[dim]\['beta'] │
├────────────────┼──────────────────────────────┼──────────────────┼─────────────────────────────────────────┤
│ 持久性 │ α + β，波动率半衰期指标 │ 0\~1, >0.9=长记忆 │ vol\['garch\_models']\[dim]\['persistence'] │
├────────────────┼──────────────────────────────┼──────────────────┼─────────────────────────────────────────┤
│ 条件波动率 std │ 条件波动率序列的标准差 │ >0 │ np.std(cond\_vol) │
└────────────────┴──────────────────────────────┴──────────────────┴─────────────────────────────────────────┘

可重现性：

- 随机种子：不适用（GARCH MLE 是确定性的，但需要用相同 statsmodels 版本）
- 环境：Python 3.8+, statsmodels≥0.13
- 预计运行时间：每次约 1 分钟

***

实验 3：频带记忆长度对比实验

子问题：#3 — 不同频带的最优预测记忆长度

假设：低频带（0-500Hz，包络慢变）需要更长 lookback（≥20 窗口），高频带（5000Hz+，快速起伏）只需短 lookback（≤8
窗口）。

自变量 (IV)：频率带 — 低频(0-42 mel)、中低频(43-85)、中高频(86-106)、高频(107-127)（4 个水平）
因变量 (DV)：

- LSTM 预测 RMSE
- 最优 lookback 窗口数
- 可学性分数（越低越好）
  控制变量：同一音频文件、forecast\_horizon=20、n\_mels=128

项目功能使用：

- band\_analysis.analyze\_band\_predictability() at audiots/band\_analysis.py:116 — 分频带并行预测
- model\_analysis.analyze\_lstm\_insights() at audiots/model\_analysis.py:573 — LSTM 记忆分析
- features.split\_mel\_bands() at audiots/features.py:87 — Mel 频带划分

实验步骤：

1. 运行分频带预测分析：
   python main.py --audio1 data/song.wav --analysis band,prediction --forecast-horizon 20 --output results/exp3\_band/
2. 使用 Python API 获取详细数据：
   from audiots import loader, features, band\_analysis

y, sr = loader.load\_audio('data/song.wav', target\_sr=16000)
mel\_spec = features.compute\_mel\_spectrogram(y, sr, n\_mels=128)
band\_results = band\_analysis.analyze\_band\_predictability(mel\_spec, forecast\_horizon=20, parallel=True)
summary = band\_analysis.compute\_band\_error\_summary(band\_results)
rank = band\_analysis.get\_predictability\_rank(summary)
band\_analysis.print\_band\_summary(summary)
3\. 使用不同 lookback 参数复现 LSTM 分析：
from audiots import model\_analysis
import numpy as np

y, sr = loader.load\_audio('data/song.wav', target\_sr=16000)
dyn = dynamics.extract\_dynamics(y, sr)

for lookback in \[5, 10, 20, 30, 50]:
lstm = model\_analysis.analyze\_lstm\_insights(
dyn, d\_model=64, n\_epochs=100, lookback\_range=(lookback, lookback+1)
)
print(f"lookback={lookback}: optimal={lstm.optimal\_lookback}, "
f"learnability={lstm.per\_dim\_learnability}")

数据需求：

- 输入格式：音频文件
- 最小样本量：≥2 个不同风格音频
- 数据来源：真实音频（不同风格更好）

预期结果：

- 若假设正确：低频带 RMSE 在 lookback 增大时持续下降，最优 lookback >20；高频带在 lookback=8 附近 RMSE 最低
- 若假设错误：所有频带最优 lookback 相近，频带间预测性无系统性差异

评估指标：

┌───────────────┬────────────────────────────┬─────────────────┬────────────────────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├───────────────┼────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
│ 最优 lookback │ 使验证损失最小的回看窗口数 │ 3\~50 │ LstmInsights.optimal\_lookback │
├───────────────┼────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
│ 可学性分数 │ 该维度的预测难度 │ 0\~∞，越小越可学 │ per\_dim\_learnability dict │
├───────────────┼────────────────────────────┼─────────────────┼────────────────────────────────────────────────┤
│ RMSE/频带 │ 各频带的最低预测 RMSE │ 取决于音频 │ band\_error\_summary\['bands']\[band]\['best\_rmse'] │
└───────────────┴────────────────────────────┴─────────────────┴────────────────────────────────────────────────┘

可重现性：

- 随机种子：torch.manual\_seed(42)（LSTM 训练）
- 环境：Python 3.8+, torch≥1.12
- 预计运行时间：每次约 5-15 分钟（取决于是否有 GPU）

***

实验 4：隐状态数量选择实验

子问题：#4 — HMM 最优状态数

假设：大多数音乐/语音音频可被划分为 3-4 个隐状态，状态转移呈现方向性（非对称转移矩阵）。

自变量 (IV)：HMM 状态数 — 2, 3, 4, 5（4 个水平）
因变量 (DV)：

- BIC（贝叶斯信息准则）— 平衡拟合度和复杂度
- 状态区分度 — 各状态的特征向量是否显著不同
- 转移矩阵的方向性指标 — 非对称程度
  控制变量：同一音频文件、covariance\_type='full'

项目功能使用：

- model\_analysis.analyze\_hmm\_insights() at audiots/model\_analysis.py:401 — HMM 隐状态发现
- dynamics.extract\_dynamics() at audiots/dynamics.py:106 — 4D 趋势输入

实验步骤：

1. 遍历不同状态数进行 HMM 分析：
   from audiots import loader, dynamics, model\_analysis

y, sr = loader.load\_audio('data/song.wav', target\_sr=16000)
dyn = dynamics.extract\_dynamics(y, sr)

for n\_states in \[2, 3, 4, 5]:
hmm = model\_analysis.analyze\_hmm\_insights(dyn, n\_states=n\_states)
print(f"\n=== n\_states={n\_states} ===")
print(f"BIC: {hmm.bic:.2f}")
print(f"区分度: {hmm.state\_separability:.2%}")
print(f"转移矩阵:\n{hmm.transition\_matrix}")
print(f"标签: {hmm.state\_labels}")
2\. CLI 运行（使用默认 3 状态）：
python main.py --audio1 data/song.wav --analysis model\_analysis --output results/exp4\_hmm/
3\. 在多首音频上重复，统计最优状态数的分布

数据需求：

- 输入格式：音频文件
- 最小样本量：≥5 个不同风格音频
- 数据来源：真实音频（包含不同风格：语音、流行音乐、古典、环境音）

预期结果：

- 若假设正确：BIC 在 n=3 或 n=4 时最小，转移矩阵非对称（某些方向转移概率 >70%）
- 若假设错误：BIC 随 n 单调变化（一直递减 = 模型越复杂越好；一直递增 = 只发现 1-2 个状态）

评估指标：

┌──────────────┬────────────────────────┬────────────────────┬─────────────────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├──────────────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────┤
│ BIC │ 贝叶斯信息准则 │ 越小越好 │ HmmInsights.bic │
├──────────────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────┤
│ 状态区分度 │ 状态间特征向量的可分性 │ 0\~100% │ HmmInsights.state\_separability │
├──────────────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────┤
│ 转移非对称度 │ 转移矩阵的不对称程度 │ 0\~1 │ np.linalg.norm(T - T.T) / np.linalg.norm(T) │
├──────────────┼────────────────────────┼────────────────────┼─────────────────────────────────────────────┤
│ 最小状态占比 │ 最小状态的样本比例 │ >5% 表示状态有意义 │ min(counts) / sum(counts) │
└──────────────┴────────────────────────┴────────────────────┴─────────────────────────────────────────────┘

可重现性：

- 随机种子：np.random.seed(42)（HMM 初始化）
- 环境：Python 3.8+, hmmlearn≥0.2
- 预计运行时间：每次约 1 分钟

***

实验 5：多头注意力时间尺度实验

子问题：#5 — Transformer 多尺度自注意力

假设：Transformer 的多个注意力头自发分化为不同时间尺度的检测器——至少一个头关注短程依赖（lag
1-3，<1s），一个关注长程结构（lag >10，>2.5s）。

自变量 (IV)：注意力头编号 — head\_0, head\_1, head\_2, head\_3（4 个水平）
因变量 (DV)：

- 每个头的主导 lag（平均注意力权重最大的滞后距离）
- 注意力熵（注意力分布的集中程度）
- 头间余弦相似度
  控制变量：同一音频、Transformer 超参（d\_model=64, n\_heads=4）

项目功能使用：

- model\_analysis.analyze\_transformer\_insights() at audiots/model\_analysis.py:717 — Transformer 注意力分析
- dynamics.extract\_dynamics() at audiots/dynamics.py:106 — 趋势输入

实验步骤：

1. 提取 Transformer 注意力模式：
   from audiots import loader, dynamics, model\_analysis

y, sr = loader.load\_audio('data/song.wav', target\_sr=16000)
dyn = dynamics.extract\_dynamics(y, sr)
tf\_insights = model\_analysis.analyze\_transformer\_insights(dyn, d\_model=64, n\_heads=4)

for head in tf\_insights.heads:
print(f"{head.head\_name}: dominant\_lag={head.dominant\_lag} "
f"({head.dominant\_lag \* 0.25:.1f}s), entropy={head.attention\_entropy:.3f}, "
f"interpretation={head.interpretation}")
print(f"\n多尺度检测: {tf\_insights.n\_scales\_detected} 个时间尺度")
2\. CLI 运行：
python main.py --audio1 data/song.wav --analysis model\_analysis --output results/exp5\_transformer/
3\. 在多首音频上重复，统计各头的 lag 分布

数据需求：

- 输入格式：音频文件，≥5 秒（足够多的窗口）
- 最小样本量：≥3 个不同风格音频
- 数据来源：真实音频

预期结果：

- 若假设正确：头间主导 lag 的标准差 >5 窗口，注意力熵差异 >0.5，检测到 ≥2 个时间尺度
- 若假设错误：所有头的 lag 和熵接近，各头退化为一模一样的注意力

评估指标：

┌────────────────┬──────────────────────────┬──────────────────┬──────────────────────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├────────────────┼──────────────────────────┼──────────────────┼──────────────────────────────────────────────────┤
│ 主导 lag │ 平均注意力权重峰值位置 │ 1\~N\_windows │ AttentionHeadInfo.dominant\_lag │
├────────────────┼──────────────────────────┼──────────────────┼──────────────────────────────────────────────────┤
│ 注意力熵 │ −Σ p·log(p) 均值 │ 0\~log(N) │ AttentionHeadInfo.attention\_entropy │
├────────────────┼──────────────────────────┼──────────────────┼──────────────────────────────────────────────────┤
│ 头间余弦相似度 │ 头 × 头的注意力向量夹角 │ -1\~1, 接近 │ cosine\_similarity(head\_i\_weights, │
│ │ │ 0=多样 │ head\_j\_weights) │
├────────────────┼──────────────────────────┼──────────────────┼──────────────────────────────────────────────────┤
│ 时间尺度数 │ 检测到的不同时间尺度数量 │ 1\~4 │ TfInsights.n\_scales\_detected │
└────────────────┴──────────────────────────┴──────────────────┴──────────────────────────────────────────────────┘

可重现性：

- 随机种子：torch.manual\_seed(42)
- 环境：Python 3.8+, torch≥1.12
- 预计运行时间：每次约 3-8 分钟

***

实验 6：无监督结构边界检测实验

子问题：#6 — 音频结构变化点检测

假设：无监督变点检测能发现音频中 energy 和 rhythm
的突变位置，这些位置对应于音频的自然结构边界（如音乐的主歌/副歌过渡点）。

自变量 (IV)：检测方法 — PELT（Pruned Exact Linear Time）、Binary Segmentation、阈值法
因变量 (DV)：

- 检测到的变点数量
- 变点间距的均值/标准差
- 变点处各维度的变化幅度
  控制变量：同一音频文件、惩罚参数、最小段长

项目功能使用：

- unsupervised 模块 at audiots/unsupervised.py — 变点检测、段落标注、异常检测
- dynamics.extract\_dynamics() at audiots/dynamics.py:106 — 趋势输入

实验步骤：

1. 运行无监督分析：
   from audiots import loader, dynamics
   from audiots.unsupervised import detect\_change\_points, explore\_unsupervised

y, sr = loader.load\_audio('data/song.wav', target\_sr=16000)
dyn = dynamics.extract\_dynamics(y, sr)

# 对每个趋势维度分别检测变点

for dim\_name, trend\_series in \[('energy', dyn\['energy']), ('rhythm', dyn\['rhythm'])]:
cps = detect\_change\_points(trend\_series, method='pelt', penalty=3.0)
print(f"{dim\_name}: {len(cps)} change points at {cps}")

# 综合无监督探索

report = explore\_unsupervised(y, sr)
print(f"Segments: {len(report.segments)}")
for seg in report.segments:
print(f" \[{seg.start:.1f}s - {seg.end:.1f}s] {seg.label}: {seg.character}")
2\. CLI 运行：
python main.py --audio1 data/song.wav --analysis unsupervised,dynamics --output results/exp6\_unsupervised/
3\. 与 dynamics.detect\_structural\_segments() at audiots/dynamics.py:208 的结果进行交叉验证

数据需求：

- 输入格式：有明确结构的音频文件
- 最小样本量：≥3 个音乐文件（有主歌/副歌结构）
- 数据来源：用户提供的标注音乐

预期结果：

- 若假设正确：变点检测在 energy 维度找到 3-5 个显著变点，对应 music structure boundaries；rhythm
  维度的变点数量可能不同
- 若假设错误：变点数量为 0（无显著变化）或过多（每个窗口都是变点）

评估指标：

┌──────────────────────────────┬────────────────────────────┬────────────────┬─────────────────────────────────┐
│ 指标 │ 定义 │ 预期范围 │ 计算方法 │
├──────────────────────────────┼────────────────────────────┼────────────────┼─────────────────────────────────┤
│ 变点数 │ 检测到的显著变化点数 │ 2\~10/分钟 │ len(change\_points) │
├──────────────────────────────┼────────────────────────────┼────────────────┼─────────────────────────────────┤
│ 平均段长 │ 相邻变点的时间间距 │ 取决于音频结构 │ np.mean(np.diff(change\_points)) │
├──────────────────────────────┼────────────────────────────┼────────────────┼─────────────────────────────────┤
│ 变点显著性 │ 变点处多维度的联合变化幅度 │ >阈值 │ unsupervised 模块内部计算 │
├──────────────────────────────┼────────────────────────────┼────────────────┼─────────────────────────────────┤
│ 与 dynamics segment 的重合度 │ 两种方法的交叠程度 │ >50% │ 自定义比较逻辑 │
└──────────────────────────────┴────────────────────────────┴────────────────┴─────────────────────────────────┘

可重现性：

- 随机种子：np.random.seed(42)
- 环境：Python 3.8+, scikit-learn≥1.0
- 预计运行时间：每次约 1-2 分钟

***

1. 能力缺口

本项目所有 6 个实验子问题均由现有模块直接支持，不存在能力缺口。

但存在以下增强性建议（非阻塞性）：

┌──────────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────┐
│ 项目 │ 现状 │ 影响 │
├──────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 实验结果的统计显著性 │ analysis.py 有白噪声/平稳性检验，但缺少跨实 │ 实验 3-6 的结果解释主要依赖描述性统计， │
│ 检验 │ 验的标准化统计报告（如 t 检验、效应量 │ 缺少推断性统计支撑 │
│ │ Cohen's d） │ │
├──────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 实验自动化脚本 │ 每次实验需手动编写 Python 代码或 CLI 命令 │ 可复现性依赖人工记录，容易出错 │
├──────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
│ 多音频批量实验对比 │ batch\_analyze.py │ 无法用一条命令完成"3 个音频 × 4 │
│ │ 提供批量分析但面向文件夹扫描，不是实验设计 │ 个状态数"的网格实验 │
└──────────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────┘

以上为优化建议，不影响当前实验的执行。

***

1. 实验执行建议

快速开始

# 单个音频完整时间序列分析（覆盖实验 1-6 的所有数据采集）

python main.py \
\--audio1 data/your\_audio.wav \
\--analysis
features,dynamics,dynamics\_analysis,model\_analysis,timeseries,unsupervised,band,prediction,visualization \
\--save-json \
\--output results/time\_series\_experiment/

# 使用合成音频快速验证

python main.py --analysis model\_analysis,dynamics\_analysis,timeseries,unsupervised --output results/quick\_test/

结果文件导航

执行后查看以下关键输出：

- results/\*/dynamics/trends.png — 4 趋势叠加图（实验 1）
- results/*/volatility/garch\_*.png — GARCH 条件波动率（实验 2）
- results/\*/results.json — 结构化数据（所有实验的原始数据）
- 控制台输出 — Model Ensemble 综合报告（实验 1/4/5/6）

实验矩阵

┌─────────────┬──────────────────────────────┬────────────────────────────────┬──────────┐
│ 实验 │ CLI 选项 │ 关键输出 │ 预计时间 │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #1 趋势结构 │ --analysis model\_analysis │ 控制台 ARIMA 报告 + JSON │ \~30s │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #2 波动聚集 │ --analysis dynamics\_analysis │ volatility/\*.png + JSON │ \~60s │
\--save-json \
\--output results/time\_series\_experiment/

# 使用合成音频快速验证

python main.py --analysis model\_analysis,dynamics\_analysis,timeseries,unsupervised --output results/quick\_test/

结果文件导航

执行后查看以下关键输出：

- results/\*/dynamics/trends.png — 4 趋势叠加图（实验 1）
- results/*/volatility/garch\_*.png — GARCH 条件波动率（实验 2）
- results/\*/results.json — 结构化数据（所有实验的原始数据）
- 控制台输出 — Model Ensemble 综合报告（实验 1/4/5/6）

实验矩阵

┌─────────────┬──────────────────────────────┬────────────────────────────────┬──────────┐
│ 实验 │ CLI 选项 │ 关键输出 │ 预计时间 │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #1 趋势结构 │ --analysis model\_analysis │ 控制台 ARIMA 报告 + JSON │ \~30s │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #2 波动聚集 │ --analysis dynamics\_analysis │ volatility/\*.png + JSON │ \~60s │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #3 频带记忆 │ --analysis band,prediction │ 控制台频带排名 + JSON │ \~5-15min │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #4 隐状态 │ --analysis model\_analysis │ 控制台 HMM 报告 + JSON │ \~60s │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #5 注意力 │ --analysis model\_analysis │ 控制台 Transformer 报告 + JSON │ \~3-8min │
├─────────────┼──────────────────────────────┼────────────────────────────────┼──────────┤
│ #6 结构边界 │ --analysis unsupervised │ 控制台段落报告 + JSON │ \~2min │
└─────────────┴──────────────────────────────┴────────────────────────────────┴──────────┘

***

质量检查清单

- 每个声称的项目能力都有 文件:行号 引用
- 每个实验步骤都是具体的命令或代码片段
- 没有实验引用不存在的功能
- 缺口分析诚实评估了可行性（无缺口）
- 不确定的地方已标注为优化建议
- 没有填充、没有捏造、没有伪装成事实的推测

