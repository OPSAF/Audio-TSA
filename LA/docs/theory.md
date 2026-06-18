# 理论基础

本文档详细介绍音频时间序列分析的理论基础，从声波物理到数字信号处理，再到时间序列分析方法。

---

## 一、声音的物理基础

### 1.1 声波的本质

声音是空气分子的振动产生的机械波。声波可以用以下参数描述：

- **频率（f）**：每秒振动次数，单位 Hz
- **振幅（A）**：振动的最大位移，决定响度
- **相位（φ）**：振动的初始状态
- **波长（λ）**：一个完整波形的空间长度

**声波的数学表示：**

$$y(t) = A \sin(2\pi f t + \phi)$$

其中：
- $y(t)$ 是时刻 $t$ 的位移
- $A$ 是振幅
- $f$ 是频率
- $\phi$ 是初相位

### 1.2 人耳听觉范围

- **频率范围**：20 Hz ~ 20,000 Hz
- **声压范围**：0 dB ~ 120 dB（痛阈）
- **等响曲线**：人耳对不同频率的响度感知不同

### 1.3 傅里叶级数展开

任何周期信号都可以分解为正弦波的叠加：

$$x(t) = a_0 + \sum_{n=1}^{\infty} [a_n \cos(2\pi n f_0 t) + b_n \sin(2\pi n f_0 t)]$$

其中：
- $f_0$ 是基频
- $a_n, b_n$ 是傅里叶系数

**傅里叶系数计算：**

$$a_n = \frac{2}{T} \int_0^T x(t) \cos(2\pi n f_0 t) \, dt$$

$$b_n = \frac{2}{T} \int_0^T x(t) \sin(2\pi n f_0 t) \, dt$$

---

## 二、数字信号处理

### 2.1 采样定理（Nyquist-Shannon）

**采样定理**：为了无失真地恢复原始信号，采样频率必须至少是信号最高频率的两倍。

$$f_s \geq 2 f_{max}$$

其中：
- $f_s$ 是采样频率
- $f_{max}$ 是信号的最高频率分量

**实际应用：**
- 人耳听觉上限 20 kHz，所以 CD 采样率为 44.1 kHz
- 语音信号通常使用 16 kHz 采样率（覆盖 8 kHz 以下频率）

### 2.2 离散傅里叶变换（DFT）

将连续傅里叶变换离散化：

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} kn}$$

其中：
- $x[n]$ 是离散时域信号
- $X[k]$ 是离散频域信号
- $N$ 是信号长度

**逆变换（IDFT）：**

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j \frac{2\pi}{N} kn}$$

### 2.3 快速傅里叶变换（FFT）

FFT 是 DFT 的高效算法，将计算复杂度从 $O(N^2)$ 降低到 $O(N \log N)$。

**Cooley-Tukey 算法原理：**

将 $N$ 点 DFT 分解为两个 $N/2$ 点 DFT：

$$X[k] = \sum_{m=0}^{N/2-1} x[2m] W_N^{2mk} + W_N^k \sum_{m=0}^{N/2-1} x[2m+1] W_N^{2mk}$$

其中 $W_N = e^{-j \frac{2\pi}{N}}$ 是旋转因子。

### 2.4 短时傅里叶变换（STFT）

STFT 将信号分成短帧，对每帧进行 FFT，得到时频表示：

$$X(t, f) = \int_{-\infty}^{\infty} x(\tau) w(\tau - t) e^{-j 2\pi f \tau} \, d\tau$$

其中：
- $w(\tau - t)$ 是以 $t$ 为中心的窗函数
- 常用窗函数：汉宁窗、汉明窗、布莱克曼窗

**参数选择：**
- **窗口大小（n_fft）**：通常 2048 或 4096
- **跳跃长度（hop_length）**：通常 n_fft / 4
- **时间分辨率**：hop_length / sample_rate
- **频率分辨率**：sample_rate / n_fft

---

## 三、音频特征提取

### 3.1 Mel 滤波器组

Mel 刻度模拟人耳对频率的非线性感知：

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$

**逆变换：**

$$f = 700 \left(10^{m/2595} - 1\right)$$

Mel 滤波器组将线性频率转换为 Mel 刻度，低频分辨率高，高频分辨率低。

### 3.2 Mel 语谱图

Mel 语谱图是音频分析的核心特征：

1. 计算 STFT 得到功率谱
2. 应用 Mel 滤波器组
3. 取对数得到 dB 单位

$$S_{mel}[m, t] = \log\left(\sum_{k} |X[t, k]|^2 \cdot H_m[k]\right)$$

其中 $H_m[k]$ 是第 $m$ 个 Mel 滤波器。

### 3.3 MFCC（Mel 倒谱系数）

MFCC 进一步压缩 Mel 语谱图：

1. 计算 Mel 语谱图
2. 应用离散余弦变换（DCT）
3. 保留低阶系数（通常 13-20 个）

$$c_n = \sum_{m=0}^{M-1} S_{mel}[m] \cos\left(\frac{\pi n (m + 0.5)}{M}\right)$$

**MFCC 的物理意义：**
- $c_0$：能量
- $c_1, c_2$：频谱倾斜
- $c_3 \sim c_{12}$：频谱细节

---

## 四、时间序列分析

### 4.1 自相关函数（ACF）

自相关函数衡量信号与其延迟版本的相关性：

$$\rho_k = \frac{\sum_{t=k+1}^{n} (x_t - \bar{x})(x_{t-k} - \bar{x})}{\sum_{t=1}^{n} (x_t - \bar{x})^2}$$

**性质：**
- $\rho_0 = 1$
- $-1 \leq \rho_k \leq 1$
- 周期信号的 ACF 也是周期的

**白噪声的 ACF：**

$$\rho_k = \begin{cases} 1 & k = 0 \\ 0 & k \neq 0 \end{cases}$$

### 4.2 偏自相关函数（PACF）

PACF 衡量在移除中间滞后影响后的直接相关性：

$$\phi_{kk} = \text{Corr}(x_t, x_{t-k} | x_{t-1}, \ldots, x_{t-k+1})$$

**Yule-Walker 方程：**

$$\rho_k = \sum_{j=1}^{p} \phi_j \rho_{k-j}$$

### 4.3 ARIMA 模型

ARIMA(p, d, q) 模型：

$$\phi(B) (1-B)^d x_t = \theta(B) \epsilon_t$$

其中：
- $B$ 是后移算子：$B x_t = x_{t-1}$
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$（AR 部分）
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$（MA 部分）
- $d$ 是差分阶数

**模型定阶：**
- **AR(p)**：PACF 在 p 阶后截尾
- **MA(q)**：ACF 在 q 阶后截尾
- **ARMA(p,q)**：ACF 和 PACF 都拖尾

### 4.4 白噪声检验

#### Ljung-Box 检验

检验多个滞后阶的自相关是否联合为零：

$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}$$

在 $H_0$（白噪声）下，$Q \sim \chi^2_h$

#### Jarque-Bera 正态性检验

检验数据是否服从正态分布：

$$JB = \frac{n}{6} \left(S^2 + \frac{(K-3)^2}{4}\right)$$

其中：
- $S$ 是偏度
- $K$ 是峰度
- 在正态分布下，$JB \sim \chi^2_2$

---

## 五、波动率建模

### 5.1 ARCH 模型

自回归条件异方差模型：

$$\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \cdots + \alpha_q \epsilon_{t-q}^2$$

适用于波动聚集现象。

### 5.2 GARCH(1,1) 模型

广义 ARCH 模型：

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**参数约束：**
- $\omega > 0$
- $\alpha \geq 0, \beta \geq 0$
- $\alpha + \beta < 1$（平稳性条件）

**持久性：**
- $\alpha + \beta$ 接近 1 表示高持久性
- 波动冲击的影响持续时间长

---

## 六、谱分析

### 6.1 功率谱密度（PSD）

功率谱密度是自相关函数的傅里叶变换：

$$S(f) = \sum_{k=-\infty}^{\infty} \rho_k e^{-j 2\pi f k}$$

**Wiener-Khinchin 定理：**

$$S(f) = \lim_{T \to \infty} \frac{1}{T} \left| \int_{-T/2}^{T/2} x(t) e^{-j 2\pi f t} \, dt \right|^2$$

### 6.2 谱平坦度（Spectral Flatness）

衡量频谱的"平坦程度"：

$$SF = \frac{\exp\left(\frac{1}{N}\sum_k \ln S_k\right)}{\frac{1}{N}\sum_k S_k}$$

**解释：**
- $SF \approx 1$：类似白噪声（平坦频谱）
- $SF \approx 0$：有明显的音调成分

---

## 七、信息论与复杂度

### 7.1 样本熵（Sample Entropy）

衡量时间序列的规则性和可预测性：

$$SampEn(m, r, N) = -\ln \frac{A}{B}$$

其中：
- $A$：匹配 $m+1$ 个点的模板对数
- $B$：匹配 $m$ 个点的模板对数
- $r$：容差阈值（通常 $r = 0.2 \times \text{std}$）

**解释：**
- 低样本熵：规则、可预测
- 高样本熵：复杂、随机

### 7.2 零交叉率

信号穿过零点的频率：

$$ZCR = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{1}[\text{sgn}(x_n) \neq \text{sgn}(x_{n-1})]$$

**应用：**
- 高 ZCR：高频成分多
- 低 ZCR：低频成分主导

---

## 八、动态时间规整（DTW）

### 8.1 DTW 距离

衡量两个不等长序列的相似度：

$$D(i,j) = d(x_i, y_j) + \min\{D(i-1,j), D(i,j-1), D(i-1,j-1)\}$$

**应用场景：**
- 语音识别：处理语速差异
- 音频相似度：忽略时间扭曲

---

返回 [文档首页](/docs) | [分析模块详解](/docs/analysis)
