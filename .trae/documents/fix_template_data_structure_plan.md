# 修复模板数据结构对齐问题

## 问题分析

之前我们修改了 app.py 的数据结构，使其与 main.py 保持一致，但忘记更新相应的模板文件，导致出现了 `jinja2.exceptions.UndefinedError` 错误。

## 需要修复的内容

### 1. 时间序列分析相关访问
- `results['periodicity']` → `results['timeseries']['periodicity']`
- `results['complexity']` → `results['timeseries']['complexity']`
- `results['spectral_flatness']` → `results['timeseries']['spectral_flatness']`

### 2. 预测相关访问
- `results['predictions']` → `results['prediction']`

## 修改文件

- `templates/results.html`

## 修复步骤

1. 更新时间序列分析数据访问路径
2. 更新预测数据访问路径
3. 验证是否有其他需要更新的地方
