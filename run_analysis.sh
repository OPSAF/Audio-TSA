#!/bin/bash
# Audio Time Series Analysis - Bash Script
# Usage: ./run_analysis.sh [audio_file]

PYTHON_PATH="/c/Users/27862/anaconda3/envs/P/python.exe"
SCRIPT_PATH="main.py"

echo "================================================"
echo "Audio Time Series Analysis & Prediction"
echo "================================================"

if [ -z "$1" ]; then
    echo "Mode: 使用合成音频进行分析"
    $PYTHON_PATH $SCRIPT_PATH
else
    echo "Mode: 分析音频文件: $1"
    $PYTHON_PATH $SCRIPT_PATH --audio1 "$1"
fi

echo ""
echo "================================================"
echo "分析完成！结果已保存到 ./outputs/ 目录"
echo "================================================"