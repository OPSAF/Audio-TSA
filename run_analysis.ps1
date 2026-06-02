<#
Audio Time Series Analysis - PowerShell Script
Usage: .\run_analysis.ps1 [-Audio1 <path>] [-Audio2 <path>] [-NoSave]
#>

param(
    [string]$Audio1 = $null,
    [string]$Audio2 = $null,
    [switch]$NoSave
)

$PYTHON_PATH = "C:\Users\27862\anaconda3\envs\P\python.exe"
$SCRIPT_PATH = "main.py"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Audio Time Series Analysis & Prediction" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$argsList = @()

if ($Audio1) {
    $argsList += "--audio1", "`"$Audio1`""
    Write-Host "音频文件1: $Audio1" -ForegroundColor Green
}

if ($Audio2) {
    $argsList += "--audio2", "`"$Audio2`""
    Write-Host "音频文件2: $Audio2" -ForegroundColor Green
}

if ($NoSave) {
    $argsList += "--no-save"
    Write-Host "模式: 不保存图片" -ForegroundColor Yellow
}

if (-not $Audio1 -and -not $Audio2) {
    Write-Host "模式: 使用合成音频" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "正在运行分析... (可能需要1-2分钟)" -ForegroundColor White

$fullCommand = "& `"$PYTHON_PATH`" `"$SCRIPT_PATH`""
if ($argsList) {
    $fullCommand += " " + ($argsList -join " ")
}

Invoke-Expression $fullCommand

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "分析完成！" -ForegroundColor Green
if (-not $NoSave) {
    Write-Host "结果已保存到 ./outputs/ 目录" -ForegroundColor Green
}
Write-Host "================================================" -ForegroundColor Cyan