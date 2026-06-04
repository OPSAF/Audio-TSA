<#
Audio Lab Web App 启动脚本
Usage:
    .\run_app.ps1
#>

param(
    [string]$PythonPath = "C:\Users\27862\anaconda3\envs\P\python.exe",
    [string]$ScriptPath = "app.py"
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "启动 Audio Lab Web 应用" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

if (-not (Test-Path $PythonPath)) {
    Write-Host "找不到 Python 可执行文件: $PythonPath" -ForegroundColor Red
    Write-Host "请修改 run_app.ps1 中的 PythonPath 为你的 Python 路径。" -ForegroundColor Yellow
    exit 1
}

Write-Host "使用 Python: $PythonPath" -ForegroundColor Green
Write-Host "运行脚本: $ScriptPath" -ForegroundColor Green
Write-Host ""

& "$PythonPath" "$ScriptPath"

Write-Host ""
Write-Host "应用已退出。" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
