@echo off
set PYTHON_PATH=C:\Users\27862\anaconda3\envs\P\python.exe
set SCRIPT_PATH=main.py

echo ================================
echo Audio Time Series Analysis
echo ================================

if "%1"=="" (
    echo Mode: Synthetic Audio
    %PYTHON_PATH% %SCRIPT_PATH%
) else (
    echo Mode: Analyzing: %1
    %PYTHON_PATH% %SCRIPT_PATH% --audio1 "%1"
)

pause