@echo off
set PYTHON_PATH=C:\Users\27862\anaconda3\envs\P\python.exe
set SCRIPT_PATH=app.py

echo ================================
echo 启动 Audio Lab Web 应用
echo ================================

if not exist "%PYTHON_PATH%" (
    echo 找不到 Python 可执行文件: %PYTHON_PATH%
    echo 请修改 run_app.bat 中的 PYTHON_PATH 为你的 Python 路径。
    pause
    exit /b 1
)

%PYTHON_PATH% %SCRIPT_PATH%

pause
