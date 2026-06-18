@echo off
chcp 65001 >nul
title 音乐分析报告查看器

echo.
echo   ============================================
echo     🎵 音乐流派分析报告
echo   ============================================
echo.
echo   [提示] 推荐直接双击 viewer_standalone.html
echo          无需服务器，即开即看！
echo.
echo   --------------------------------------------
echo   如需使用在线版 viewer.html，请选择服务器：
echo   --------------------------------------------
echo.
echo   [1] 尝试 Python / Node.js 服务器
echo   [2] 直接打开 viewer_standalone.html（推荐）
echo.
choice /c 12 /n /m "请选择 [1] 或 [2]: "

if errorlevel 2 goto :standalone
if errorlevel 1 goto :server

:standalone
start "" viewer_standalone.html
goto :end

:server
set PORT=8080

:: Try py
where py >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] 使用 Python (py) 启动服务器...
    echo   URL: http://localhost:%PORT%/viewer.html
    echo   按 Ctrl+C 停止服务器
    start "" http://localhost:%PORT%/viewer.html
    py -m http.server %PORT%
    goto :end
)

:: Try python3
where python3 >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] 使用 Python3 启动服务器...
    echo   URL: http://localhost:%PORT%/viewer.html
    start "" http://localhost:%PORT%/viewer.html
    python3 -m http.server %PORT%
    goto :end
)

:: Try Node.js
where npx >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] 使用 Node.js 启动服务器...
    echo   URL: http://localhost:%PORT%/viewer.html
    start "" http://localhost:%PORT%/viewer.html
    npx serve . -p %PORT% --no-clipboard
    goto :end
)

echo   [FAIL] 未找到 Python 或 Node.js！
echo   改为直接打开 standalone 版本...
start "" viewer_standalone.html

:end

