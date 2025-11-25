@echo off
chcp 65001 >nul
REM 视频流检测系统启动脚本 (Windows)

echo =========================================
echo   视频流检测系统 - 启动中...
echo =========================================
echo.

REM 检查Python环境
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

echo ✓ Python 已安装
python --version
echo.

REM 检查依赖
echo 检查依赖包...
python -c "import cv2, yaml, mediapipe, ultralytics, PyQt5, requests" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️  警告: 部分依赖包未安装
    echo 正在安装依赖...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ 依赖安装失败，请手动执行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo ✓ 依赖检查完成
echo.

REM 检查权重文件
if not exist "weight\best.pt" (
    echo ⚠️  警告: 未找到模型权重文件 weight\best.pt
    echo 请将模型文件放到 weight\ 目录下
    echo.
    set /p continue="是否继续启动？(y/n) "
    if /i not "%continue%"=="y" exit /b 1
)

REM 检查配置文件
if not exist "config.yaml" (
    echo ❌ 错误: 未找到配置文件 config.yaml
    pause
    exit /b 1
)

echo ✓ 配置文件检查完成
echo.

REM 创建报警目录
if not exist "alarms" mkdir alarms
echo ✓ 报警目录: alarms
echo.

echo =========================================
echo 配置信息已加载
echo =========================================
echo.

REM 启动程序
echo 🚀 启动检测系统...
echo 按 Ctrl+C 停止程序
echo.

python detect_video.py

echo.
echo =========================================
echo   系统已退出
echo =========================================
pause
