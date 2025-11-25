#!/bin/bash

# 视频流检测系统启动脚本
# 用途：快速启动检测系统

echo "========================================="
echo "  视频流检测系统 - 启动中..."
echo "========================================="
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

echo "✓ Python 版本: $(python3 --version)"
echo ""

# 检查依赖
echo "检查依赖包..."
python3 -c "import cv2, yaml, mediapipe, ultralytics, PyQt5, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告: 部分依赖包未安装"
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ 依赖安装失败，请手动执行: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "✓ 依赖检查完成"
echo ""

# 检查权重文件
if [ ! -f "weight/best.pt" ]; then
    echo "⚠️  警告: 未找到模型权重文件 weight/best.pt"
    echo "请将模型文件放到 weight/ 目录下"
    echo ""
    read -p "是否继续启动？(y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "❌ 错误: 未找到配置文件 config.yaml"
    exit 1
fi

echo "✓ 配置文件检查完成"
echo ""

# 创建报警目录
ALARM_DIR=$(grep "alarm_dir:" config.yaml | awk '{print $2}' | tr -d "'\"")
if [ -z "$ALARM_DIR" ]; then
    ALARM_DIR="alarms"
fi

mkdir -p "$ALARM_DIR"
echo "✓ 报警目录: $ALARM_DIR"
echo ""

# 显示配置信息
echo "========================================="
echo "配置信息:"
echo "========================================="
DEVICE=$(grep "device:" config.yaml | head -1 | awk '{print $2}' | tr -d "'\"")
CONFIDENCE=$(grep "confidence:" config.yaml | head -1 | awk '{print $2}' | tr -d "'\"")
STREAM_COUNT=$(grep -c "name:" config.yaml)

echo "  设备: $DEVICE"
echo "  置信度: $CONFIDENCE"
echo "  视频流数量: $STREAM_COUNT"
echo "========================================="
echo ""

# 启动程序
echo "🚀 启动检测系统..."
echo "按 Ctrl+C 停止程序"
echo ""

# 使用 nohup 在后台运行（可选）
# nohup python3 detect_video.py > output.log 2>&1 &
# echo "系统已在后台启动，日志文件: output.log"
# echo "进程ID: $!"

# 前台运行
python3 detect_video.py

echo ""
echo "========================================="
echo "  系统已退出"
echo "========================================="
