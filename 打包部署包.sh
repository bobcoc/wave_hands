#!/bin/bash

# 部署包打包脚本
# 用途：将部署包打包成tar.gz文件，方便传输

PACKAGE_NAME="wave_hands_deploy_$(date +%Y%m%d_%H%M%S)"
PACKAGE_FILE="${PACKAGE_NAME}.tar.gz"

echo "========================================="
echo "  部署包打包工具"
echo "========================================="
echo ""

# 检查部署包目录
if [ ! -d "deploy_package" ]; then
    echo "❌ 错误: 未找到 deploy_package 目录"
    exit 1
fi

echo "📦 正在打包部署包..."
echo "目标文件: ${PACKAGE_FILE}"
echo ""

# 打包
tar -czf "${PACKAGE_FILE}" deploy_package/

if [ $? -eq 0 ]; then
    FILE_SIZE=$(du -h "${PACKAGE_FILE}" | cut -f1)
    echo "✓ 打包成功!"
    echo ""
    echo "========================================="
    echo "文件信息:"
    echo "========================================="
    echo "  文件名: ${PACKAGE_FILE}"
    echo "  大小: ${FILE_SIZE}"
    echo "  位置: $(pwd)/${PACKAGE_FILE}"
    echo "========================================="
    echo ""
    echo "📤 传输命令示例:"
    echo ""
    echo "# 通过SCP传输到服务器:"
    echo "scp ${PACKAGE_FILE} user@server:/path/to/destination/"
    echo ""
    echo "# 解压命令:"
    echo "tar -xzf ${PACKAGE_FILE}"
    echo ""
else
    echo "❌ 打包失败"
    exit 1
fi
