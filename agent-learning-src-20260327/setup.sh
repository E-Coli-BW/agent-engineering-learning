#!/bin/bash
# ============================================================
# Agent Learning 项目一键安装脚本
# ============================================================
# 用法: bash setup.sh
# ============================================================

set -e

echo "🧠 Agent Learning — 一键安装"
echo "================================"
echo ""

# ---- 检查 Python ----
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION"

# ---- 检查 Ollama ----
if command -v ollama &> /dev/null; then
    echo "✅ Ollama 已安装"
else
    echo "⚠️  Ollama 未安装"
    echo "   请访问 https://ollama.com 安装后再运行本脚本"
    echo "   macOS: brew install ollama"
    exit 1
fi

# ---- 创建虚拟环境 ----
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 创建虚拟环境..."
    python3 -m venv .venv
fi

echo "📦 激活虚拟环境..."
source .venv/bin/activate

# ---- 安装依赖 ----
echo ""
echo "📦 安装 Python 依赖..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q fastapi uvicorn qrcode

echo ""
echo "✅ Python 依赖安装完成"

# ---- 拉取 Ollama 模型 ----
echo ""
echo "🤖 检查 Ollama 模型..."

if ollama list 2>/dev/null | grep -q "qwen2.5:7b"; then
    echo "✅ qwen2.5:7b 已存在"
else
    echo "📥 下载 qwen2.5:7b (约 4.4GB)..."
    ollama pull qwen2.5:7b
fi

if ollama list 2>/dev/null | grep -q "mxbai-embed-large"; then
    echo "✅ mxbai-embed-large 已存在"
else
    echo "📥 下载 mxbai-embed-large (约 670MB)..."
    ollama pull mxbai-embed-large
fi

# ---- 创建数据目录 ----
mkdir -p data/wechat_bridge
mkdir -p outputs

# ---- 完成 ----
echo ""
echo "============================================================"
echo "🎉 安装完成！"
echo "============================================================"
echo ""
echo "快速开始:"
echo ""
echo "  1. 激活虚拟环境:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. 启动 Expert Agent (终端 1):"
echo "     python project/a2a_agent.py --serve"
echo ""
echo "  3. 启动微信桥接器 (终端 2):"
echo "     python project/wechat_bridge.py"
echo ""
echo "  4. 直接 CLI 测试 (无需微信):"
echo "     python project/a2a_agent.py '什么是Transformer?'"
echo ""
echo "  5. 启动 API Server:"
echo "     python project/api_server.py"
echo ""
echo "更多信息: README.md"
echo ""
