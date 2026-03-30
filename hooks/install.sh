#!/usr/bin/env bash
# 安装 git hooks
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cp "$ROOT_DIR/hooks/pre-commit" "$ROOT_DIR/.git/hooks/pre-commit"
chmod +x "$ROOT_DIR/.git/hooks/pre-commit"
echo "✅ Git pre-commit hook 已安装"
echo "   每次 git commit 前会自动运行 Code Review + 测试"
echo "   跳过: git commit --no-verify"
