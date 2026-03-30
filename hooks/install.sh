#!/usr/bin/env bash
# 安装 git hooks
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cp "$ROOT_DIR/hooks/pre-commit" "$ROOT_DIR/.git/hooks/pre-commit"
chmod +x "$ROOT_DIR/.git/hooks/pre-commit"

cp "$ROOT_DIR/hooks/pre-push" "$ROOT_DIR/.git/hooks/pre-push"
chmod +x "$ROOT_DIR/.git/hooks/pre-push"

echo "✅ Git hooks 已安装:"
echo "   pre-commit: Code Review + pytest (每次 commit)"
echo "   pre-push:   阻止直接 push master (提醒走 PR)"
echo ""
echo "   跳过: git commit --no-verify / git push --no-verify"
