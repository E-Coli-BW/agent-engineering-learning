"""
Code Review Agent — 自动审查代码变更
====================================

两种使用方式:
  1. CLI: PYTHONPATH=. python project/review/reviewer.py
     → 审查当前 git 暂存区 (staged) 的变更

  2. Git pre-commit hook:
     → 每次 git commit 前自动运行

审查维度:
  - 代码风格 (命名规范, import 顺序)
  - 测试覆盖 (新代码是否有对应测试)
  - 安全检查 (hardcode 密码/路径, eval 使用)
  - 文档完整性 (docstring, 注释)
  - 架构一致性 (是否遵循项目约定)

注意: 这个 Agent 不依赖 Ollama，用纯规则检查 + 可选 LLM 审查。
这样即使 Ollama 没运行也能用。
"""

import os
import re
import sys
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger("review")


@dataclass
class ReviewIssue:
    """一个审查问题"""
    severity: str  # error | warning | info
    file: str
    line: int
    rule: str
    message: str

    def __str__(self):
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(self.severity, "?")
        return f"  {icon} [{self.rule}] {self.file}:{self.line} — {self.message}"


class CodeReviewer:
    """代码审查器"""

    def __init__(self):
        self.issues: list[ReviewIssue] = []

    def review_staged(self) -> list[ReviewIssue]:
        """审查 git staged 的变更"""
        self.issues = []

        # 获取 staged 文件
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True, text=True,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

        if not files:
            logger.info("没有 staged 的文件")
            return []

        logger.info("审查 %d 个文件: %s", len(files), files)

        for file_path in files:
            if file_path.endswith(".py"):
                self._review_python(file_path)
            elif file_path.endswith(".java"):
                self._review_java(file_path)
            elif file_path.endswith(".tsx") or file_path.endswith(".ts"):
                self._review_typescript(file_path)

        # 检查测试覆盖
        self._check_test_coverage(files)

        # 检查安全问题
        self._check_security(files)

        return self.issues

    def _review_python(self, file_path: str):
        """Python 代码审查"""
        try:
            content = Path(file_path).read_text()
            lines = content.split("\n")
        except Exception:
            return

        # 检查 docstring
        if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
            if file_path.startswith("project/") and "__init__" not in file_path:
                self.issues.append(ReviewIssue(
                    "warning", file_path, 1, "missing-docstring",
                    "模块缺少 docstring (项目约定: 每个 Python 文件开头需要 docstring)",
                ))

        for i, line in enumerate(lines, 1):
            # 检查 hardcoded localhost (除了默认值)
            if "localhost" in line and "getenv" not in line and "default" not in line.lower():
                if not line.strip().startswith("#") and not line.strip().startswith('"""'):
                    self.issues.append(ReviewIssue(
                        "warning", file_path, i, "hardcoded-url",
                        "硬编码 localhost，应该用 os.getenv() + 默认值",
                    ))

            # 检查 bare except
            if re.match(r'\s*except\s*:', line):
                self.issues.append(ReviewIssue(
                    "error", file_path, i, "bare-except",
                    "不要用 bare except，至少 except Exception:",
                ))

            # 检查 print (生产代码应该用 logger)
            if file_path.startswith("project/") and re.match(r'\s*print\(', line):
                if "__main__" not in content[max(0, content.rfind("if __name__")):]:
                    self.issues.append(ReviewIssue(
                        "info", file_path, i, "print-in-prod",
                        "生产代码建议用 logger 替代 print",
                    ))

    def _review_java(self, file_path: str):
        """Java 代码审查"""
        try:
            content = Path(file_path).read_text()
            lines = content.split("\n")
        except Exception:
            return

        has_javadoc = "/**" in content
        if not has_javadoc and "Controller" in file_path:
            self.issues.append(ReviewIssue(
                "warning", file_path, 1, "missing-javadoc",
                "Controller 类缺少 Javadoc",
            ))

        for i, line in enumerate(lines, 1):
            # 检查 System.out.println
            if "System.out.println" in line:
                self.issues.append(ReviewIssue(
                    "warning", file_path, i, "sysout",
                    "用 @Slf4j log.info() 替代 System.out.println",
                ))

    def _review_typescript(self, file_path: str):
        """TypeScript 代码审查"""
        try:
            content = Path(file_path).read_text()
            lines = content.split("\n")
        except Exception:
            return

        for i, line in enumerate(lines, 1):
            # 检查 console.log
            if "console.log" in line and not line.strip().startswith("//"):
                self.issues.append(ReviewIssue(
                    "info", file_path, i, "console-log",
                    "记得删除 console.log",
                ))

            # 检查 any 类型
            if ": any" in line or "as any" in line:
                self.issues.append(ReviewIssue(
                    "warning", file_path, i, "typescript-any",
                    "避免使用 any 类型",
                ))

    def _check_test_coverage(self, files: list[str]):
        """检查新增/修改的代码是否有对应测试"""
        py_files = [f for f in files if f.endswith(".py")
                    and f.startswith("project/")
                    and "test" not in f
                    and "__init__" not in f]

        test_files = set(f for f in files if f.startswith("tests/"))

        for f in py_files:
            module_name = Path(f).stem
            expected_test = f"tests/test_{module_name}.py"
            has_test = any(module_name in tf for tf in test_files) or Path(expected_test).exists()
            if not has_test:
                self.issues.append(ReviewIssue(
                    "warning", f, 0, "no-test",
                    f"新增/修改的文件没有对应测试 (建议: {expected_test})",
                ))

    def _check_security(self, files: list[str]):
        """安全检查"""
        for file_path in files:
            try:
                content = Path(file_path).read_text()
            except Exception:
                continue

            for i, line in enumerate(content.split("\n"), 1):
                # 密码/密钥硬编码
                if re.search(r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']', line, re.I):
                    if "getenv" not in line and "default" not in line.lower() and "test" not in file_path:
                        self.issues.append(ReviewIssue(
                            "error", file_path, i, "hardcoded-secret",
                            "疑似硬编码密码/密钥，应使用环境变量",
                        ))

                # eval 使用
                if "eval(" in line and "safe" not in line.lower():
                    self.issues.append(ReviewIssue(
                        "warning", file_path, i, "eval-usage",
                        "eval() 有安全风险，确认输入已验证",
                    ))


def print_report(issues: list[ReviewIssue]):
    """打印审查报告"""
    if not issues:
        print("✅ Code Review 通过 — 没有发现问题")
        return

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    print(f"\n📋 Code Review Report: {len(issues)} 个问题")
    print(f"   ❌ {len(errors)} errors | ⚠️ {len(warnings)} warnings | ℹ️ {len(infos)} info")
    print()

    for issue in sorted(issues, key=lambda i: ({"error": 0, "warning": 1, "info": 2}[i.severity], i.file)):
        print(str(issue))

    print()
    if errors:
        print("❌ 有 error 级别问题，建议修复后再 commit")
    else:
        print("⚠️ 有 warning，建议检查但可以 commit")


# ============================================================
# LLM Code Review (可选, 需要 Ollama 运行)
# ============================================================

LLM_REVIEW_PROMPT = """你是一个资深代码审查员。请审查以下 git diff，从这几个维度给出简洁的反馈:

1. **Bug 风险**: 有没有明显的逻辑错误、边界情况遗漏、空指针风险?
2. **性能**: 有没有 O(n²) 循环、不必要的重复计算、内存泄漏?
3. **安全**: 有没有注入风险、硬编码密钥、不安全的反序列化?
4. **可读性**: 命名是否清晰、函数是否过长、有没有魔法数字?
5. **测试**: 这个改动是否需要新增或修改测试?

规则:
- 只指出有问题的地方，不要说"看起来没问题"
- 每个问题一行，格式: [severity] 文件:行号 — 描述
  severity: 🔴 BUG | 🟡 WARNING | 🔵 SUGGESTION
- 如果没有问题，只说 "LGTM ✅"
- 用中文回答

Git Diff:
```
{diff}
```
"""


class LLMReviewer:
    """
    LLM 驱动的代码审查

    使用 Ollama LLM 分析 git diff，给出智能审查意见。
    比纯规则检查更能发现逻辑问题和设计问题。

    使用:
      PYTHONPATH=. python project/review/reviewer.py --llm

    注意:
      - 需要 Ollama 运行 + qwen2.5:7b 模型
      - 每次审查约 5-15 秒 (取决于 diff 大小)
      - 不放在 pre-commit hook 里 (太慢)
      - 适合手动审查或 CI pipeline
    """

    @staticmethod
    def review_staged_diff() -> str:
        """获取 staged diff 并用 LLM 审查"""
        # 获取 diff
        result = subprocess.run(
            ["git", "diff", "--cached", "--no-color"],
            capture_output=True, text=True,
        )
        diff = result.stdout.strip()

        if not diff:
            return "没有 staged 的变更。"

        # 截断过长的 diff (LLM context 限制)
        if len(diff) > 6000:
            diff = diff[:6000] + "\n... (diff 太长，已截断)"

        print(f"📝 Diff 大小: {len(diff)} 字符")
        print("🤖 正在调用 LLM 审查 (需要 5-15 秒)...")
        print()

        # 调用 Ollama
        try:
            import json
            import urllib.request

            ollama_url = os.getenv("OLLAMA_BASE_URL",
                                   os.getenv("OLLAMA_URL", "http://localhost:11434"))
            model = os.getenv("CHAT_MODEL", "qwen2.5:7b")

            prompt = LLM_REVIEW_PROMPT.format(diff=diff)

            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024},
            }).encode()

            req = urllib.request.Request(
                f"{ollama_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                return result.get("message", {}).get("content", "LLM 没有返回内容")

        except Exception as e:
            return f"❌ LLM 审查失败 (Ollama 可能未运行): {e}\n\n提示: 确保 ollama serve 在运行"


# ============================================================
# CLI + Git Hook
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reviewer = CodeReviewer()

    if "--llm" in sys.argv:
        # LLM 审查模式 — 调 Ollama 做智能 review
        print("=" * 60)
        print("🤖 LLM Code Review (Ollama)")
        print("=" * 60)
        print()

        # 先跑规则检查
        issues = reviewer.review_staged()
        if issues:
            print_report(issues)
            print()
            print("-" * 60)
            print()

        # 再跑 LLM 审查
        llm_feedback = LLMReviewer.review_staged_diff()
        print("🤖 LLM 审查意见:")
        print("-" * 40)
        print(llm_feedback)
        print("-" * 40)

    elif "--hook" in sys.argv:
        # Git pre-commit hook 模式 — error 时阻止 commit
        issues = reviewer.review_staged()
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            print_report(issues)
            sys.exit(1)  # 阻止 commit
        elif issues:
            print_report(issues)
        sys.exit(0)
    else:
        # CLI 模式 — 只打印规则检查报告
        issues = reviewer.review_staged()
        print_report(issues)
        print()
        print("💡 提示: 用 --llm 启用 LLM 智能审查 (需要 Ollama)")
