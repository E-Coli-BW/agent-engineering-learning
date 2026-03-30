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
# CLI + Git Hook
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reviewer = CodeReviewer()

    if "--hook" in sys.argv:
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
        # CLI 模式 — 只打印报告
        issues = reviewer.review_staged()
        print_report(issues)
