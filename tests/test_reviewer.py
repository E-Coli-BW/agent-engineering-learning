"""
Code Reviewer 测试
"""

from project.review.reviewer import CodeReviewer, ReviewIssue
import tempfile
from pathlib import Path


class TestReviewIssue:
    def test_str_format(self):
        issue = ReviewIssue("error", "test.py", 10, "test-rule", "test message")
        s = str(issue)
        assert "❌" in s
        assert "test.py:10" in s

    def test_warning_icon(self):
        issue = ReviewIssue("warning", "f.py", 1, "r", "msg")
        assert "⚠️" in str(issue)


class TestPythonReview:
    def test_detect_bare_except(self):
        reviewer = CodeReviewer()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
            f.write('"""docstring"""\ntry:\n    pass\nexcept:\n    pass\n')
            f.flush()
            reviewer._review_python(f.name)
        issues = [i for i in reviewer.issues if i.rule == "bare-except"]
        assert len(issues) >= 1

    def test_no_issues_clean_code(self):
        reviewer = CodeReviewer()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
            f.write('"""Clean module."""\n\ndef hello():\n    return "world"\n')
            f.flush()
            reviewer._review_python(f.name)
        errors = [i for i in reviewer.issues if i.severity == "error"]
        assert len(errors) == 0


class TestSecurityCheck:
    def test_detect_hardcoded_password(self):
        reviewer = CodeReviewer()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
            f.write('password = "my_secret_123"\n')
            f.flush()
            reviewer._check_security([f.name])
        secrets = [i for i in reviewer.issues if i.rule == "hardcoded-secret"]
        assert len(secrets) >= 1

    def test_allow_env_var_secret(self):
        reviewer = CodeReviewer()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, dir="/tmp") as f:
            f.write('password = os.getenv("PASSWORD", "default")\n')
            f.flush()
            reviewer._check_security([f.name])
        secrets = [i for i in reviewer.issues if i.rule == "hardcoded-secret"]
        assert len(secrets) == 0
