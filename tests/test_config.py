"""
Config 单元测试
================
测试环境变量覆盖、默认值、路径创建
"""

import os
import pytest
from pathlib import Path


class TestConfigDefaults:
    """测试默认配置值"""

    def test_default_chat_model(self):
        from project.config import Config
        c = Config()
        assert c.CHAT_MODEL == os.getenv("CHAT_MODEL", "qwen2.5:7b")

    def test_default_embed_model(self):
        from project.config import Config
        c = Config()
        assert c.EMBED_MODEL == os.getenv("EMBED_MODEL", "mxbai-embed-large")

    def test_default_ollama_url(self):
        from project.config import Config
        c = Config()
        assert "11434" in c.OLLAMA_BASE_URL

    def test_default_top_k(self):
        from project.config import Config
        c = Config()
        assert c.TOP_K == int(os.getenv("TOP_K", "5"))

    def test_default_chunk_size(self):
        from project.config import Config
        c = Config()
        assert c.CHUNK_SIZE == int(os.getenv("CHUNK_SIZE", "500"))

    def test_project_root_exists(self):
        from project.config import Config
        c = Config()
        assert c.PROJECT_ROOT.exists()

    def test_data_dir_created(self):
        from project.config import Config
        c = Config()
        assert c.DATA_DIR.exists()


class TestConfigEnvOverride:
    """测试环境变量覆盖 — 通过 subprocess 验证（dataclass field 在定义时求值）"""

    def test_chat_model_override(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-c",
             "from project.config import Config; c = Config(); print(c.CHAT_MODEL)"],
            env={**__import__('os').environ, "CHAT_MODEL": "test-model:1b"},
            capture_output=True, text=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent),
        )
        assert "test-model:1b" in result.stdout

    def test_top_k_override(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-c",
             "from project.config import Config; c = Config(); print(c.TOP_K)"],
            env={**__import__('os').environ, "TOP_K": "10"},
            capture_output=True, text=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent),
        )
        assert "10" in result.stdout

    def test_api_port_override(self):
        import subprocess
        result = subprocess.run(
            ["python3", "-c",
             "from project.config import Config; c = Config(); print(c.API_PORT)"],
            env={**__import__('os').environ, "API_PORT": "9999"},
            capture_output=True, text=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent),
        )
        assert "9999" in result.stdout
