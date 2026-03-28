"""
项目配置管理
============

真实工程不该到处硬编码模型名、路径。
统一配置，支持环境变量覆盖。
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """项目全局配置"""

    # ---- 路径 ----
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VECTOR_DB_DIR: Path = DATA_DIR / "project_chroma_db"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    ETL_LOG_DIR: Path = DATA_DIR / "etl_logs"

    # ---- 模型 ----
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "qwen2.5:7b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ---- OpenAI (Responses API / Codex) ----
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "codex-mini-latest")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # ---- RAG ----
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_base")

    # ---- API ----
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))

    def __post_init__(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.ETL_LOG_DIR.mkdir(parents=True, exist_ok=True)


# 单例
config = Config()
