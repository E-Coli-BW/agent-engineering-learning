"""
Code Agent — 代码生成 + 解释 + 审查
=====================================

能力:
  1. 代码生成: 根据需求生成 Python/Java 代码
  2. 代码解释: 解释给定代码的作用
  3. 代码审查: 指出代码问题和改进建议

本 Agent 直接调用 Ollama LLM，用专门的 code prompt。
"""

import os
import json
import logging
import urllib.request

logger = logging.getLogger("orchestrator.code")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
CODE_MODEL = os.getenv("CODE_MODEL", os.getenv("CHAT_MODEL", "qwen2.5:7b"))

CODE_SYSTEM_PROMPT = """你是一个专业的编程助手。请根据用户需求：
1. 如果需要生成代码，提供完整、可运行的代码，包含注释
2. 如果需要解释代码，逐行解释关键逻辑
3. 如果需要审查代码，指出 bug、性能问题、最佳实践

用中文回答，代码中的注释也用中文。技术术语保留英文。"""


def code_agent(query: str) -> str:
    """
    Code Agent 入口

    通过 Ollama /api/chat 调用 LLM 生成代码相关回答。
    """
    try:
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        data = json.dumps({
            "model": CODE_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("message", {}).get("content", "代码生成失败")

    except Exception as e:
        logger.error("Code Agent 调用失败: %s", e)
        return f"代码生成失败: {e}"
