"""
Ollama HTTP 客户端
==================

封装对 Ollama API 的调用，支持:
  - /api/generate  (同步 / 流式)
  - /api/chat      (多轮对话，ReAct Agent 用)

与学习版的区别:
  - 学习版在 a2a_agent.py 里 call_ollama 是嵌在 create_expert_app() 内部的局部函数
  - 生产版独立出来，可被 expert_server / react_agent 等任意模块复用
"""

import json
import logging
import urllib.request
from typing import Generator
from http.client import HTTPResponse

logger = logging.getLogger("app.ollama")


class OllamaClient:
    """Ollama HTTP 客户端"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ----------------------------------------------------------------
    # /api/generate
    # ----------------------------------------------------------------

    def generate(self, prompt: str, **options) -> str:
        """
        同步生成 — POST /api/generate (stream=false)

        返回完整文本。
        """
        resp = self._raw_generate(prompt, stream=False, **options)
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "")

    def generate_stream(self, prompt: str, **options) -> Generator[str, None, None]:
        """
        流式生成 — POST /api/generate (stream=true)

        逐 token yield。
        """
        resp = self._raw_generate(prompt, stream=True, **options)
        for line in resp:
            chunk = json.loads(line.decode("utf-8"))
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done", False):
                break

    def _raw_generate(self, prompt: str, stream: bool = False, **options) -> HTTPResponse:
        """底层 /api/generate 请求"""
        merged = {"temperature": 0.3, "num_predict": 1024, "num_ctx": 2048}
        merged.update(options)
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": merged,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    # ----------------------------------------------------------------
    # /api/chat
    # ----------------------------------------------------------------

    def chat(self, messages: list[dict], **options) -> str:
        """
        多轮对话 — POST /api/chat (stream=false)

        messages 格式: [{"role": "system/user/assistant", "content": "..."}]
        返回 assistant 的 content。
        """
        merged = {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096}
        merged.update(options)
        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": merged,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")
