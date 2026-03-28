"""
OpenAI Responses API 客户端
============================

封装 OpenAI Responses API（新接口），支持:
  - Codex 系列模型 (codex-mini-latest 等)
  - o 系列推理模型 (o4-mini, o3 等)
  - 标准 GPT 模型 (gpt-4.1 等)

与传统 Chat Completions API 的区别:
  - 端点: /v1/responses  (而非 /v1/chat/completions)
  - 参数: input          (而非 messages)
  - 返回: output_text    (而非 choices[0].message.content)

用法:
    client = OpenAIClient(model="codex-mini-latest")
    result = client.ask("解释这段代码的作用")
    print(result)

环境变量:
    OPENAI_API_KEY  — 必须设置
    OPENAI_MODEL    — 可选，默认 codex-mini-latest
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("app.openai")


class OpenAIClient:
    """OpenAI Responses API 客户端"""

    def __init__(
        self,
        model: str = "codex-mini-latest",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "请安装 openai SDK: pip install 'openai>=1.75.0'"
            )

        self.model = model
        self.timeout = timeout
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
        )

    # ----------------------------------------------------------------
    # 简单问答
    # ----------------------------------------------------------------

    def ask(self, prompt: str, **kwargs) -> str:
        """
        单轮问答 — 发送一条 user message，返回文本。

        等价于:
            client.responses.create(
                model="codex-mini-latest",
                input="你的问题"
            )
        """
        resp = self._client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
        return resp.output_text

    # ----------------------------------------------------------------
    # 多轮对话
    # ----------------------------------------------------------------

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        多轮对话 — 传入 messages 列表。

        messages 格式与 Chat Completions 相同:
            [{"role": "user", "content": "..."}, ...]

        Responses API 的 input 参数同时支持字符串和消息列表。
        """
        resp = self._client.responses.create(
            model=self.model,
            input=messages,
            **kwargs,
        )
        return resp.output_text

    # ----------------------------------------------------------------
    # 带工具调用
    # ----------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs,
    ):
        """
        带工具定义的对话 — 返回原始 response 对象，
        调用方可检查 response.output 中是否有 function_call。

        tools 格式:
            [{"type": "function", "function": {"name": ..., "parameters": ...}}]
        """
        resp = self._client.responses.create(
            model=self.model,
            input=messages,
            tools=tools,
            **kwargs,
        )
        return resp

    # ----------------------------------------------------------------
    # 流式输出
    # ----------------------------------------------------------------

    def ask_stream(self, prompt: str, **kwargs):
        """
        流式问答 — yield 逐个文本片段。

        用法:
            for chunk in client.ask_stream("写一首诗"):
                print(chunk, end="")
        """
        stream = self._client.responses.create(
            model=self.model,
            input=prompt,
            stream=True,
            **kwargs,
        )
        for event in stream:
            # Responses API 流式事件中，文本增量在
            # response.output_text.delta 类型的事件里
            if hasattr(event, "delta") and event.delta:
                yield event.delta


# ----------------------------------------------------------------
# 便捷工厂函数
# ----------------------------------------------------------------

def get_openai_client(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIClient:
    """
    从环境变量创建客户端。

    优先级: 参数 > 环境变量 > 默认值
    """
    model = model or os.getenv("OPENAI_MODEL", "codex-mini-latest")
    return OpenAIClient(model=model, api_key=api_key)
