"""
ReAct Agent — 带工具调用的智能体
=================================

ReAct 循环: Thought → Action → Observation → ... → Final Answer

与学习版 react_agent.py 的区别:
  - 学习版把工具注册表、Agent 循环、FastAPI Server、CLI 堆在一个文件
  - 生产版: 工具 → tools.py, Agent 循环 → 本文件 (agent.py)
  - Ollama 调用复用 ollama_client.py
"""

import os
import re
import json
import uuid
import logging
from typing import Optional

from ..ollama_client import OllamaClient
from .tools import ToolRegistry, create_default_registry

logger = logging.getLogger("app.react.agent")

# ============================================================
# System Prompt
# ============================================================

REACT_SYSTEM_PROMPT = """你是一个能使用工具的智能助手。请用中文回答。

## 可用工具
{tool_descriptions}

## 输出格式

当你需要使用工具时，严格按以下格式输出（每行一个字段）：

Thought: 我需要查询知识图谱来回答这个问题
Action: knowledge_graph_query
Action Input: Transformer

当你收到 Observation 后，继续用同样格式思考。

当你已有足够信息时，输出：

Thought: 我已经获得了足够的信息
Final Answer: 这里是最终回答

## 重要规则
- 每次只调用一个工具
- 简单问题（闲聊、打招呼）直接给 Final Answer，不需要调工具
- Action 必须是上面列出的工具名之一
- Action Input 是传给工具的参数（字符串）"""


class ReActAgent:
    """
    ReAct Agent — Thought → Action → Observation 循环

    用法:
        agent = ReActAgent()
        answer = agent.run("Transformer中Self-Attention的缩放因子是什么?")
    """

    def __init__(self, model: str = None, max_steps: int = 4, registry: ToolRegistry = None):
        self.llm = OllamaClient(
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=model or os.getenv("LLM_MODEL", "qwen2.5:7b"),
        )
        self.max_steps = max_steps
        self.registry = registry or create_default_registry()

    def run(self, question: str) -> str:
        """执行 ReAct 循环"""
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.registry.get_tool_descriptions()
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            logger.info("ReAct Step %d/%d", step, self.max_steps)

            output = self.llm.chat(messages)
            logger.info("LLM [step %d]:\n%s", step, output[:300])

            messages.append({"role": "assistant", "content": output})

            # Final Answer?
            final = self._extract_final_answer(output)
            if final:
                logger.info("✅ Final Answer (step %d): %s", step, final[:100])
                return final

            # Action?
            action, action_input = self._parse_action(output)
            if action:
                logger.info("🔧 工具调用: %s(%s)", action, action_input[:80])
                observation = self.registry.execute(action, action_input)
                logger.info("📋or 工具结果: %s", observation[:200])
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请根据以上信息继续思考，给出 Final Answer 或调用下一个工具。",
                })
            else:
                logger.info("⚠️  无标准格式，当作直接回答")
                return output.strip()

        # 超过最大步数
        logger.warning("达到最大步数 %d，强制结束", self.max_steps)
        messages.append({
            "role": "user",
            "content": "你已经获得了足够的信息。请现在直接给出 Final Answer:",
        })
        final_output = self.llm.chat(messages)
        final = self._extract_final_answer(final_output)
        return final if final else final_output.strip()

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """提取 Final Answer"""
        match = re.search(r"Final\s*Answer\s*[:：]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.split(r"\n(?:Thought|Action)\s*[:：]", answer)[0].strip()
            return answer
        return None

    def _parse_action(self, text: str) -> tuple[Optional[str], str]:
        """解析 Action 和 Action Input"""
        action_match = re.search(r"Action\s*[:：]\s*(\w+)", text, re.IGNORECASE)
        input_match = re.search(r"Action\s*Input\s*[:：]\s*(.+?)(?:\n|$)", text, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            action_input = action_input.strip('"\'')
            return action, action_input
        return None, ""


# ============================================================
# A2A 兼容的 FastAPI Server
# ============================================================

def create_react_app():
    """创建 ReAct Agent 的 FastAPI 应用 — 兼容 A2A 协议"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="ReAct Agent with Tools", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    port = int(os.getenv("REACT_PORT", "5002"))
    agent = ReActAgent()

    agent_card = {
        "name": "ReAct Agent with Tools",
        "description": "带工具调用的 ReAct Agent。支持知识图谱查询、数学计算、RAG 检索等。",
        "url": f"http://localhost:{port}",
        "version": "2.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False, "toolCalling": True},
        "skills": [
            {"id": "tool_agent", "name": "工具调用 Agent",
             "description": "自主决策调用工具回答问题",
             "tags": ["agent", "tools", "react"]},
        ],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
    }

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        return JSONResponse(agent_card)

    @app.post("/tasks/send")
    async def send_task(request: Request):
        body = await request.json()
        message = body.get("message", {})
        parts = message.get("parts", [])
        user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not user_text:
            return JSONResponse({"error": "No text"}, status_code=400)

        task_id = body.get("id", f"task-{uuid.uuid4().hex[:8]}")
        logger.info("Task %s: '%s'", task_id, user_text[:50])

        try:
            answer = agent.run(user_text)
            return JSONResponse({
                "id": task_id,
                "status": {"state": "completed", "message": {
                    "role": "agent", "parts": [{"type": "text", "text": answer}]
                }},
            })
        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e)
            return JSONResponse({
                "id": task_id,
                "status": {"state": "failed", "message": {
                    "role": "agent", "parts": [{"type": "text", "text": f"错误: {e}"}]
                }},
            })

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": "ReAct Agent", "tools": list(agent.registry.tools.keys())}

    return app
