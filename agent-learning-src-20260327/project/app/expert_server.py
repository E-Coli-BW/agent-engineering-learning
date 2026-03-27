"""
Expert Agent Server (FastAPI)
=============================

A2A 协议的 Expert Agent 服务端，对外暴露:
  GET  /.well-known/agent.json    Agent Card 发现
  POST /tasks/send                同步执行 Task
  POST /tasks/sendSubscribe       SSE 流式执行 Task
  GET  /tasks/{task_id}           查询 Task 状态
  GET  /health                    健康检查

与学习版 a2a_agent.py 的区别:
  - 学习版把数据模型、服务器、客户端、CLI 全部塞在一个 717 行的文件里
  - 生产版 server 只关心"接收请求 → 调用 Ollama → 返回 A2A 响应"
"""

import os
import json
import uuid
import logging
from typing import AsyncGenerator

from .models import (
    TaskState, Part, Message, TaskStatus, Task,
    Skill, AgentCard,
)
from .ollama_client import OllamaClient

logger = logging.getLogger("app.expert")

# Task 存储 (内存; 生产环境替换为 Redis/DB)
task_store: dict[str, Task] = {}

# 技能 → system prompt 前缀
SKILL_PROMPTS = {
    "transformer_theory":  "你是 Transformer 原理专家。请用清晰的中文解释，包含关键公式和直觉解释：\n\n",
    "lora_finetuning":     "你是 LoRA 微调专家。请详细解释以下关于参数高效微调的问题：\n\n",
    "rag_system":          "你是 RAG 系统专家。请解释以下关于检索增强生成的问题：\n\n",
    "agent_development":   "你是大模型 Agent 开发专家。请解释以下关于 Agent 架构和开发的问题：\n\n",
    "inference_deployment": "你是大模型推理部署专家。请解释模型推理优化和部署的问题：\n\n",
    "knowledge_graph":     "你是知识图谱专家。请解释以下关于知识图谱和 Graph RAG 的问题：\n\n",
}

DEFAULT_PROMPT_PREFIX = "你是一个有帮助的 AI 助手。请用中文回答：\n\n"


def route_skill(text: str) -> str:
    """关键词路由 — 根据用户输入匹配技能"""
    t = text.lower()
    rules = [
        (["transformer", "attention", "qkv", "注意力", "self-attention", "multi-head", "mask"], "transformer_theory"),
        (["lora", "qlora", "微调", "fine-tun", "sft", "rlhf", "dpo", "peft"], "lora_finetuning"),
        (["rag", "检索", "向量", "embedding", "chromadb", "chunk", "retrieve"], "rag_system"),
        (["agent", "tool call", "react", "langgraph", "工具调用", "function call", "mcp", "a2a"], "agent_development"),
        (["推理", "部署", "kv cache", "量化", "quantiz", "ollama", "vllm", "inference"], "inference_deployment"),
        (["知识图谱", "knowledge graph", "graph rag", "triple", "三元组"], "knowledge_graph"),
    ]
    for keywords, skill in rules:
        if any(kw in t for kw in keywords):
            return skill
    return "general_qa"


def _build_agent_card(port: int) -> AgentCard:
    """构建 Agent Card"""
    return AgentCard(
        name="DeepLearning Expert",
        description="深度学习和大模型领域专家。擅长 Transformer、LoRA、RAG、Agent、推理部署、知识图谱等技术。",
        url=f"http://localhost:{port}",
        skills=[
            Skill(id="transformer_theory", name="Transformer 原理",
                  description="Self-Attention, Multi-Head Attention, 位置编码等",
                  tags=["transformer", "attention", "qkv"],
                  examples=["Self-Attention为什么要除以sqrt(d_k)?"]),
            Skill(id="lora_finetuning", name="LoRA 微调",
                  description="LoRA, QLoRA, SFT, RLHF, DPO 等参数高效微调",
                  tags=["lora", "finetune", "sft", "rlhf"],
                  examples=["LoRA的低秩分解原理"]),
            Skill(id="rag_system", name="RAG 系统",
                  description="检索增强生成系统的设计和实现",
                  tags=["rag", "embedding", "retrieval"],
                  examples=["如何优化RAG的召回率?"]),
            Skill(id="agent_development", name="Agent 开发",
                  description="Agent 架构、Tool Calling、ReAct、LangGraph",
                  tags=["agent", "react", "tool-calling"],
                  examples=["ReAct循环是什么?"]),
            Skill(id="inference_deployment", name="推理部署",
                  description="KV Cache、量化、模型推理优化",
                  tags=["inference", "kv-cache", "quantization"],
                  examples=["KV Cache如何加速推理?"]),
            Skill(id="knowledge_graph", name="知识图谱",
                  description="知识图谱构建、Graph RAG",
                  tags=["knowledge-graph", "graph-rag"],
                  examples=["Graph RAG比普通RAG好在哪?"]),
            Skill(id="general_qa", name="通用问答",
                  description="其他通用技术问题",
                  tags=["general"], examples=[]),
        ],
    )


def create_expert_app():
    """创建 Expert Agent 的 FastAPI 应用"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="A2A Expert Agent", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    port = int(os.getenv("EXPERT_PORT", "5001"))
    llm = OllamaClient(
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        model=os.getenv("LLM_MODEL", "qwen2.5:7b"),
    )
    agent_card = _build_agent_card(port)

    # ---- A2A 端点 ----

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        return JSONResponse(agent_card.to_dict())

    @app.post("/tasks/send")
    async def send_task(request: Request):
        """POST /tasks/send — 同步执行 Task"""
        body = await request.json()
        task_id = body.get("id", f"task-{uuid.uuid4().hex[:8]}")
        message = body.get("message", {})
        metadata = body.get("metadata", {})

        parts = message.get("parts", [])
        user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not user_text:
            return JSONResponse({"error": "No text content in message parts"}, status_code=400)

        skill_id = metadata.get("skill") or route_skill(user_text)

        task = Task(
            id=task_id,
            status=TaskStatus(state=TaskState.WORKING),
            history=[Message(role="user", parts=[Part(text=user_text)])],
            metadata={"skill": skill_id},
        )
        task_store[task_id] = task
        logger.info("Task %s: skill=%s, input='%s'", task_id, skill_id, user_text[:50])

        try:
            prompt = SKILL_PROMPTS.get(skill_id, DEFAULT_PROMPT_PREFIX) + user_text
            answer = llm.generate(prompt)

            agent_msg = Message(role="agent", parts=[Part(text=answer)])
            task.history.append(agent_msg)
            task.artifacts = [{"parts": [Part(text=answer).to_dict()]}]
            task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg.to_dict())
        except Exception as e:
            logger.error("Task %s failed: %s", task_id, e)
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(role="agent", parts=[Part(text=f"错误: {e}")]).to_dict(),
            )

        task_store[task_id] = task
        return JSONResponse(task.to_dict())

    @app.post("/tasks/sendSubscribe")
    async def send_subscribe(request: Request):
        """POST /tasks/sendSubscribe — SSE 流式执行 Task"""
        body = await request.json()
        task_id = body.get("id", f"task-{uuid.uuid4().hex[:8]}")
        message = body.get("message", {})
        metadata = body.get("metadata", {})

        parts = message.get("parts", [])
        user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
        if not user_text:
            return JSONResponse({"error": "No text content"}, status_code=400)

        skill_id = metadata.get("skill") or route_skill(user_text)

        task = Task(
            id=task_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[Message(role="user", parts=[Part(text=user_text)])],
            metadata={"skill": skill_id},
        )
        task_store[task_id] = task
        logger.info("Task %s (stream): skill=%s, input='%s'", task_id, skill_id, user_text[:50])

        async def event_stream() -> AsyncGenerator[str, None]:
            task.status = TaskStatus(state=TaskState.WORKING)
            yield f"event: status\ndata: {json.dumps(task.status.to_dict(), ensure_ascii=False)}\n\n"

            try:
                prompt = SKILL_PROMPTS.get(skill_id, DEFAULT_PROMPT_PREFIX) + user_text
                full_text = ""
                for token in llm.generate_stream(prompt):
                    full_text += token
                    artifact = {"parts": [{"type": "text", "text": token}], "index": 0, "append": True}
                    yield f"event: artifact\ndata: {json.dumps(artifact, ensure_ascii=False)}\n\n"

                agent_msg = Message(role="agent", parts=[Part(text=full_text)])
                task.history.append(agent_msg)
                task.artifacts = [{"parts": [Part(text=full_text).to_dict()]}]
                task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_msg.to_dict())
                yield f"event: status\ndata: {json.dumps(task.status.to_dict(), ensure_ascii=False)}\n\n"
            except Exception as e:
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=Message(role="agent", parts=[Part(text=f"错误: {e}")]).to_dict(),
                )
                yield f"event: status\ndata: {json.dumps(task.status.to_dict(), ensure_ascii=False)}\n\n"
            task_store[task_id] = task

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Task-ID": task_id},
        )

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        task = task_store.get(task_id)
        if not task:
            return JSONResponse({"error": f"Task {task_id} not found"}, status_code=404)
        return JSONResponse(task.to_dict())

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": agent_card.name, "model": llm.model}

    return app
