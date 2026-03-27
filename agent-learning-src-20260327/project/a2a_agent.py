"""
A2A (Agent-to-Agent) 协作系统 — v2 HTTP REST 版
==============================================

对齐 Google A2A 协议规范 (https://google.github.io/A2A/)

旧版 (v1) 保留在 a2a_agent_v1_stdio.py，使用 subprocess + stdin/stdout 管道。
新版 (v2) 使用 HTTP REST + SSE，完全对齐 A2A 规范。

┌─────────────────────────────────────────────────────────┐
│  真实 A2A 协议的三大核心:                                │
│                                                         │
│  1. Agent Card — GET /.well-known/agent.json            │
│     Agent 用 JSON 声明自己的能力、技能、端点              │
│                                                         │
│  2. Task 生命周期 — POST /tasks/send                    │
│     submitted → working → completed / failed             │
│     每个 Task 携带 Message (role + parts)                │
│                                                         │
│  3. Streaming — POST /tasks/sendSubscribe               │
│     通过 SSE 推送 Task 状态变更和增量输出                 │
└─────────────────────────────────────────────────────────┘

架构:
  Expert Agent (FastAPI Server, port 5001)
    ├── GET  /.well-known/agent.json   → Agent Card
    ├── POST /tasks/send               → 同步执行 Task
    ├── POST /tasks/sendSubscribe      → SSE 流式执行
    ├── GET  /tasks/{task_id}          → 查询 Task 状态
    └── 内部调用 Ollama 生成回答

  Coordinator Agent (Client)
    ├── 发现 Expert: GET /.well-known/agent.json
    ├── 匹配 skill → 创建 Task
    ├── POST /tasks/send 或 /tasks/sendSubscribe
    └── 汇总结果返回给用户

运行方式:
  # 终端 1: 启动 Expert Agent Server
  python project/a2a_agent.py --serve

  # 终端 2: 客户端调用
  python project/a2a_agent.py "Transformer的Self-Attention为什么要除以sqrt(d_k)?"
  python project/a2a_agent.py --stream "LoRA的原理是什么？"

对比旧版:
  旧版 v1 (a2a_agent_v1_stdio.py): subprocess + stdin/stdout, 自定义 JSON-RPC
  新版 v2 (本文件):                 HTTP REST + SSE, 对齐 Google A2A 规范
"""

import os
import sys
import json
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, AsyncGenerator
from enum import Enum
from datetime import datetime, timezone

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)
logger = logging.getLogger("a2a")


# ============================================================
# A2A 协议数据模型 (对齐 Google A2A spec)
# ============================================================

class TaskState(str, Enum):
    """
    Task 状态机 (A2A 规范):
      submitted → working → completed
                         ↘ failed
                         ↘ input-required
                         ↘ canceled
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    INPUT_REQUIRED = "input-required"
    CANCELED = "canceled"


@dataclass
class Part:
    """
    A2A Message Part — 消息的最小单元
    规范支持 TextPart / FilePart / DataPart, 我们只用 TextPart
    """
    type: str = "text"
    text: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class Message:
    """
    A2A Message — Task 中的对话消息
    """
    role: str = "user"     # "user" | "agent"
    parts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() if isinstance(p, Part) else p for p in self.parts],
        }


@dataclass
class TaskStatus:
    """A2A Task Status — 当前状态快照"""
    state: str = TaskState.SUBMITTED
    message: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = {"state": self.state, "timestamp": self.timestamp}
        if self.message:
            d["message"] = self.message
        return d


@dataclass
class Task:
    """
    A2A Task — 协议的核心工作单元

    结构:
    {
      "id": "task-abc123",
      "status": { "state": "completed", "message": {...} },
      "history": [ {"role":"user","parts":[...]}, {"role":"agent","parts":[...]} ],
      "artifacts": [ {"parts": [...]} ]
    }
    """
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    status: TaskStatus = field(default_factory=TaskStatus)
    history: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.to_dict(),
            "history": [m.to_dict() if isinstance(m, Message) else m for m in self.history],
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass
class Skill:
    """A2A Agent Skill — 能力声明"""
    id: str
    name: str
    description: str
    tags: list = field(default_factory=list)
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentCard:
    """
    A2A Agent Card — GET /.well-known/agent.json

    {
      "name": "...", "description": "...", "url": "http://...",
      "version": "1.0.0",
      "capabilities": { "streaming": true, "pushNotifications": false },
      "skills": [ { "id": "...", "name": "...", "tags": [...] } ],
      "defaultInputModes": ["text"],
      "defaultOutputModes": ["text"]
    }
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list = field(default_factory=list)
    capabilities: dict = field(default_factory=lambda: {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    })
    defaultInputModes: list = field(default_factory=lambda: ["text"])
    defaultOutputModes: list = field(default_factory=lambda: ["text"])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": [s.to_dict() if isinstance(s, Skill) else s for s in self.skills],
            "defaultInputModes": self.defaultInputModes,
            "defaultOutputModes": self.defaultOutputModes,
        }


# ============================================================
# Task 存储 (内存, 生产环境用 Redis/DB)
# ============================================================
task_store: dict[str, Task] = {}


# ============================================================
# Expert Agent — FastAPI Server
# ============================================================

def create_expert_app():
    """创建 Expert Agent 的 FastAPI 应用"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="A2A Expert Agent", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    EXPERT_PORT = int(os.getenv("EXPERT_PORT", "5001"))
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

    # ---- Agent Card ----
    agent_card = AgentCard(
        name="DeepLearning Expert",
        description="深度学习和大模型领域专家。擅长 Transformer、LoRA、RAG、Agent、推理部署、知识图谱等技术。",
        url=f"http://localhost:{EXPERT_PORT}",
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

    # ---- Ollama 调用 ----
    def call_ollama(prompt: str, stream: bool = False):
        """调用本地 Ollama, 返回 http.client.HTTPResponse"""
        import urllib.request
        data = json.dumps({
            "model": LLM_MODEL, "prompt": prompt, "stream": stream,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,     # 允许长回答完整生成
                "num_ctx": 2048,         # 上下文窗口
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=120)

    SKILL_PROMPTS = {
        "transformer_theory": "你是 Transformer 原理专家。请用清晰的中文解释，包含关键公式和直觉解释：\n\n",
        "lora_finetuning":    "你是 LoRA 微调专家。请详细解释以下关于参数高效微调的问题：\n\n",
        "rag_system":         "你是 RAG 系统专家。请解释以下关于检索增强生成的问题：\n\n",
        "agent_development":  "你是大模型 Agent 开发专家。请解释以下关于 Agent 架构和开发的问题：\n\n",
        "inference_deployment":"你是大模型推理部署专家。请解释模型推理优化和部署的问题：\n\n",
        "knowledge_graph":    "你是知识图谱专家。请解释以下关于知识图谱和 Graph RAG 的问题：\n\n",
    }

    def route_skill(text: str) -> str:
        """关键词路由"""
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

    # ================================================================
    # A2A 端点
    # ================================================================

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        """A2A: Agent Card 发现端点"""
        return JSONResponse(agent_card.to_dict())

    @app.post("/tasks/send")
    async def send_task(request: Request):
        """
        A2A: POST /tasks/send — 同步执行 Task

        请求体:
        {
          "id": "task-xxx",  // 可选
          "message": { "role": "user", "parts": [{"type":"text","text":"问题"}] },
          "metadata": { "skill": "transformer_theory" }  // 可选
        }
        """
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
            prompt = SKILL_PROMPTS.get(skill_id, "你是一个有帮助的 AI 助手。请用中文回答：\n\n") + user_text
            resp = call_ollama(prompt, stream=False)
            result = json.loads(resp.read().decode("utf-8"))
            answer = result.get("response", "无法生成回答")

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
        """
        A2A: POST /tasks/sendSubscribe — SSE 流式执行 Task

        SSE 事件:
          event: status   → {"state": "working", ...}
          event: artifact → {"parts": [{"type":"text","text":"增量"}], "append": true}
          event: status   → {"state": "completed", ...}
        """
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
            # submitted → working
            task.status = TaskStatus(state=TaskState.WORKING)
            yield f"event: status\ndata: {json.dumps(task.status.to_dict(), ensure_ascii=False)}\n\n"

            try:
                prompt = SKILL_PROMPTS.get(skill_id, "你是一个有帮助的 AI 助手。请用中文回答：\n\n") + user_text
                resp = call_ollama(prompt, stream=True)

                full_text = ""
                for line in resp:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        full_text += token
                        artifact = {"parts": [{"type": "text", "text": token}], "index": 0, "append": True}
                        yield f"event: artifact\ndata: {json.dumps(artifact, ensure_ascii=False)}\n\n"
                    if chunk.get("done", False):
                        break

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
            event_stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Task-ID": task_id},
        )

    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """A2A: 查询 Task 状态"""
        task = task_store.get(task_id)
        if not task:
            return JSONResponse({"error": f"Task {task_id} not found"}, status_code=404)
        return JSONResponse(task.to_dict())

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": agent_card.name, "model": LLM_MODEL}

    return app


# ============================================================
# Coordinator Agent (HTTP Client)
# ============================================================

class CoordinatorAgent:
    """
    协调者 Agent — 通过 HTTP 调用 Expert Agent

    对齐 A2A 规范的客户端行为:
      1. GET /.well-known/agent.json  发现 Agent
      2. 根据 Agent Card 的 skills  匹配
      3. POST /tasks/send 或 /tasks/sendSubscribe
    """

    def __init__(self, expert_url: str = "http://localhost:5001"):
        self.expert_url = expert_url.rstrip("/")
        self.agent_card: Optional[dict] = None

    def discover(self) -> dict:
        """A2A 发现: GET /.well-known/agent.json"""
        import urllib.request
        url = f"{self.expert_url}/.well-known/agent.json"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                self.agent_card = json.loads(resp.read().decode("utf-8"))
                return self.agent_card
        except Exception as e:
            raise RuntimeError(f"无法连接 Expert Agent ({url}): {e}")

    def match_skill(self, question: str) -> str:
        """根据 Agent Card 的 skills + tags 匹配最佳 skill"""
        if not self.agent_card:
            return "general_qa"

        q = question.lower()
        best_skill, best_score = "general_qa", 0

        for skill in self.agent_card.get("skills", []):
            score = sum(2 for tag in skill.get("tags", []) if tag.lower() in q)
            if skill.get("name", "").lower() in q:
                score += 1
            if score > best_score:
                best_score = score
                best_skill = skill.get("id", "general_qa")

        # fallback 关键词路由
        if best_score == 0:
            rules = [
                (["transformer", "attention", "qkv", "注意力", "self-attention"], "transformer_theory"),
                (["lora", "qlora", "微调", "fine-tun", "sft", "rlhf", "dpo"], "lora_finetuning"),
                (["rag", "检索", "向量", "embedding", "chunk"], "rag_system"),
                (["agent", "tool call", "react", "langgraph", "工具调用", "mcp"], "agent_development"),
                (["推理", "部署", "kv cache", "量化", "ollama", "inference"], "inference_deployment"),
                (["知识图谱", "knowledge graph", "graph rag"], "knowledge_graph"),
            ]
            for keywords, sid in rules:
                if any(kw in q for kw in keywords):
                    return sid
        return best_skill

    def send_task(self, question: str, skill: str = None) -> dict:
        """POST /tasks/send — 同步"""
        import urllib.request
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/send", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def send_subscribe(self, question: str, skill: str = None):
        """
        POST /tasks/sendSubscribe — 流式
        Yields (event_type, data) 元组
        """
        import urllib.request
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/sendSubscribe", data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            event_type = None
            data_lines = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line == "":
                    if event_type and data_lines:
                        yield (event_type, json.loads("".join(data_lines)))
                    event_type = None
                    data_lines = []


# ============================================================
# CLI Demo
# ============================================================

def demo_sync(question: str):
    """同步模式"""
    print("=" * 60)
    print("🤝 A2A Agent-to-Agent 协作 (HTTP REST)")
    print("=" * 60)
    print()

    coordinator = CoordinatorAgent()

    print("📡 发现 Expert Agent...")
    card = coordinator.discover()
    print(f"  Name:    {card['name']}")
    print(f"  URL:     {card['url']}")
    print(f"  Skills:  {[s['id'] for s in card['skills']]}")
    print(f"  Stream:  {card['capabilities'].get('streaming', False)}")
    print()

    skill = coordinator.match_skill(question)
    print(f"❓ 问题: {question}")
    print(f"🔀 路由: {skill}")
    print()

    print("📤 POST /tasks/send ...")
    result = coordinator.send_task(question, skill)
    print(f"📋 Task ID: {result['id']}")
    print(f"📋 Status:  {result['status']['state']}")
    print()

    if result["status"]["state"] == "completed":
        msg = result["status"].get("message", {})
        answer = " ".join(p["text"] for p in msg.get("parts", []) if p.get("type") == "text")
        print(f"💡 回答:\n{answer}")
    else:
        print(f"❌ Task 失败: {result['status']}")

    _print_comparison()


def demo_stream(question: str):
    """流式模式"""
    print("=" * 60)
    print("🤝 A2A Agent-to-Agent 协作 (SSE Streaming)")
    print("=" * 60)
    print()

    coordinator = CoordinatorAgent()

    print("📡 发现 Expert Agent...")
    card = coordinator.discover()
    print(f"  Name: {card['name']}  streaming={card['capabilities'].get('streaming')}")
    print()

    skill = coordinator.match_skill(question)
    print(f"❓ 问题: {question}")
    print(f"🔀 路由: {skill}")
    print()
    print("📤 POST /tasks/sendSubscribe ...")
    print("-" * 40)

    answer_parts = []
    for event_type, event_data in coordinator.send_subscribe(question, skill):
        if event_type == "status":
            state = event_data.get("state", "?")
            icon = {"working": "⏳", "completed": "✅", "failed": "❌"}.get(state, "❓")
            print(f"\n  {icon} [status] state={state}")
        elif event_type == "artifact":
            for p in event_data.get("parts", []):
                if p.get("type") == "text":
                    token = p["text"]
                    answer_parts.append(token)
                    sys.stdout.write(token)
                    sys.stdout.flush()

    print("\n" + "-" * 40)
    print(f"\n💡 完整回答:\n{''.join(answer_parts)}")

    _print_comparison()


def _print_comparison():
    """打印新旧版对比表"""
    print()
    print("=" * 60)
    print("📊 A2A 协议对比: v1 (stdio) vs v2 (HTTP REST)")
    print("=" * 60)
    print("""
┌──────────────┬──────────────────────┬──────────────────────────┐
│              │ v1 stdio (旧)         │ v2 HTTP REST (新)         │
├──────────────┼──────────────────────┼──────────────────────────┤
│ 传输层       │ stdin/stdout 管道     │ HTTP REST + SSE           │
│ 发现机制     │ 无 (硬编码)           │ GET /.well-known/agent.json│
│ Agent Card   │ 自定义 JSON-RPC      │ 标准 A2A Agent Card       │
│ Task 格式    │ 自定义 flat dict     │ message + parts 结构      │
│ 流式输出     │ ❌                   │ ✅ SSE event-stream       │
│ 跨机器调用   │ ❌ 同机器进程管道     │ ✅ 任意 HTTP 可达          │
│ 跨语言       │ ❌ 仅 Python          │ ✅ 任意语言               │
│ 状态查询     │ ❌                   │ ✅ GET /tasks/{id}        │
│ 多客户端     │ ❌ 1对1              │ ✅ 并发                   │
│ 部署         │ ❌ 本地进程           │ ✅ Docker/K8s             │
└──────────────┴──────────────────────┴──────────────────────────┘
""")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    if "--serve" in sys.argv:
        import uvicorn
        port = int(os.getenv("EXPERT_PORT", "5001"))
        print(f"🚀 A2A Expert Agent Server")
        print(f"   Agent Card:  http://localhost:{port}/.well-known/agent.json")
        print(f"   Task (sync): POST http://localhost:{port}/tasks/send")
        print(f"   Task (SSE):  POST http://localhost:{port}/tasks/sendSubscribe")
        print(f"   Task query:  GET  http://localhost:{port}/tasks/{{task_id}}")
        app = create_expert_app()
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    elif "--stream" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--stream"]
        q = " ".join(args) if args else "什么是Transformer?"
        demo_stream(q)

    elif len(sys.argv) > 1:
        demo_sync(" ".join(sys.argv[1:]))

    else:
        print("用法:")
        print("  启动 Expert:  python project/a2a_agent.py --serve")
        print("  同步查询:     python project/a2a_agent.py '你的问题'")
        print("  流式查询:     python project/a2a_agent.py --stream '你的问题'")
        print()
        print("旧版 (stdio):   python project/a2a_agent_v1_stdio.py '你的问题'")
