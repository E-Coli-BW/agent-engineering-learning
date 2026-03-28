"""
A2A (Agent-to-Agent) 协作系统
==============================================

实现两个 Agent 通过 stdio 管道协作：

  Coordinator Agent (协调者)
    ├── 接收用户问题
    ├── 分析需要调用哪个专家
    └── 通过 stdin/stdout JSON-RPC 发送 Task 给 Expert

  Expert Agent (专家)
    ├── 通过 stdin 接收 Task (JSON-RPC)
    ├── 调用本地 Ollama 回答
    └── 通过 stdout 返回结果

通信协议 (简化版 A2A):
  - 基于 JSON-RPC 2.0 over stdio
  - Task 生命周期: submitted → working → completed / failed
  - Agent Card: 自描述能力 (支持的技能列表)

架构:
  用户 ──question──> Coordinator ──stdio──> Expert(子进程)
                          │                    │
                          │                 Ollama
                          │                    │
                          <───response────<────┘

运行方式:
  python project/a2a_agent.py "Transformer的Self-Attention为什么要除以sqrt(d_k)?"
"""

import os
import sys
import json
import subprocess
import time
import uuid
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum

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
# A2A 协议定义
# ============================================================

class TaskState(str, Enum):
    """Task 生命周期状态"""
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentCard:
    """
    Agent Card — A2A 协议的核心概念
    每个 Agent 用 Agent Card 声明自己的能力
    """
    name: str
    description: str
    skills: list[str]
    version: str = "1.0.0"
    protocol: str = "a2a-stdio"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Task:
    """
    Task — A2A 协议的工作单元
    Coordinator 创建 Task，Expert 执行后返回结果
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    skill: str = ""
    input_text: str = ""
    state: str = TaskState.SUBMITTED
    output_text: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class JSONRPC:
    """
    JSON-RPC 2.0 消息封装
    A2A 协议通过 JSON-RPC 传递消息
    """
    @staticmethod
    def request(method: str, params: dict, req_id: str = None) -> str:
        msg = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": req_id or str(uuid.uuid4())[:8],
        }
        return json.dumps(msg, ensure_ascii=False)

    @staticmethod
    def response(result: dict, req_id: str) -> str:
        msg = {
            "jsonrpc": "2.0",
            "result": result,
            "id": req_id,
        }
        return json.dumps(msg, ensure_ascii=False)

    @staticmethod
    def error(code: int, message: str, req_id: str) -> str:
        msg = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }
        return json.dumps(msg, ensure_ascii=False)

    @staticmethod
    def parse(line: str) -> dict:
        return json.loads(line.strip())


# ============================================================
# Expert Agent (子进程模式)
# ============================================================

class ExpertAgent:
    """
    专家 Agent — 通过 stdin/stdout 提供服务

    运行为独立子进程，通过 stdio 接收 JSON-RPC 请求：
      - agent/card     → 返回 Agent Card (能力声明)
      - task/execute    → 执行 Task，调用 Ollama 生成回答

    这是 A2A 协议中的 Remote Agent 角色
    """

    def __init__(self):
        self.card = AgentCard(
            name="DeepLearning Expert",
            description="深度学习和大模型领域专家，擅长 Transformer、LoRA、RAG、Agent 等技术",
            skills=[
                "transformer_theory",     # Transformer 原理
                "lora_finetuning",        # LoRA 微调
                "rag_system",             # RAG 系统
                "agent_development",      # Agent 开发
                "inference_deployment",   # 推理部署
                "knowledge_graph",        # 知识图谱
                "general_qa",             # 通用问答
            ],
        )
        self.model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    def call_ollama(self, prompt: str) -> str:
        """调用本地 Ollama 生成回答"""
        import urllib.request

        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 500},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "无法生成回答")
        except Exception as e:
            return f"Ollama 调用失败: {e}"

    def handle_request(self, msg: dict) -> str:
        """处理一条 JSON-RPC 请求"""
        method = msg.get("method", "")
        params = msg.get("params", {})
        req_id = msg.get("id", "0")

        if method == "agent/card":
            # 返回 Agent Card
            return JSONRPC.response(self.card.to_dict(), req_id)

        elif method == "task/execute":
            # 执行 Task
            task = Task(
                id=params.get("task_id", str(uuid.uuid4())[:8]),
                skill=params.get("skill", "general_qa"),
                input_text=params.get("input", ""),
                state=TaskState.WORKING,
            )

            # 根据 skill 构造 prompt
            skill_prompts = {
                "transformer_theory": "你是 Transformer 原理专家。请用清晰的中文解释以下问题，包含数学公式和直觉解释：\n\n",
                "lora_finetuning": "你是 LoRA 微调专家。请详细解释以下关于参数高效微调的问题：\n\n",
                "rag_system": "你是 RAG 系统专家。请解释以下关于检索增强生成的问题：\n\n",
                "agent_development": "你是大模型 Agent 开发专家。请解释以下关于 Agent 架构和开发的问题：\n\n",
                "inference_deployment": "你是大模型推理部署专家。请解释以下关于模型推理优化和部署的问题：\n\n",
                "knowledge_graph": "你是知识图谱专家。请解释以下关于知识图谱和 Graph RAG 的问题：\n\n",
                "general_qa": "你是一个有帮助的 AI 助手。请用中文回答以下问题：\n\n",
            }

            prompt = skill_prompts.get(task.skill, skill_prompts["general_qa"])
            prompt += task.input_text

            # 调用 Ollama
            answer = self.call_ollama(prompt)

            task.output_text = answer
            task.state = TaskState.COMPLETED

            return JSONRPC.response(task.to_dict(), req_id)

        else:
            return JSONRPC.error(-32601, f"Method not found: {method}", req_id)

    def run_stdio_loop(self):
        """
        主循环 — 从 stdin 读 JSON-RPC 请求，向 stdout 写响应
        每行一个完整的 JSON 消息 (newline-delimited JSON)
        """
        # 写一条启动消息到 stderr (不干扰 stdout 通信)
        sys.stderr.write(f"[Expert] Agent '{self.card.name}' started, model={self.model}\n")
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = JSONRPC.parse(line)
                response = self.handle_request(msg)
            except json.JSONDecodeError as e:
                response = JSONRPC.error(-32700, f"Parse error: {e}", "0")
            except Exception as e:
                response = JSONRPC.error(-32603, f"Internal error: {e}", "0")

            # 写回 stdout (一行一个 JSON)
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


# ============================================================
# Coordinator Agent (主进程)
# ============================================================

class CoordinatorAgent:
    """
    协调者 Agent — 管理 Expert 子进程，分发 Task

    协调者的职责:
      1. 启动 Expert Agent 子进程
      2. 获取 Expert 的 Agent Card (了解其能力)
      3. 根据用户问题，匹配最合适的 skill
      4. 创建 Task，通过 stdio 发送给 Expert
      5. 收集结果返回给用户

    这是 A2A 协议中的 Client Agent 角色
    """

    def __init__(self):
        self.card = AgentCard(
            name="Coordinator",
            description="协调者 Agent，负责分发任务给专家",
            skills=["coordination", "skill_routing"],
        )
        self.expert_process: Optional[subprocess.Popen] = None
        self.expert_card: Optional[dict] = None

    def start_expert(self):
        """启动 Expert Agent 子进程"""
        python = sys.executable
        script = str(Path(__file__).resolve())

        self.expert_process = subprocess.Popen(
            [python, script, "--expert"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
            cwd=str(PROJECT_ROOT),
        )

        logger.info("Expert Agent 子进程已启动 (PID: %d)", self.expert_process.pid)

        # 获取 Agent Card
        self.expert_card = self._send_to_expert("agent/card", {})
        if self.expert_card:
            logger.info("Expert Card: %s", self.expert_card.get("name", "unknown"))
            logger.info("Expert Skills: %s", self.expert_card.get("skills", []))

    def _send_to_expert(self, method: str, params: dict) -> Optional[dict]:
        """通过 stdio 向 Expert 发送 JSON-RPC 请求"""
        if not self.expert_process:
            raise RuntimeError("Expert not started")

        req = JSONRPC.request(method, params)
        try:
            self.expert_process.stdin.write(req + "\n")
            self.expert_process.stdin.flush()

            response_line = self.expert_process.stdout.readline()
            if not response_line:
                logger.error("Expert returned empty response")
                return None

            msg = JSONRPC.parse(response_line)

            if "error" in msg:
                logger.error("Expert error: %s", msg["error"])
                return None

            return msg.get("result")

        except Exception as e:
            logger.error("Communication error: %s", e)
            return None

    def route_skill(self, question: str) -> str:
        """
        根据问题内容路由到最合适的 skill
        简单关键词匹配 (生产环境可以用 LLM 路由)
        """
        question_lower = question.lower()

        routing_rules = [
            (["transformer", "attention", "qkv", "注意力", "self-attention", "multi-head", "位置编码", "mask"],
             "transformer_theory"),
            (["lora", "qlora", "微调", "fine-tun", "sft", "rlhf", "dpo", "peft"],
             "lora_finetuning"),
            (["rag", "检索", "向量", "embedding", "chromadb", "chunk", "retrieve"],
             "rag_system"),
            (["agent", "tool call", "react", "langgraph", "工具调用", "function call", "mcp", "a2a"],
             "agent_development"),
            (["推理", "部署", "kv cache", "量化", "quantiz", "ollama", "vllm", "inference", "deploy"],
             "inference_deployment"),
            (["知识图谱", "knowledge graph", "graph rag", "triple", "三元组", "实体"],
             "knowledge_graph"),
        ]

        for keywords, skill in routing_rules:
            for kw in keywords:
                if kw in question_lower:
                    return skill

        return "general_qa"

    def ask(self, question: str) -> str:
        """
        处理用户问题的完整流程:
          1. 路由到合适的 skill
          2. 创建 A2A Task
          3. 发送给 Expert
          4. 返回结果
        """
        skill = self.route_skill(question)
        logger.info("问题路由到技能: %s", skill)

        task_id = str(uuid.uuid4())[:8]

        # 发送 Task
        result = self._send_to_expert("task/execute", {
            "task_id": task_id,
            "skill": skill,
            "input": question,
        })

        if not result:
            return "❌ Expert Agent 未返回结果"

        state = result.get("state", "unknown")
        if state == TaskState.COMPLETED:
            return result.get("output_text", "无内容")
        else:
            return f"❌ Task 状态异常: {state}, 错误: {result.get('error', 'unknown')}"

    def stop(self):
        """停止 Expert 子进程"""
        if self.expert_process:
            self.expert_process.stdin.close()
            self.expert_process.wait(timeout=5)
            logger.info("Expert Agent 子进程已停止")


# ============================================================
# 交互式 Demo
# ============================================================

def interactive_demo():
    """交互式 A2A Demo"""
    print("=" * 60)
    print("🤝 A2A Agent-to-Agent 协作系统")
    print("=" * 60)
    print()

    coordinator = CoordinatorAgent()

    # 1. 启动 Expert 子进程
    print("📡 启动 Expert Agent 子进程...")
    coordinator.start_expert()
    print(f"✅ Expert 已连接: {coordinator.expert_card.get('name', 'unknown')}")
    print(f"   技能列表: {coordinator.expert_card.get('skills', [])}")
    print()

    # 2. 展示通信过程
    print("-" * 60)
    print("📋 A2A 协议通信演示:")
    print("-" * 60)
    print()
    print("通信方式: JSON-RPC 2.0 over stdio (管道)")
    print("消息格式: newline-delimited JSON (每行一个消息)")
    print()

    # 示例请求
    sample_req = JSONRPC.request("task/execute", {
        "task_id": "demo-001",
        "skill": "transformer_theory",
        "input": "什么是Self-Attention?",
    })
    print(f"→ Coordinator 发送:\n  {sample_req}")
    print()

    sample_resp = JSONRPC.response({
        "id": "demo-001", "state": "completed",
        "output_text": "Self-Attention 是...",
    }, "req-123")
    print(f"← Expert 返回:\n  {sample_resp}")
    print()

    # 3. 处理命令行参数或进入交互模式
    if len(sys.argv) > 1 and sys.argv[1] != "--expert":
        # 命令行模式: 直接回答一个问题
        question = " ".join(sys.argv[1:])
        print("-" * 60)
        print(f"❓ 问题: {question}")
        skill = coordinator.route_skill(question)
        print(f"🔀 路由: {skill}")
        print(f"📤 发送 Task 给 Expert...")
        print("-" * 60)

        answer = coordinator.ask(question)
        print(f"\n💡 Expert 回答:\n{answer}")

    else:
        # 交互模式
        print("-" * 60)
        print("💬 交互模式 (输入问题，输入 'quit' 退出)")
        print("-" * 60)
        print()

        while True:
            try:
                question = input("❓ 你的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            skill = coordinator.route_skill(question)
            print(f"🔀 路由到: {skill}")
            print(f"📤 发送 Task...")

            answer = coordinator.ask(question)
            print(f"\n💡 回答:\n{answer}\n")

    coordinator.stop()
    print("\n👋 A2A 系统已关闭")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    if "--expert" in sys.argv:
        # 作为 Expert 子进程运行
        expert = ExpertAgent()
        expert.run_stdio_loop()
    else:
        # 作为 Coordinator 主进程运行
        interactive_demo()
