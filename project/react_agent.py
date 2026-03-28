"""
ReAct Agent + MCP 工具 — 微信 AI Bot v2
========================================

演进路线 (学习路径):
  v0  agent/03_react_agent.py     — 手写 ReAct Agent (教学版，硬编码工具)
  v1  project/a2a_agent.py        — A2A Expert Agent (HTTP REST，纯 LLM 问答)
      project/wechat_bridge.py    — 微信桥接器 (iLink 协议)
  v2  project/react_agent.py      — 本文件: ReAct Agent + 工具调用 + 微信集成

v1 → v2 的核心升级:
  v1: 用户提问 → LLM 直接回答 → 返回 (纯问答，无法执行动作)
  v2: 用户提问 → LLM 思考 → 决定调什么工具 → 执行 → 拿结果 → 继续推理 → 最终回答

架构:
  ┌─────────────────────────────────────────────────────────────┐
  │  微信用户                                                    │
  │    ↓                                                        │
  │  WeChatBridge (wechat_bridge.py)                            │
  │    ↓                                                        │
  │  ReActExpertAgent (本文件)                                   │
  │    ├── LLM (Ollama) — 思考 + 决策                           │
  │    └── 工具注册表:                                           │
  │        ├── knowledge_graph_query — 知识图谱查询              │
  │        ├── rag_query             — RAG 向量检索              │
  │        ├── calculator            — 数学计算                  │
  │        ├── get_time              — 获取当前时间              │
  │        └── (可扩展更多工具...)                                │
  │                                                             │
  │  ReAct 循环:                                                │
  │    Thought → Action → Observation → Thought → ... → Answer  │
  └─────────────────────────────────────────────────────────────┘

运行方式:
  # 独立测试 ReAct Agent (无需微信)
  python project/react_agent.py "Transformer中Self-Attention的缩放因子是什么?"
  python project/react_agent.py "计算 768 的平方根"
  python project/react_agent.py "知识图谱里LoRA和什么有关?"

  # 作为 Expert Agent 服务启动 (供微信桥接器调用)
  python project/react_agent.py --serve

  # 微信桥接器连接 ReAct Agent (注意端口改为 5002)
  python project/wechat_bridge.py --expert http://localhost:5002

与 v1 的对比:
  ┌──────────────────┬─────────────────────┬──────────────────────────┐
  │                  │ v1 (a2a_agent.py)    │ v2 (react_agent.py)       │
  ├──────────────────┼─────────────────────┼──────────────────────────┤
  │ 决策方式         │ 关键词路由 (硬编码)   │ LLM 自主决策 (ReAct)      │
  │ 工具调用         │ ❌ 无               │ ✅ 动态工具注册 + 调用     │
  │ 多步推理         │ ❌ 单轮问答          │ ✅ 多步 Think-Act-Observe  │
  │ 知识图谱         │ ❌                  │ ✅ knowledge_graph_query   │
  │ RAG 检索         │ ❌                  │ ✅ rag_query               │
  │ 可扩展性         │ 加 skill 要改代码    │ 注册新工具即可             │
  │ A2A 兼容         │ ✅ 完整 A2A 端点     │ ✅ 复用同一套 A2A 端点     │
  └──────────────────┴─────────────────────┴──────────────────────────┘

已完成:
  - [x] ReAct 循环 (Thought/Action/Observation 解析)
  - [x] /api/chat 多轮消息格式 (比 /api/generate 更稳定)
  - [x] 知识图谱查询工具
  - [x] 数学计算工具
  - [x] RAG 向量检索工具
  - [x] 与 wechat_bridge.py 无缝切换 (A2A 兼容)

可扩展:
  - [ ] LangChain/LangGraph 集成 (更强的 Agent 框架)
  - [ ] 更多工具: 网页搜索、文件读写、公众号发布等
  - [ ] 工具调用结果的流式输出
"""

import os
import sys
import json
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---- 日志 ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("react-agent")


# ============================================================
# 工具注册表
# ============================================================

class ToolRegistry:
    """
    工具注册中心 — 管理 Agent 可调用的所有工具

    与 agent/03_react_agent.py 的 ToolRegistry 类似，
    但增加了 get_tool_prompt() 用于生成 system prompt。
    """

    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register(self, name: str, func, description: str, params: str = ""):
        """注册一个工具"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "params": params,
        }
        logger.info("注册工具: %s — %s", name, description)

    def execute(self, name: str, args_str: str) -> str:
        """执行一个工具"""
        if name not in self.tools:
            return f"错误: 工具 '{name}' 不存在。可用工具: {list(self.tools.keys())}"
        try:
            # 尝试解析 JSON 参数
            if args_str.strip():
                args = json.loads(args_str)
                if isinstance(args, dict):
                    result = self.tools[name]["function"](**args)
                else:
                    result = self.tools[name]["function"](args)
            else:
                result = self.tools[name]["function"]()
            return str(result)
        except json.JSONDecodeError:
            # 非 JSON，当作单个字符串参数
            try:
                result = self.tools[name]["function"](args_str.strip().strip('"\''))
                return str(result)
            except Exception as e:
                return f"工具执行错误: {e}"
        except Exception as e:
            return f"工具执行错误: {e}"

    def get_tool_descriptions(self) -> str:
        """生成工具描述 (用于 system prompt)"""
        lines = []
        for name, info in self.tools.items():
            params = f" 参数: {info['params']}" if info["params"] else ""
            lines.append(f"  - {name}: {info['description']}{params}")
        return "\n".join(lines)


# ============================================================
# 内置工具
# ============================================================

def _knowledge_graph_query(entity: str) -> str:
    """查询知识图谱 (复用 mcp_server.py 的数据)"""
    kg_triples = [
        ("Transformer", "包含", "Self-Attention"),
        ("Transformer", "包含", "Feed-Forward Network"),
        ("Transformer", "提出时间", "2017年"),
        ("Transformer", "论文", "Attention Is All You Need"),
        ("Self-Attention", "计算", "QKV矩阵"),
        ("Self-Attention", "缩放因子", "sqrt(d_k)"),
        ("Self-Attention", "变体", "Multi-Head Attention"),
        ("Multi-Head Attention", "特点", "多子空间并行计算"),
        ("GPT", "基于", "Transformer Decoder"),
        ("BERT", "基于", "Transformer Encoder"),
        ("LoRA", "用于", "参数高效微调"),
        ("LoRA", "原理", "低秩矩阵分解 ΔW=BA"),
        ("QLoRA", "改进自", "LoRA"),
        ("QLoRA", "特点", "4-bit量化+LoRA"),
        ("RAG", "步骤", "Retrieve-Augment-Generate"),
        ("RAG", "使用", "向量检索"),
        ("Graph RAG", "结合", "知识图谱+RAG"),
        ("ReAct", "循环", "Thought-Action-Observation"),
        ("Agent", "核心能力", "Tool Calling"),
        ("KV Cache", "作用", "加速自回归推理"),
        ("MCP", "连接", "Agent与工具/数据"),
        ("A2A", "连接", "Agent与Agent"),
    ]
    entity_lower = entity.lower()
    related = [f"  {h} --[{r}]--> {t}" for h, r, t in kg_triples
               if entity_lower in h.lower() or entity_lower in t.lower()]
    if not related:
        return f"未找到与 '{entity}' 相关的知识图谱条目。"
    return f"'{entity}' 的知识图谱关系:\n" + "\n".join(related)


def _calculator(expression: str) -> str:
    """安全计算数学表达式"""
    import math as _math
    allowed = set("0123456789+-*/().% sqrtabcdefghijklmnopqrstuvwxyz_")
    if not all(c in allowed for c in expression.lower().replace(" ", "")):
        return f"非法表达式: {expression}"
    # 支持 sqrt
    expr = expression.replace("sqrt", "_math.sqrt")
    try:
        return str(eval(expr))
    except Exception as e:
        return f"计算错误: {e}"


def _get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rag_query(question: str) -> str:
    """RAG 向量检索 — 查询项目知识库"""
    VECTOR_DB_DIR = DATA_DIR / "project_chroma_db"
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        if not VECTOR_DB_DIR.exists():
            return "向量库不存在。请先运行 python project/etl_pipeline.py 构建索引。"

        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings,
            collection_name=os.getenv("COLLECTION_NAME", "project_knowledge_v2"),
        )
        results = vector_store.similarity_search_with_score(question, k=3)
        if not results:
            return "未找到相关内容。"

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (相关度:{1-score:.2f}) [{source}]\n{doc.page_content[:300]}")
        return "\n---\n".join(parts)
    except ImportError:
        return "RAG 依赖未安装 (需要 langchain-ollama, chromadb)。"
    except Exception as e:
        return f"RAG 查询失败: {e}"


def _list_tools() -> str:
    """列出所有可用工具"""
    return create_default_registry().get_tool_descriptions()


def create_default_registry() -> ToolRegistry:
    """创建默认工具注册表"""
    registry = ToolRegistry()
    registry.register(
        "knowledge_graph_query",
        _knowledge_graph_query,
        "查询知识图谱中技术概念的关系。",
        'entity (str): 技术实体名，如 "Transformer"、"LoRA"',
    )
    registry.register(
        "calculator",
        _calculator,
        "计算数学表达式。支持 sqrt。",
        'expression (str): 如 "768**0.5"、"sqrt(512)"',
    )
    registry.register(
        "get_time",
        _get_time,
        "获取当前日期和时间。",
        "无参数",
    )
    registry.register(
        "rag_query",
        _rag_query,
        "从项目知识库中检索相关文档。适合查找具体实现细节。",
        'question (str): 检索问题，如 "LoRA的实现代码"',
    )
    return registry


# ============================================================
# ReAct Agent
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
    ReAct Agent — 带工具调用的智能体

    对比 agent/03_react_agent.py:
      - 那个是教学版，展示 ReAct 原理
      - 这个是生产版，集成到 A2A + 微信体系

    对比 project/a2a_agent.py:
      - 那个是纯 LLM 问答 (prompt → response)
      - 这个是 ReAct 循环 (Think → Act → Observe → ...)
    """

    def __init__(self, model: str = None, max_steps: int = 4):
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.max_steps = max_steps
        self.registry = create_default_registry()

    def _call_ollama_chat(self, messages: list[dict]) -> str:
        """调用 Ollama /api/chat — 支持 messages 格式，多轮更稳定"""
        import urllib.request
        data = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")

    def run(self, question: str) -> str:
        """
        执行 ReAct 循环

        使用 /api/chat 的 messages 格式:
          system → user → assistant → user(observation) → assistant → ...
        """
        system_prompt = REACT_SYSTEM_PROMPT.format(
            tool_descriptions=self.registry.get_tool_descriptions()
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_steps + 1):
            logger.info("ReAct Step %d/%d", step, self.max_steps)

            output = self._call_ollama_chat(messages)
            logger.info("LLM [step %d]:\n%s", step, output[:300])

            # 把 LLM 输出加到消息历史
            messages.append({"role": "assistant", "content": output})

            # 检查是否有 Final Answer
            final = self._extract_final_answer(output)
            if final:
                logger.info("✅ Final Answer (step %d): %s", step, final[:100])
                return final

            # 解析 Action
            action, action_input = self._parse_action(output)
            if action:
                logger.info("🔧 工具调用: %s(%s)", action, action_input[:80])
                observation = self.registry.execute(action, action_input)
                logger.info("📋 工具结果: %s", observation[:200])
                # 把工具结果作为 user 消息追加 (模拟 Observation)
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请根据以上信息继续思考，给出 Final Answer 或调用下一个工具。"
                })
            else:
                # 没有 Action 也没有 Final Answer — 可能 LLM 直接回答了
                logger.info("⚠️  无标准格式，当作直接回答")
                return output.strip()

        # 超过最大步数，强制要求 Final Answer
        logger.warning("达到最大步数 %d，强制结束", self.max_steps)
        messages.append({
            "role": "user",
            "content": "你已经获得了足够的信息。请现在直接给出 Final Answer:"
        })
        final_output = self._call_ollama_chat(messages)
        final = self._extract_final_answer(final_output)
        return final if final else final_output.strip()

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """提取 Final Answer"""
        # 支持 "Final Answer:" 后跟内容
        match = re.search(r"Final\s*Answer\s*[:：]\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # 去掉可能的 trailing Thought/Action
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
            # 清理引号
            action_input = action_input.strip('"\'')
            return action, action_input
        return None, ""


# ============================================================
# A2A 兼容的 FastAPI Server (复用 a2a_agent.py 的协议)
# ============================================================

def create_react_app():
    """创建 ReAct Agent 的 FastAPI 应用 — 兼容 A2A 协议"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="ReAct Agent with Tools", version="2.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    AGENT_PORT = int(os.getenv("REACT_PORT", "5002"))
    agent = ReActAgent()

    # Agent Card
    agent_card = {
        "name": "ReAct Agent with Tools",
        "description": "带工具调用的 ReAct Agent。支持知识图谱查询、数学计算、RAG 检索等。",
        "url": f"http://localhost:{AGENT_PORT}",
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


# ============================================================
# CLI 入口
# ============================================================

def main():
    if "--serve" in sys.argv:
        import uvicorn
        port = int(os.getenv("REACT_PORT", "5002"))
        print(f"🚀 ReAct Agent Server (port {port})")
        print(f"   Tools: {list(create_default_registry().tools.keys())}")
        print(f"   Agent Card: http://localhost:{port}/.well-known/agent.json")
        print(f"   Task: POST http://localhost:{port}/tasks/send")
        print()
        print(f"   微信桥接器连接: python project/wechat_bridge.py --expert http://localhost:{port}")
        app = create_react_app()
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    elif len(sys.argv) > 1:
        question = " ".join(a for a in sys.argv[1:] if a != "--serve")
        if not question:
            question = "Transformer的Self-Attention缩放因子是什么？"
        agent = ReActAgent()
        print(f"\n{'='*60}")
        print(f"❓ 问题: {question}")
        print(f"🔧 可用工具: {list(agent.registry.tools.keys())}")
        print(f"{'='*60}\n")
        answer = agent.run(question)
        print(f"\n{'='*60}")
        print(f"💡 最终回答:\n{answer}")
        print(f"{'='*60}")

    else:
        print("ReAct Agent with Tools (v2)")
        print()
        print("用法:")
        print("  测试:   python project/react_agent.py '你的问题'")
        print("  服务:   python project/react_agent.py --serve")
        print()
        print("示例:")
        print("  python project/react_agent.py '知识图谱里LoRA和什么有关?'")
        print("  python project/react_agent.py '计算 768 的平方根'")
        print("  python project/react_agent.py '现在几点了?'")
        print()
        print("对比 v1 (纯问答):  python project/a2a_agent.py --serve")
        print("对比 v2 (带工具):  python project/react_agent.py --serve")


if __name__ == "__main__":
    main()
