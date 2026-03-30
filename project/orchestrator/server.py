"""
Orchestrator Server — Multi-Agent 编排 API
============================================

端口: 5003
兼容 A2A 协议: /.well-known/agent.json, /tasks/send

同时提供:
  - POST /orchestrate — 简洁调用接口
  - GET /health
"""

import os
import sys
import uuid
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("orchestrator.server")


def create_orchestrator_app():
    """创建 Orchestrator FastAPI 应用"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Multi-Agent Orchestrator", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    PORT = int(os.getenv("ORCHESTRATOR_PORT", "5003"))

    # ---- Agent Card ----
    agent_card = {
        "name": "Multi-Agent Orchestrator",
        "description": "LangGraph 驱动的多 Agent 编排系统。自主决策调用 Knowledge/Calculator/Code Agent 协作回答复杂问题。",
        "url": f"http://localhost:{PORT}",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "multiAgent": True,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "multi_agent_qa",
                "name": "多 Agent 协作问答",
                "description": "自动分析问题，调用 Knowledge (RAG)/Calculator/Code Agent 协作回答",
                "tags": ["multi-agent", "orchestrator", "rag", "calculator", "code"],
                "examples": [
                    "帮我分析 LoRA 和 QLoRA 的区别，并计算 rank=16 时参数减少了多少",
                    "写一个 Python 函数实现 scaled dot-product attention",
                    "Transformer 的参数量怎么算？以 d_model=4096, 32层为例计算",
                ],
            },
        ],
        "subAgents": ["knowledge", "calculator", "code"],
    }

    @app.get("/.well-known/agent.json")
    async def get_agent_card():
        return JSONResponse(agent_card)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "agent": "Multi-Agent Orchestrator",
            "sub_agents": ["knowledge", "calculator", "code"],
        }

    # ---- A2A 兼容: POST /tasks/send ----
    @app.post("/tasks/send")
    async def send_task(request: Request):
        body = await request.json()
        message = body.get("message", {})
        parts = message.get("parts", [])
        user_text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")

        if not user_text:
            return JSONResponse({"error": "No text content"}, status_code=400)

        task_id = body.get("id", f"task-{uuid.uuid4().hex[:8]}")
        logger.info("Orchestrator task %s: '%s'", task_id, user_text[:80])

        try:
            from project.orchestrator.graph import run_orchestrator
            result = run_orchestrator(user_text)

            return JSONResponse({
                "id": task_id,
                "status": {
                    "state": "completed",
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": result["answer"]}],
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "metadata": {
                    "plan": result["plan"],
                    "agents_used": result["agents_used"],
                    "iterations": result["iterations"],
                },
            })
        except Exception as e:
            logger.error("Orchestrator task %s failed: %s", task_id, e)
            return JSONResponse({
                "id": task_id,
                "status": {
                    "state": "failed",
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": f"编排失败: {e}"}],
                    },
                },
            })

    # ---- 简洁接口 ----
    @app.post("/orchestrate")
    async def orchestrate(request: Request):
        """
        简洁调用接口

        请求: {"query": "你的问题"}
        响应: {"answer": "...", "plan": "...", "agents_used": [...]}
        """
        body = await request.json()
        query = body.get("query", "")
        if not query:
            return JSONResponse({"error": "query is required"}, status_code=400)

        from project.orchestrator.graph import run_orchestrator
        result = run_orchestrator(query)
        return JSONResponse(result)

    return app


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import uvicorn

    # 结构化日志
    try:
        from project.infra.logging import setup_logging
        setup_logging("orchestrator")
    except ImportError:
        logging.basicConfig(level=logging.INFO)

    port = int(os.getenv("ORCHESTRATOR_PORT", "5003"))
    print(f"🎯 Multi-Agent Orchestrator")
    print(f"   Agent Card:  http://localhost:{port}/.well-known/agent.json")
    print(f"   Task:        POST http://localhost:{port}/tasks/send")
    print(f"   Orchestrate: POST http://localhost:{port}/orchestrate")
    print(f"   Sub-Agents:  knowledge, calculator, code")
    print()

    # 支持 --test 快速测试
    if "--test" in sys.argv:
        query = " ".join(a for a in sys.argv[1:] if a != "--test") or "LoRA 的原理是什么？"
        print(f"🧪 测试: {query}\n")
        from project.orchestrator.graph import run_orchestrator
        result = run_orchestrator(query)
        print(f"📋 Plan: {result['plan']}")
        print(f"🤖 Agents: {result['agents_used']}")
        print(f"🔄 Iterations: {result['iterations']}")
        print(f"\n💡 Answer:\n{result['answer']}")
        sys.exit(0)

    app = create_orchestrator_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
