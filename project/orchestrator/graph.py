"""
Orchestrator Graph — LangGraph 驱动的 Multi-Agent 编排
======================================================

状态机流转:
  START → plan (LLM 分析需要哪些 Agent)
        → execute (并行/串行调用 Sub-Agent)
        → synthesize (LLM 整合结果)
        → should_continue? (判断是否需要补充)
            → Yes: 回到 plan
            → No:  END

关键设计:
  - Orchestrator LLM 做路由决策，不是 if/else 硬编码
  - Sub-Agent 返回结构化结果，Orchestrator 负责整合
  - max_iterations 防止死循环
  - 每个 Sub-Agent 是纯函数，可以独立测试
"""

import os
import json
import logging
import urllib.request
from typing import Literal

from langgraph.graph import StateGraph, END

from project.orchestrator.state import OrchestratorState, AgentResult
from project.orchestrator.agents.knowledge import knowledge_agent
from project.orchestrator.agents.calculator import calculator_agent
from project.orchestrator.agents.code import code_agent

logger = logging.getLogger("orchestrator.graph")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")
MAX_ITERATIONS = 3


def _call_llm(messages: list[dict], temperature: float = 0.3) -> str:
    """调用 Ollama LLM"""
    data = json.dumps({
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")


# ============================================================
# Graph 节点
# ============================================================

def plan_node(state: OrchestratorState) -> dict:
    """
    Plan 节点 — LLM 分析用户问题，决定调用哪些 Agent

    输出: agents_to_call 列表 + 执行计划文本
    """
    query = state["user_query"]
    iteration = state.get("iteration", 0)

    # 如果是重新规划（iteration > 0），带上之前的结果
    context = ""
    if state.get("agent_results"):
        context = "\n已有的 Agent 结果:\n"
        for r in state["agent_results"]:
            context += f"- {r['agent']}: {r['content'][:200]}...\n"
        context += "\n根据以上结果，判断是否还需要调用其他 Agent 补充信息。\n"

    plan_prompt = f"""你是一个 Multi-Agent 系统的调度器。分析用户问题，决定需要调用哪些 Agent。

可用的 Agent:
- knowledge: 知识检索 (RAG 向量搜索 + 知识图谱)。适合: 概念解释、原理问答、技术对比
- calculator: 数学计算 + 参数量分析。适合: 数值计算、LoRA/Transformer 参数量、统计分析
- code: 代码生成/解释/审查。适合: 写代码、解释代码、代码优化

规则:
1. 只返回 JSON，不要其他内容
2. 可以选 1-3 个 Agent，按需选择
3. 如果问题很简单，不需要 Agent，agents 返回空数组
4. 每个 agent 配一个具体的子任务 query
{context}
用户问题: {query}

返回格式:
{{"plan": "简要执行计划", "agents": [{{"name": "agent_name", "query": "给这个agent的具体问题"}}]}}"""

    response = _call_llm([{"role": "user", "content": plan_prompt}])

    # 结构化输出: Pydantic 校验 LLM 返回的 JSON
    from project.infra.structured import parse_llm_json, PlanOutput
    fallback = {"plan": "解析失败，默认使用知识检索", "agents": [{"name": "knowledge", "query": query}]}
    plan = parse_llm_json(response, PlanOutput, fallback=fallback)

    agents_to_call = [a.name for a in plan.agents]
    valid = {"knowledge", "calculator", "code"}
    agents_to_call = [a for a in agents_to_call if a in valid]

    if not agents_to_call:
        agents_to_call = ["knowledge"]

    logger.info("Plan: %s → agents=%s", plan.plan, agents_to_call)

    return {
        "plan": plan.plan,
        "agents_to_call": agents_to_call,
        "iteration": iteration + 1,
        "_agent_queries": {a.name: a.query for a in plan.agents},
    }


def execute_node(state: OrchestratorState) -> dict:
    """
    Execute 节点 — 调用 Sub-Agent 并收集结果

    根据 plan 中的 agents_to_call，逐个调用对应的 Agent。
    """
    agents_to_call = state.get("agents_to_call", [])
    agent_queries = state.get("_agent_queries", {})
    user_query = state["user_query"]
    results = []

    agent_map = {
        "knowledge": knowledge_agent,
        "calculator": calculator_agent,
        "code": code_agent,
    }

    for agent_name in agents_to_call:
        agent_fn = agent_map.get(agent_name)
        if not agent_fn:
            continue

        query = agent_queries.get(agent_name, user_query)
        logger.info("Calling %s: %s", agent_name, query[:80])

        try:
            content = agent_fn(query)
            results.append(AgentResult(
                agent=agent_name,
                content=content,
                success=True,
            ))
            logger.info("%s completed: %d chars", agent_name, len(content))
        except Exception as e:
            logger.error("%s failed: %s", agent_name, e)
            results.append(AgentResult(
                agent=agent_name,
                content=f"Agent 执行失败: {e}",
                success=False,
            ))

    return {"agent_results": results}


def synthesize_node(state: OrchestratorState) -> dict:
    """
    Synthesize 节点 — LLM 整合所有 Agent 的结果生成最终答案
    """
    query = state["user_query"]
    results = state.get("agent_results", [])

    if not results:
        return {"final_answer": "没有获取到任何信息，无法回答。", "needs_more": False}

    # 构造合并 prompt
    results_text = ""
    for r in results:
        status = "✅" if r["success"] else "❌"
        results_text += f"\n--- {status} {r['agent']} ---\n{r['content']}\n"

    synth_prompt = f"""你是一个 AI 助手。根据多个专业 Agent 收集的信息，为用户生成一个完整、结构化的回答。

用户问题: {query}

各 Agent 提供的信息:
{results_text}

要求:
1. 整合所有 Agent 的信息，不要简单拼接
2. 如果有计算结果，确保数字准确
3. 如果有代码，保持代码格式
4. 用中文回答，结构清晰 (可以用标题、列表、表格)
5. 如果某个 Agent 失败了，用其他 Agent 的结果补充"""

    answer = _call_llm([{"role": "user", "content": synth_prompt}])

    return {
        "final_answer": answer,
        "needs_more": False,  # 简单版: 一轮合成就够
    }


def should_continue(state: OrchestratorState) -> Literal["plan", "end"]:
    """
    条件路由: 是否需要再调更多 Agent
    """
    if state.get("needs_more", False) and state.get("iteration", 0) < MAX_ITERATIONS:
        logger.info("需要更多信息，重新规划 (iteration=%d)", state.get("iteration", 0))
        return "plan"
    return "end"


# ============================================================
# 构建 Graph
# ============================================================

def build_orchestrator_graph() -> StateGraph:
    """
    构建 Orchestrator StateGraph

    流程:
      plan → execute → synthesize → should_continue?
                                        ├── "plan" (循环)
                                        └── "end"  (结束)
    """
    graph = StateGraph(OrchestratorState)

    # 添加节点
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)

    # 添加边
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_conditional_edges("synthesize", should_continue, {
        "plan": "plan",
        "end": END,
    })

    return graph.compile()


# ============================================================
# 便捷调用函数
# ============================================================

def run_orchestrator(query: str) -> dict:
    """
    运行 Orchestrator，返回完整结果

    Returns:
        {"answer": str, "plan": str, "agents_used": [str], "iterations": int}
    """
    graph = build_orchestrator_graph()

    initial_state = {
        "user_query": query,
        "plan": "",
        "agents_to_call": [],
        "agent_results": [],
        "final_answer": "",
        "iteration": 0,
        "needs_more": False,
    }

    result = graph.invoke(initial_state)

    agents_used = list(set(r["agent"] for r in result.get("agent_results", [])))

    return {
        "answer": result.get("final_answer", ""),
        "plan": result.get("plan", ""),
        "agents_used": agents_used,
        "iterations": result.get("iteration", 0),
        "agent_results": result.get("agent_results", []),
    }
