"""
Orchestrator 共享状态
=====================

LangGraph StateGraph 的核心是 TypedDict 状态对象，
所有节点读写同一个状态，实现 Agent 间数据传递。
"""

from typing import TypedDict, Annotated, Literal
from operator import add


class AgentResult(TypedDict):
    """单个 Sub-Agent 的执行结果"""
    agent: str        # agent 名称
    content: str      # 结果内容
    success: bool     # 是否成功


class OrchestratorState(TypedDict):
    """
    Orchestrator 全局状态

    LangGraph 的 Annotated[list, add] 表示:
    每个节点 return 的值会 append 到列表，不是覆盖。
    """
    # 用户输入
    user_query: str

    # Orchestrator 的路由决策
    plan: str  # LLM 生成的执行计划
    agents_to_call: list[str]  # 需要调用的 agent 列表

    # Sub-Agent 执行结果 (累积)
    agent_results: Annotated[list[AgentResult], add]

    # 最终合成的回答
    final_answer: str

    # 控制流
    iteration: int  # 当前迭代次数 (防止死循环)
    needs_more: bool  # 是否需要再调更多 Agent
