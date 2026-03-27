"""
Level 4: LangGraph Agent —— 生产级 Agent 框架
===============================================

为什么需要 LangGraph？
--------------------
Level 3 手写的 ReAct Agent 有几个痛点:
  1. 解析 LLM 输出格式不稳定 (正则匹配容易出错)
  2. 流程控制靠 if-else，复杂了就难维护
  3. 没有状态管理、检查点、回溯等能力
  4. 很难支持多 Agent 协作

LangGraph 的核心思想:
  把 Agent 的工作流建模为一个 **有向图 (Graph)**:
    - 节点 (Node) = 一个处理步骤 (调 LLM、执行工具、做判断等)
    - 边 (Edge) = 流转条件 (成功→下一步，需要工具→执行工具，完成→结束)
    - 状态 (State) = 贯穿整个图的共享数据

  这和你在数据结构课学的有向图是一回事!

LangGraph vs LangChain 的关系 (面试题!):
  - LangChain: 提供 LLM、Tool、Prompt 等基础组件 (积木)
  - LangGraph: 提供图结构来编排这些组件 (搭建方式)
  - 类比: LangChain = 砖瓦水泥, LangGraph = 建筑设计图纸

本文件会:
  1. 解释 LangGraph 的核心概念 (State, Node, Edge)
  2. 从最简单的图开始，逐步增加复杂度
  3. 用 LangGraph 重写 ReAct Agent
  4. 展示条件分支、循环等高级流程
"""

import json
import datetime
import operator
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

MODEL_NAME = "qwen2.5:7b"


# ============================================================
# Part 1: LangGraph 核心概念 —— 最简单的图
# ============================================================
def part1_simple_graph():
    """
    最简单的 LangGraph: 一个节点，接收输入，返回输出。

    图结构:
      START → greet → END

    让你理解:
      - State: 在图中流转的数据
      - Node: 处理数据的函数
      - Edge: 节点间的连接
    """
    print("=" * 60)
    print("Part 1: 最简单的 LangGraph (理解核心概念)")
    print("=" * 60)

    # ---- Step 1: 定义 State (图中流转的数据) ----
    # TypedDict 就是一个有类型约束的字典
    class SimpleState(TypedDict):
        input: str       # 用户输入
        output: str      # 处理结果

    # ---- Step 2: 定义 Node (处理函数) ----
    def greet(state: SimpleState) -> dict:
        """节点函数：接收 state，返回要更新的字段"""
        name = state["input"]
        return {"output": f"你好 {name}！欢迎学习 LangGraph！"}

    # ---- Step 3: 构建图 ----
    graph = StateGraph(SimpleState)
    graph.add_node("greet", greet)           # 添加节点
    graph.add_edge(START, "greet")           # START → greet
    graph.add_edge("greet", END)             # greet → END

    # ---- Step 4: 编译并运行 ----
    app = graph.compile()

    result = app.invoke({"input": "LangGraph 学习者"})
    print(f"  输入: {result['input']}")
    print(f"  输出: {result['output']}")

    # ---- 打印图结构 ----
    print(f"\n  📊 图结构:")
    print(f"    START → greet → END")
    print(f"    (这就是最简单的 LangGraph!)")


# ============================================================
# Part 2: 条件分支 —— 让图变智能
# ============================================================
def part2_conditional_graph():
    """
    带条件分支的图:

      START → classify → [需要工具?]
                            ├── Yes → use_tool → respond → END
                            └── No  → respond → END

    让你理解:
      - 条件边 (conditional_edge): 根据状态决定走哪条路
      - 这就是 Agent 决策的基础!
    """
    print("\n" + "=" * 60)
    print("Part 2: 条件分支 (Agent 决策的基础)")
    print("=" * 60)

    class BranchState(TypedDict):
        question: str
        needs_tool: bool
        tool_result: str
        answer: str

    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def classify(state: BranchState) -> dict:
        """判断问题是否需要工具"""
        q = state["question"]
        # 简单规则判断 (实际中可以让 LLM 判断)
        needs_tool = any(kw in q for kw in ["计算", "天气", "时间", "几点"])
        print(f"  🔍 分类: '{q}' → {'需要工具' if needs_tool else '直接回答'}")
        return {"needs_tool": needs_tool}

    def use_tool(state: BranchState) -> dict:
        """模拟工具调用"""
        q = state["question"]
        if "时间" in q or "几点" in q:
            result = datetime.datetime.now().strftime("%H:%M:%S")
        elif "天气" in q:
            result = "北京: 晴，25°C"
        elif "计算" in q:
            result = "42 (模拟计算结果)"
        else:
            result = "工具执行完毕"
        print(f"  🔧 工具返回: {result}")
        return {"tool_result": result}

    def respond(state: BranchState) -> dict:
        """生成最终回答"""
        if state.get("tool_result"):
            prompt = f"根据工具结果 '{state['tool_result']}' 回答: {state['question']}"
        else:
            prompt = state["question"]

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content[:200]}

    # ---- 条件路由函数 ----
    def should_use_tool(state: BranchState) -> str:
        """根据 state 决定走哪条边"""
        return "use_tool" if state["needs_tool"] else "respond"

    # ---- 构建图 ----
    graph = StateGraph(BranchState)
    graph.add_node("classify", classify)
    graph.add_node("use_tool", use_tool)
    graph.add_node("respond", respond)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", should_use_tool, {
        "use_tool": "use_tool",
        "respond": "respond",
    })
    graph.add_edge("use_tool", "respond")
    graph.add_edge("respond", END)

    app = graph.compile()

    # ---- 测试 ----
    questions = ["现在几点了？", "什么是 Transformer？"]
    for q in questions:
        print(f"\n  🧑 问题: {q}")
        result = app.invoke({"question": q})
        print(f"  🤖 回答: {result['answer'][:150]}")


# ============================================================
# Part 3: LangGraph ReAct Agent (生产级实现)
# ============================================================
def part3_react_agent():
    """
    用 LangGraph 实现 ReAct Agent

    图结构:
      START → agent → [有 tool_calls?]
                         ├── Yes → tools → agent (循环!)
                         └── No  → END

    对比 Level 3 手写版:
      - 不用手动解析输出格式 (模型原生支持 tool_calls)
      - 不用手动管理消息历史 (State 自动维护)
      - 循环和终止条件由图结构定义
      - 可以加检查点、回溯、人工审批等
    """
    print("\n" + "=" * 60)
    print("Part 3: LangGraph ReAct Agent (生产级)")
    print("=" * 60)

    # ---- 定义工具 ----
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式。输入应该是一个合法的数学表达式字符串，如 '2+3*4'"""
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return f"错误: 包含不允许的字符"
            return f"计算结果: {eval(expression)}"
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气情况。输入城市名，如'北京'"""
        mock = {
            "北京": "晴天，气温 25°C，湿度 40%，东北风 3 级",
            "上海": "多云，气温 22°C，湿度 65%，东南风 2 级",
            "深圳": "小雨，气温 28°C，湿度 80%，南风 4 级",
            "广州": "阴天，气温 27°C，湿度 70%，西南风 2 级",
        }
        return mock.get(city, f"暂无 {city} 的天气数据")

    @tool
    def get_current_time() -> str:
        """获取当前的日期和时间"""
        return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    tools = [calculator, get_weather, get_current_time]

    # ---- State 定义 ----
    # 关键: messages 用 Annotated + operator.add 表示 "追加" 而非 "覆盖"
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], operator.add]

    # ---- 节点: 调用 LLM ----
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Agent 节点: 调用 LLM，让它决定是否使用工具"""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ---- 路由: 判断是否需要工具 ----
    def should_continue(state: AgentState) -> str:
        """
        检查最后一条消息是否包含 tool_calls:
          - 有 → 去 tools 节点执行
          - 没有 → 结束 (模型已经给出最终答案)
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    # ---- 构建图 ----
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))  # LangGraph 内置的工具执行节点

    # 添加边
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "agent")  # 工具执行完 → 回到 agent (循环!)

    # 编译
    app = graph.compile()

    # ---- 打印图结构 ----
    print("\n  📊 图结构:")
    print("    START → agent → [tool_calls?]")
    print("                      ├── Yes → tools → agent (循环)")
    print("                      └── No  → END")

    # ---- 测试 ----
    test_questions = [
        "帮我计算 (15 + 27) * 3",
        "北京今天天气怎么样？",
        "北京和深圳哪个更热？",
        "你好！",
    ]

    for q in test_questions:
        print(f"\n{'─'*50}")
        print(f"🧑 问题: {q}")

        result = app.invoke({
            "messages": [HumanMessage(content=q)]
        })

        # 打印完整的消息流 (展示 Agent 的思考过程)
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                continue  # 跳过用户输入
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  🔧 调用工具: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"  🤖 回答: {msg.content[:200]}")
            elif isinstance(msg, ToolMessage):
                print(f"  📊 工具返回: {msg.content[:100]}")


# ============================================================
# Part 4: 对比总结
# ============================================================
def part4_comparison():
    print("\n" + "=" * 60)
    print("Part 4: 三种实现方式对比")
    print("=" * 60)

    print("""
    ┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
    │              │ Level 2 单步调用    │ Level 3 手写 ReAct  │ Level 4 LangGraph  │
    ├──────────────┼────────────────────┼────────────────────┼────────────────────┤
    │ 多步调用      │ ❌ 不支持          │ ✅ while 循环       │ ✅ 图循环边        │
    │ 格式解析      │ 手动 JSON          │ 手动正则           │ 原生 tool_calls    │
    │ 状态管理      │ 无                 │ 消息列表           │ TypedDict State    │
    │ 流程控制      │ 线性              │ if-else            │ 条件边/图结构      │
    │ 错误恢复      │ try-except         │ 手动重试           │ 内置机制           │
    │ 可维护性      │ 简单               │ 中等               │ 好                 │
    │ 多 Agent      │ ❌                 │ 很难              │ ✅ 子图 / 多节点    │
    │ 检查点/回溯   │ ❌                 │ ❌                │ ✅ 内置             │
    │ 适用场景      │ 简单任务           │ 学习原理           │ 生产环境           │
    └──────────────┴────────────────────┴────────────────────┴────────────────────┘

    面试回答要点:
    1. LangGraph 把 Agent 工作流建模为有向图
    2. 节点 = 处理步骤，边 = 流转条件，状态 = 共享数据
    3. 相比手写循环: 更结构化、可维护、支持复杂流程
    4. 相比 LangChain 的 AgentExecutor: 更灵活、可定制
    5. 核心优势: 状态管理 + 条件分支 + 循环 + 检查点
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 4: LangGraph Agent\n")

    # Part 1: 最简单的图
    part1_simple_graph()

    # Part 2: 条件分支
    part2_conditional_graph()

    # Part 3: 完整 ReAct Agent
    part3_react_agent()

    # Part 4: 对比总结
    part4_comparison()

    print("\n" + "=" * 60)
    print("✅ Level 4 完成！Agent 模块全部学完！")
    print()
    print("关键收获:")
    print("  1. LangGraph 核心 = State + Node + Edge")
    print("  2. 条件边实现 Agent 的决策分支")
    print("  3. tools → agent 的循环边实现 ReAct 多步推理")
    print("  4. ToolNode 自动处理工具执行和结果注入")
    print("  5. 比手写更稳定、更易扩展、更适合生产")
    print()
    print("📊 你现在的技能树:")
    print("  ✅ Transformer 原理 (multi_head_attention.py)")
    print("  ✅ 模型训练 + 可视化 (char_transformer.py)")
    print("  ✅ LLM 调用原理 (agent/01)")
    print("  ✅ 工具调用原理 (agent/02)")
    print("  ✅ ReAct Agent 原理 (agent/03)")
    print("  ✅ LangGraph 生产级 Agent (agent/04)")
    print("  🔲 RAG 系统 (下一阶段)")
    print("  🔲 LoRA 微调 (之后)")
    print("=" * 60)
