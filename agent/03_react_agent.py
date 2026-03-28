"""
Level 3: 手写 ReAct Agent —— 理解 Agent 的思维循环
====================================================

核心问题: 如果一个问题需要多步工具调用，怎么办？

答案: ReAct (Reasoning + Acting) 循环。

ReAct 原理:
-----------
ReAct 论文 (Yao et al., 2022) 提出了一个简单而强大的框架:

  循环 {
    1. Thought (思考): 分析当前状况，决定下一步做什么
    2. Action  (行动): 调用一个工具
    3. Observation (观察): 获取工具的返回结果
  } 直到 → Final Answer (最终答案)

示例:
  问题: "北京和上海哪个城市更热？"

  Thought 1: 我需要先查北京的天气
  Action 1: get_weather("北京")
  Observation 1: 晴，25°C

  Thought 2: 再查上海的天气
  Action 2: get_weather("上海")
  Observation 2: 多云，22°C

  Thought 3: 北京 25°C > 上海 22°C，北京更热
  Final Answer: 北京更热，北京 25°C，上海 22°C。

这个 "思考-行动-观察" 的循环就是 Agent 的本质！

本文件会:
  1. 完全手写一个 ReAct Agent (不用 LangGraph)
  2. 让你看到每一步的思考过程
  3. 理解循环终止条件和最大步数保护
"""

import json
import re
import datetime
import math
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

MODEL_NAME = "qwen2.5:7b"


# ============================================================
# 1. 工具定义 (和 Level 2 类似，多加几个)
# ============================================================
class ToolRegistry:
    """工具注册中心：管理所有可用工具"""

    def __init__(self):
        self.tools = {}

    def register(self, name: str, func, description: str):
        self.tools[name] = {"function": func, "description": description}

    def execute(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return f"错误: 工具 '{name}' 不存在。可用工具: {list(self.tools.keys())}"
        try:
            return str(self.tools[name]["function"](**args))
        except Exception as e:
            return f"工具执行错误: {e}"

    def get_descriptions(self) -> str:
        lines = []
        for name, info in self.tools.items():
            lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)


# 创建工具注册中心并注册工具
registry = ToolRegistry()

registry.register(
    "calculator", 
    lambda expression: str(eval(expression)) if all(c in "0123456789+-*/().% " for c in expression) else "非法表达式",
    "计算数学表达式。参数: expression (str)"
)

registry.register(
    "get_weather",
    lambda city: {"北京": "晴，25°C，湿度40%", "上海": "多云，22°C，湿度65%", 
                  "深圳": "小雨，28°C，湿度80%", "广州": "阴，27°C，湿度70%"}.get(city, f"{city}: 暂无数据"),
    "查询城市天气。参数: city (str)"
)

registry.register(
    "get_time",
    lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "获取当前时间。无参数。"
)

registry.register(
    "search_knowledge",
    lambda query: {
        "注意力机制": "注意力机制是 Transformer 的核心组件，通过 Q、K、V 矩阵计算序列元素间的相关性。",
        "梯度下降": "梯度下降是最基本的优化算法，沿着损失函数梯度的反方向更新参数。",
        "反向传播": "反向传播利用链式法则，从输出层向输入层逐层计算梯度。",
    }.get(query, f"未找到关于 '{query}' 的知识 (模拟知识库)"),
    "搜索知识库。参数: query (str)"
)


# ============================================================
# 2. 手写 ReAct Agent
# ============================================================
class ReActAgent:
    """
    手写的 ReAct Agent

    核心循环:
      while not done and step < max_steps:
          1. LLM 生成 Thought + Action
          2. 解析并执行 Action
          3. 把 Observation 加入消息历史
          4. 检查是否到达 Final Answer
    """

    # ReAct 的 system prompt —— 这是 Agent 的 "灵魂"
    SYSTEM_PROMPT = """你是一个能使用工具的智能助手。

可用工具:
{tool_descriptions}

你必须按照以下格式来思考和行动:

Thought: <分析当前状况，思考下一步该做什么>
Action: <工具名>
Action Input: <参数，JSON 格式>

当你收到工具返回的结果后，会以 Observation 的形式呈现给你。

如果你已经有了足够的信息来回答问题，使用:
Thought: <总结推理过程>
Final Answer: <最终回答>

重要规则:
1. 每次只调用一个工具
2. 必须先 Thought 再 Action
3. 如果工具返回错误，分析原因并尝试换一种方式
4. 不要编造工具不存在的信息"""

    def __init__(self, model_name: str = MODEL_NAME, max_steps: int = 6, verbose: bool = True):
        self.llm = ChatOllama(model=model_name, temperature=0)
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, question: str) -> str:
        """运行 ReAct 循环"""
        print(f"\n{'='*60}")
        print(f"🧑 问题: {question}")
        print(f"{'='*60}")

        # 初始化消息
        system_msg = self.SYSTEM_PROMPT.format(
            tool_descriptions=registry.get_descriptions()
        )
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=question),
        ]

        # ---- ReAct 循环 ----
        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n--- Step {step}/{self.max_steps} ---")

            # 1. LLM 生成 Thought + Action (或 Final Answer)
            response = self.llm.invoke(messages)
            output = response.content.strip()

            if self.verbose:
                print(f"🤖 LLM 输出:\n{output}")

            # 2. 检查是否是最终答案
            if "Final Answer:" in output:
                final = output.split("Final Answer:")[-1].strip()
                print(f"\n✅ 最终答案: {final}")
                return final

            # 3. 解析 Action
            action_name, action_input = self._parse_action(output)

            if action_name is None:
                # 解析失败，让 LLM 重试
                if self.verbose:
                    print(f"⚠️ 无法解析 Action，要求重试")
                messages.append(AIMessage(content=output))
                messages.append(HumanMessage(
                    content="请按格式输出。如果已有答案，用 'Final Answer: xxx'。否则用 'Action: 工具名' 和 'Action Input: 参数'。"
                ))
                continue

            if self.verbose:
                print(f"🔧 Action: {action_name}({action_input})")

            # 4. 执行工具
            observation = registry.execute(action_name, action_input)
            if self.verbose:
                print(f"📊 Observation: {observation}")

            # 5. 把 AI 输出和工具结果加入消息历史
            messages.append(AIMessage(content=output))
            messages.append(HumanMessage(content=f"Observation: {observation}"))

        # 超过最大步数
        print(f"\n⚠️ 达到最大步数 ({self.max_steps})，强制结束")
        return "抱歉，我无法在有限步骤内完成这个问题。"

    def _parse_action(self, text: str) -> tuple:
        """解析 LLM 输出中的 Action 和 Action Input"""
        try:
            # 提取 Action
            action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
            if not action_match:
                return None, None
            action_name = action_match.group(1).strip()

            # 提取 Action Input
            input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.DOTALL)
            if input_match:
                raw_input = input_match.group(1).strip()
                try:
                    # 尝试 JSON 解析
                    action_input = json.loads(raw_input)
                except json.JSONDecodeError:
                    # 如果不是 JSON，作为单参数处理
                    action_input = {"expression": raw_input} if action_name == "calculator" else \
                                   {"city": raw_input} if action_name == "get_weather" else \
                                   {"query": raw_input} if action_name == "search_knowledge" else {}
            else:
                action_input = {}

            return action_name, action_input

        except Exception:
            return None, None


# ============================================================
# 3. 测试
# ============================================================
def main():
    agent = ReActAgent(verbose=True)

    # ---- 测试用例 ----
    test_questions = [
        # 单步工具调用
        "现在几点了？",

        # 多步工具调用 (需要比较)
        "北京和深圳哪个城市更热？温差多少度？",

        # 需要计算 + 知识
        "请帮我计算 (25 + 37) * 12 的结果",

        # 不需要工具
        "你好，你是谁？",
    ]

    results = []
    for q in test_questions:
        answer = agent.run(q)
        results.append((q, answer))

    # ---- 总结 ----
    print("\n" + "=" * 60)
    print("📋 所有结果汇总:")
    print("=" * 60)
    for q, a in results:
        print(f"  Q: {q}")
        print(f"  A: {a[:100]}")
        print()


if __name__ == "__main__":
    print("🚀 Level 3: 手写 ReAct Agent\n")

    main()

    print("\n" + "=" * 60)
    print("✅ Level 3 完成！")
    print()
    print("关键收获:")
    print("  1. ReAct = Thought → Action → Observation 循环")
    print("  2. Agent 的 '智能' 来自 system prompt 的设计")
    print("  3. 每一步 LLM 只是在生成文本，程序负责解析和执行")
    print("  4. 需要最大步数保护，防止无限循环")
    print("  5. 解析 LLM 输出是最脆弱的环节 (格式不一定稳定)")
    print()
    print("💡 手写的问题:")
    print("   - 输出格式解析脆弱")
    print("   - 状态管理靠消息列表，复杂了就乱")
    print("   - 没有条件分支、并行等高级流程控制")
    print("   → 这就是为什么需要 LangGraph!")
    print()
    print("👉 下一步: python agent/04_langgraph_agent.py")
    print("=" * 60)
