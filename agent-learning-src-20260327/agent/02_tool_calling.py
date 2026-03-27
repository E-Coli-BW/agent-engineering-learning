"""
Level 2: 工具调用 —— Agent 的核心能力
=======================================

核心问题: LLM 只能生成文本，但现实世界需要 "动作"。怎么办？

答案: 让 LLM 输出 "我想调用 XXX 工具"，然后程序去执行，把结果喂回给 LLM。

原理说明:
---------
所谓 "工具调用" 并不是什么黑魔法，本质上是：

  1. 你把工具的描述 (名字 + 参数 + 用途) 塞进 system prompt
  2. LLM 看到问题后，判断需要哪个工具，输出一个 JSON 格式的调用请求
  3. 你的程序解析这个 JSON，执行对应的 Python 函数
  4. 把执行结果作为新消息喂回给 LLM
  5. LLM 根据工具返回的结果，生成最终回复

整个过程中，LLM 只是在 "生成文本"，只不过有些文本恰好是 JSON 格式的工具调用。

  [用户问题] → LLM 判断需要工具 → 输出 JSON → 程序执行 → 结果回传 → LLM 生成答案
                                    ↑
                            这就是 "function calling"
                            本质是结构化文本生成
"""

import json
import math
import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

MODEL_NAME = "qwen2.5:7b"


# ============================================================
# Part 1: 手动实现工具调用 (不用任何框架魔法)
# ============================================================

# ---- 定义几个工具 (就是普通 Python 函数) ----
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        # 只允许数学运算，防止代码注入
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"错误: 表达式包含不允许的字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


def get_weather(city: str) -> str:
    """获取天气信息 (模拟)"""
    # 真实场景会调 API，这里用模拟数据演示原理
    mock_data = {
        "北京": "晴，25°C，湿度 40%",
        "上海": "多云，22°C，湿度 65%",
        "深圳": "小雨，28°C，湿度 80%",
    }
    return mock_data.get(city, f"{city}: 数据暂不可用 (这是模拟数据)")


# ---- 工具注册表 ----
TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "计算数学表达式。参数: expression (字符串，如 '2+3*4')",
    },
    "get_current_time": {
        "function": get_current_time,
        "description": "获取当前时间。无参数。",
    },
    "get_weather": {
        "function": get_weather,
        "description": "获取指定城市的天气。参数: city (字符串，如 '北京')",
    },
}


def manual_tool_calling():
    """
    手动实现工具调用，不依赖任何 LangChain 的 tool calling 机制。
    让你看清楚底层到底在做什么。
    """
    print("=" * 60)
    print("Part 1: 手动实现工具调用 (看清底层原理)")
    print("=" * 60)

    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # ---- 构造 system prompt：告诉模型有哪些工具 ----
    tool_descriptions = "\n".join([
        f"  - {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])

    system_prompt = f"""你是一个助手。你可以使用以下工具来回答问题：

可用工具:
{tool_descriptions}

当你需要使用工具时，请严格按以下 JSON 格式输出（不要有其他文字）：
{{"tool": "工具名", "args": {{"参数名": "参数值"}}}}

如果不需要工具，直接回答即可。
每次只调用一个工具。"""

    print(f"📋 System Prompt (塞进了工具描述):")
    print(f"   {system_prompt[:200]}...")
    print()

    # ---- 测试几个问题 ----
    test_questions = [
        "计算 (15 + 27) * 3 等于多少？",
        "现在几点了？",
        "北京今天天气怎么样？",
        "你好，请介绍一下你自己",  # 不需要工具
    ]

    for q in test_questions:
        print(f"\n🧑 问题: {q}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=q),
        ]

        # Step 1: LLM 判断是否需要工具
        response = llm.invoke(messages)
        raw_output = response.content.strip()
        print(f"🤖 LLM 原始输出: {raw_output[:200]}")

        # Step 2: 尝试解析工具调用
        try:
            # 尝试提取 JSON
            # 有时模型会在 JSON 前后加其他文字，尝试清理
            json_str = raw_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            tool_call = json.loads(json_str.strip())
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            print(f"🔧 解析到工具调用: {tool_name}({tool_args})")

            # Step 3: 执行工具
            if tool_name in TOOLS:
                func = TOOLS[tool_name]["function"]
                if tool_args:
                    result = func(**tool_args)
                else:
                    result = func()
                print(f"📊 工具返回: {result}")

                # Step 4: 把工具结果喂回给 LLM
                messages.append(AIMessage(content=raw_output))
                messages.append(HumanMessage(content=f"工具 {tool_name} 的执行结果是: {result}\n请根据这个结果回答用户的问题。"))

                final_response = llm.invoke(messages)
                print(f"🤖 最终回复: {final_response.content[:200]}")
            else:
                print(f"❌ 未知工具: {tool_name}")

        except (json.JSONDecodeError, KeyError):
            # 不是工具调用，直接就是回答
            print(f"💬 直接回复 (未使用工具)")


# ============================================================
# Part 2: 使用 LangChain 的 Tool 机制 (封装后的版本)
# ============================================================
def langchain_tool_calling():
    """
    LangChain 的 @tool 装饰器做了什么？
      1. 自动从函数签名和 docstring 提取工具描述
      2. 自动构造符合模型要求的 tool schema (JSON Schema)
      3. 自动解析模型返回的 tool_calls
      4. 自动执行并把结果注入消息流

    但本质和 Part 1 完全一样：让 LLM 输出结构化 JSON → 程序执行 → 结果回传。
    """
    print("\n" + "=" * 60)
    print("Part 2: LangChain @tool 机制")
    print("=" * 60)

    # ---- 用 @tool 装饰器定义工具 ----
    @tool
    def calc(expression: str) -> str:
        """计算数学表达式，如 '2+3*4'"""
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return f"错误: 表达式包含不允许的字符"
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {e}"

    @tool
    def current_time() -> str:
        """获取当前日期和时间"""
        return datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    @tool
    def weather(city: str) -> str:
        """查询指定城市的天气情况"""
        mock = {"北京": "晴，25°C", "上海": "多云，22°C", "深圳": "小雨，28°C"}
        return mock.get(city, f"{city}: 暂无数据")

    tools = [calc, current_time, weather]

    # ---- 查看 LangChain 生成的 tool schema ----
    print("\n📋 LangChain 自动生成的 Tool Schema:")
    for t in tools:
        print(f"  {t.name}: {t.description}")
        print(f"    参数 Schema: {t.args_schema.model_json_schema()}")
    print()

    # ---- 绑定工具到模型 ----
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # ---- 测试 ----
    questions = [
        "帮我算一下 123 * 456",
        "现在几点？",
    ]

    tools_by_name = {t.name: t for t in tools}

    for q in questions:
        print(f"\n🧑 问题: {q}")

        response = llm_with_tools.invoke([HumanMessage(content=q)])

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"🔧 模型请求调用: {tc['name']}({tc['args']})")

                # 执行工具
                tool_func = tools_by_name[tc["name"]]
                result = tool_func.invoke(tc["args"])
                print(f"📊 执行结果: {result}")
        else:
            print(f"💬 直接回复: {response.content[:200]}")


# ============================================================
# Part 3: 理解 Tool Calling 的局限性
# ============================================================
def limitations_demo():
    """
    工具调用不是万能的，理解它的局限很重要 (面试题!)

    常见问题:
      1. 模型可能选错工具
      2. 模型可能提取错参数
      3. 模型可能不调工具而直接编造答案
      4. 工具执行可能失败

    这些问题在实际项目中都需要处理，也是面试高频考点。
    """
    print("\n" + "=" * 60)
    print("Part 3: 工具调用的局限性与注意事项")
    print("=" * 60)

    print("""
    ⚠️ 面试必知的工具调用陷阱:

    1. 幻觉风险:
       模型可能不调工具，直接编造答案
       → 解决: 强制要求 tool_choice="required"

    2. 参数提取错误:
       用户说 "北京天气" → 模型可能传 "Beijing" 而非 "北京"
       → 解决: 工具描述要清晰，参数要有 enum 约束

    3. 工具选择错误:
       类似的工具太多时，模型可能选错
       → 解决: 工具命名和描述要有区分度

    4. 错误处理:
       工具执行失败 (网络超时、参数非法等)
       → 解决: 工具函数内部 try-except，返回友好错误信息

    5. 多步工具调用:
       "北京和上海哪个热？" → 需要调两次 weather
       → 解决: 循环调用 (这就是 ReAct Agent 的思路!)
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 2: 工具调用 —— Agent 的核心能力\n")

    # Part 1: 手动实现 (理解原理)
    manual_tool_calling()

    # Part 2: LangChain 封装版
    langchain_tool_calling()

    # Part 3: 局限性
    limitations_demo()

    print("\n" + "=" * 60)
    print("✅ Level 2 完成！")
    print()
    print("关键收获:")
    print("  1. 工具调用 = 让 LLM 输出结构化 JSON + 程序执行 + 结果回传")
    print("  2. LLM 本身不执行任何代码，它只是'建议'调用哪个工具")
    print("  3. @tool 装饰器自动提取函数签名和描述，构造 schema")
    print("  4. 需要处理: 选错工具、参数错误、幻觉、执行失败等")
    print("  5. 多步工具调用 → 需要循环 → 这就是 Agent 的核心!")
    print()
    print("👉 下一步: python agent/03_react_agent.py")
    print("=" * 60)
