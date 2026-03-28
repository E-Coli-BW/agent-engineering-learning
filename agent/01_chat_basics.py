"""
Level 1: LLM 调用的本质
========================
在学 Agent 之前，先搞清楚 "调用一个大模型" 底层到底发生了什么。

原理说明:
---------
1. Ollama 在本地启动了一个 HTTP 服务 (默认 http://localhost:11434)
2. 你发一个 POST 请求，body 里包含 model 名字和 messages
3. 模型做 next-token prediction，流式返回结果
4. LangChain 的 ChatOllama 只是把这个过程封装成了 Python 对象

本质上就是：
  输入: [消息列表]  →  模型推理  →  输出: 一条回复

这和你在 char_transformer.py 里做的 model.generate() 是一模一样的原理，
只不过模型更大 (7B vs 你的几百K)，训练数据更多，经过了 RLHF 对齐。
"""

import json
import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

MODEL_NAME = "qwen2.5:7b"

# ============================================================
# Part 1: 不用任何框架，直接用 HTTP 请求调用 Ollama
# ============================================================
def raw_ollama_call():
    """
    最底层的调用方式：直接发 HTTP 请求。
    这就是所有框架底层在做的事。
    """
    print("=" * 60)
    print("Part 1: 裸 HTTP 请求调用 Ollama")
    print("=" * 60)

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手，用中文回答。"},
            {"role": "user", "content": "什么是注意力机制？用一句话解释。"}
        ],
        "stream": False  # 非流式，等全部生成完再返回
    }

    print(f"📤 请求 URL: {url}")
    print(f"📤 请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print()

    response = requests.post(url, json=payload)
    result = response.json()

    print(f"📥 响应状态: {response.status_code}")
    print(f"📥 模型回复: {result['message']['content']}")
    print(f"📥 Token 统计: prompt={result.get('prompt_eval_count', '?')}, "
          f"completion={result.get('eval_count', '?')}")
    print()

    return result


# ============================================================
# Part 2: 用 LangChain 封装调用（本质完全一样）
# ============================================================
def langchain_call():
    """
    LangChain 的 ChatOllama 做了什么？
      1. 帮你构造 HTTP 请求
      2. 帮你解析响应
      3. 把消息封装成 Python 对象 (HumanMessage, AIMessage 等)
      4. 支持流式、回调、重试等生产功能

    但核心就是 Part 1 里的 requests.post()。
    """
    print("=" * 60)
    print("Part 2: LangChain ChatOllama 调用")
    print("=" * 60)

    # 初始化模型 (底层：记住了 Ollama 的地址和模型名)
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.7,  # 控制随机性：0=确定性，1=随机
    )

    # 构造消息列表 (和 HTTP 请求里的 messages 完全对应)
    messages = [
        SystemMessage(content="你是一个简洁的助手，用中文回答。"),
        HumanMessage(content="什么是注意力机制？用一句话解释。"),
    ]

    print(f"📤 消息列表:")
    for msg in messages:
        print(f"   [{msg.type}] {msg.content}")
    print()

    # 调用模型 (底层：发 HTTP 请求到 Ollama)
    response = llm.invoke(messages)

    print(f"📥 回复类型: {type(response).__name__}")
    print(f"📥 回复内容: {response.content}")
    print(f"📥 元数据: {response.response_metadata}")
    print()

    return response


# ============================================================
# Part 3: 理解消息格式 —— 对话历史的管理
# ============================================================
def conversation_demo():
    """
    大模型是无状态的！

    每次调用都是独立的。"多轮对话" 的实现方式是：
    把之前所有的对话历史都放进 messages 里重新发送。

    这就是为什么对话越长，推理越慢 —— 输入的 token 越来越多。
    这也是为什么有 context window 限制 —— 超过就会遗忘。
    """
    print("=" * 60)
    print("Part 3: 多轮对话 (理解对话历史管理)")
    print("=" * 60)

    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # 模拟多轮对话
    history = [
        SystemMessage(content="你是一个深度学习老师，简洁回答。"),
    ]

    questions = [
        "Transformer 的核心创新是什么？",
        "你刚才说的那个东西，它的时间复杂度是多少？",  # 指代消解：需要上下文
        "如何降低这个复杂度？举一个具体方法。",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n🧑 第 {i} 轮提问: {q}")

        # 添加用户消息
        history.append(HumanMessage(content=q))

        # 发送 **完整历史** 给模型
        print(f"   (发送 {len(history)} 条消息给模型，约 {sum(len(m.content) for m in history)} 字符)")
        response = llm.invoke(history)

        print(f"🤖 回复: {response.content[:200]}{'...' if len(response.content) > 200 else ''}")

        # 把模型回复加入历史 (下一轮会带上)
        history.append(AIMessage(content=response.content))

    print()
    print("💡 关键理解:")
    print("   - 模型本身是无状态的，每次都从零开始")
    print("   - '记忆' 是通过把历史消息不断重复发送实现的")
    print("   - 这就是为什么 context window 很重要")
    print("   - 这也是后面学 RAG 的动机：context 放不下怎么办？")


# ============================================================
# Part 4: 理解 temperature 和采样策略
# ============================================================
def temperature_demo():
    """
    temperature 控制生成的随机性：
      - temperature=0: 贪心解码，每次选概率最高的 token (确定性输出)
      - temperature=1: 按原始概率分布采样 (有随机性)
      - temperature>1: 放大随机性 (更多创造性/胡说八道)
      - temperature<1: 压缩分布 (更确定/更保守)

    这和你在 char_transformer.py 里的 generate() 函数中
    logits / temperature 做的事情是完全一样的！
    """
    print("=" * 60)
    print("Part 4: Temperature 对比")
    print("=" * 60)

    prompt = [HumanMessage(content="用一句话描述月亮。")]

    for temp in [0.0, 0.5, 1.0, 1.5]:
        llm = ChatOllama(model=MODEL_NAME, temperature=temp)
        responses = []
        for _ in range(2):
            r = llm.invoke(prompt)
            responses.append(r.content.strip()[:80])

        print(f"\n  temperature={temp}:")
        for j, r in enumerate(responses):
            print(f"    尝试 {j+1}: {r}")

    print()
    print("💡 观察: temperature=0 时两次结果一样，越高越不同")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 1: LLM 调用的本质\n")

    # Part 1: 最底层的 HTTP 调用
    raw_ollama_call()

    # Part 2: LangChain 封装 (本质一样)
    langchain_call()

    # Part 3: 多轮对话原理
    conversation_demo()

    # Part 4: 采样策略
    temperature_demo()

    print("\n" + "=" * 60)
    print("✅ Level 1 完成！")
    print()
    print("关键收获:")
    print("  1. LLM 调用本质是 HTTP 请求 → 模型推理 → 返回文本")
    print("  2. LangChain 只是封装，底层就是你在 char_transformer 里做的事")
    print("  3. 多轮对话靠重复发送历史消息实现，模型本身无状态")
    print("  4. temperature 控制采样随机性 (和你手写的 generate 一样)")
    print()
    print("👉 下一步: python agent/02_tool_calling.py")
    print("=" * 60)
