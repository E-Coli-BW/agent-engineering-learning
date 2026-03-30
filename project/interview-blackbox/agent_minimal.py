"""
Black Box Optimization Agent - 面试核心版（~80行）
只保留最关键的逻辑，面试时在白板/编辑器上能写完的量。
"""
import asyncio, json, os
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# 配置
MCP_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8339/mcp")
llm = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
)
MODEL = os.environ.get("MODEL", "qwen2.5:7b")

SYSTEM = """You are an optimization agent. Find (x,y) that MAXIMIZES a black-box function.
- query(x,y) returns score. judge(x,y) submits answer. reset() restarts.
- Strategy: coarse grid scan (step=5) → analyze → fine search (±2) → judge best point.
- You MUST call tools, not just describe."""


async def run():
    # 1. 连接 MCP Server
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"Tools: {[t.name for t in tools]}")

            # 2. 转换 tool schema → OpenAI function calling 格式
            oai_tools = [
                {"type": "function", "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {},
                }}
                for t in tools
            ]

            # 3. Agent Loop
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Reset then find the maximum. Go!"},
            ]

            for i in range(50):
                resp = llm.chat.completions.create(
                    model=MODEL, messages=messages,
                    tools=oai_tools, tool_choice="auto",
                )
                msg = resp.choices[0].message
                messages.append(msg.model_dump())

                if msg.content:
                    print(f"\n🤖 {msg.content[:200]}")

                # 没有 tool call → 提醒继续
                if not msg.tool_calls:
                    messages.append({"role": "user", "content": "Call a tool now!"})
                    continue

                # 执行 tool calls
                done = False
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = await session.call_tool(name, args)
                    text = result.content[0].text
                    print(f"🔧 {name}({args}) → {text[:120]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text,
                    })
                    if name == "judge":
                        done = True

                if done:
                    print("\n✅ Done!")
                    break


if __name__ == "__main__":
    asyncio.run(run())
