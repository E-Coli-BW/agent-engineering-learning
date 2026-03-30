"""
Black Box Optimization Agent - 参考答案 v1（全 LLM 驱动）

=== 设计思路 ===

- 完全依赖 LLM 进行决策和 tool calling
- LLM 负责每一步 query、分析、judge
- 适合大模型（如 gpt-4o），小模型容易跑偏
"""

import asyncio
import json
import os
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8339/mcp")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("MODEL", "qwen2.5:7b")

SYSTEM_PROMPT = """\
You are an optimization agent. Your goal is to find the (x, y) that MAXIMIZES a black-box function.

Constraints:
- x and y are integers in [0, 20]
- You can call query(x, y) to get the score at any point
- You want to find the global maximum using as FEW queries as possible

Strategy:
1. First, do a coarse grid scan (step=5) to understand the landscape: query points like (0,0), (0,5), (0,10)... (5,0), (5,5)... etc. This gives 25 points.
2. Analyze the results to identify the high-score region.
3. Do a fine-grained search around the best region (check all neighbors within ±2 of the best point found).
4. Once you're confident you've found the maximum, call judge(x, y) to submit.

IMPORTANT:
- After the coarse scan, ANALYZE the data before doing fine search.
- Keep track of the best score found so far.
- When you call judge(), use the (x, y) with the highest score you've seen.
- Be systematic, don't randomly guess.
- You MUST call tools to proceed. Do NOT just describe what you would do.

Available tools: query, get_history, reset, judge
"""

class MCPToolClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self._streams = None

    async def connect(self):
        self._streams = streamablehttp_client(MCP_SERVER_URL)
        (read_stream, write_stream, _) = await self._streams.__aenter__()
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()
        await self.session.initialize()
        tools = await self.session.list_tools()
        print(f"✅ Connected to MCP Server at {MCP_SERVER_URL}")
        print(f"📦 Available tools: {[t.name for t in tools.tools]}")
        return tools.tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(name, arguments)
        text = result.content[0].text if result.content else "{}"
        return text

    async def close(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._streams:
            await self._streams.__aexit__(None, None, None)

def build_openai_tools(mcp_tools) -> list[dict]:
    openai_tools = []
    for tool in mcp_tools:
        func_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
            },
        }
        openai_tools.append(func_def)
    return openai_tools

async def run_agent():
    client = MCPToolClient()
    llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    try:
        mcp_tools = await client.connect()
        openai_tools = build_openai_tools(mcp_tools)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Start the optimization game! First reset the game, then begin your systematic search. "
                "Remember: coarse grid scan first (step=5), then analyze, then fine-tune."
            )},
        ]

        max_iterations = 50
        judge_called = False

        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"🔄 Iteration {iteration + 1}")
            print(f"{'='*60}")

            response = llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())

            if assistant_message.content:
                print(f"\n🤖 LLM says: {assistant_message.content[:500]}")

            if not assistant_message.tool_calls:
                if not judge_called:
                    print("\n⚠️  LLM stopped without tool calls. Nudging it to continue...")
                    messages.append({
                        "role": "user",
                        "content": (
                            "You stopped without calling any tools. "
                            "If you have enough data, call judge(x, y) with the best (x, y) you found. "
                            "If you need more data, keep querying. You MUST call a tool now."
                        ),
                    })
                    continue
                else:
                    print("\n✅ LLM finished (no more tool calls)")
                    break

            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"\n🔧 Calling: {func_name}({func_args})")

                result_text = await client.call_tool(func_name, func_args)

                try:
                    result_obj = json.loads(result_text)
                    print(f"   📊 Result: {json.dumps(result_obj, indent=2)[:300]}")
                except json.JSONDecodeError:
                    print(f"   📊 Result: {result_text[:300]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

                if func_name == "judge":
                    judge_called = True

            if judge_called:
                final_response = llm.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                final_msg = final_response.choices[0].message.content
                print(f"\n🏆 Final summary: {final_msg}")
                break

        if not judge_called:
            print("\n⚠️  Agent loop ended without judge. Auto-submitting best known result...")
            history_text = await client.call_tool("get_history", {})
            history = json.loads(history_text)
            best = history.get("best_so_far")
            if best and best.get("x") is not None:
                print(f"   📤 Auto-judging with best: ({best['x']}, {best['y']}) score={best['score']}")
                judge_result = await client.call_tool("judge", {"x": best["x"], "y": best["y"]})
                result_obj = json.loads(judge_result)
                print(f"   🏆 Judge result: {json.dumps(result_obj, indent=2)}")
            else:
                print("   ❌ No queries were made, nothing to submit.")

    finally:
        await client.close()
        print("\n🔌 Disconnected from MCP Server")

if __name__ == "__main__":
    print("🚀 Black Box Optimization Agent v1 (Full LLM)")
    print(f"🔗 MCP Server: {MCP_SERVER_URL}")
    print(f"🧠 LLM Model: {MODEL}")
    print()
    asyncio.run(run_agent())
