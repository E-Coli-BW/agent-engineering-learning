"""
Black Box Optimization Agent - 参考答案 (v2, 适配本地小模型)

=== 设计思路 ===

v1 的问题：完全靠 LLM 驱动每一步 query，但 7B 小模型的 tool calling
不够稳定，容易跑偏（比如还没扫完就急着 judge）。

v2 的改进：程序化 + LLM 混合策略
  - Phase 0（程序化）：探测 x, y 的合法范围（题目未给数据范围）
  - Phase 1（程序化）：代码自动做粗粒度网格扫描，收集数据
  - Phase 2（LLM 决策）：把扫描数据交给 LLM 分析，让它决定精细搜索的区域
  - Phase 3（LLM 执行）：LLM 在感兴趣的区域调用 query 做精细搜索
  - Phase 4（LLM 或 fallback）：LLM 调用 judge，或程序自动提交最优解

这种设计更务实 —— 用程序做机械性工作，用 LLM 做需要推理的工作。
"""

import asyncio
import json
import os
from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# ============================================================
# 配置
# ============================================================
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8339/mcp")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("MODEL", "qwen2.5:7b")


# ============================================================
# MCP Client
# ============================================================
class MCPToolClient:
    """封装 MCP Client，提供简洁的 tool 调用接口"""

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
        tool_names = [t.name for t in tools.tools]
        print(f"✅ Connected to MCP Server | Tools: {tool_names}")
        return tools.tools

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """调用 MCP tool 并返回解析后的 dict"""
        result = await self.session.call_tool(name, arguments)
        text = result.content[0].text if result.content else "{}"
        return json.loads(text)

    async def close(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._streams:
            await self._streams.__aexit__(None, None, None)


# ============================================================
# OpenAI 工具格式转换
# ============================================================
def build_openai_tools(mcp_tools) -> list[dict]:
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return openai_tools


# ============================================================
# Phase 0: 探测合法范围（题目未给数据范围）
# ============================================================
async def phase0_probe_bounds(client: MCPToolClient) -> tuple[int, int, int, int]:
    """
    通过二分法探测 x 和 y 的合法上下界。
    思路：先用指数增长快速找到一个失败的上界，再二分精确定位。
    
    Returns:
        (x_min, x_max, y_min, y_max)
    """
    print(f"\n{'='*60}")
    print(f"🔍 Phase 0: Probing Valid Bounds")
    print(f"{'='*60}")

    async def is_valid(x: int, y: int) -> bool:
        """测试一个点是否在合法范围内"""
        r = await client.call_tool("query", {"x": x, "y": y})
        return "error" not in r

    async def find_upper_bound(axis: str) -> int:
        """用指数探测 + 二分法找某个轴的上界"""
        # 先用指数增长找到一个会失败的值
        probe = 1
        if axis == "x":
            while await is_valid(probe, 0):
                print(f"  probe {axis}={probe} ✓")
                probe *= 2
            print(f"  probe {axis}={probe} ✗ (out of bounds)")
        else:
            while await is_valid(0, probe):
                print(f"  probe {axis}={probe} ✓")
                probe *= 2
            print(f"  probe {axis}={probe} ✗ (out of bounds)")

        # 二分法精确定位
        lo, hi = probe // 2, probe
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if axis == "x":
                valid = await is_valid(mid, 0)
            else:
                valid = await is_valid(0, mid)

            if valid:
                lo = mid
            else:
                hi = mid - 1
        print(f"  → {axis}_max = {lo}")
        return lo

    async def find_lower_bound(axis: str) -> int:
        """探测负数下界"""
        # 先测 0 是否合法（通常是的）
        if axis == "x":
            if not await is_valid(0, 0):
                # 0 都不合法，往正方向找
                return 0
        # 测试负数
        probe = -1
        if axis == "x":
            valid = await is_valid(probe, 0)
        else:
            valid = await is_valid(0, probe)

        if not valid:
            print(f"  → {axis}_min = 0 (negatives not allowed)")
            return 0

        # 负数合法，继续用指数+二分法找下界
        probe = -1
        if axis == "x":
            while await is_valid(probe, 0):
                probe *= 2
        else:
            while await is_valid(0, probe):
                probe *= 2

        lo, hi = probe, probe // 2  # lo < hi, both negative
        while lo < hi:
            mid = (lo + hi) // 2
            if axis == "x":
                valid = await is_valid(mid, 0)
            else:
                valid = await is_valid(0, mid)

            if valid:
                hi = mid
            else:
                lo = mid + 1
        print(f"  → {axis}_min = {lo}")
        return lo

    # 探测四个边界
    print("\n📐 Probing x bounds...")
    x_min = await find_lower_bound("x")
    x_max = await find_upper_bound("x")

    print("\n📐 Probing y bounds...")
    y_min = await find_lower_bound("y")
    y_max = await find_upper_bound("y")

    print(f"\n✅ Detected bounds: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
    range_size = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"   Search space: {x_max - x_min + 1} × {y_max - y_min + 1} = {range_size} points")

    # 重置，因为探测过程产生了一些 query 记录
    await client.call_tool("reset", {})
    print("🔄 Reset after probing.")

    return x_min, x_max, y_min, y_max


# ============================================================
# Phase 1: 程序化粗粒度扫描（自适应步长）
# ============================================================
async def phase1_coarse_scan(
    client: MCPToolClient,
    x_min: int, x_max: int,
    y_min: int, y_max: int,
    target_points_per_axis: int = 6,
) -> list[dict]:
    """
    用代码自动做网格扫描，根据探测到的范围自适应选择步长。
    target_points_per_axis=6 → 每轴约 6 个采样点，总共 ~36 个点
    """
    x_range = x_max - x_min
    y_range = y_max - y_min

    step_x = max(1, x_range // (target_points_per_axis - 1))
    step_y = max(1, y_range // (target_points_per_axis - 1))

    print(f"\n{'='*60}")
    print(f"📡 Phase 1: Coarse Grid Scan")
    print(f"   Range: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
    print(f"   Step:  x_step={step_x}, y_step={step_y}")
    print(f"{'='*60}")

    results = []
    # 生成坐标（确保包含边界点）
    x_coords = list(range(x_min, x_max + 1, step_x))
    if x_coords[-1] != x_max:
        x_coords.append(x_max)
    y_coords = list(range(y_min, y_max + 1, step_y))
    if y_coords[-1] != y_max:
        y_coords.append(y_max)

    total = len(x_coords) * len(y_coords)

    idx = 0
    for x in x_coords:
        for y in y_coords:
            r = await client.call_tool("query", {"x": x, "y": y})
            idx += 1
            marker = "⭐" if r.get("is_new_best") else "  "
            print(f"  [{idx:2d}/{total}] query({x:2d}, {y:2d}) → score={r['score']:>10.4f} {marker}")
            results.append({"x": x, "y": y, "score": r["score"]})

    # 按 score 降序排列
    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n📊 Top 5 points from coarse scan:")
    for r in results[:5]:
        print(f"   ({r['x']:2d}, {r['y']:2d}) → score={r['score']:.4f}")

    return results


# ============================================================
# Phase 2 & 3: LLM 分析 + 精细搜索
# ============================================================
async def phase2_llm_fine_search(
    client: MCPToolClient,
    llm: OpenAI,
    openai_tools: list[dict],
    coarse_results: list[dict],
) -> dict | None:
    """
    把粗扫描结果交给 LLM，让它决定在哪里做精细搜索。
    返回 judge 的结果（如果 LLM 调用了 judge），否则返回 None。
    """
    print(f"\n{'='*60}")
    print(f"🧠 Phase 2-3: LLM Analysis + Fine-grained Search")
    print(f"{'='*60}")

    # 构建数据摘要给 LLM
    top_10 = coarse_results[:10]
    data_summary = "Coarse scan results (top 10 by score, descending):\n"
    for r in top_10:
        data_summary += f"  ({r['x']}, {r['y']}) → score = {r['score']:.4f}\n"

    best = coarse_results[0]
    data_summary += f"\nCurrent best: ({best['x']}, {best['y']}) with score {best['score']:.4f}"
    data_summary += f"\nTotal points scanned: {len(coarse_results)}"

    system_prompt = """\
You are an optimization agent. A coarse grid scan has already been done for you.
Your job now is to do a FINE-GRAINED search around the promising regions to find the exact maximum.

Rules:
- x and y are integers within the valid bounds (already determined by the scan)
- Use query(x, y) to test points near the best regions found in the coarse scan
- Search ALL integer points within ±2 of the top candidates
- After thorough fine search, call judge(x, y) with your best found point
- You MUST call tools. Do NOT just describe what you would do.
- Be systematic: search a small grid, don't skip points
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Here are the coarse scan results:\n\n{data_summary}\n\n"
            f"Now do a fine-grained search around the best region. "
            f"Query all integer points within ±2 of the top 1-2 candidates, "
            f"then call judge(x, y) with the overall best point."
        )},
    ]

    max_iterations = 40
    judge_result = None

    for iteration in range(max_iterations):
        print(f"\n--- LLM Iteration {iteration + 1} ---")

        response = llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if msg.content:
            print(f"🤖 {msg.content[:300]}")

        if not msg.tool_calls:
            if judge_result is None:
                print("⚠️  No tool calls. Nudging LLM...")
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call tools! If you've searched enough, "
                        "call judge(x, y) with the best point. Otherwise keep querying."
                    ),
                })
                continue
            else:
                break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"🔧 {name}({args})")

            result = await client.call_tool(name, args)
            result_text = json.dumps(result)

            if name == "query":
                score = result.get("score", "?")
                is_best = "⭐" if result.get("is_new_best") else ""
                print(f"   → score={score} {is_best}")
            elif name == "judge":
                print(f"   → {json.dumps(result, indent=2)}")
                judge_result = result
            else:
                print(f"   → {result_text[:200]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

        if judge_result is not None:
            # 让 LLM 总结
            final = llm.chat.completions.create(model=MODEL, messages=messages)
            if final.choices[0].message.content:
                print(f"\n🤖 Summary: {final.choices[0].message.content[:500]}")
            break

    return judge_result


# ============================================================
# 主流程
# ============================================================
async def run_agent():
    client = MCPToolClient()
    llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    try:
        mcp_tools = await client.connect()
        openai_tools = build_openai_tools(mcp_tools)

        # Phase 0: 探测合法范围
        x_min, x_max, y_min, y_max = await phase0_probe_bounds(client)

        # Phase 1: 程序化粗扫描（自适应步长）
        coarse_results = await phase1_coarse_scan(
            client, x_min, x_max, y_min, y_max
        )

        # Phase 2-3: LLM 分析 + 精细搜索 + judge
        judge_result = await phase2_llm_fine_search(
            client, llm, openai_tools, coarse_results
        )

        # Phase 4: Fallback —— 如果 LLM 没 judge，程序自动提交
        if judge_result is None:
            print(f"\n{'='*60}")
            print(f"🔧 Fallback: Auto-submitting best known result")
            print(f"{'='*60}")
            history = await client.call_tool("get_history", {})
            best = history.get("best_so_far", {})
            if best.get("x") is not None:
                judge_result = await client.call_tool(
                    "judge", {"x": best["x"], "y": best["y"]}
                )
                print(f"📤 Auto-judge: {json.dumps(judge_result, indent=2)}")

        # 最终结果
        print(f"\n{'='*60}")
        print(f"🏆 FINAL RESULT")
        print(f"{'='*60}")
        if judge_result:
            answer = judge_result.get("your_answer", {})
            print(f"   Answer: ({answer.get('x')}, {answer.get('y')})")
            print(f"   Score:  {answer.get('score')}")
            print(f"   Found optimal: {judge_result.get('found_optimal')}")
            print(f"   Gap:    {judge_result.get('score_gap')}")
            print(f"   {judge_result.get('message')}")
            print(f"   Queries used: {judge_result.get('total_queries_used')}")

    finally:
        await client.close()
        print("\n🔌 Disconnected.")


if __name__ == "__main__":
    print("🚀 Black Box Optimization Agent v2")
    print(f"🔗 MCP: {MCP_SERVER_URL}")
    print(f"🧠 Model: {MODEL}")
    asyncio.run(run_agent())
