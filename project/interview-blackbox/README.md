# 🎯 Black Box Optimization — MCP Agent 面试题

## 题目概述

这是一道 **LLM Agent 工程面试题**。有一个黑盒函数接受离散整数 `(x, y)`，你需要构建一个 Agent，通过 MCP 协议与服务端交互，找到使函数值最大的 `(x, y)`。

### 给面试者的规则
- `x` 和 `y` 都是整数，**具体范围未知**（需要你自己探测）
- 超出范围的查询会返回错误信息
- 每次 `query` 可以得到一个 score
- 目标是找到全局最大值
- **关键决策需要 LLM 来做**（而不是暴力遍历）

---

## MCP Server

**地址：** `http://localhost:8339/mcp`

### 可用工具

| Tool | 描述 |
|------|------|
| `query(x, y)` | 查询黑盒函数，返回该点的 score |
| `get_history()` | 获取所有历史查询记录和当前最优 |
| `reset()` | 重置游戏，清空历史记录 |
| `judge(x, y)` | 提交你认为的最优解，系统告诉你对不对 |

### 启动服务

```bash
python server.py
```

服务将在 `http://localhost:8339` 运行。

---

## 🧠 这道题到底在考什么

表面是"找最大值"，实际考察的是：

| 考察维度 | 具体内容 |
|---------|---------|
| **MCP 协议理解** | 能否正确连接 MCP Server、调用 tools？ |
| **Agent 架构设计** | 能否写出 LLM + Tool 调用的 Agent Loop？ |
| **Prompt Engineering** | 如何让 LLM 成为一个"优化器"？ |
| **搜索策略** | 粗扫描 → 分析 → 精细搜索，而非暴力枚举 |
| **边界处理** | 范围未知时，能否主动探测边界？ |
| **工程能力** | 代码是否结构清晰、可运行？ |

---

## 💡 解题思路（面试官参考）

### 1. 探测边界（Phase 0）

题目故意不给数据范围。优秀的面试者应该先探测：

```
query(0, 0)     → 成功
query(100, 0)   → 失败（"out of the valid range"）
query(50, 0)    → 失败
query(25, 0)    → 失败
query(10, 0)    → 成功
query(20, 0)    → 成功
query(21, 0)    → 失败
→ 上界是 20
```

指数探测 + 二分法，O(log N) 次确定范围。

### 2. 粗粒度网格扫描（Phase 1）

根据探测到的范围，自适应选择步长，快速了解函数地形：

```
步长 = max(1, 范围 // 5)
→ 每轴 ~6 个采样点 → ~36 个点
```

### 3. LLM 分析 + 精细搜索（Phase 2-3）

把粗扫描的 Top 10 数据交给 LLM，让它：
- 分析哪个区域分值最高
- 在该区域 ±2 范围内做精细搜索
- 找到最优点后调用 `judge(x, y)` 提交

### 4. Fallback 兜底（Phase 4）

如果 LLM 没有成功调用 judge（小模型常见问题），程序自动提交当前最优解。

---

## 📂 项目结构

```
├── server.py              # MCP Server（面试题服务端）
├── black_box.py           # 黑盒函数（不要给面试者看！）
├── agent_minimal.py       # 参考答案 - 面试核心版（~80行，面试现场能写完）
├── agent.py               # 参考答案 - 完整版 v2（程序化 + LLM 混合策略）
├── agent_v1_full_llm.py   # 参考答案 - v1（全 LLM 驱动，适合大模型）
└── README.md              # 本文档
```

### 三个参考答案的对比

| 版本 | 行数 | 策略 | 适用场景 |
|------|------|------|---------|
| `agent_minimal.py` | **~80** | 全靠 LLM 驱动 | **面试现场手写** |
| `agent_v1_full_llm.py` | ~170 | 全靠 LLM，含 fallback | 大模型（GPT-4o 级别） |
| `agent.py` | ~300 | 程序化扫描 + LLM 精细搜索 | 小模型（7B）/ 生产环境 |

---

## 🎤 面试流程建议（给面试官）

### 阶段 1：思路讨论（15 min）

让面试者描述解题思路，期望听到：
- ✅ "我会先探测数据范围"
- ✅ "做粗粒度扫描了解地形"
- ✅ "让 LLM 分析数据，在高分区域精细搜索"
- ✅ "最后调用 judge 提交"
- 🌟 加分："如果范围是百万级，我会用多级网格 / 梯度估计"

### 阶段 2：写核心代码（30 min）

让面试者实现 Agent 核心逻辑，参考 `agent_minimal.py`：
1. MCP Client 连接（3 行）
2. Tool schema 转 OpenAI function calling 格式（5 行）
3. Agent Loop：LLM 调用 → tool call → 执行 → 反馈（30 行）
4. System Prompt（5 行）

**总共 ~50 行有效代码**，30 分钟足够。

### 阶段 3：运行调试（15 min）

跑起来看效果，讨论实际运行中的问题。

### 阶段 4：追问（15 min）

| 追问 | 期望回答 |
|------|---------|
| "范围未知怎么办？" | 指数探测 + 二分法 |
| "小模型 tool calling 不稳定？" | 程序化做机械性工作，LLM 只做决策；加 nudge 和 fallback |
| "搜索空间是百万级？" | 多级网格 / 随机采样 + LLM 推理 / 梯度估计 |
| "为什么不全暴力？" | 不 scalable，题目要求 LLM 参与决策 |
| "用 AI 写代码算不算作弊？" | 用 AI 写实现没问题，关键是能解释设计、讨论取舍 |

### 评分标准

| 等级 | 表现 |
|------|------|
| ❌ 不通过 | 不知道 MCP 是什么 / 写不出 agent loop |
| ⚠️ 勉强 | 能连接 MCP、调 tool，但策略是暴力枚举 |
| ✅ 通过 | 有搜索策略、agent loop 完整、能跑起来 |
| 🌟 优秀 | 主动探测边界、讨论 v1 vs v2 取舍、考虑扩展性 |

---

## ⚙️ 环境要求

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate
pip install mcp openai uvicorn

# 如果用本地模型
# 确保 Ollama 运行中，且有 qwen2.5:7b 等模型
ollama pull qwen2.5:7b
```

### 运行

```bash
# 终端 1：启动 MCP Server
python server.py

# 终端 2：运行 Agent
python agent_minimal.py          # 面试核心版
# 或
python agent.py                  # 完整版
# 或
OPENAI_API_KEY=sk-xxx MODEL=gpt-4o python agent_v1_full_llm.py  # 用 OpenAI
```
