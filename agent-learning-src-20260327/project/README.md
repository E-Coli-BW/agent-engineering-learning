# 🏗️ 工程化 RAG 系统 (`project/`)

> 之前所有模块都是"教学脚本"——跑一下看效果。
> 这个目录是**真实工程级别**的实现，弥补和生产系统的差距。

## 解决的差距

| 差距 | 之前 | 现在 |
|------|------|------|
| 数据处理 | 硬编码 fake data | 真实 ETL Pipeline (多格式文件 → 清洗 → 分块 → 入库) |
| 可服务性 | 独立 `.py` 脚本 | FastAPI REST API (可供前端/其他系统调用) |
| 错误处理 | 几乎为零 | 超时重试、降级策略、结构化日志 |
| 配置管理 | 硬编码到处都是 | 统一 Config + 环境变量覆盖 |
| 增量更新 | 每次全量重建 | content_hash 去重 + 增量入库 |
| 流式输出 | 无 | SSE 流式响应 |
| 可观测性 | 无 | /health + /metrics 端点 |

## 文件说明

```
project/
├── config.py              # 统一配置管理 (环境变量覆盖)
├── etl_pipeline.py        # ETL: Extract → Transform → Load
├── api_server.py          # FastAPI RAG API Server
├── mcp_server.py          # MCP Server (RAG/知识图谱/计算工具)
├── a2a_agent.py           # A2A Expert Agent (学习版, 单文件 717 行)
├── a2a_agent_v1_stdio.py  # A2A v0 (旧版 stdio)
├── react_agent.py         # ReAct Agent (学习版, 单文件 542 行)
├── wechat_bridge.py       # 微信桥接器 (学习版, 单文件 935 行)
├── README.md              # 本文件
│
└── app/                   # ← 生产版 package (从学习版重构)
    ├── __init__.py         # 包入口 + 版本号
    ├── __main__.py         # python -m project.app 入口
    ├── models.py           # A2A 协议数据模型 (Part/Message/Task/AgentCard)
    ├── ollama_client.py    # Ollama HTTP 客户端 (generate + chat)
    ├── expert_server.py    # FastAPI Expert Agent (A2A 端点)
    ├── coordinator.py      # Coordinator Agent HTTP 客户端
    ├── cli.py              # 统一 CLI 入口 (serve/ask/stream/react/wechat)
    ├── wechat/
    │   ├── ilink.py        # iLink 协议层 (凭证/QR登录/消息收发)
    │   └── bridge.py       # 微信 ↔ Agent 桥接编排
    └── react/
        ├── tools.py        # 工具注册表 + 内置工具
        └── agent.py        # ReAct Agent 循环
```

### 学习版 vs 生产版对照

| 学习版 (单文件)          | 生产版 (app/ package)        | 说明 |
|--------------------------|------------------------------|------|
| `a2a_agent.py` (717行)   | `models.py` + `ollama_client.py` + `expert_server.py` + `coordinator.py` | 数据模型 / LLM客户端 / 服务端 / 客户端 各司其职 |
| `react_agent.py` (542行) | `react/agent.py` + `react/tools.py` | Agent 循环和工具注册表分离 |
| `wechat_bridge.py` (935行) | `wechat/ilink.py` + `wechat/bridge.py` | 协议层和编排层分离 |
| 各文件各有 `__main__`    | `cli.py` 统一入口            | `python -m project.app <cmd>` |

### 生产版运行方式

```bash
# 启动 Expert Agent
python -m project.app serve

# 启动 ReAct Agent
python -m project.app serve --react

# 同步查询
python -m project.app ask '什么是Transformer?'

# 流式查询
python -m project.app stream '什么是LoRA?'

# 本地 ReAct 测试
python -m project.app react '计算 sqrt(768)'

# 微信桥接器
python -m project.app wechat
python -m project.app wechat --expert http://localhost:5002 --stream
```

## 🎓 演进路线 — 从"基础问答"到"ReAct Agent"的学习历程

本目录展示了一个 AI Agent 系统从最简原型到带工具调用的完整演进过程。
每个版本都保留了代码，可以独立运行对比效果。

### 演进全景图

```
v0: a2a_agent_v1_stdio.py     v1: a2a_agent.py          v2: react_agent.py
    (subprocess + stdio)           (HTTP REST + SSE)         (ReAct + 工具)
         ↓                              ↓                         ↓
    最原始的 Agent 通信           对齐 A2A 规范             LLM 自主决策 + 工具

┌─────────────┐         ┌──────────────────┐       ┌──────────────────────┐
│ 用户 → LLM  │  ──→    │ 用户 → 关键词路由 │ ──→   │ 用户 → LLM 思考      │
│      → 回答  │         │      → LLM 回答   │       │      → 调工具?        │
│              │         │      → HTTP 返回   │       │      → 执行 → 观察    │
│ 一问一答     │         │ 技能路由           │       │      → 继续思考       │
│ 无法扩展     │         │ 可远程调用         │       │      → Final Answer   │
└─────────────┘         └──────────────────┘       └──────────────────────┘
```

---

### v0: `a2a_agent_v1_stdio.py` — 最原始的 Agent 通信

**核心思路**: 用 subprocess 启动一个子进程，通过 stdin/stdout 管道互发 JSON 消息。

**学到了什么**:
- Agent-to-Agent 通信的本质就是"发消息 → 收回复"
- stdin/stdout 是最简单的进程间通信方式
- 但局限很大：只能同机器、只能 Python、没有服务发现

**关键代码模式**:
```python
# 启动子进程
proc = subprocess.Popen(["python", "expert.py"], stdin=PIPE, stdout=PIPE)
# 发请求
proc.stdin.write(json.dumps({"question": "..."}).encode() + b"\n")
# 收响应
response = json.loads(proc.stdout.readline())
```

**为什么要升级**: 无法跨机器、无法多客户端并发、没有标准协议。

---

### v1: `a2a_agent.py` — A2A 协议版 (HTTP REST)

**核心思路**: 用 HTTP REST API 替代 stdio，对齐 Google A2A 协议规范。

**相比 v0 的升级**:

| 维度 | v0 (stdio) | v1 (HTTP REST) |
|------|-----------|---------------|
| 传输层 | stdin/stdout 管道 | HTTP REST + SSE |
| 服务发现 | 无，硬编码路径 | `GET /.well-known/agent.json` |
| 能力声明 | 无 | Agent Card (skills, tags) |
| 路由策略 | 无 | 关键词匹配 Skill |
| 跨机器 | ❌ | ✅ |
| 并发 | ❌ 1对1 | ✅ |
| 流式输出 | ❌ | ✅ SSE |

**学到了什么**:
- A2A 协议三要素: Agent Card、Task 生命周期、Streaming
- `关键词路由`虽然简单但有效——匹配 tags 决定交给哪个 Skill
- `sendmessage` 需要 `client_id` 做幂等，否则会被去重
- Session 过期 (`errcode: -14`) 需要自动处理

**踩坑经验 (微信桥接器)**:
1. iLink API 端点有 `ilink/bot/` 前缀（文档没写清楚）
2. `getupdates` 响应可能没有 `ret` 字段，不能用默认值 `-1` 判断失败
3. `sendmessage` 必须带 `client_id`，否则只有第一条能送达
4. `context_token` 每条消息都不同，是消息路由的关键
5. `message_type=2` 是 Bot 自己的回复，必须过滤防止无限循环

**关键限制**: LLM 只能"回答问题"，不能"执行动作"。路由靠关键词硬编码，加新能力要改代码。

---

### v2: `react_agent.py` — ReAct Agent (LLM + 工具调用)

**核心思路**: 让 LLM 自己决定要不要调工具、调哪个工具，实现 Think → Act → Observe 循环。

**相比 v1 的升级**:

| 维度 | v1 (纯问答) | v2 (ReAct) |
|------|-----------|-----------|
| 决策方式 | 关键词路由 (if/else) | LLM 自主推理 |
| 工具调用 | ❌ 无 | ✅ 注册制，动态调用 |
| 多步推理 | ❌ 一问一答 | ✅ 多轮 Think-Act-Observe |
| 扩展新能力 | 改代码加 Skill | `registry.register()` 一行搞定 |
| 数据查询 | ❌ | ✅ 知识图谱 + RAG |

**ReAct 循环详解**:
```
用户: "知识图谱里 LoRA 和什么有关?"

Step 1:
  LLM Thought: 用户想查 LoRA 的知识图谱关系，我需要调用 knowledge_graph_query
  LLM Action: knowledge_graph_query
  LLM Action Input: LoRA
  → 工具执行 → 返回: "LoRA --[用于]--> 参数高效微调, LoRA --[原理]--> 低秩矩阵分解..."

Step 2:
  LLM Thought: 我已经拿到了知识图谱的结果，可以回答了
  LLM Final Answer: LoRA 在知识图谱中有以下关系：用于参数高效微调，原理是...
```

**学到了什么**:
- ReAct 的核心是**让 LLM 自己写结构化的"思考-行动"文本**，然后我们解析执行
- `/api/chat` (messages 格式) 比 `/api/generate` (单 prompt) 更适合多轮 ReAct
- System Prompt 必须给**具体示例**，否则 LLM 输出格式不稳定
- 工具注册表模式让扩展变得极其简单
- 需要 `max_steps` 保护，防止 LLM 陷入无限循环

**关键设计决策**:
- 用 Ollama `/api/chat` 的 messages 格式而非拼接 prompt，多轮对话更稳定
- 工具结果作为 `user` 角色的消息追加（模拟"环境反馈"）
- Final Answer 解析支持中英文冒号，容错性更好
- 与 v1 共享同一套 A2A HTTP 接口，微信桥接器无缝切换

---

### 运行对比

```bash
# v1: 纯问答 (关键词路由 → LLM 回答)
python project/a2a_agent.py --serve        # 终端 1
python project/a2a_agent.py "什么是LoRA?"  # 终端 2
# → LLM 直接回答，不调任何工具

# v2: ReAct Agent (LLM 决策 → 可能调工具 → 回答)
python project/react_agent.py --serve      # 终端 1
python project/react_agent.py "知识图谱里LoRA和什么有关?"  # 终端 2
# → LLM 思考 → 调 knowledge_graph_query → 拿结果 → 回答

# 微信桥接器可以连任意版本:
python project/wechat_bridge.py --expert http://localhost:5001  # 连 v1
python project/wechat_bridge.py --expert http://localhost:5002  # 连 v2
```

---

### 未来演进方向 (v3+)

```
v2 (当前)                    v3 (未来)
ReAct + 本地工具      →      LangGraph + MCP + 外部 API

┌─────────────────┐         ┌──────────────────────────┐
│ 知识图谱查询     │         │ 公众号发文               │
│ RAG 检索         │  ──→    │ 抖音视频发布             │
│ 数学计算         │         │ 邮件发送                 │
│ 时间查询         │         │ 数据库操作               │
└─────────────────┘         │ 网页搜索                 │
                            │ 文件读写                 │
  本地函数调用                │ ... (任意 MCP 工具)      │
                            └──────────────────────────┘
                              通过 MCP 协议调外部服务
```

## 使用方式

### Step 1: 安装额外依赖

```bash
pip install fastapi uvicorn
```

### Step 2: 运行 ETL (构建索引)

```bash
python project/etl_pipeline.py
```

会扫描项目中所有 `.py` / `.md` 文件 → 清洗 → 分块 → 向量化 → 存入 ChromaDB。

### Step 3: 启动 API Server

```bash
python project/api_server.py
```

访问 http://localhost:8000/docs 查看自动生成的 API 文档。

### Step 4: 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "LoRA 微调的原理是什么？"}'

# 流式查询
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Transformer 的注意力机制如何工作？"}'

# 指标
curl http://localhost:8000/metrics
```

### 环境变量配置

```bash
# 换模型
export CHAT_MODEL=qwen2.5:3b
export EMBED_MODEL=mxbai-embed-large

# 换端口
export API_PORT=3000

# 调参数
export TOP_K=3
export CHUNK_SIZE=800
```

---

## 🌉 微信桥接器 (`wechat_bridge.py`)

将微信私聊消息转发到 A2A Expert Agent，实现"微信上问深度学习问题"。

### 架构

```
微信用户 ←→ iLink Server ←→ WeChatBridge ←→ A2A Expert Agent
             (weixin.qq.com)   (本桥接器)      (port 5001)
```

### 微信配置要求

你的微信需要能使用 **"爪机器人 (ClawBot)"** 的 iLink Bot API：
- 在微信 "发现" → "小程序" 搜索 **"爪机器人"** 并开通
- 这是微信官方的 iLink Bot API，非第三方逆向
- 扫码登录后，别人给你发的 **私聊消息** 会转发到本 Bot
- 仅支持私聊，不支持群聊

### 运行方式

```bash
# 终端 1: 启动 Expert Agent
python project/a2a_agent.py --serve

# 终端 2: 启动微信桥接器
python project/wechat_bridge.py

# 流式模式
python project/wechat_bridge.py --stream

# 连接远程 Expert
python project/wechat_bridge.py --expert http://192.168.1.100:5001

# 强制重新扫码
python project/wechat_bridge.py --login
```

### 可选依赖

```bash
pip install qrcode   # 终端内直接显示 QR 码图案 (否则需要在浏览器打开 URL 扫码)
```
