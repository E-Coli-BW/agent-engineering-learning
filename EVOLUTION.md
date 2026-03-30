# 🛤️ 项目演进日志 — 从零到全栈 AI Agent 系统

> 记录每一步的决策、踩坑和学到的东西，便于复习和面试准备。

---

## Phase 0: 底层原理 + 教学脚本

**做了什么**: 手写 Transformer、Agent、RAG、LoRA、知识图谱、推理部署

**关键文件**: `multi_head_attention.py`, `char_transformer.py`, `agent/01-04`, `rag/01-04`

**学到的**:
- QKV 注意力的每一步 shape 变化
- ReAct 循环的本质: LLM 输出结构化文本 → 解析执行 → 结果反馈
- RAG = Embedding + 向量检索 + LLM 生成
- LoRA = 低秩分解，只训练 ΔW = BA

---

## Phase 1: 工程化后端 (project/)

### A2A Agent v0 → v1 → v2

| 版本 | 文件 | 通信方式 | 决策方式 |
|------|------|----------|---------|
| v0 | `a2a_agent_v1_stdio.py` | subprocess + stdin/stdout | 无 |
| v1 | `a2a_agent.py` | HTTP REST + SSE | 关键词路由 |
| v2 | `react_agent.py` | HTTP REST (同步) | LLM 自主 ReAct |

**踩坑**:
- `uvicorn.run("project.api_server:app", reload=True)` 需要 `PYTHONPATH=.`，否则 import 失败
- Ollama `/api/chat` (messages 格式) 比 `/api/generate` (单 prompt) 更适合多轮 ReAct
- A2A 的 SSE 格式: `event: artifact\ndata: {...}\n\n`，空行才表示事件结束

### ETL Pipeline + RAG API

**设计决策**:
- ChromaDB 做向量库（轻量，嵌入式，不需要额外服务）
- content_hash 去重实现增量更新
- `/query/stream` SSE 流式输出 — TTFT 0.5s vs 同步 5-10s

### 微信桥接器

**踩坑**:
- iLink API 的 `client_id` 必须带，否则只有第一条消息能送达
- `message_type=2` 是 bot 自己的回复，必须过滤防死循环
- `context_token` 每条消息不同，是消息路由的 key

---

## Phase 2: 前端 (frontend/)

**技术选型**: React 19 + Vite 6 + Tailwind CSS 3 + TypeScript

**做了什么**: Chat 页面（RAG/A2A/ReAct 三模式切换）、Dashboard、Agent 管理、ETL 管理

**踩坑**:
- `React.StrictMode` 在 dev 模式下双重执行 effects → SSE 连接创建两次 → 每个 token 重复
  - 修复: 移除 StrictMode + AbortController 防护
- SSE 解析: 必须在空行才分发事件，不能在 `data:` 行立即处理
- ReAct Agent 不支持 streaming，前端需要对 ReAct 用同步 `/tasks/send`
- ReAct 的回答在 `status.message.parts` 里，不在 `artifacts` 里

---

## Phase 3: Java API Gateway

**技术选型**: Spring Cloud Gateway + Sentinel + Nacos + Spring Security

### 演进过程

1. **v1 (手写轮子)**: Resilience4j 手动配、自己写 JWT filter、Redis 限流
2. **被 challenge**: "为什么不用 Spring Cloud 生态标准组件？"
3. **v2 (标准做法)**:
   - Sentinel 替代 Resilience4j (动态规则 + Dashboard)
   - Spring Security ReactiveAuthenticationManager 替代手写 GlobalFilter
   - Nacos 可选 (服务发现 + 配置中心)
   - Spring Cloud LoadBalancer 替代 Ribbon (已废弃)

**踩坑**:
- 公司 WiFi Captive Portal 劫持 HTTP → Maven 下载的 jar 是 HTML 登录页面
- Sentinel Nacos datasource 在 Nacos 未运行时会导致启动失败 → 移到 profile
- `application.yml` 里不应该混用 `${ENV_VAR:default}` 和硬编码 → 拆成 Profile

### Profile 分离

```
application.yml          → 本地开发 (localhost)
application-docker.yml   → Docker (service name)
application-nacos.yml    → Nacos (Sentinel 规则持久化)
```

---

## Phase 4: 可观测性

### Gateway 侧
- Micrometer + Prometheus: `gateway_requests_total`, `gateway_request_duration_seconds` (按 route 分组)
- `X-Request-Id` 全链路追踪
- 结构化日志: `[traceId] GET /path → 200 (42ms) [ip=..., route=...]`

### Python 侧
- `/metrics` 双格式: JSON (前端) + Prometheus text (采集)
- P50/P95/P99 延迟百分位
- 从 Gateway `X-Request-Id` 继承追踪 ID

### Prometheus + Grafana
- Docker Compose 一键拉起
- 8 个面板: 服务状态/QPS/延迟/P50-P99/向量数/JVM堆/GC

---

## Phase 5: 测试 + CI

### Python (62 tests)
- `test_config.py`: 默认值 + 环境变量覆盖 (subprocess)
- `test_models.py`: A2A 数据模型序列化
- `test_api_server.py`: /health /metrics /query 校验
- `test_a2a_agent.py`: Agent Card + Task send + 空消息
- `test_react_agent.py`: ToolRegistry + ReAct 解析
- `test_infra.py`: TaskStore + Error + Logging

### Java (11 tests)
- Actuator/Auth/Fallback/HealthAggregation

### GitHub Actions CI
- `python-test`: Python 3.12 + pytest
- `java-test`: JDK 21 + mvn verify
- `java-build`: Package JAR + upload artifact

---

## Phase 6: 容器化

### Docker 文件
- `Dockerfile.rag/a2a/react`: Python 3.12-slim + 精简依赖
- `Dockerfile.gateway`: Maven build → JRE 21 (multi-stage)
- `Dockerfile.frontend`: npm build → nginx (multi-stage)

### docker-compose.yml
- 8 个服务: rag + a2a + react + gateway + frontend + redis + prometheus + grafana
- Ollama 在宿主机运行，容器通过 `host.docker.internal` 访问

---

## Phase 7: 数据持久化 + 基础设施

### project/infra/
- `TaskStore`: MemoryStore (默认) + RedisStore (REDIS_URL)，自动降级
- `AppError`: 统一错误模型 `{"error":{"code":"...","message":"...","request_id":"..."}}`
- `setup_logging()`: JSON (生产) / Text (开发) 结构化日志，`LOG_FORMAT` 环境变量切换

---

## Phase 8: Multi-Agent Orchestrator

### 从"多服务"到"真 Multi-Agent"

**之前**: 用户手动选 RAG/A2A/ReAct → 各自独立回答
**现在**: LLM 自主决策 → 调用多个 Sub-Agent → 合成结果

### LangGraph StateGraph

```
plan (LLM 分析需要哪些 Agent)
  → execute (调用 Knowledge/Calculator/Code Agent)
  → synthesize (LLM 整合多个 Agent 结果)
  → should_continue? (判断是否需要补充)
      → Yes: 回到 plan (最多 3 轮)
      → No:  END
```

### Sub-Agent 职责

| Agent | 能力 | 实现 |
|-------|------|------|
| Knowledge | RAG 向量检索 + 知识图谱 | 复用 ChromaDB + 图谱数据 |
| Calculator | 数学计算 + LoRA/Transformer 参数量 | 安全 eval + 领域公式 |
| Code | 代码生成/解释/审查 | Ollama LLM + code prompt |

### 关键学到的

1. **不是所有场景都需要 Multi-Agent** — 单 ReAct Agent + 工具注册表能解决 90% 的问题
2. **Multi-Agent 的价值在于专业分工** — 不同 Agent 有不同能力、不同 prompt、不同数据源
3. **LangGraph 比手写更可靠** — 状态机 + 条件路由 + 内置重试
4. **Orchestrator LLM 是关键** — 路由决策的质量取决于 plan prompt 的设计

---

## 技术债务 & 未来方向

### 当前已知问题
- [ ] Orchestrator 不支持 SSE streaming (当前同步等待全部 Agent 完成)
- [ ] 没有聊天历史持久化 (前端刷新丢失)
- [ ] Sentinel 没有配实际限流规则
- [ ] 没有端到端测试
- [ ] 没有 LLM 输出质量评估

### 可扩展方向
- [ ] Spring AI 结构化输出 (Structured Output)
- [ ] 多模态支持 (图片理解)
- [ ] 聊天记忆 (长期对话上下文)
- [ ] 模型评估 (RAG 准确性 + Agent 决策质量)
- [ ] Orchestrator SSE streaming (逐 Agent 推送结果)
