# Copilot Instructions — Agent Learning Project

## Architecture Overview

This is a **full-stack AI learning project** with two distinct layers:

1. **Teaching scripts** (root-level dirs: `agent/`, `rag/`, `finetune/`, `deploy/`, `knowledge_graph/`) — standalone `.py` files meant to run sequentially for learning. Do not refactor these into packages.
2. **Production system** (`project/` + `frontend/`) — engineering-grade services:
   - `project/api_server.py` — FastAPI RAG API (`:8000`): `/query`, `/query/stream` (SSE), `/health`, `/metrics`, `/etl/run`
   - `project/a2a_agent.py` — A2A Expert Agent (`:5001`): `/.well-known/agent.json`, `/tasks/send`, `/tasks/sendSubscribe` (SSE)
   - `project/react_agent.py` — ReAct Agent (`:5002`): `/tasks/send` only (no streaming), same A2A task format
   - `project/wechat_bridge.py` — 微信 ↔ Agent 桥接器: iLink Bot API → 消息轮询 → 转发到 A2A Expert
   - `project/mcp_server.py` — MCP Server (stdio): 暴露 `rag_query`, `knowledge_graph`, `calculate`, `list_modules` 给 VS Code Copilot / Claude Desktop
   - `frontend/` — React 19 + Vite + Tailwind Chat & Manage UI (`:3000`), proxies via `/api/rag`, `/api/a2a`, `/api/react`
   - `gateway/` — Java Spring Cloud Gateway (`:8080`): 统一入口、JWT 认证、IP 限流、熔断降级、聚合健康检查
   - `project/app/` — refactored production package of the above (same functionality, proper module separation)

## Key Conventions

- **All LLM calls go through local Ollama** (`http://localhost:11434`). Default models: `qwen2.5:7b` (chat), `mxbai-embed-large` (embeddings). Never add external API calls without env-var gating.
- **Configuration**: `project/config.py` — dataclass singleton `config`, every setting overridable via env vars (e.g., `CHAT_MODEL`, `OLLAMA_BASE_URL`, `TOP_K`).
- **A2A protocol**: Messages use `{ role, parts: [{ type: "text", text }] }` structure. Tasks have states: `submitted → working → completed/failed`.
- **ReAct Agent does NOT support streaming** — frontend must use sync `a2aSendTask()` for ReAct, `a2aSendSubscribe()` (SSE) for A2A. See `ChatPage.tsx` backend branching logic.
- **Frontend API layer**: All backend communication is in `frontend/src/api.ts`. Vite proxy rewrites `/api/rag/*` → `:8000`, `/api/a2a/*` → `:5001`, `/api/react/*` → `:5002`.
- **Chinese-first**: UI, comments, docstrings, and commit messages are in Chinese. Technical terms stay in English.

## WeChat Bridge (`wechat_bridge.py`)

- Uses **iLink Bot API** (微信"爪机器人"小程序), NOT reverse-engineered WeChat protocol.
- Flow: `微信用户 ←→ iLink Server ←→ WeChatBridge (polling) ←→ A2A Expert Agent`
- Credentials stored in `data/wechat_bridge/credentials.json` (gitignored). QR login on first run.
- Key pitfalls documented in `project/README.md`: `client_id` required for `sendmessage`, `message_type=2` must be filtered (bot echo), `context_token` per-message routing.
- Connect to either agent: `python project/wechat_bridge.py --expert http://localhost:5001` (A2A) or `:5002` (ReAct).

## MCP Server (`mcp_server.py`)

- Runs via **stdio** (not HTTP) — launched and managed by the MCP client (VS Code Copilot, Claude Desktop).
- Configured in `.vscode/mcp.json`; Copilot auto-connects on workspace open.
- Exposes tools: `rag_query` (vector search), `knowledge_graph` (entity lookup), `calculate` (math eval), `list_modules` (project structure).
- Logs to `data/mcp_server.log` (not stdout, to avoid polluting stdio protocol).

## Developer Workflows

### Git 规范 (每次改动必须遵守)

1. **永远不要直接在 main 上改代码**。先创建 feature branch:
   ```bash
   git checkout -b feat/xxx   # 新功能
   git checkout -b fix/xxx    # 修 bug
   git checkout -b refactor/xxx  # 重构
   ```

2. **Commit 前必须通过测试**:
   ```bash
   PYTHONPATH=. python -m pytest tests/ --tb=short  # Python 93+ tests
   cd gateway && mvn test                            # Java 11 tests
   ```

3. **Commit message 格式** (Conventional Commits):
   ```
   feat: 新功能描述
   fix: 修复了什么问题
   refactor: 重构了什么
   docs: 文档更新
   test: 测试相关
   ```

4. **Push 前 checklist**:
   - [ ] 所有测试通过
   - [ ] 新代码有对应测试
   - [ ] 没有 hardcode 的路径/密码
   - [ ] Python 代码有 docstring
   - [ ] Java 代码有 Javadoc

### 代码风格

**Python**:
- 函数/变量: `snake_case`
- 类: `PascalCase`
- 常量: `UPPER_SNAKE_CASE`
- 每个模块开头有 `"""docstring"""`
- Type hints on all public functions
- Import 顺序: stdlib → third-party → project

**Java**:
- 遵循 Spring Boot 标准分层: `config/`, `filter/`, `controller/`, `security/`
- `@Slf4j` + Lombok
- 配置走 `application.yml`, 不要在代码里硬编码

**TypeScript**:
- 组件: `PascalCase.tsx`
- 工具函数: `camelCase`
- API 调用统一走 `src/api.ts`

### 添加新功能时的 Checklist

每次添加新功能，按以下顺序思考:

1. **需要改哪些文件?**
   - 后端: `project/` 新文件或改现有文件
   - 前端: `api.ts` + 页面 + `vite.config.ts` proxy
   - Gateway: `application.yml` 路由 + `application-docker.yml`
   - Docker: `docker-compose.yml` + 可能新 Dockerfile

2. **需要加哪些测试?**
   - `tests/test_*.py` — 单元测试 (不依赖 Ollama)
   - Mock 外部依赖 (Ollama, Redis)
   - 边界情况: 空输入、超长输入、服务不可用

3. **需要更新哪些文档?**
   - `README.md` — 架构图 / 完成度表
   - `EVOLUTION.md` — 演进日志
   - `.github/copilot-instructions.md` — AI 开发指引
   - `start.sh` — 如果加了新服务

### 启动命令

```bash
# 一键启动全栈:
./start.sh              # start all 6 services
./start.sh status       # check health
./start.sh stop         # kill all

# 单独启动:
./start.sh rag          # RAG API :8000
./start.sh a2a          # A2A Agent :5001
./start.sh react        # ReAct Agent :5002
./start.sh orchestrator # Multi-Agent :5003
./start.sh front        # Frontend :3000
./start.sh gateway      # Gateway :8080

# 测试:
PYTHONPATH=. pytest tests/ -v           # Python
cd gateway && mvn test                  # Java
PYTHONPATH=. python project/eval/benchmark.py  # 评估

# 前端 dev:
cd frontend && npm run dev

# Python backend (需要 PYTHONPATH):
PYTHONPATH=. python project/api_server.py
```

## When Adding New Features

**必须按 checklist 执行，不要跳步:**

- **New backend tool for ReAct/Orchestrator**: 
  1. 实现: `project/orchestrator/agents/` 或 `project/react_agent.py` ToolRegistry
  2. 测试: `tests/test_*.py` 至少覆盖正常路径 + 边界情况
  3. 注册: Orchestrator `graph.py` 的 agent_map 或 ReAct 的 registry.register()

- **New API endpoint**: 
  1. 后端: `project/api_server.py` 或新 router
  2. 前端: `frontend/src/api.ts` + `vite.config.ts` proxy
  3. Gateway: `application.yml` + `application-docker.yml` 路由
  4. 测试: `tests/test_api_server.py` 加端点测试

- **New frontend page**: 
  1. 页面: `frontend/src/pages/XxxPage.tsx`
  2. 路由: `App.tsx` 加 Route
  3. 导航: `Layout.tsx` NAV 数组加入口

- **New Sub-Agent (Orchestrator)**: 
  1. 实现: `project/orchestrator/agents/xxx.py`
  2. 注册: `graph.py` 的 `agent_map` + plan prompt 的可用 Agent 列表
  3. 测试: 单元测试 (mock LLM) + benchmark 评估集加新场景

## File Patterns

- `project/*.py` (learning versions) and `project/app/**` (production versions) coexist intentionally — never merge them.
- Logs go to `data/logs/`, PID files also in `data/logs/`. Both are gitignored.
- `data/project_chroma_db/` is the vector store — rebuilt via `python project/etl_pipeline.py`.
