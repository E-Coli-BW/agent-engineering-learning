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

```bash
# One-command full-stack start/stop (requires Ollama running):
./start.sh              # start all 4 services
./start.sh status       # check health
./start.sh stop         # kill all

# Individual services:
./start.sh front        # frontend only
./start.sh rag          # RAG API only
./start.sh a2a          # A2A agent only

# Frontend dev (from frontend/):
npm run dev             # Vite dev server :3000

# Python backend needs PYTHONPATH for uvicorn reload:
PYTHONPATH=. python project/api_server.py

# Production-style CLI:
python -m project.app serve           # A2A Expert :5001
python -m project.app serve --react   # ReAct Agent :5002
python -m project.app ask 'question'  # sync query
```

## When Adding New Features

- **New backend tool for ReAct**: Register in `project/react_agent.py` `ToolRegistry` or `project/app/react/tools.py`. Pattern: `registry.register(name, description, func)`.
- **New API endpoint**: Add to `project/api_server.py`, then add corresponding proxy + client function in `frontend/src/api.ts` + `frontend/vite.config.ts`.
- **New frontend page**: Create in `frontend/src/pages/`, add route in `App.tsx`, add nav entry in `components/Layout.tsx` `NAV` array.
- **New teaching module**: Create as self-contained script (not package). Include heavy docstrings explaining concepts. Follow the `Level 1→N` naming pattern.

## File Patterns

- `project/*.py` (learning versions) and `project/app/**` (production versions) coexist intentionally — never merge them.
- Logs go to `data/logs/`, PID files also in `data/logs/`. Both are gitignored.
- `data/project_chroma_db/` is the vector store — rebuilt via `python project/etl_pipeline.py`.
