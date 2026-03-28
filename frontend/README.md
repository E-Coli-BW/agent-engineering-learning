# 🖥️ Agent Learning Frontend

> Chat & Manage 前端应用 — 连接后端 RAG / A2A / ReAct Agent

## 功能

| 页面 | 功能 | 对接后端 |
|------|------|----------|
| 💬 Chat | 流式智能问答 | RAG API (:8000) / A2A (:5001) / ReAct (:5002) |
| 📊 Dashboard | 系统健康状态 & 性能指标 | RAG `/health` `/metrics` + Agent `/health` |
| 🤖 Agents | A2A Agent Card 查看 & 管理 | `/.well-known/agent.json` |
| 🗄️ ETL | 数据导入管道触发 & 历史 | RAG `/etl/run` |

## 技术栈

- **React 19** + **TypeScript**
- **Vite 6** (开发 & 构建)
- **Tailwind CSS 3** (样式)
- **React Router 7** (路由)
- **Lucide React** (图标)
- **React Markdown** (Markdown 渲染)

## 快速开始

```bash
# 1. 安装依赖
cd frontend
npm install

# 2. 启动开发服务器 (端口 3000)
npm run dev

# 3. 确保后端服务已启动 (任选其一或全部):
#    RAG API:     python project/api_server.py       → :8000
#    A2A Expert:  python project/a2a_agent.py --serve → :5001
#    ReAct Agent: python project/react_agent.py --serve → :5002
```

## 架构

```
前端 (localhost:3000)
  ├── /api/rag/*   → proxy → localhost:8000 (RAG API Server)
  ├── /api/a2a/*   → proxy → localhost:5001 (A2A Expert Agent)
  └── /api/react/* → proxy → localhost:5002 (ReAct Agent)
```

Vite 的 proxy 配置自动代理请求，无需处理 CORS。

## 目录结构

```
frontend/
├── index.html
├── package.json
├── vite.config.ts          # Vite 配置 + API 代理
├── tailwind.config.js
├── tsconfig.json
└── src/
    ├── main.tsx            # 入口
    ├── App.tsx             # 路由配置
    ├── api.ts              # API 客户端 (RAG + A2A + ReAct)
    ├── index.css           # Tailwind + 全局样式
    ├── components/
    │   └── Layout.tsx      # 侧边栏布局
    └── pages/
        ├── ChatPage.tsx    # 💬 聊天页面 (流式输出)
        ├── DashboardPage.tsx # 📊 仪表盘
        ├── AgentsPage.tsx  # 🤖 Agent 管理
        └── ETLPage.tsx     # 🗄️ ETL 管道
```
