# 🤖 Agent 学习路线：从原理到实战

> **100% 本地运行**，使用 Ollama + qwen2.5:7b，不依赖任何外部 API。

## 学习顺序（由浅入深）

```
Level 1: 01_chat_basics.py       — 理解 LLM 调用的本质
Level 2: 02_tool_calling.py      — 理解 Agent 的核心：工具调用
Level 3: 03_react_agent.py       — 手写 ReAct 循环，理解 Agent 思维链
Level 4: 04_langgraph_agent.py   — 用 LangGraph 构建生产级 Agent
```

## 运行方式

确保 Ollama 正在运行，然后逐个执行：

```bash
cd agent-learning
.venv/bin/python agent/01_chat_basics.py
.venv/bin/python agent/02_tool_calling.py
.venv/bin/python agent/03_react_agent.py
.venv/bin/python agent/04_langgraph_agent.py
```
