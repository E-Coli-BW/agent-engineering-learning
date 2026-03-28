"""
project.app — 生产版 Agent 系统
================================

从学习版代码（project/*.py）重构而来的 package 结构。
学习版代码保持原样不动，供学习参考。

模块划分:
  models.py          A2A 协议数据模型 (Part, Message, Task, AgentCard, Skill)
  ollama_client.py   Ollama HTTP 客户端 (同步 + 流式)
  expert_server.py   FastAPI Expert Agent 服务 (A2A 端点)
  coordinator.py     Coordinator Agent HTTP 客户端
  wechat/
    ilink.py         iLink 协议层 (凭证、QR 登录、消息收发)
    bridge.py        微信 ↔ Agent 桥接器
  react/
    tools.py         工具注册表 + 内置工具
    agent.py         ReAct Agent 循环
  cli.py             CLI 入口

运行方式:
  python -m project.app serve          # 启动 Expert Agent
  python -m project.app ask '问题'     # 同步查询
  python -m project.app stream '问题'  # 流式查询
  python -m project.app react '问题'   # ReAct Agent
  python -m project.app wechat         # 微信桥接器
"""

__version__ = "2.0.0"
