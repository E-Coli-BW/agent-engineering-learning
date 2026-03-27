"""
CLI 入口
=========

统一的命令行入口，支持:
  python -m project.app serve            启动 Expert Agent
  python -m project.app serve --react    启动 ReAct Agent
  python -m project.app ask '问题'       同步查询
  python -m project.app stream '问题'    流式查询
  python -m project.app react '问题'     ReAct Agent 本地测试
  python -m project.app wechat           启动微信桥接器

与学习版的区别:
  - 学习版每个文件 (a2a_agent.py / react_agent.py / wechat_bridge.py) 各有自己的 __main__
  - 生产版统一入口，一个 CLI 管所有
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
)


def cmd_serve(args):
    """启动 Agent Server"""
    import uvicorn

    if args.react:
        from .react.agent import create_react_app
        port = int(os.getenv("REACT_PORT", "5002"))
        app = create_react_app()
        print(f"🚀 ReAct Agent Server (port {port})")
        print(f"   Agent Card: http://localhost:{port}/.well-known/agent.json")
        print(f"   Task:       POST http://localhost:{port}/tasks/send")
    else:
        from .expert_server import create_expert_app
        port = int(os.getenv("EXPERT_PORT", "5001"))
        app = create_expert_app()
        print(f"🚀 A2A Expert Agent Server (port {port})")
        print(f"   Agent Card:  http://localhost:{port}/.well-known/agent.json")
        print(f"   Task (sync): POST http://localhost:{port}/tasks/send")
        print(f"   Task (SSE):  POST http://localhost:{port}/tasks/sendSubscribe")
        print(f"   Task query:  GET  http://localhost:{port}/tasks/{{task_id}}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def cmd_ask(args):
    """同步查询 Expert Agent"""
    from .coordinator import CoordinatorAgent

    question = " ".join(args.question)
    if not question:
        print("请提供问题")
        sys.exit(1)

    coordinator = CoordinatorAgent(args.expert)

    print("📡 发现 Expert Agent...")
    card = coordinator.discover()
    print(f"  Name:   {card['name']}")
    print(f"  Skills: {[s['id'] for s in card['skills']]}")
    print()

    skill = coordinator.match_skill(question)
    print(f"❓ 问题: {question}")
    print(f"🔀 路由: {skill}")
    print()

    print("📤 POST /tasks/send ...")
    result = coordinator.send_task(question, skill)
    print(f"📋 Task ID: {result['id']}")
    print(f"📋 Status:  {result['status']['state']}")
    print()

    if result["status"]["state"] == "completed":
        msg = result["status"].get("message", {})
        answer = " ".join(p["text"] for p in msg.get("parts", []) if p.get("type") == "text")
        print(f"💡 回答:\n{answer}")
    else:
        print(f"❌ Task 失败: {result['status']}")


def cmd_stream(args):
    """流式查询 Expert Agent"""
    from .coordinator import CoordinatorAgent

    question = " ".join(args.question)
    if not question:
        print("请提供问题")
        sys.exit(1)

    coordinator = CoordinatorAgent(args.expert)

    print("📡 发现 Expert Agent...")
    card = coordinator.discover()
    print(f"  Name: {card['name']}  streaming={card['capabilities'].get('streaming')}")
    print()

    skill = coordinator.match_skill(question)
    print(f"❓ 问题: {question}")
    print(f"🔀 路由: {skill}")
    print()
    print("📤 POST /tasks/sendSubscribe ...")
    print("-" * 40)

    answer_parts = []
    for event_type, event_data in coordinator.send_subscribe(question, skill):
        if event_type == "status":
            state = event_data.get("state", "?")
            icon = {"working": "⏳", "completed": "✅", "failed": "❌"}.get(state, "❓")
            print(f"\n  {icon} [status] state={state}")
        elif event_type == "artifact":
            for p in event_data.get("parts", []):
                if p.get("type") == "text":
                    token = p["text"]
                    answer_parts.append(token)
                    sys.stdout.write(token)
                    sys.stdout.flush()

    print("\n" + "-" * 40)
    print(f"\n💡 完整回答:\n{''.join(answer_parts)}")


def cmd_react(args):
    """本地 ReAct Agent 测试"""
    from .react.agent import ReActAgent

    question = " ".join(args.question)
    if not question:
        question = "Transformer的Self-Attention缩放因子是什么？"

    agent = ReActAgent()
    print(f"\n{'=' * 60}")
    print(f"❓ 问题: {question}")
    print(f"🔧 可用工具: {list(agent.registry.tools.keys())}")
    print(f"{'=' * 60}\n")

    answer = agent.run(question)
    print(f"\n{'=' * 60}")
    print(f"💡 最终回答:\n{answer}")
    print(f"{'=' * 60}")


def cmd_wechat(args):
    """启动微信桥接器"""
    from .wechat.bridge import WeChatBridge

    bridge = WeChatBridge(
        expert_url=args.expert,
        storage_dir=Path(args.storage),
        stream_mode=args.stream,
        force_login=args.login,
    )
    bridge.start()


def main():
    parser = argparse.ArgumentParser(
        prog="python -m project.app",
        description="Agent Learning — 生产版 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- serve ----
    sp_serve = subparsers.add_parser("serve", help="启动 Agent Server")
    sp_serve.add_argument("--react", action="store_true", help="启动 ReAct Agent (默认 Expert Agent)")
    sp_serve.set_defaults(func=cmd_serve)

    # ---- ask ----
    sp_ask = subparsers.add_parser("ask", help="同步查询 Expert Agent")
    sp_ask.add_argument("question", nargs="+", help="问题")
    sp_ask.add_argument("--expert", default="http://localhost:5001", help="Expert Agent URL")
    sp_ask.set_defaults(func=cmd_ask)

    # ---- stream ----
    sp_stream = subparsers.add_parser("stream", help="流式查询 Expert Agent")
    sp_stream.add_argument("question", nargs="+", help="问题")
    sp_stream.add_argument("--expert", default="http://localhost:5001", help="Expert Agent URL")
    sp_stream.set_defaults(func=cmd_stream)

    # ---- react ----
    sp_react = subparsers.add_parser("react", help="本地 ReAct Agent 测试")
    sp_react.add_argument("question", nargs="*", help="问题")
    sp_react.set_defaults(func=cmd_react)

    # ---- wechat ----
    sp_wechat = subparsers.add_parser("wechat", help="启动微信桥接器")
    sp_wechat.add_argument("--expert", default="http://localhost:5001", help="Expert Agent URL")
    sp_wechat.add_argument("--stream", action="store_true", help="流式模式")
    sp_wechat.add_argument("--login", action="store_true", help="强制重新扫码")
    sp_wechat.add_argument("--storage", default=str(Path(__file__).parent.parent.parent / "data" / "wechat_bridge"),
                           help="凭证存储目录")
    sp_wechat.set_defaults(func=cmd_wechat)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        print()
        print("示例:")
        print("  python -m project.app serve              # 启动 Expert Agent")
        print("  python -m project.app serve --react      # 启动 ReAct Agent")
        print("  python -m project.app ask '什么是LoRA?'  # 同步查询")
        print("  python -m project.app stream '什么是RAG?' # 流式查询")
        print("  python -m project.app react '计算 sqrt(768)' # ReAct 测试")
        print("  python -m project.app wechat             # 微信桥接器")
        sys.exit(0)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
