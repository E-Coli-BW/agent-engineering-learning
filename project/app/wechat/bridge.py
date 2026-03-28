"""
微信 ↔ Agent 桥接器
====================

编排层: 把 iLink 协议 (ilink.py) 和 Expert Agent (coordinator.py) 串联起来。

  微信用户 ←→ iLink Server ←→ WeChatBridge ←→ Expert Agent
              (weixin.qq.com)  (本模块)         (port 5001)

与学习版 wechat_bridge.py 的区别:
  - 学习版 ExpertClient 是内嵌在 wechat_bridge.py 的独立类
  - 生产版复用 coordinator.CoordinatorAgent (send_task / send_subscribe)
  - iLink 协议细节拆到 ilink.py，桥接器只做编排
"""

import json
import sys
import time
import signal
import logging
from pathlib import Path
from typing import Optional, Callable

from .ilink import (
    Credentials, SessionExpiredError,
    qr_login, poll_messages, send_text, send_typing, extract_user_message,
)
from ..coordinator import CoordinatorAgent

logger = logging.getLogger("app.wechat.bridge")

# 默认存储路径
_DEFAULT_STORAGE = Path(__file__).parent.parent.parent.parent / "data" / "wechat_bridge"


class ExpertClient:
    """
    调用 A2A Expert Agent 的 HTTP 客户端

    基于 CoordinatorAgent 封装, 提供更简洁的 ask / ask_stream 接口。
    """

    def __init__(self, expert_url: str = "http://localhost:5001"):
        self._coordinator = CoordinatorAgent(expert_url)
        self.agent_card: Optional[dict] = None

    def discover(self) -> dict:
        self.agent_card = self._coordinator.discover()
        return self.agent_card

    def match_skill(self, text: str) -> str:
        return self._coordinator.match_skill(text)

    def ask(self, question: str, skill: str = None) -> str:
        """同步调用 Expert Agent → 返回回答文本"""
        result = self._coordinator.send_task(question, skill)
        state = result.get("status", {}).get("state", "")
        if state == "completed":
            msg = result["status"].get("message", {})
            parts = msg.get("parts", [])
            return " ".join(p["text"] for p in parts if p.get("type") == "text")
        else:
            error_msg = result.get("status", {}).get("message", {})
            return f"[Expert Agent 错误] state={state}: {error_msg}"

    def ask_stream(self, question: str, skill: str = None, on_token: Callable[[str], None] = None) -> str:
        """流式调用 Expert Agent → 逐 token 回调 → 返回完整回答"""
        full_text = []
        for event_type, event_data in self._coordinator.send_subscribe(question, skill):
            if event_type == "artifact":
                for p in event_data.get("parts", []):
                    if p.get("type") == "text":
                        token = p["text"]
                        full_text.append(token)
                        if on_token:
                            on_token(token)
        return "".join(full_text)


class WeChatBridge:
    """
    微信 ↔ A2A Expert Agent 桥接器

    使用方式:
        bridge = WeChatBridge(expert_url="http://localhost:5001")
        bridge.start()
    """

    def __init__(
        self,
        expert_url: str = "http://localhost:5001",
        storage_dir: Path = _DEFAULT_STORAGE,
        stream_mode: bool = False,
        force_login: bool = False,
    ):
        self.expert = ExpertClient(expert_url)
        self.storage_dir = storage_dir
        self.stream_mode = stream_mode
        self.force_login = force_login
        self.cred: Optional[Credentials] = None
        self.running = False

        self._context_tokens: dict[str, str] = {}
        self._chat_history: dict[str, list[tuple[str, str]]] = {}
        self._max_history_rounds: int = 5
        self._cursor: str = ""
        self._cursor_path = self.storage_dir / "poll_cursor.json"
        self._cred_path = self.storage_dir / "credentials.json"

    # ---- 公共接口 ----

    def start(self):
        """一键启动: 登录 → 发现 Expert → 消息循环"""
        print()
        print("=" * 60)
        print("🌉 WeChat ↔ A2A Expert Agent 桥接器")
        print("=" * 60)
        print()

        self.login()

        print()
        print("📡 发现 Expert Agent...")
        try:
            card = self.expert.discover()
            print(f"  ✅ Agent: {card['name']}")
            print(f"  📋 Skills: {[s['id'] for s in card['skills']]}")
            print(f"  🌊 Streaming: {card['capabilities'].get('streaming', False)}")
        except Exception as e:
            print(f"  ❌ 无法连接 Expert Agent: {e}")
            print(f"  请先启动: python -m project.app serve")
            sys.exit(1)

        print()
        print(f"🔄 开始消息循环 (stream_mode={self.stream_mode})")
        print(f"   按 Ctrl+C 停止")
        print()

        self.run_loop()

    def login(self):
        """登录 (加载已保存凭证或 QR 扫码)"""
        if not self.force_login:
            self.cred = Credentials.load(self._cred_path)
        if not self.cred:
            self.cred = qr_login()
            self.cred.save(self._cred_path)
        else:
            print(f"  📂 已加载保存的登录凭证")
            print(f"     (使用 --login 强制重新扫码)")
        self._load_cursor()

    def run_loop(self):
        """消息循环 — 长轮询收消息 + 转发到 Expert"""
        self.running = True

        def _shutdown(signum, frame):
            print("\n\n🛑 正在停止...")
            self.running = False
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        consecutive_errors = 0
        while self.running:
            try:
                msgs, new_cursor = poll_messages(self.cred, self._cursor)
                self._cursor = new_cursor
                self._save_cursor()
                consecutive_errors = 0
                if msgs:
                    logger.info("📬 收到 %d 条消息", len(msgs))
                for msg in msgs:
                    self._handle_message(msg)
            except SessionExpiredError:
                logger.error("Session 过期，需要重新登录")
                print("\n⚠️  微信 Session 过期，请重新运行并扫码登录")
                self._cred_path.unlink(missing_ok=True)
                self._cursor_path.unlink(missing_ok=True)
                self.running = False
            except Exception as e:
                consecutive_errors += 1
                wait = min(consecutive_errors * 5, 60)
                logger.error("轮询出错 (%d次): %s, %ds 后重试", consecutive_errors, e, wait)
                time.sleep(wait)

        print("👋 桥接器已停止")

    # ---- 内部方法 ----

    def _handle_message(self, raw_msg: dict):
        """处理单条消息"""
        logger.info("📨 原始消息: from=%s, msg_type=%s, group=%s",
                     raw_msg.get("from_user_id", "?")[:8] if raw_msg.get("from_user_id") else "?",
                     raw_msg.get("message_type", "?"),
                     raw_msg.get("group_id", ""))

        user_id, context_token, text = extract_user_message(
            raw_msg,
            bot_user_id=self.cred.ilink_bot_id if self.cred else "",
        )
        if not text:
            return

        self._context_tokens[user_id] = context_token

        # 特殊指令
        if text.strip() in ("清除历史", "重置", "reset", "clear"):
            self._chat_history.pop(user_id, None)
            send_text(self.cred, user_id, context_token, "✅ 对话历史已清除")
            logger.info("🗑️ 清除 %s 的对话历史", user_id[:8])
            return

        logger.info("📩 来自 %s: %s", user_id[:8] + "...", text[:50])

        try:
            send_typing(self.cred, user_id, context_token, status=1)
            question_with_history = self._build_prompt_with_history(user_id, text)

            if self.stream_mode:
                answer = self._ask_stream(question_with_history, user_id, context_token)
            else:
                answer = self._ask_sync(question_with_history)

            if answer and answer.strip():
                send_typing(self.cred, user_id, context_token, status=2)
                send_text(self.cred, user_id, context_token, answer)
                logger.info("📤 回复 %s: %s", user_id[:8] + "...", answer[:50])
                self._append_history(user_id, "user", text)
                self._append_history(user_id, "assistant", answer)
            else:
                send_text(self.cred, user_id, context_token, "抱歉，我暂时无法回答这个问题。")
        except Exception as e:
            logger.error("处理消息失败: %s", e)
            send_typing(self.cred, user_id, context_token, status=2)
            send_text(self.cred, user_id, context_token, f"⚠️ 处理出错: {e}")

    def _ask_sync(self, question: str) -> str:
        skill = self.expert.match_skill(question)
        logger.info("  🔀 路由到 skill=%s", skill)
        return self.expert.ask(question, skill)

    def _ask_stream(self, question: str, user_id: str, context_token: str) -> str:
        skill = self.expert.match_skill(question)
        logger.info("  🔀 路由到 skill=%s (stream)", skill)

        last_typing = time.time()

        def on_token(token: str):
            nonlocal last_typing
            if time.time() - last_typing > 5:
                send_typing(self.cred, user_id, context_token, status=1)
                last_typing = time.time()

        return self.expert.ask_stream(question, skill, on_token=on_token)

    def _build_prompt_with_history(self, user_id: str, current_question: str) -> str:
        history = self._chat_history.get(user_id, [])
        if not history:
            return current_question

        lines = ["[对话历史]"]
        for role, text in history:
            label = "用户" if role == "user" else "助手"
            short = text[:200] + "..." if len(text) > 200 else text
            lines.append(f"{label}: {short}")
        lines.append("")
        lines.append("[当前问题]")
        lines.append(current_question)
        return "\n".join(lines)

    def _append_history(self, user_id: str, role: str, text: str):
        if user_id not in self._chat_history:
            self._chat_history[user_id] = []
        self._chat_history[user_id].append((role, text))
        max_items = self._max_history_rounds * 2
        if len(self._chat_history[user_id]) > max_items:
            self._chat_history[user_id] = self._chat_history[user_id][-max_items:]

    def _load_cursor(self):
        if self._cursor_path.exists():
            try:
                data = json.loads(self._cursor_path.read_text())
                self._cursor = data.get("cursor", "")
                logger.info("已加载轮询游标")
            except Exception:
                self._cursor = ""

    def _save_cursor(self):
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
        self._cursor_path.write_text(json.dumps({"cursor": self._cursor}))
