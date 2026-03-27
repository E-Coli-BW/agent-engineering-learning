"""
WeChat ↔ A2A Expert Agent 桥接器
================================

直接使用 iLink Bot 协议 (https://www.wechatbot.dev/en/protocol)
将微信消息转发到你的 A2A Expert Agent (HTTP REST)，然后把回复发回微信。

┌──────────────────────────────────────────────────────────────┐
│  架构:                                                       │
│                                                              │
│  微信用户 ←→ iLink Server ←→ WeChatBridge ←→ A2A Expert     │
│                (weixin.qq.com)  (本文件)        (port 5001)   │
│                                                              │
│  核心流程:                                                    │
│    1. QR 码登录 → 拿到 bot_token                              │
│    2. 长轮询收消息 (getupdates, 35s hold)                     │
│    3. 提取 context_token + 用户文本                           │
│    4. POST /tasks/send → Expert Agent                        │
│    5. 拿到回答 → sendmessage 发回微信                         │
│    6. 可选: typing 指示器 ("对方正在输入中...")                 │
└──────────────────────────────────────────────────────────────┘

运行前提:
  1. Expert Agent 已启动:  python project/a2a_agent.py --serve
  2. 微信账号能使用 iLink Bot API (ClawBot/爪机器人 功能)

运行方式:
  python project/wechat_bridge.py                  # 默认连接 localhost:5001
  python project/wechat_bridge.py --expert http://192.168.1.100:5001  # 远程 Expert
  python project/wechat_bridge.py --stream          # 流式模式 (SSE)

微信配置说明:
  你的微信需要能使用 "爪机器人 (ClawBot)" 功能:
  - 这是微信官方的 iLink Bot API，并非第三方逆向接口
  - 需要在微信 "发现" → "小程序" 中搜索 "爪机器人" 并开通
  - 开通后，本脚本的 QR 码扫描就是 ClawBot 的授权扫码
  - 扫码后，别人给你发的私聊消息会被转发到本 Bot
  - 仅支持私聊，不支持群聊
"""

import os
import sys
import json
import time
import base64
import struct
import logging
import signal
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_DIR = PROJECT_ROOT / "data" / "wechat_bridge"

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("wechat-bridge")


# ============================================================
# iLink 协议常量
# ============================================================
ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
ILINK_API_PREFIX = "ilink/bot"  # 所有 API 端点的前缀
CHANNEL_VERSION = "2.0.0"


# ============================================================
# 凭证存储
# ============================================================

@dataclass
class Credentials:
    """登录凭证"""
    bot_token: str = ""
    base_url: str = ILINK_BASE_URL
    ilink_bot_id: str = ""
    ilink_user_id: str = ""
    saved_at: str = ""

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False))
        logger.info("凭证已保存到 %s", path)

    @classmethod
    def load(cls, path: Path) -> Optional["Credentials"]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            cred = cls(**data)
            logger.info("已加载保存的凭证 (bot_id=%s, saved_at=%s)", cred.ilink_bot_id, cred.saved_at)
            return cred
        except Exception as e:
            logger.warning("加载凭证失败: %s", e)
            return None


# ============================================================
# iLink HTTP 工具
# ============================================================

def _make_uin_header() -> str:
    """生成 X-WECHAT-UIN: base64(str(random_uint32))"""
    random_bytes = os.urandom(4)
    uint32 = struct.unpack("<I", random_bytes)[0]
    return base64.b64encode(str(uint32).encode()).decode()


def _ilink_request(base_url: str, endpoint: str, token: str = "", body: dict = None, method: str = "POST", timeout: int = 40) -> dict:
    """
    发送 iLink 协议请求

    所有请求都需要:
      - AuthorizationType: ilink_bot_token
      - X-WECHAT-UIN: base64(random_uint32_str)
    POST 请求额外:
      - Authorization: Bearer <bot_token>
      - body 里带 base_info.channel_version
    """
    url = f"{base_url.rstrip('/')}/{ILINK_API_PREFIX}/{endpoint}"

    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _make_uin_header(),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if body is not None:
        if "base_info" not in body:
            body["base_info"] = {"channel_version": CHANNEL_VERSION}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    else:
        data = None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.readable() else ""
        logger.error("HTTP %d: %s — %s", e.code, url, error_body[:200])
        raise
    except urllib.error.URLError as e:
        logger.error("URL Error: %s — %s", url, e.reason)
        raise


def _ilink_get(base_url: str, endpoint: str, token: str = "", timeout: int = 10) -> dict:
    """
    发送 iLink GET 请求

    GET 请求也需要带 AuthorizationType 和 X-WECHAT-UIN 头
    """
    url = f"{base_url.rstrip('/')}/{ILINK_API_PREFIX}/{endpoint}"

    headers = {
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _make_uin_header(),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace") if e.readable() else ""
        logger.error("HTTP %d: %s — %s", e.code, url, error_body[:200])
        raise
    except urllib.error.URLError as e:
        logger.error("URL Error: %s — %s", url, e.reason)
        raise


# ============================================================
# QR 登录
# ============================================================

def qr_login(base_url: str = ILINK_BASE_URL) -> Credentials:
    """
    iLink QR 登录流程:
      1. GET /ilink/bot/get_bot_qrcode?bot_type=3  → 拿到 QR 码 URL
      2. 轮询 GET /ilink/bot/get_qrcode_status      → wait → scaned → confirmed
      3. confirmed 返回 bot_token + base_url
    """
    # Step 1: 获取 QR 码
    logger.info("请求 QR 码...")

    qr_data = _ilink_get(base_url, "get_bot_qrcode?bot_type=3", timeout=15)

    qrcode = qr_data.get("qrcode", "")
    qr_img_url = qr_data.get("qrcode_img_content", "")

    if not qrcode:
        raise RuntimeError(f"获取 QR 码失败: {qr_data}")

    print()
    print("=" * 50)
    print("📱 请用微信扫描以下 QR 码登录")
    print("=" * 50)
    print()
    print(f"  QR URL: {qr_img_url}")
    print()

    # 尝试用 qrcode-terminal 库在终端显示 (可选)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(qr_img_url)
        qr.print_ascii(invert=True)
    except ImportError:
        print("  (提示: pip install qrcode 可在终端直接显示 QR 码图案)")
        print(f"  请在浏览器打开上面的 URL 扫码")
    print()

    # Step 2: 轮询扫码状态
    print("⏳ 等待扫码...")
    while True:
        status_data = _ilink_get(base_url, f"get_qrcode_status?qrcode={urllib.parse.quote(qrcode)}", timeout=60)

        status = status_data.get("status", "")

        if status == "wait":
            pass  # 继续等待
        elif status == "scaned":
            print("  📲 已扫码，请在微信上确认登录...")
        elif status == "confirmed":
            print("  ✅ 登录成功!")
            cred = Credentials(
                bot_token=status_data.get("bot_token", ""),
                base_url=status_data.get("baseurl", base_url),
                ilink_bot_id=status_data.get("ilink_bot_id", ""),
                ilink_user_id=status_data.get("ilink_user_id", ""),
                saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return cred
        elif status == "expired":
            raise RuntimeError("QR 码已过期，请重新运行")
        else:
            logger.debug("未知状态: %s", status)

        time.sleep(2)


# ============================================================
# 消息收发
# ============================================================

def poll_messages(cred: Credentials, cursor: str = "") -> tuple[list[dict], str]:
    """
    POST /getupdates — 长轮询收消息

    返回: (消息列表, 新游标)
    - 首次调用: cursor="" (空字符串)
    - 后续调用: 用上次返回的 get_updates_buf
    - 服务器 hold 连接最多 35 秒
    """
    body = {"get_updates_buf": cursor}
    result = _ilink_request(
        cred.base_url,
        "getupdates",
        cred.bot_token,
        body,
        timeout=40,  # 略大于 35 秒的 long-poll
    )

    ret = result.get("ret", -1)
    errcode = result.get("errcode")

    # 调试: 首次或异常时打印完整响应
    if ret != 0 or errcode:
        logger.info("getupdates 完整响应: %s", json.dumps(result, ensure_ascii=False)[:500])

    # 检查 session 过期 (ret=-14 或 errcode=-14)
    if ret == -14 or errcode == -14:
        raise SessionExpiredError("Session 过期 (errcode: -14), 需要重新登录")
    if ret != 0 and ret != -1:
        logger.warning("getupdates 返回 ret=%s: %s", ret, result.get("errmsg", ""))
        return [], cursor

    # ret=0 或者 ret 字段不存在(-1 是我们的默认值) 都认为是成功

    msgs = result.get("msgs", [])
    new_cursor = result.get("get_updates_buf", cursor)
    return msgs, new_cursor


def send_text(cred: Credentials, to_user_id: str, context_token: str, text: str):
    """
    POST /sendmessage — 发送文本消息

    关键: 必须带 context_token, 否则消息无法路由到正确的对话
    """
    # 微信单条消息有长度限制，做分段
    segments = _split_text(text, max_len=2000)  # 保守一些，2000字一段
    logger.info("sendmessage: %d 段, 总长 %d", len(segments), len(text))
    for i, segment in enumerate(segments):
        import uuid as _uuid
        client_id = f"wechat-bridge-{_uuid.uuid4().hex[:16]}"
        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": client_id,
                "context_token": context_token,
                "message_type": 2,   # BOT message
                "message_state": 2,  # FINISH
                "item_list": [
                    {"type": 1, "text_item": {"text": segment}}
                ],
            }
        }
        resp = _ilink_request(cred.base_url, "sendmessage", cred.bot_token, body)
        logger.info("sendmessage [%d/%d] 响应: %s | len=%d",
                     i + 1, len(segments),
                     json.dumps(resp, ensure_ascii=False)[:200], len(segment))
        if i < len(segments) - 1:
            time.sleep(1.0)  # 分段间隔 1 秒，避免被限流


def send_typing(cred: Credentials, user_id: str, context_token: str, status: int = 1):
    """
    显示 "对方正在输入中..." 指示器

    分两步:
      1. POST /getconfig → 拿 typing_ticket
      2. POST /sendtyping → 发送 typing 状态 (1=开始, 2=停止)
    """
    try:
        # Step 1: 获取 typing_ticket
        config_body = {
            "ilink_user_id": user_id,
            "context_token": context_token,
        }
        config_resp = _ilink_request(cred.base_url, "getconfig", cred.bot_token, config_body)
        ticket = config_resp.get("typing_ticket", "")
        if not ticket:
            return

        # Step 2: 发送 typing 状态
        typing_body = {
            "ilink_user_id": user_id,
            "typing_ticket": ticket,
            "status": status,
        }
        _ilink_request(cred.base_url, "sendtyping", cred.bot_token, typing_body)
    except Exception as e:
        logger.debug("发送 typing 失败 (非关键): %s", e)


def _split_text(text: str, max_len: int = 4000) -> list[str]:
    """按自然边界分割文本"""
    if len(text) <= max_len:
        return [text]

    segments = []
    while text:
        if len(text) <= max_len:
            segments.append(text)
            break

        # 在 max_len 范围内找最后一个换行
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
            # 找不到换行，找句号/问号
            for sep in ["。", "！", "？", ". ", "! ", "? "]:
                cut = text.rfind(sep, 0, max_len)
                if cut > 0:
                    cut += len(sep)
                    break
        if cut <= 0:
            cut = max_len

        segments.append(text[:cut])
        text = text[cut:]

    return segments


def extract_user_message(msg: dict, bot_user_id: str = "") -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从 iLink 消息中提取 (user_id, context_token, text)

    消息结构:
    {
      "from_user_id": "...",
      "context_token": "...",
      "item_list": [
        {"type": 1, "text_item": {"text": "用户消息"}},
        {"type": 2, "image_item": {...}},   // 图片
        {"type": 3, "voice_item": {...}},   // 语音
        ...
      ]
    }
    """
    user_id = msg.get("from_user_id")
    context_token = msg.get("context_token")

    # ---- message_type 过滤 ----
    # iLink 协议: MessageType.USER=1, MessageType.BOT=2, MessageType.NONE=0
    #
    # 默认情况: 别人发给你的私聊 → message_type=1 (USER)
    # 你在 ClawBot 界面发的   → message_type=1 (也算 USER，因为是你主动输入的)
    # Bot 通过 sendmessage 发的回复 → message_type=2 (BOT)
    #
    # 关键: 必须排除 message_type=2，否则 Bot 自己的回复会触发无限循环
    msg_type = msg.get("message_type")
    if msg_type == 2:  # BOT 发出的消息 → 跳过 (防止无限循环)
        return None, None, None

    # 额外防护: 如果发送者是 Bot 账号本身 (ilink_bot_id)，跳过
    # 注意: bot_user_id 这里传的是 ilink_bot_id (如 xxx@im.bot)，不是 ilink_user_id
    if bot_user_id and user_id == bot_user_id:
        return None, None, None

    # 跳过群消息
    if msg.get("group_id"):
        return None, None, None

    if not user_id or not context_token:
        return None, None, None

    # 提取文本
    text_parts = []
    for item in msg.get("item_list", []):
        item_type = item.get("type", 0)
        if item_type == 1:  # 文本
            t = item.get("text_item", {}).get("text", "")
            if t:
                text_parts.append(t)
        elif item_type == 3:  # 语音转文字
            t = item.get("voice_item", {}).get("text", "")
            if t:
                text_parts.append(f"[语音] {t}")

    text = " ".join(text_parts) if text_parts else None
    return user_id, context_token, text


class SessionExpiredError(Exception):
    """iLink session 过期"""
    pass


# ============================================================
# A2A Expert Agent 客户端 (复用你的 CoordinatorAgent 的逻辑)
# ============================================================

class ExpertClient:
    """
    调用 A2A Expert Agent 的 HTTP 客户端

    对接你的 a2a_agent.py 的 REST API:
      POST /tasks/send          → 同步
      POST /tasks/sendSubscribe → SSE 流式
    """

    def __init__(self, expert_url: str = "http://localhost:5001"):
        self.expert_url = expert_url.rstrip("/")
        self.agent_card: Optional[dict] = None

    def discover(self) -> dict:
        """GET /.well-known/agent.json — 发现 Agent"""
        url = f"{self.expert_url}/.well-known/agent.json"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.agent_card = json.loads(resp.read().decode("utf-8"))
        return self.agent_card

    def match_skill(self, text: str) -> str:
        """根据 Agent Card 的 skills + tags 匹配"""
        if not self.agent_card:
            return "general_qa"

        q = text.lower()
        best_skill, best_score = "general_qa", 0

        for skill in self.agent_card.get("skills", []):
            score = sum(2 for tag in skill.get("tags", []) if tag.lower() in q)
            if score > best_score:
                best_score = score
                best_skill = skill.get("id", "general_qa")

        # fallback
        if best_score == 0:
            rules = [
                (["transformer", "attention", "qkv", "注意力"], "transformer_theory"),
                (["lora", "qlora", "微调", "sft", "rlhf"], "lora_finetuning"),
                (["rag", "检索", "向量", "embedding"], "rag_system"),
                (["agent", "tool call", "react", "langgraph", "mcp"], "agent_development"),
                (["推理", "部署", "kv cache", "量化"], "inference_deployment"),
                (["知识图谱", "knowledge graph", "graph rag"], "knowledge_graph"),
            ]
            for keywords, sid in rules:
                if any(kw in q for kw in keywords):
                    return sid
        return best_skill

    def ask(self, question: str, skill: str = None) -> str:
        """
        同步调用 Expert Agent

        POST /tasks/send → 返回回答文本
        """
        import uuid as _uuid
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{_uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/send", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        state = result.get("status", {}).get("state", "")
        if state == "completed":
            msg = result["status"].get("message", {})
            parts = msg.get("parts", [])
            return " ".join(p["text"] for p in parts if p.get("type") == "text")
        else:
            error_msg = result.get("status", {}).get("message", {})
            return f"[Expert Agent 错误] state={state}: {error_msg}"

    def ask_stream(self, question: str, skill: str = None, on_token: Callable[[str], None] = None) -> str:
        """
        流式调用 Expert Agent

        POST /tasks/sendSubscribe → SSE 逐 token 回调
        返回完整回答
        """
        import uuid as _uuid
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{_uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/sendSubscribe", data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )

        full_text = []
        with urllib.request.urlopen(req, timeout=120) as resp:
            event_type = None
            data_lines = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line == "":
                    if event_type and data_lines:
                        event_data = json.loads("".join(data_lines))
                        if event_type == "artifact":
                            for p in event_data.get("parts", []):
                                if p.get("type") == "text":
                                    token = p["text"]
                                    full_text.append(token)
                                    if on_token:
                                        on_token(token)
                    event_type = None
                    data_lines = []

        return "".join(full_text)


# ============================================================
# WeChat Bridge — 核心桥接器
# ============================================================

class WeChatBridge:
    """
    微信 ↔ A2A Expert Agent 桥接器

    使用方式:
        bridge = WeChatBridge(expert_url="http://localhost:5001")
        bridge.start()   # 阻塞运行

    或者编程式使用:
        bridge = WeChatBridge(expert_url="http://localhost:5001")
        bridge.login()   # 扫码登录
        bridge.run_loop() # 开始消息循环

    架构:
        微信用户 → iLink Server → [长轮询] → WeChatBridge → ExpertClient → A2A Expert
                 ← iLink Server ← [sendmessage] ←     ←       ←       ←
    """

    def __init__(
        self,
        expert_url: str = "http://localhost:5001",
        storage_dir: Path = STORAGE_DIR,
        stream_mode: bool = False,
        force_login: bool = False,
    ):
        self.expert = ExpertClient(expert_url)
        self.storage_dir = storage_dir
        self.stream_mode = stream_mode
        self.force_login = force_login
        self.cred: Optional[Credentials] = None
        self.running = False

        # 每个用户缓存最新的 context_token
        self._context_tokens: dict[str, str] = {}
        # typing_ticket 缓存 (userId -> (ticket, expire_time))
        self._typing_tickets: dict[str, tuple[str, float]] = {}
        # 每个用户的对话历史 (userId -> [(role, text), ...])，最多保留最近 5 轮
        self._chat_history: dict[str, list[tuple[str, str]]] = {}
        self._max_history_rounds: int = 5
        # 消息游标
        self._cursor: str = ""
        # 游标持久化路径
        self._cursor_path = self.storage_dir / "poll_cursor.json"
        # 凭证路径
        self._cred_path = self.storage_dir / "credentials.json"

    # ---- 公共接口 ----

    def start(self):
        """一键启动: 登录 → 发现 Expert → 消息循环"""
        print()
        print("=" * 60)
        print("🌉 WeChat ↔ A2A Expert Agent 桥接器")
        print("=" * 60)
        print()

        # 1. 登录
        self.login()

        # 2. 发现 Expert Agent
        print()
        print("📡 发现 Expert Agent...")
        try:
            card = self.expert.discover()
            print(f"  ✅ Agent: {card['name']}")
            print(f"  📋 Skills: {[s['id'] for s in card['skills']]}")
            print(f"  🌊 Streaming: {card['capabilities'].get('streaming', False)}")
        except Exception as e:
            print(f"  ❌ 无法连接 Expert Agent: {e}")
            print(f"  请先启动: python project/a2a_agent.py --serve")
            sys.exit(1)

        # 3. 消息循环
        print()
        print(f"🔄 开始消息循环 (stream_mode={self.stream_mode})")
        print(f"   按 Ctrl+C 停止")
        print()

        self.run_loop()

    def login(self):
        """登录 (加载已保存的凭证或 QR 扫码)"""
        if not self.force_login:
            self.cred = Credentials.load(self._cred_path)

        if not self.cred:
            self.cred = qr_login()
            self.cred.save(self._cred_path)
        else:
            print(f"  📂 已加载保存的登录凭证")
            print(f"     (使用 --login 强制重新扫码)")

        # 加载保存的游标
        self._load_cursor()

    def run_loop(self):
        """消息循环 — 长轮询收消息 + 转发到 Expert"""
        self.running = True

        # 优雅退出
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
                # 清除凭证
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
        # 调试: 打印原始消息的关键字段
        logger.info("📨 原始消息: from=%s, msg_type=%s, group=%s, keys=%s",
                      raw_msg.get("from_user_id", "?")[:8] if raw_msg.get("from_user_id") else "?",
                      raw_msg.get("message_type", "?"),
                      raw_msg.get("group_id", ""),
                      list(raw_msg.keys()))

        user_id, context_token, text = extract_user_message(
            raw_msg,
            bot_user_id=self.cred.ilink_bot_id if self.cred else "",
        )

        if not text:
            # 调试: 说明为什么跳过
            if raw_msg.get("from_user_id"):
                logger.info("⏭️  跳过消息: user=%s, msg_type=%s, has_items=%s, text_extracted=%s",
                            raw_msg.get("from_user_id", "?")[:8],
                            raw_msg.get("message_type", "?"),
                            bool(raw_msg.get("item_list")),
                            text is not None)
            return

        # 缓存 context_token
        self._context_tokens[user_id] = context_token

        # 特殊指令: 清除对话历史
        if text.strip() in ("清除历史", "重置", "reset", "clear"):
            self._chat_history.pop(user_id, None)
            send_text(self.cred, user_id, context_token, "✅ 对话历史已清除")
            logger.info("🗑️ 清除 %s 的对话历史", user_id[:8])
            return

        logger.info("📩 来自 %s: %s", user_id[:8] + "...", text[:50])

        try:
            # 发送 typing 指示器 (best-effort, 不影响主流程)
            send_typing(self.cred, user_id, context_token, status=1)

            # 构建带历史的问题
            question_with_history = self._build_prompt_with_history(user_id, text)

            if self.stream_mode:
                answer = self._ask_stream(question_with_history, user_id, context_token)
            else:
                answer = self._ask_sync(question_with_history)

            if answer and answer.strip():
                send_typing(self.cred, user_id, context_token, status=2)
                send_text(self.cred, user_id, context_token, answer)
                logger.info("📤 回复 %s: %s", user_id[:8] + "...", answer[:50])
                # 记录对话历史
                self._append_history(user_id, "user", text)
                self._append_history(user_id, "assistant", answer)
            else:
                send_text(self.cred, user_id, context_token, "抱歉，我暂时无法回答这个问题。")

        except Exception as e:
            logger.error("处理消息失败: %s", e)
            send_typing(self.cred, user_id, context_token, status=2)
            send_text(self.cred, user_id, context_token, f"⚠️ 处理出错: {e}")

    def _ask_sync(self, question: str) -> str:
        """同步调用 Expert"""
        skill = self.expert.match_skill(question)
        logger.info("  🔀 路由到 skill=%s", skill)
        return self.expert.ask(question, skill)

    def _ask_stream(self, question: str, user_id: str, context_token: str) -> str:
        """
        流式调用 Expert

        在流式过程中定期发送 typing keepalive
        """
        skill = self.expert.match_skill(question)
        logger.info("  🔀 路由到 skill=%s (stream)", skill)

        last_typing = time.time()

        def on_token(token: str):
            nonlocal last_typing
            # 每 5 秒发一次 typing keepalive
            if time.time() - last_typing > 5:
                send_typing(self.cred, user_id, context_token, status=1)
                last_typing = time.time()

        return self.expert.ask_stream(question, skill, on_token=on_token)

    def _build_prompt_with_history(self, user_id: str, current_question: str) -> str:
        """
        把对话历史拼到当前问题前面

        格式:
          [对话历史]
          用户: xxx
          助手: xxx
          用户: xxx
          助手: xxx

          [当前问题]
          xxx
        """
        history = self._chat_history.get(user_id, [])
        if not history:
            return current_question

        lines = ["[对话历史]"]
        for role, text in history:
            label = "用户" if role == "user" else "助手"
            # 历史消息截断，避免 prompt 太长
            short = text[:200] + "..." if len(text) > 200 else text
            lines.append(f"{label}: {short}")
        lines.append("")
        lines.append(f"[当前问题]")
        lines.append(current_question)
        return "\n".join(lines)

    def _append_history(self, user_id: str, role: str, text: str):
        """记录对话历史，保留最近 N 轮"""
        if user_id not in self._chat_history:
            self._chat_history[user_id] = []
        self._chat_history[user_id].append((role, text))
        # 每轮 = 2条 (user + assistant)，保留最近 N 轮
        max_items = self._max_history_rounds * 2
        if len(self._chat_history[user_id]) > max_items:
            self._chat_history[user_id] = self._chat_history[user_id][-max_items:]

    def _load_cursor(self):
        """加载持久化的轮询游标"""
        if self._cursor_path.exists():
            try:
                data = json.loads(self._cursor_path.read_text())
                self._cursor = data.get("cursor", "")
                logger.info("已加载轮询游标")
            except Exception:
                self._cursor = ""

    def _save_cursor(self):
        """持久化轮询游标"""
        self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
        self._cursor_path.write_text(json.dumps({"cursor": self._cursor}))


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="WeChat ↔ A2A Expert Agent 桥接器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行前提:
  1. 启动 Expert Agent:
     python project/a2a_agent.py --serve

  2. 启动桥接器 (本脚本):
     python project/wechat_bridge.py

  3. 用微信扫 QR 码登录

微信配置要求:
  你的微信需要能使用 "爪机器人 (ClawBot)" 的 iLink Bot API。
  具体见文档: https://www.wechatbot.dev/en/protocol

示例:
  # 同步模式
  python project/wechat_bridge.py

  # 流式模式 (Expert 会 SSE 流式回复)
  python project/wechat_bridge.py --stream

  # 连接远程 Expert Agent
  python project/wechat_bridge.py --expert http://192.168.1.100:5001

  # 强制重新扫码
  python project/wechat_bridge.py --login
        """,
    )
    parser.add_argument("--expert", default="http://localhost:5001",
                        help="Expert Agent 的 URL (default: http://localhost:5001)")
    parser.add_argument("--stream", action="store_true",
                        help="使用 SSE 流式模式调用 Expert")
    parser.add_argument("--login", action="store_true",
                        help="强制重新 QR 扫码登录")
    parser.add_argument("--storage", default=str(STORAGE_DIR),
                        help=f"凭证和状态存储目录 (default: {STORAGE_DIR})")

    args = parser.parse_args()

    bridge = WeChatBridge(
        expert_url=args.expert,
        storage_dir=Path(args.storage),
        stream_mode=args.stream,
        force_login=args.login,
    )
    bridge.start()


if __name__ == "__main__":
    main()
