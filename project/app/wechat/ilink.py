"""
iLink 协议层
=============

封装微信 iLink Bot API 的底层通信:
  - Credentials       凭证管理 (保存/加载)
  - _ilink_request    POST 请求
  - _ilink_get        GET 请求
  - qr_login          QR 码扫码登录
  - poll_messages     长轮询收消息
  - send_text         发送文本消息
  - send_typing       发送 typing 指示器
  - extract_user_message  从原始消息提取用户文本

与学习版 wechat_bridge.py 的区别:
  - 学习版把 iLink 协议、ExpertClient、WeChatBridge、CLI 全部堆在一个 935 行文件
  - 生产版把纯协议层拆到 ilink.py，桥接逻辑拆到 bridge.py
"""

import os
import json
import time
import base64
import struct
import logging
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("app.wechat.ilink")

# iLink 协议常量
ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
ILINK_API_PREFIX = "ilink/bot"
CHANNEL_VERSION = "2.0.0"


# ============================================================
# 凭证存储
# ============================================================

@dataclass
class Credentials:
    """iLink 登录凭证"""
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


class SessionExpiredError(Exception):
    """iLink session 过期"""
    pass


# ============================================================
# HTTP 工具
# ============================================================

def _make_uin_header() -> str:
    """生成 X-WECHAT-UIN: base64(str(random_uint32))"""
    random_bytes = os.urandom(4)
    uint32 = struct.unpack("<I", random_bytes)[0]
    return base64.b64encode(str(uint32).encode()).decode()


def _ilink_request(
    base_url: str,
    endpoint: str,
    token: str = "",
    body: dict = None,
    method: str = "POST",
    timeout: int = 40,
) -> dict:
    """发送 iLink POST 请求"""
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
    """发送 iLink GET 请求"""
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
      1. GET /get_bot_qrcode?bot_type=3 → QR 码 URL
      2. 轮询 GET /get_qrcode_status    → wait → scaned → confirmed
      3. confirmed 返回 bot_token
    """
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

    try:
        import qrcode as qr_lib
        qr = qr_lib.QRCode(border=1)
        qr.add_data(qr_img_url)
        qr.print_ascii(invert=True)
    except ImportError:
        print("  (提示: pip install qrcode 可在终端直接显示 QR 码图案)")
        print(f"  请在浏览器打开上面的 URL 扫码")
    print()

    print("⏳ 等待扫码...")
    while True:
        status_data = _ilink_get(
            base_url,
            f"get_qrcode_status?qrcode={urllib.parse.quote(qrcode)}",
            timeout=60,
        )
        status = status_data.get("status", "")

        if status == "wait":
            pass
        elif status == "scaned":
            print("  📲 已扫码，请在微信上确认登录...")
        elif status == "confirmed":
            print("  ✅ 登录成功!")
            return Credentials(
                bot_token=status_data.get("bot_token", ""),
                base_url=status_data.get("baseurl", base_url),
                ilink_bot_id=status_data.get("ilink_bot_id", ""),
                ilink_user_id=status_data.get("ilink_user_id", ""),
                saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
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
    """
    body = {"get_updates_buf": cursor}
    result = _ilink_request(cred.base_url, "getupdates", cred.bot_token, body, timeout=40)

    ret = result.get("ret", -1)
    errcode = result.get("errcode")

    if ret != 0 or errcode:
        logger.info("getupdates 完整响应: %s", json.dumps(result, ensure_ascii=False)[:500])

    if ret == -14 or errcode == -14:
        raise SessionExpiredError("Session 过期 (errcode: -14), 需要重新登录")
    if ret != 0 and ret != -1:
        logger.warning("getupdates 返回 ret=%s: %s", ret, result.get("errmsg", ""))
        return [], cursor

    msgs = result.get("msgs", [])
    new_cursor = result.get("get_updates_buf", cursor)
    return msgs, new_cursor


def send_text(cred: Credentials, to_user_id: str, context_token: str, text: str):
    """POST /sendmessage — 发送文本消息 (自动分段)"""
    import uuid as _uuid
    segments = _split_text(text, max_len=2000)
    logger.info("sendmessage: %d 段, 总长 %d", len(segments), len(text))

    for i, segment in enumerate(segments):
        client_id = f"wechat-bridge-{_uuid.uuid4().hex[:16]}"
        body = {
            "msg": {
                "from_user_id": "",
                "to_user_id": to_user_id,
                "client_id": client_id,
                "context_token": context_token,
                "message_type": 2,
                "message_state": 2,
                "item_list": [{"type": 1, "text_item": {"text": segment}}],
            }
        }
        resp = _ilink_request(cred.base_url, "sendmessage", cred.bot_token, body)
        logger.info("sendmessage [%d/%d] 响应: %s | len=%d",
                     i + 1, len(segments),
                     json.dumps(resp, ensure_ascii=False)[:200], len(segment))
        if i < len(segments) - 1:
            time.sleep(1.0)


def send_typing(cred: Credentials, user_id: str, context_token: str, status: int = 1):
    """显示 "对方正在输入中..." 指示器"""
    try:
        config_body = {
            "ilink_user_id": user_id,
            "context_token": context_token,
        }
        config_resp = _ilink_request(cred.base_url, "getconfig", cred.bot_token, config_body)
        ticket = config_resp.get("typing_ticket", "")
        if not ticket:
            return
        typing_body = {
            "ilink_user_id": user_id,
            "typing_ticket": ticket,
            "status": status,
        }
        _ilink_request(cred.base_url, "sendtyping", cred.bot_token, typing_body)
    except Exception as e:
        logger.debug("发送 typing 失败 (非关键): %s", e)


# ============================================================
# 工具函数
# ============================================================

def _split_text(text: str, max_len: int = 4000) -> list[str]:
    """按自然边界分割文本"""
    if len(text) <= max_len:
        return [text]

    segments = []
    while text:
        if len(text) <= max_len:
            segments.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut <= 0:
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

    过滤逻辑:
      - 排除 message_type=2 (Bot 自己的回复，防止无限循环)
      - 排除发送者是 Bot 自身
      - 排除群消息
    """
    user_id = msg.get("from_user_id")
    context_token = msg.get("context_token")

    msg_type = msg.get("message_type")
    if msg_type == 2:
        return None, None, None

    if bot_user_id and user_id == bot_user_id:
        return None, None, None

    if msg.get("group_id"):
        return None, None, None

    if not user_id or not context_token:
        return None, None, None

    text_parts = []
    for item in msg.get("item_list", []):
        item_type = item.get("type", 0)
        if item_type == 1:
            t = item.get("text_item", {}).get("text", "")
            if t:
                text_parts.append(t)
        elif item_type == 3:
            t = item.get("voice_item", {}).get("text", "")
            if t:
                text_parts.append(f"[语音] {t}")

    text = " ".join(text_parts) if text_parts else None
    return user_id, context_token, text
