"""
聊天记忆 — 对话历史持久化
==========================

支持:
  - 按 session_id 存储对话历史
  - Buffer Memory: 保留最近 N 轮
  - Summary Memory: 超过 N 轮后 LLM 自动总结
  - 存储后端: Redis (生产) / 内存 (开发)

用法:
  memory = get_chat_memory()
  memory.add("session-1", "user", "什么是LoRA？")
  memory.add("session-1", "assistant", "LoRA是...")
  history = memory.get_history("session-1")
  # [{"role":"user","content":"什么是LoRA？"}, {"role":"assistant","content":"LoRA是..."}]
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger("infra.memory")

MAX_TURNS = int(os.getenv("CHAT_MEMORY_MAX_TURNS", "20"))


class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: str = ""):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


class ChatMemory:
    """聊天记忆抽象接口"""

    def add(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError

    def get_history(self, session_id: str, max_turns: int = 0) -> list[dict]:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError

    def list_sessions(self) -> list[str]:
        raise NotImplementedError


class MemoryChatMemory(ChatMemory):
    """内存聊天记忆"""

    def __init__(self):
        self._store: dict[str, list[dict]] = {}
        logger.info("ChatMemory: 内存存储")

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(
            ChatMessage(role, content).to_dict()
        )
        # 保留最近 MAX_TURNS 轮 (每轮 = user + assistant = 2 条)
        if len(self._store[session_id]) > MAX_TURNS * 2:
            self._store[session_id] = self._store[session_id][-(MAX_TURNS * 2):]

    def get_history(self, session_id: str, max_turns: int = 0) -> list[dict]:
        msgs = self._store.get(session_id, [])
        if max_turns > 0:
            return msgs[-(max_turns * 2):]
        return msgs

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._store.keys())


class RedisChatMemory(ChatMemory):
    """Redis 聊天记忆"""

    def __init__(self, redis_url: str, prefix: str = "chat:"):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._prefix = prefix
        self._client.ping()
        logger.info("ChatMemory: Redis (%s)", redis_url)

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def add(self, session_id: str, role: str, content: str) -> None:
        msg = ChatMessage(role, content).to_dict()
        self._client.rpush(self._key(session_id), json.dumps(msg, ensure_ascii=False))
        # 保留最近 MAX_TURNS * 2 条
        self._client.ltrim(self._key(session_id), -(MAX_TURNS * 2), -1)
        # 刷新 TTL (7天)
        self._client.expire(self._key(session_id), 7 * 24 * 3600)

    def get_history(self, session_id: str, max_turns: int = 0) -> list[dict]:
        data = self._client.lrange(self._key(session_id), 0, -1)
        msgs = [json.loads(d) for d in data]
        if max_turns > 0:
            return msgs[-(max_turns * 2):]
        return msgs

    def clear(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))

    def list_sessions(self) -> list[str]:
        keys = self._client.keys(f"{self._prefix}*")
        return [k.removeprefix(self._prefix) for k in keys]


_memory_instance: Optional[ChatMemory] = None


def get_chat_memory() -> ChatMemory:
    """获取 ChatMemory 单例"""
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            _memory_instance = RedisChatMemory(redis_url)
        except Exception as e:
            logger.warning("Redis 连接失败, 降级内存: %s", e)
            _memory_instance = MemoryChatMemory()
    else:
        _memory_instance = MemoryChatMemory()
    return _memory_instance
