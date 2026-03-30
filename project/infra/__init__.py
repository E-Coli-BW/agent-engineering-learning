"""
Task Store — 统一的 Task 持久化层
==================================

支持两种后端:
  - MemoryStore: 内存字典 (开发/测试, 默认)
  - RedisStore:  Redis (生产, 通过环境变量 REDIS_URL 启用)

用法:
  from project.infra.task_store import get_task_store
  store = get_task_store()
  store.save(task)
  task = store.get("task-xxx")
  store.delete("task-xxx")

设计原则:
  - 接口统一，调用方无需关心底层存储
  - 零配置启动: 没有 Redis 就用内存，有就自动切换
  - Task 序列化为 JSON，Redis TTL 默认 24h
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional
from datetime import timedelta

logger = logging.getLogger("infra.task_store")


class TaskStore(ABC):
    """Task 存储抽象接口"""

    @abstractmethod
    def save(self, task_id: str, task_data: dict) -> None:
        """保存 Task (upsert)"""
        ...

    @abstractmethod
    def get(self, task_id: str) -> Optional[dict]:
        """获取 Task, 不存在返回 None"""
        ...

    @abstractmethod
    def delete(self, task_id: str) -> bool:
        """删除 Task, 返回是否存在"""
        ...

    @abstractmethod
    def list_all(self) -> list[str]:
        """列出所有 Task ID"""
        ...


class MemoryStore(TaskStore):
    """内存 Task Store — 开发/测试用"""

    def __init__(self):
        self._store: dict[str, dict] = {}
        logger.info("TaskStore: 使用内存存储 (重启丢失)")

    def save(self, task_id: str, task_data: dict) -> None:
        self._store[task_id] = task_data

    def get(self, task_id: str) -> Optional[dict]:
        return self._store.get(task_id)

    def delete(self, task_id: str) -> bool:
        return self._store.pop(task_id, None) is not None

    def list_all(self) -> list[str]:
        return list(self._store.keys())


class RedisStore(TaskStore):
    """Redis Task Store — 生产用, 持久化 + TTL"""

    def __init__(self, redis_url: str, ttl_hours: int = 24, prefix: str = "task:"):
        import redis
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._ttl = timedelta(hours=ttl_hours)
        self._prefix = prefix
        # 验证连接
        self._client.ping()
        logger.info("TaskStore: 使用 Redis (%s), TTL=%dh", redis_url, ttl_hours)

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    def save(self, task_id: str, task_data: dict) -> None:
        self._client.setex(self._key(task_id), self._ttl, json.dumps(task_data, ensure_ascii=False))

    def get(self, task_id: str) -> Optional[dict]:
        data = self._client.get(self._key(task_id))
        return json.loads(data) if data else None

    def delete(self, task_id: str) -> bool:
        return self._client.delete(self._key(task_id)) > 0

    def list_all(self) -> list[str]:
        keys = self._client.keys(f"{self._prefix}*")
        return [k.removeprefix(self._prefix) for k in keys]


# ---- 工厂函数 ----

_store_instance: Optional[TaskStore] = None


def get_task_store() -> TaskStore:
    """
    获取 TaskStore 单例。

    优先级:
      1. REDIS_URL 环境变量存在 → RedisStore
      2. 否则 → MemoryStore
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            _store_instance = RedisStore(redis_url)
        except Exception as e:
            logger.warning("Redis 连接失败 (%s), 降级为内存存储: %s", redis_url, e)
            _store_instance = MemoryStore()
    else:
        _store_instance = MemoryStore()

    return _store_instance
