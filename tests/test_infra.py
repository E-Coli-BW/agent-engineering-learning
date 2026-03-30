"""
基础设施模块测试
=================
TaskStore, 错误模型, 结构化日志
"""

import json
import logging
from io import StringIO


class TestMemoryStore:
    def test_save_and_get(self):
        from project.infra import MemoryStore
        store = MemoryStore()
        store.save("t1", {"id": "t1", "status": "completed"})
        assert store.get("t1")["status"] == "completed"

    def test_get_nonexistent(self):
        from project.infra import MemoryStore
        store = MemoryStore()
        assert store.get("nope") is None

    def test_delete(self):
        from project.infra import MemoryStore
        store = MemoryStore()
        store.save("t1", {"id": "t1"})
        assert store.delete("t1") is True
        assert store.get("t1") is None
        assert store.delete("t1") is False

    def test_list_all(self):
        from project.infra import MemoryStore
        store = MemoryStore()
        store.save("t1", {})
        store.save("t2", {})
        ids = store.list_all()
        assert "t1" in ids
        assert "t2" in ids

    def test_upsert(self):
        from project.infra import MemoryStore
        store = MemoryStore()
        store.save("t1", {"v": 1})
        store.save("t1", {"v": 2})
        assert store.get("t1")["v"] == 2


class TestGetTaskStore:
    def test_default_is_memory(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        # 重置单例
        import project.infra as infra
        infra._store_instance = None
        store = infra.get_task_store()
        assert isinstance(store, infra.MemoryStore)

    def test_invalid_redis_falls_back(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://invalid-host:9999")
        import project.infra as infra
        infra._store_instance = None
        store = infra.get_task_store()
        # 应降级为 MemoryStore
        assert isinstance(store, infra.MemoryStore)
        # 清理
        infra._store_instance = None
        monkeypatch.delenv("REDIS_URL")


class TestAppError:
    def test_error_response_format(self):
        from project.infra.errors import error_response
        resp = error_response("TEST_ERROR", "test message", 400, "req-123")
        body = json.loads(resp.body)
        assert body["error"]["code"] == "TEST_ERROR"
        assert body["error"]["message"] == "test message"
        assert body["error"]["request_id"] == "req-123"
        assert resp.status_code == 400

    def test_app_error_exception(self):
        from project.infra.errors import AppError
        err = AppError("NOT_FOUND", "找不到", 404)
        assert err.code == "NOT_FOUND"
        assert err.status_code == 404
        assert str(err) == "找不到"


class TestStructuredLogging:
    def test_json_formatter(self):
        from project.infra.logging import JsonFormatter
        formatter = JsonFormatter(service="test-svc")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello world", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["service"] == "test-svc"
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_json_formatter_with_extra(self):
        from project.infra.logging import JsonFormatter
        formatter = JsonFormatter(service="test")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="query done", args=(), exc_info=None,
        )
        record.request_id = "abc123"
        record.latency_ms = 42
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "abc123"
        assert parsed["latency_ms"] == 42

    def test_setup_logging_text_mode(self, monkeypatch):
        monkeypatch.setenv("LOG_FORMAT", "text")
        from project.infra.logging import setup_logging
        setup_logging("test")
        root = logging.getLogger()
        assert len(root.handlers) > 0
