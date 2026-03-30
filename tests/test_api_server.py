"""
RAG API Server 集成测试
========================
使用 httpx.AsyncClient + FastAPI TestClient
不依赖 Ollama（mock LLM 调用）
"""

import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

# 在 import api_server 之前 mock 掉 Ollama 依赖，避免启动时连接
with patch("project.api_server.OllamaEmbeddings"), \
     patch("project.api_server.ChatOllama"), \
     patch("project.api_server.Chroma"):
    from project.api_server import app


@pytest.fixture
def client():
    """同步测试客户端"""
    from starlette.testclient import TestClient
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "timestamp" in data
        assert "ollama" in data

    def test_health_has_vector_info(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "vector_store" in data
        assert "vector_count" in data


class TestMetricsEndpoint:
    def test_metrics_json(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_seconds" in data
        assert "total_requests" in data
        assert "p50_latency_ms" in data
        assert "p95_latency_ms" in data

    def test_metrics_prometheus_format(self, client):
        resp = client.get("/metrics", headers={"Accept": "text/plain"})
        assert resp.status_code == 200
        text = resp.text
        assert "rag_uptime_seconds" in text
        assert "rag_requests_total" in text
        assert "rag_vector_count" in text
        assert "quantile" in text


class TestQueryEndpoint:
    def test_query_validates_input(self, client):
        """空问题应返回 422"""
        resp = client.post("/query", json={
            "question": "",
        })
        assert resp.status_code == 422

    def test_query_requires_question(self, client):
        """缺少 question 字段应返回 422"""
        resp = client.post("/query", json={})
        assert resp.status_code == 422


class TestTimingHeaders:
    def test_response_has_timing_header(self, client):
        resp = client.get("/health")
        assert "x-response-time-ms" in resp.headers

    def test_response_has_request_id(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_request_id_propagated(self, client):
        """传入的 X-Request-Id 应该被返回"""
        resp = client.get("/health", headers={"X-Request-Id": "test-123"})
        assert resp.headers.get("x-request-id") == "test-123"
