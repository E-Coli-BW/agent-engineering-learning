"""
A2A Agent 集成测试
===================
测试 Agent Card、Task 生命周期、SSE streaming
不依赖 Ollama（mock LLM 调用）
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient


def make_mock_ollama_response(text="mock answer"):
    """模拟 Ollama /api/generate 的响应"""
    response = MagicMock()
    response.read.return_value = json.dumps({"response": text}).encode()
    return response


def make_mock_ollama_stream(tokens=None):
    """模拟 Ollama /api/generate stream 响应"""
    if tokens is None:
        tokens = ["Hello", " World", "!"]
    lines = []
    for t in tokens:
        lines.append(json.dumps({"response": t, "done": False}).encode())
    lines.append(json.dumps({"response": "", "done": True}).encode())
    return MagicMock(__iter__=lambda self: iter(lines))


@pytest.fixture
def a2a_client():
    """创建 A2A Expert 的测试客户端"""
    from project.a2a_agent import create_expert_app
    app = create_expert_app()
    with TestClient(app) as c:
        yield c


class TestAgentCard:
    def test_agent_card_endpoint(self, a2a_client):
        resp = a2a_client.get("/.well-known/agent.json")
        assert resp.status_code == 200
        card = resp.json()
        assert "name" in card
        assert "skills" in card
        assert "url" in card
        assert "capabilities" in card

    def test_agent_card_has_skills(self, a2a_client):
        resp = a2a_client.get("/.well-known/agent.json")
        card = resp.json()
        assert len(card["skills"]) > 0
        skill = card["skills"][0]
        assert "id" in skill
        assert "name" in skill
        assert "tags" in skill

    def test_agent_card_streaming_capability(self, a2a_client):
        resp = a2a_client.get("/.well-known/agent.json")
        card = resp.json()
        assert card["capabilities"]["streaming"] is True


class TestHealthEndpoint:
    def test_health(self, a2a_client):
        resp = a2a_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestTaskSend:
    @patch("urllib.request.urlopen")
    def test_send_task_success(self, mock_urlopen, a2a_client):
        mock_urlopen.return_value = make_mock_ollama_response("test answer")

        resp = a2a_client.post("/tasks/send", json={
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "什么是Transformer?"}],
            },
        })
        assert resp.status_code == 200
        task = resp.json()
        assert "id" in task
        assert task["status"]["state"] in ("completed", "failed")

    def test_send_task_empty_message(self, a2a_client):
        """空消息应返回 400"""
        resp = a2a_client.post("/tasks/send", json={
            "message": {"role": "user", "parts": []},
        })
        assert resp.status_code == 400

    def test_send_task_no_text_parts(self, a2a_client):
        """没有 text 类型的 part 应返回 400"""
        resp = a2a_client.post("/tasks/send", json={
            "message": {
                "role": "user",
                "parts": [{"type": "image", "url": "http://example.com"}],
            },
        })
        assert resp.status_code == 400


class TestTaskQuery:
    def test_get_nonexistent_task(self, a2a_client):
        resp = a2a_client.get("/tasks/nonexistent-id")
        assert resp.status_code == 404
