"""
Coordinator Agent — A2A HTTP 客户端
====================================

负责:
  1. GET /.well-known/agent.json   发现 Expert
  2. 根据 Agent Card 的 skills     匹配技能
  3. POST /tasks/send              同步调用
  4. POST /tasks/sendSubscribe     流式调用 (SSE)

与学习版 a2a_agent.py 中 CoordinatorAgent 的区别:
  - 学习版和 server 代码混在同一个文件
  - 生产版独立模块，可单独 import
"""

import json
import uuid
import logging
import urllib.request
from typing import Optional, Generator

logger = logging.getLogger("app.coordinator")


class CoordinatorAgent:
    """
    协调者 Agent — 通过 HTTP 调用 Expert Agent

    对齐 A2A 规范的客户端行为:
      1. GET /.well-known/agent.json  发现 Agent
      2. 根据 Agent Card 的 skills  匹配
      3. POST /tasks/send 或 /tasks/sendSubscribe
    """

    def __init__(self, expert_url: str = "http://localhost:5001"):
        self.expert_url = expert_url.rstrip("/")
        self.agent_card: Optional[dict] = None

    def discover(self) -> dict:
        """A2A 发现: GET /.well-known/agent.json"""
        url = f"{self.expert_url}/.well-known/agent.json"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                self.agent_card = json.loads(resp.read().decode("utf-8"))
                return self.agent_card
        except Exception as e:
            raise RuntimeError(f"无法连接 Expert Agent ({url}): {e}")

    def match_skill(self, question: str) -> str:
        """根据 Agent Card 的 skills + tags 匹配最佳 skill"""
        if not self.agent_card:
            return "general_qa"

        q = question.lower()
        best_skill, best_score = "general_qa", 0

        for skill in self.agent_card.get("skills", []):
            score = sum(2 for tag in skill.get("tags", []) if tag.lower() in q)
            if skill.get("name", "").lower() in q:
                score += 1
            if score > best_score:
                best_score = score
                best_skill = skill.get("id", "general_qa")

        # fallback 关键词路由
        if best_score == 0:
            from .expert_server import route_skill
            return route_skill(question)
        return best_skill

    def send_task(self, question: str, skill: str = None) -> dict:
        """POST /tasks/send — 同步"""
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/send",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def send_subscribe(self, question: str, skill: str = None) -> Generator[tuple[str, dict], None, None]:
        """
        POST /tasks/sendSubscribe — 流式

        Yields (event_type, data) 元组
        """
        skill_id = skill or self.match_skill(question)
        payload = {
            "id": f"task-{uuid.uuid4().hex[:8]}",
            "message": {"role": "user", "parts": [{"type": "text", "text": question}]},
            "metadata": {"skill": skill_id},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.expert_url}/tasks/sendSubscribe",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
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
                        yield (event_type, json.loads("".join(data_lines)))
                    event_type = None
                    data_lines = []
