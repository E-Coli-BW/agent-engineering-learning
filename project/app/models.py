"""
A2A 协议数据模型
================

对齐 Google A2A 规范 (https://google.github.io/A2A/)

所有 A2A 协议的数据结构集中在此：
  - TaskState   任务状态枚举
  - Part        消息最小单元 (TextPart)
  - Message     对话消息 (role + parts)
  - TaskStatus  任务状态快照
  - Task        核心工作单元
  - Skill       Agent 能力声明
  - AgentCard   Agent 发现卡片

与学习版 a2a_agent.py 的区别:
  - 学习版把模型、服务器、客户端都堆在一个文件
  - 生产版拆到独立模块，可复用、可测试
"""

import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
from datetime import datetime, timezone


class TaskState(str, Enum):
    """
    Task 状态机 (A2A 规范):
      submitted → working → completed
                         ↘ failed
                         ↘ input-required
                         ↘ canceled
    """
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    INPUT_REQUIRED = "input-required"
    CANCELED = "canceled"


@dataclass
class Part:
    """A2A Message Part — 消息最小单元 (目前只用 TextPart)"""
    type: str = "text"
    text: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "text": self.text}


@dataclass
class Message:
    """A2A Message — Task 中的对话消息"""
    role: str = "user"  # "user" | "agent"
    parts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() if isinstance(p, Part) else p for p in self.parts],
        }


@dataclass
class TaskStatus:
    """A2A Task Status — 当前状态快照"""
    state: str = TaskState.SUBMITTED
    message: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = {"state": self.state, "timestamp": self.timestamp}
        if self.message:
            d["message"] = self.message
        return d


@dataclass
class Task:
    """
    A2A Task — 协议的核心工作单元

    {
      "id": "task-abc123",
      "status": { "state": "completed", "message": {...} },
      "history": [ ... ],
      "artifacts": [ ... ]
    }
    """
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    status: TaskStatus = field(default_factory=TaskStatus)
    history: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.to_dict(),
            "history": [m.to_dict() if isinstance(m, Message) else m for m in self.history],
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass
class Skill:
    """A2A Agent Skill — 能力声明"""
    id: str
    name: str
    description: str
    tags: list = field(default_factory=list)
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentCard:
    """
    A2A Agent Card — GET /.well-known/agent.json

    Agent 用此 JSON 声明自己的能力、技能、端点。
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list = field(default_factory=list)
    capabilities: dict = field(default_factory=lambda: {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    })
    defaultInputModes: list = field(default_factory=lambda: ["text"])
    defaultOutputModes: list = field(default_factory=lambda: ["text"])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": self.capabilities,
            "skills": [s.to_dict() if isinstance(s, Skill) else s for s in self.skills],
            "defaultInputModes": self.defaultInputModes,
            "defaultOutputModes": self.defaultOutputModes,
        }
