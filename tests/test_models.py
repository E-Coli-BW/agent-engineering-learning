"""
A2A 数据模型单元测试
=====================
测试 Part, Message, Task, AgentCard 的序列化和状态机
"""

from project.app.models import (
    Part, Message, TaskStatus, Task, TaskState,
    Skill, AgentCard,
)


class TestPart:
    def test_default_text_part(self):
        p = Part(text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_to_dict(self):
        p = Part(text="hello")
        d = p.to_dict()
        assert d == {"type": "text", "text": "hello"}

    def test_empty_text(self):
        p = Part()
        assert p.text == ""


class TestMessage:
    def test_user_message(self):
        m = Message(role="user", parts=[Part(text="question")])
        assert m.role == "user"
        assert len(m.parts) == 1

    def test_to_dict(self):
        m = Message(role="agent", parts=[Part(text="answer")])
        d = m.to_dict()
        assert d["role"] == "agent"
        assert d["parts"][0]["text"] == "answer"

    def test_mixed_parts(self):
        """parts 可以是 Part 对象或 dict"""
        m = Message(parts=[Part(text="a"), {"type": "text", "text": "b"}])
        d = m.to_dict()
        assert d["parts"][0]["text"] == "a"
        assert d["parts"][1]["text"] == "b"


class TestTaskStatus:
    def test_default_state(self):
        s = TaskStatus()
        assert s.state == TaskState.SUBMITTED

    def test_working_state(self):
        s = TaskStatus(state=TaskState.WORKING)
        d = s.to_dict()
        assert d["state"] == "working"

    def test_completed_with_message(self):
        msg = Message(role="agent", parts=[Part(text="done")]).to_dict()
        s = TaskStatus(state=TaskState.COMPLETED, message=msg)
        d = s.to_dict()
        assert d["state"] == "completed"
        assert d["message"]["role"] == "agent"

    def test_timestamp_present(self):
        s = TaskStatus()
        d = s.to_dict()
        assert "timestamp" in d


class TestTask:
    def test_auto_id(self):
        t = Task()
        assert t.id.startswith("task-")

    def test_custom_id(self):
        t = Task(id="task-custom123")
        assert t.id == "task-custom123"

    def test_to_dict_structure(self):
        t = Task(
            id="task-test",
            status=TaskStatus(state=TaskState.COMPLETED),
            history=[Message(role="user", parts=[Part(text="q")])],
            artifacts=[{"parts": [{"type": "text", "text": "a"}]}],
            metadata={"skill": "test"},
        )
        d = t.to_dict()
        assert d["id"] == "task-test"
        assert d["status"]["state"] == "completed"
        assert d["history"][0]["role"] == "user"
        assert d["artifacts"][0]["parts"][0]["text"] == "a"
        assert d["metadata"]["skill"] == "test"

    def test_state_transitions(self):
        """测试 Task 状态机流转"""
        t = Task()
        assert t.status.state == TaskState.SUBMITTED

        t.status = TaskStatus(state=TaskState.WORKING)
        assert t.status.state == TaskState.WORKING

        t.status = TaskStatus(state=TaskState.COMPLETED)
        assert t.status.state == TaskState.COMPLETED


class TestAgentCard:
    def test_basic_card(self):
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="http://localhost:5001",
        )
        d = card.to_dict()
        assert d["name"] == "Test Agent"
        assert d["version"] == "1.0.0"
        assert d["capabilities"]["streaming"] is True

    def test_with_skills(self):
        card = AgentCard(
            name="Expert",
            description="Expert Agent",
            url="http://localhost:5001",
            skills=[
                Skill(id="qa", name="问答", description="通用问答", tags=["general"]),
            ],
        )
        d = card.to_dict()
        assert len(d["skills"]) == 1
        assert d["skills"][0]["id"] == "qa"
        assert "general" in d["skills"][0]["tags"]
