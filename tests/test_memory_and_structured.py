"""
聊天记忆 + 结构化输出测试
"""

import json


class TestMemoryChatMemory:
    def test_add_and_get(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        m.add("s1", "user", "hello")
        m.add("s1", "assistant", "hi")
        history = m.get_history("s1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "hi"

    def test_max_turns(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        # 添加超过 MAX_TURNS 的消息
        for i in range(50):
            m.add("s1", "user", f"q{i}")
            m.add("s1", "assistant", f"a{i}")
        history = m.get_history("s1")
        assert len(history) <= 40  # MAX_TURNS * 2

    def test_clear(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        m.add("s1", "user", "hello")
        m.clear("s1")
        assert m.get_history("s1") == []

    def test_separate_sessions(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        m.add("s1", "user", "hello")
        m.add("s2", "user", "world")
        assert len(m.get_history("s1")) == 1
        assert len(m.get_history("s2")) == 1

    def test_list_sessions(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        m.add("s1", "user", "a")
        m.add("s2", "user", "b")
        sessions = m.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

    def test_get_with_max_turns(self):
        from project.infra.memory import MemoryChatMemory
        m = MemoryChatMemory()
        for i in range(10):
            m.add("s1", "user", f"q{i}")
            m.add("s1", "assistant", f"a{i}")
        # 只取最后 3 轮
        history = m.get_history("s1", max_turns=3)
        assert len(history) == 6  # 3 轮 × 2


class TestStructuredOutput:
    def test_extract_json_plain(self):
        from project.infra.structured import extract_json
        result = extract_json('{"key": "value"}')
        assert '"key"' in result

    def test_extract_json_code_block(self):
        from project.infra.structured import extract_json
        text = 'Here is the plan:\n```json\n{"plan": "test"}\n```\nDone.'
        result = extract_json(text)
        assert '"plan"' in result

    def test_extract_json_surrounded(self):
        from project.infra.structured import extract_json
        text = 'I think we should do this:\n{"plan": "analyze"}\nThat is my plan.'
        result = extract_json(text)
        assert "analyze" in result

    def test_parse_plan_output(self):
        from project.infra.structured import parse_llm_json, PlanOutput
        text = '{"plan": "test plan", "agents": [{"name": "knowledge", "query": "what is LoRA"}]}'
        plan = parse_llm_json(text, PlanOutput)
        assert plan.plan == "test plan"
        assert len(plan.agents) == 1
        assert plan.agents[0].name == "knowledge"

    def test_parse_with_code_block(self):
        from project.infra.structured import parse_llm_json, PlanOutput
        text = '```json\n{"plan": "compute", "agents": [{"name": "calculator", "query": "2+2"}]}\n```'
        plan = parse_llm_json(text, PlanOutput)
        assert plan.agents[0].name == "calculator"

    def test_parse_fallback(self):
        from project.infra.structured import parse_llm_json, PlanOutput
        text = 'this is not json at all'
        fallback = {"plan": "fallback", "agents": []}
        plan = parse_llm_json(text, PlanOutput, fallback=fallback)
        assert plan.plan == "fallback"

    def test_eval_report_model(self):
        from project.infra.structured import EvalReport, EvalScore
        report = EvalReport(
            total=2, passed=1, failed=1, avg_score=0.75,
            details=[
                EvalScore(question="q1", score=1.0, passed=True),
                EvalScore(question="q2", score=0.5, passed=False),
            ],
        )
        d = report.model_dump()
        assert d["total"] == 2
        assert d["avg_score"] == 0.75
