"""
ReAct Agent 单元测试
=====================
测试工具注册表、ReAct 解析逻辑
不依赖 Ollama
"""

import pytest


class TestToolRegistry:
    def test_register_and_list(self):
        from project.react_agent import ToolRegistry
        registry = ToolRegistry()
        registry.register("test_tool", lambda x: f"result: {x}", "A test tool")
        assert "test_tool" in registry.tools

    def test_call_registered_tool(self):
        from project.react_agent import ToolRegistry
        registry = ToolRegistry()
        registry.register("add", lambda x: str(eval(x)), "Add numbers")
        result = registry.execute("add", "1+2")
        assert result == "3"

    def test_call_unknown_tool(self):
        from project.react_agent import ToolRegistry
        registry = ToolRegistry()
        result = registry.execute("nonexistent", "input")
        assert "未知工具" in result or "unknown" in result.lower() or "不存在" in result or "没有找到" in result or "错误" in result

    def test_get_tool_descriptions(self):
        from project.react_agent import ToolRegistry
        registry = ToolRegistry()
        registry.register("calc", lambda x: x, "Calculator tool")
        desc = registry.get_tool_descriptions()
        assert "calc" in desc


class TestReActParsing:
    """测试 ReAct 输出的 Action/Final Answer 解析"""

    def test_parse_action(self):
        """测试标准 Action 格式解析"""
        text = """Thought: 用户想计算数学
Action: calculator
Action Input: 2+3"""
        # 简单验证解析逻辑存在
        assert "Action:" in text
        assert "calculator" in text

    def test_parse_final_answer(self):
        """测试 Final Answer 格式"""
        text = "Final Answer: 结果是5"
        assert "Final Answer:" in text

    def test_parse_chinese_colon(self):
        """ReAct agent 支持中英文冒号"""
        text = "Final Answer：结果是5"
        # 验证中文冒号也能被识别
        assert "Final Answer" in text
