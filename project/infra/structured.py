"""
结构化输出 — Pydantic 模型定义 + LLM 输出校验
================================================

用 Pydantic 严格校验 LLM 返回的 JSON，
替代手动 json.loads + dict.get 的脆弱解析。

用法:
  from project.infra.structured import parse_llm_json, PlanOutput
  plan = parse_llm_json(llm_response, PlanOutput)
  # plan.plan → str
  # plan.agents → [AgentCall(name="knowledge", query="...")]
"""

import json
import re
import logging
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("infra.structured")


# ============================================================
# Orchestrator 结构化输出模型
# ============================================================

class AgentCall(BaseModel):
    """单个 Agent 调用指令"""
    name: str = Field(description="Agent 名称: knowledge / calculator / code")
    query: str = Field(description="给该 Agent 的具体问题")


class PlanOutput(BaseModel):
    """Orchestrator Plan 的结构化输出"""
    plan: str = Field(description="简要执行计划")
    agents: list[AgentCall] = Field(default_factory=list, description="需要调用的 Agent 列表")


class EvalScore(BaseModel):
    """单个评估结果"""
    question: str
    expected_keywords: list[str] = []
    actual_answer: str = ""
    keyword_hits: int = 0
    keyword_total: int = 0
    score: float = 0.0
    passed: bool = False


class EvalReport(BaseModel):
    """评估报告"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    details: list[EvalScore] = []


# ============================================================
# LLM JSON 解析工具
# ============================================================

def extract_json(text: str) -> str:
    """
    从 LLM 回复中提取 JSON 字符串。

    LLM 可能返回:
      - 纯 JSON
      - ```json ... ```  包裹的 JSON
      - 前后有解释文字的 JSON
    """
    # 尝试 markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 尝试找 { ... } 或 [ ... ]
    for start, end in [('{', '}'), ('[', ']')]:
        idx_start = text.find(start)
        idx_end = text.rfind(end)
        if idx_start != -1 and idx_end > idx_start:
            return text[idx_start:idx_end + 1]

    return text.strip()


def parse_llm_json(text: str, model_class: type[BaseModel], fallback: Optional[dict] = None):
    """
    解析 LLM 输出为 Pydantic 模型。

    流程:
      1. 提取 JSON 字符串
      2. json.loads 解析
      3. Pydantic 校验
      4. 失败则使用 fallback

    Returns:
        model_class 实例
    """
    try:
        json_str = extract_json(text)
        data = json.loads(json_str)
        return model_class.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("LLM JSON 解析失败: %s — 原文: %s", e, text[:200])
        if fallback:
            return model_class.model_validate(fallback)
        raise
