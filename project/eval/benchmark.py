"""
模型评估 — RAG + Agent 质量 Benchmark
=======================================

评估维度:
  1. RAG 检索: 关键词覆盖率 (keywords in answer?)
  2. Agent 路由: Orchestrator 选对了 Agent 吗?
  3. 计算准确性: Calculator Agent 结果正确吗?
  4. 综合评分: 关键词覆盖 + 结构完整性

运行:
  PYTHONPATH=. python project/eval/benchmark.py
  PYTHONPATH=. python project/eval/benchmark.py --rag-only
  PYTHONPATH=. python project/eval/benchmark.py --orchestrator-only
"""

import os
import sys
import json
import time
import logging
import urllib.request
from pathlib import Path

# 添加项目根到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project.infra.structured import EvalScore, EvalReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("eval")

# ============================================================
# 评估数据集
# ============================================================

RAG_EVAL_SET = [
    {
        "question": "什么是 LoRA?",
        "expected_keywords": ["低秩", "微调", "参数"],
        "category": "knowledge",
    },
    {
        "question": "Self-Attention 为什么要除以 sqrt(d_k)?",
        "expected_keywords": ["方差", "softmax", "梯度", "缩放"],
        "category": "knowledge",
    },
    {
        "question": "RAG 系统的核心流程是什么?",
        "expected_keywords": ["检索", "向量", "生成", "Embedding"],
        "category": "knowledge",
    },
    {
        "question": "ReAct Agent 和普通 LLM 有什么区别?",
        "expected_keywords": ["工具", "思考", "行动", "观察"],
        "category": "knowledge",
    },
    {
        "question": "KV Cache 如何加速推理?",
        "expected_keywords": ["缓存", "Key", "Value", "重复计算"],
        "category": "knowledge",
    },
]

ORCHESTRATOR_EVAL_SET = [
    {
        "question": "LoRA rank=8 d_model=2048 参数量减少多少?",
        "expected_agents": ["calculator"],
        "expected_keywords": ["2048", "参数", "减少"],
        "category": "calculator",
    },
    {
        "question": "写一个 Python 的 softmax 函数",
        "expected_agents": ["code"],
        "expected_keywords": ["def", "softmax", "exp"],
        "category": "code",
    },
    {
        "question": "Transformer 的 Multi-Head Attention 原理是什么?",
        "expected_agents": ["knowledge"],
        "expected_keywords": ["多头", "注意力", "Q", "K", "V"],
        "category": "knowledge",
    },
    {
        "question": "帮我分析 LoRA 和 QLoRA 的区别，并计算 rank=16 时参数减少了多少",
        "expected_agents": ["knowledge", "calculator"],
        "expected_keywords": ["LoRA", "QLoRA", "量化", "参数", "减少"],
        "category": "multi-agent",
    },
]


# ============================================================
# 评估函数
# ============================================================

def call_rag(question: str, base_url: str = "http://localhost:8000") -> str:
    """调用 RAG API"""
    data = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{base_url}/query",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result.get("answer", "")
    except Exception as e:
        return f"ERROR: {e}"


def call_orchestrator(question: str, base_url: str = "http://localhost:5003") -> dict:
    """调用 Orchestrator"""
    data = json.dumps({"query": question}).encode()
    req = urllib.request.Request(
        f"{base_url}/orchestrate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"answer": f"ERROR: {e}", "agents_used": []}


def eval_keywords(answer: str, keywords: list[str]) -> tuple[int, int, float]:
    """关键词覆盖率评估"""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    total = len(keywords)
    score = hits / total if total > 0 else 0
    return hits, total, score


def eval_agent_routing(actual_agents: list[str], expected_agents: list[str]) -> bool:
    """Agent 路由准确性"""
    return set(expected_agents).issubset(set(actual_agents))


# ============================================================
# 评估流程
# ============================================================

def run_rag_eval() -> EvalReport:
    """评估 RAG 检索 + 回答质量"""
    print("\n" + "=" * 60)
    print("📚 RAG 评估")
    print("=" * 60)

    details = []
    for item in RAG_EVAL_SET:
        q = item["question"]
        print(f"\n  ❓ {q}")

        start = time.time()
        answer = call_rag(q)
        elapsed = time.time() - start

        hits, total, score = eval_keywords(answer, item["expected_keywords"])
        passed = score >= 0.5  # 覆盖一半以上算通过

        print(f"  ⏱️  {elapsed:.1f}s | 关键词: {hits}/{total} ({score:.0%}) | {'✅' if passed else '❌'}")

        details.append(EvalScore(
            question=q,
            expected_keywords=item["expected_keywords"],
            actual_answer=answer[:200],
            keyword_hits=hits,
            keyword_total=total,
            score=score,
            passed=passed,
        ))

    total = len(details)
    passed = sum(1 for d in details if d.passed)
    avg = sum(d.score for d in details) / total if total else 0

    report = EvalReport(total=total, passed=passed, failed=total - passed, avg_score=avg, details=details)
    print(f"\n  📊 RAG 总计: {passed}/{total} 通过, 平均覆盖率 {avg:.0%}")
    return report


def run_orchestrator_eval() -> EvalReport:
    """评估 Orchestrator 路由 + 回答质量"""
    print("\n" + "=" * 60)
    print("🎯 Orchestrator 评估")
    print("=" * 60)

    details = []
    for item in ORCHESTRATOR_EVAL_SET:
        q = item["question"]
        print(f"\n  ❓ {q}")

        start = time.time()
        result = call_orchestrator(q)
        elapsed = time.time() - start

        answer = result.get("answer", "")
        agents_used = result.get("agents_used", [])

        # 关键词评估
        hits, total, kw_score = eval_keywords(answer, item["expected_keywords"])

        # 路由评估
        route_ok = eval_agent_routing(agents_used, item.get("expected_agents", []))

        # 综合评分 (关键词 70% + 路由 30%)
        score = kw_score * 0.7 + (1.0 if route_ok else 0.0) * 0.3
        passed = score >= 0.5

        route_icon = "✅" if route_ok else "❌"
        print(f"  ⏱️  {elapsed:.1f}s | Agents: {agents_used} {route_icon}")
        print(f"  📝 关键词: {hits}/{total} ({kw_score:.0%}) | 综合: {score:.0%} | {'✅' if passed else '❌'}")

        details.append(EvalScore(
            question=q,
            expected_keywords=item["expected_keywords"],
            actual_answer=answer[:200],
            keyword_hits=hits,
            keyword_total=total,
            score=score,
            passed=passed,
        ))

    total = len(details)
    passed = sum(1 for d in details if d.passed)
    avg = sum(d.score for d in details) / total if total else 0

    report = EvalReport(total=total, passed=passed, failed=total - passed, avg_score=avg, details=details)
    print(f"\n  📊 Orchestrator 总计: {passed}/{total} 通过, 平均分 {avg:.0%}")
    return report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print("🧪 Agent Learning — 模型评估 Benchmark")
    print(f"   RAG API: http://localhost:8000")
    print(f"   Orchestrator: http://localhost:5003")

    rag_only = "--rag-only" in sys.argv
    orch_only = "--orchestrator-only" in sys.argv

    reports = {}

    if not orch_only:
        reports["rag"] = run_rag_eval()

    if not rag_only:
        reports["orchestrator"] = run_orchestrator_eval()

    # 总结
    print("\n" + "=" * 60)
    print("📋 评估总结")
    print("=" * 60)
    for name, report in reports.items():
        icon = "✅" if report.failed == 0 else "⚠️" if report.avg_score >= 0.5 else "❌"
        print(f"  {icon} {name}: {report.passed}/{report.total} 通过, 平均分 {report.avg_score:.0%}")

    # 保存报告
    report_path = Path(__file__).parent.parent.parent / "data" / "eval_reports"
    report_path.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    for name, report in reports.items():
        out = report_path / f"eval_{name}_{ts}.json"
        with open(out, "w") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"  📄 报告: {out}")
