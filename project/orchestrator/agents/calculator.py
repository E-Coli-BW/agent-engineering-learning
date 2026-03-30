"""
Calculator Agent — 数学计算 + 单位换算 + 统计分析
==================================================

能力:
  1. 数学表达式求值 (安全 eval)
  2. 参数量计算 (Transformer/LoRA 相关)
  3. 单位换算 (MB↔GB, 参数↔显存)
"""

import math
import logging

logger = logging.getLogger("orchestrator.calculator")

# 安全的数学函数白名单
SAFE_MATH = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "pow": pow, "int": int, "float": float,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
}


def calculator_agent(query: str) -> str:
    """
    Calculator Agent 入口

    支持:
      - 数学表达式: "2048 * 2048", "sqrt(768)"
      - 参数量问题: "LoRA rank=16 d_model=4096 参数量"
      - 对比计算: "4096^2 vs 2*4096*16"
    """
    # 尝试直接计算表达式
    expr_result = _safe_eval(query)
    if expr_result is not None:
        return f"计算结果: {query} = {expr_result}"

    # 尝试从自然语言中提取计算
    extracted = _extract_and_compute(query)
    if extracted:
        return extracted

    return f"无法解析计算表达式: {query}"


def _safe_eval(expr: str) -> str | None:
    """安全的表达式求值 (不执行任意代码)"""
    # 清理常见的自然语言干扰
    expr = expr.strip()
    for remove in ["计算", "算一下", "等于多少", "=?", "是多少"]:
        expr = expr.replace(remove, "")
    expr = expr.strip()

    if not expr:
        return None

    # 替换常见符号
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")

    try:
        result = eval(expr, {"__builtins__": {}}, SAFE_MATH)
        # 大数字格式化
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return f"{int(result):,}"
            return f"{result:,.4f}"
        elif isinstance(result, int):
            return f"{result:,}"
        return str(result)
    except Exception:
        return None


def _extract_and_compute(query: str) -> str | None:
    """从自然语言中提取参数并计算"""
    import re

    results = []

    # LoRA 参数量计算
    d_match = re.search(r'd_model\s*=?\s*(\d+)', query)
    r_match = re.search(r'rank\s*=?\s*(\d+)', query) or re.search(r'r\s*=\s*(\d+)', query)

    if d_match and r_match:
        d = int(d_match.group(1))
        r = int(r_match.group(1))
        full_params = d * d
        lora_params = 2 * d * r
        reduction = (1 - lora_params / full_params) * 100

        results.append(
            f"LoRA 参数量计算 (d_model={d}, rank={r}):\n"
            f"  全量微调参数: d² = {d}² = {full_params:,}\n"
            f"  LoRA 参数: 2 × d × r = 2 × {d} × {r} = {lora_params:,}\n"
            f"  参数减少: {reduction:.1f}%\n"
            f"  压缩比: {full_params / lora_params:.0f}x"
        )

    # Transformer 参数量
    layers_match = re.search(r'(\d+)\s*层', query) or re.search(r'layers?\s*=?\s*(\d+)', query)
    if d_match and layers_match:
        d = int(d_match.group(1))
        n_layers = int(layers_match.group(1))
        attn_params = 4 * (d * d + d)  # Q/K/V/O
        ffn_params = 2 * (d * 4 * d + 4 * d)  # 2-layer MLP
        block_params = attn_params + ffn_params
        total = block_params * n_layers

        results.append(
            f"Transformer 参数量 (d_model={d}, layers={n_layers}):\n"
            f"  Attention 每层: {attn_params:,}\n"
            f"  FFN 每层: {ffn_params:,}\n"
            f"  每层总计: {block_params:,}\n"
            f"  {n_layers} 层总计: {total:,} ({total / 1e6:.1f}M)"
        )

    return "\n\n".join(results) if results else None
