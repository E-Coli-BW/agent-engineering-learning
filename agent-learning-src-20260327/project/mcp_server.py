"""
MCP Server — 把我们的知识库和工具暴露给 Copilot
================================================

MCP (Model Context Protocol) 让任何支持 MCP 的客户端
（VS Code Copilot, Claude Desktop 等）都能调用我们的工具。

本 Server 通过 stdio 运行（Copilot 会自动启动和管理进程），
暴露以下能力：

  Tools:
    - rag_query:      查询我们的 RAG 知识库
    - knowledge_graph: 查询知识图谱中的实体关系
    - calculate:       数学计算
    - list_modules:    列出项目所有学习模块

  Resources:
    - project://readme  项目 README 全文

使用方式:
  1. 运行本文件会通过 stdio 启动 MCP Server
  2. 在 .vscode/mcp.json 中配置后 Copilot 自动连接
  3. 在 Copilot Chat 中直接使用我们的工具
"""

import os
import sys
import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---- 路径 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "project_chroma_db"

# ---- 日志 (写文件，不污染 stdio) ----
logging.basicConfig(
    level=logging.INFO,
    filename=str(DATA_DIR / "mcp_server.log"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mcp_server")

# ============================================================
# 创建 MCP Server
# ============================================================
mcp = FastMCP("Agent Learning Knowledge Base")


# ============================================================
# Tool 1: RAG 知识库查询
# ============================================================
@mcp.tool()
def rag_query(question: str, top_k: int = 3) -> str:
    """
    查询 Agent Learning 项目的 RAG 知识库。

    可以回答关于 Transformer、Attention、LoRA、RAG、Agent、
    知识图谱、推理部署等所有项目内容的问题。

    Args:
        question: 要查询的问题
        top_k: 返回的相关文档数量（默认 3）
    """
    logger.info(f"rag_query called: {question}")

    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        collection = os.getenv("COLLECTION_NAME", "project_knowledge_v2")

        if not VECTOR_DB_DIR.exists():
            return "❌ 向量库不存在。请先运行 `python project/etl_pipeline.py` 构建索引。"

        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings,
            collection_name=collection,
        )

        results = vector_store.similarity_search_with_score(question, k=top_k)

        if not results:
            return "未找到相关内容。"

        output_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            output_parts.append(
                f"[{i}] (相关度: {1 - score:.2f}) [{source}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        logger.error(f"rag_query error: {e}")
        return f"查询失败: {str(e)}"


# ============================================================
# Tool 2: 知识图谱查询
# ============================================================
@mcp.tool()
def knowledge_graph_query(entity: str) -> str:
    """
    查询知识图谱中某个技术概念的相关关系。

    可以查询 Transformer、Self-Attention、LoRA、GPT、BERT、
    RAG、Agent、ReAct 等概念之间的关系。

    Args:
        entity: 要查询的技术实体名称，如 "Transformer"、"LoRA"、"RAG"
    """
    logger.info(f"knowledge_graph_query called: {entity}")

    # 内嵌一份精简版知识图谱 (和 knowledge_graph/01_graph_rag.py 中的一致)
    kg_triples = [
        ("Transformer", "包含", "Self-Attention"),
        ("Transformer", "包含", "Feed-Forward Network"),
        ("Transformer", "包含", "Layer Normalization"),
        ("Transformer", "提出时间", "2017年"),
        ("Transformer", "论文", "Attention Is All You Need"),
        ("Transformer", "提出者", "Vaswani et al."),
        ("Self-Attention", "计算", "QKV矩阵"),
        ("Self-Attention", "变体", "Multi-Head Attention"),
        ("Self-Attention", "缩放因子", "sqrt(d_k)"),
        ("Multi-Head Attention", "特点", "多子空间并行计算"),
        ("GPT", "基于", "Transformer Decoder"),
        ("BERT", "基于", "Transformer Encoder"),
        ("GPT", "训练目标", "Next Token Prediction"),
        ("GPT", "使用", "Causal Mask"),
        ("Causal Mask", "作用", "防止看到未来token"),
        ("LoRA", "用于", "参数高效微调"),
        ("LoRA", "原理", "低秩矩阵分解 ΔW=BA"),
        ("QLoRA", "改进自", "LoRA"),
        ("QLoRA", "特点", "4-bit量化+LoRA"),
        ("SFT", "全称", "Supervised Fine-Tuning"),
        ("RLHF", "用于", "模型对齐"),
        ("DPO", "替代", "RLHF"),
        ("ReAct", "循环", "Thought-Action-Observation"),
        ("LangGraph", "建模为", "有向图"),
        ("Agent", "核心能力", "Tool Calling"),
        ("RAG", "步骤", "Retrieve-Augment-Generate"),
        ("RAG", "使用", "向量检索"),
        ("Graph RAG", "结合", "知识图谱+RAG"),
        ("KV Cache", "作用", "加速自回归推理"),
        ("MCP", "连接", "Agent与工具/数据"),
        ("A2A", "连接", "Agent与Agent"),
    ]

    # 查找与实体相关的三元组 (模糊匹配)
    related = []
    entity_lower = entity.lower()
    for h, r, t in kg_triples:
        if entity_lower in h.lower() or entity_lower in t.lower():
            related.append(f"  {h} --[{r}]--> {t}")

    if not related:
        return f"未找到与 '{entity}' 相关的知识图谱条目。\n可用实体: Transformer, Self-Attention, GPT, BERT, LoRA, QLoRA, RAG, Agent, ReAct, LangGraph, KV Cache, MCP, A2A"

    return f"'{entity}' 的知识图谱关系:\n" + "\n".join(related)


# ============================================================
# Tool 3: 数学计算
# ============================================================
@mcp.tool()
def calculate(expression: str) -> str:
    """
    安全计算数学表达式。

    Args:
        expression: 数学表达式，如 "2+3*4"、"(15+27)*3"、"4096*4096"
    """
    logger.info(f"calculate called: {expression}")
    try:
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return f"错误: 表达式包含不允许的字符。只支持数字和 +-*/()."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# Tool 4: 列出项目模块
# ============================================================
@mcp.tool()
def list_modules() -> str:
    """
    列出 Agent Learning 项目的所有学习模块及其文件。
    用于了解项目结构和各模块的内容。
    """
    logger.info("list_modules called")

    modules = {
        "📐 底层原理": {
            "multi_head_attention.py": "手写 QKV 注意力机制",
            "char_transformer.py": "完整 Mini-GPT 训练 + 可视化",
        },
        "🤖 Agent 开发": {
            "agent/01_chat_basics.py": "LLM 调用本质",
            "agent/02_tool_calling.py": "工具调用原理 (MCP 的底层)",
            "agent/03_react_agent.py": "手写 ReAct Agent",
            "agent/04_langgraph_agent.py": "LangGraph 生产级 Agent",
        },
        "📚 RAG 系统": {
            "rag/01_embedding_basics.py": "Embedding 与向量检索",
            "rag/02_naive_rag.py": "手写最简 RAG",
            "rag/03_langchain_rag.py": "LangChain 标准 RAG Pipeline",
            "rag/04_advanced_rag.py": "高级 RAG (Multi-Query/CRAG)",
        },
        "🔧 LoRA 微调": {
            "finetune/01_lora.py": "手写 LoRA + SFT + RLHF/DPO",
        },
        "🚀 推理部署": {
            "deploy/01_inference.py": "KV Cache + 量化 + 生产架构",
        },
        "🕸️ 知识图谱": {
            "knowledge_graph/01_graph_rag.py": "手写知识图谱 + Graph RAG",
        },
        "🏗️ 工程化": {
            "project/etl_pipeline.py": "ETL 数据处理管线",
            "project/api_server.py": "FastAPI RAG API 服务",
            "project/mcp_server.py": "MCP Server (本文件)",
        },
    }

    lines = ["Agent Learning 项目模块:\n"]
    for category, files in modules.items():
        lines.append(f"\n{category}:")
        for file, desc in files.items():
            lines.append(f"  - {file}: {desc}")

    return "\n".join(lines)


# ============================================================
# Resource: 项目 README
# ============================================================
@mcp.resource("project://readme")
def get_readme() -> str:
    """项目 README 文档全文"""
    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "README.md 未找到"


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    # 确保 data 目录存在 (日志需要)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mcp.run(transport="stdio")
