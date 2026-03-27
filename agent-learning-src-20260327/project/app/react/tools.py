"""
工具注册表 + 内置工具
=====================

工具注册表管理 Agent 可调用的所有工具。
内置工具: 知识图谱查询、数学计算、时间查询、RAG 检索。

与学习版 react_agent.py 的区别:
  - 学习版把 ToolRegistry + 工具函数 + ReActAgent + Server + CLI 全堆在一个文件
  - 生产版拆开: tools.py (注册表 + 工具) + agent.py (ReAct 循环)
"""

import os
import json
import logging
from pathlib import Path

logger = logging.getLogger("app.react.tools")

# 数据目录 (从 project root 推算)
_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class ToolRegistry:
    """
    工具注册中心 — 管理 Agent 可调用的所有工具

    用法:
        registry = ToolRegistry()
        registry.register("calc", calc_fn, "数学计算", 'expression (str)')
        result = registry.execute("calc", "1+1")
    """

    def __init__(self):
        self.tools: dict[str, dict] = {}

    def register(self, name: str, func, description: str, params: str = ""):
        """注册一个工具"""
        self.tools[name] = {
            "function": func,
            "description": description,
            "params": params,
        }
        logger.info("注册工具: %s — %s", name, description)

    def execute(self, name: str, args_str: str) -> str:
        """执行一个工具"""
        if name not in self.tools:
            return f"错误: 工具 '{name}' 不存在。可用工具: {list(self.tools.keys())}"
        try:
            if args_str.strip():
                args = json.loads(args_str)
                if isinstance(args, dict):
                    result = self.tools[name]["function"](**args)
                else:
                    result = self.tools[name]["function"](args)
            else:
                result = self.tools[name]["function"]()
            return str(result)
        except json.JSONDecodeError:
            try:
                result = self.tools[name]["function"](args_str.strip().strip('"\''))
                return str(result)
            except Exception as e:
                return f"工具执行错误: {e}"
        except Exception as e:
            return f"工具执行错误: {e}"

    def get_tool_descriptions(self) -> str:
        """生成工具描述 (用于 system prompt)"""
        lines = []
        for name, info in self.tools.items():
            params = f" 参数: {info['params']}" if info["params"] else ""
            lines.append(f"  - {name}: {info['description']}{params}")
        return "\n".join(lines)


# ============================================================
# 内置工具
# ============================================================

def knowledge_graph_query(entity: str) -> str:
    """查询知识图谱 (内置数据)"""
    kg_triples = [
        ("Transformer", "包含", "Self-Attention"),
        ("Transformer", "包含", "Feed-Forward Network"),
        ("Transformer", "提出时间", "2017年"),
        ("Transformer", "论文", "Attention Is All You Need"),
        ("Self-Attention", "计算", "QKV矩阵"),
        ("Self-Attention", "缩放因子", "sqrt(d_k)"),
        ("Self-Attention", "变体", "Multi-Head Attention"),
        ("Multi-Head Attention", "特点", "多子空间并行计算"),
        ("GPT", "基于", "Transformer Decoder"),
        ("BERT", "基于", "Transformer Encoder"),
        ("LoRA", "用于", "参数高效微调"),
        ("LoRA", "原理", "低秩矩阵分解 ΔW=BA"),
        ("QLoRA", "改进自", "LoRA"),
        ("QLoRA", "特点", "4-bit量化+LoRA"),
        ("RAG", "步骤", "Retrieve-Augment-Generate"),
        ("RAG", "使用", "向量检索"),
        ("Graph RAG", "结合", "知识图谱+RAG"),
        ("ReAct", "循环", "Thought-Action-Observation"),
        ("Agent", "核心能力", "Tool Calling"),
        ("KV Cache", "作用", "加速自回归推理"),
        ("MCP", "连接", "Agent与工具/数据"),
        ("A2A", "连接", "Agent与Agent"),
    ]
    entity_lower = entity.lower()
    related = [f"  {h} --[{r}]--> {t}" for h, r, t in kg_triples
               if entity_lower in h.lower() or entity_lower in t.lower()]
    if not related:
        return f"未找到与 '{entity}' 相关的知识图谱条目。"
    return f"'{entity}' 的知识图谱关系:\n" + "\n".join(related)


def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    import math as _math
    allowed = set("0123456789+-*/().% sqrtabcdefghijklmnopqrstuvwxyz_")
    if not all(c in allowed for c in expression.lower().replace(" ", "")):
        return f"非法表达式: {expression}"
    expr = expression.replace("sqrt", "_math.sqrt")
    try:
        return str(eval(expr))
    except Exception as e:
        return f"计算错误: {e}"


def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rag_query(question: str) -> str:
    """RAG 向量检索 — 查询项目知识库"""
    vector_db_dir = _DATA_DIR / "project_chroma_db"
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        if not vector_db_dir.exists():
            return "向量库不存在。请先运行 python project/etl_pipeline.py 构建索引。"

        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=str(vector_db_dir),
            embedding_function=embeddings,
            collection_name=os.getenv("COLLECTION_NAME", "project_knowledge_v2"),
        )
        results = vector_store.similarity_search_with_score(question, k=3)
        if not results:
            return "未找到相关内容。"

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (相关度:{1 - score:.2f}) [{source}]\n{doc.page_content[:300]}")
        return "\n---\n".join(parts)
    except ImportError:
        return "RAG 依赖未安装 (需要 langchain-ollama, chromadb)。"
    except Exception as e:
        return f"RAG 查询失败: {e}"


# ============================================================
# 默认注册表工厂
# ============================================================

def create_default_registry() -> ToolRegistry:
    """创建包含所有内置工具的注册表"""
    registry = ToolRegistry()
    registry.register(
        "knowledge_graph_query",
        knowledge_graph_query,
        "查询知识图谱中技术概念的关系。",
        'entity (str): 技术实体名，如 "Transformer"、"LoRA"',
    )
    registry.register(
        "calculator",
        calculator,
        "计算数学表达式。支持 sqrt。",
        'expression (str): 如 "768**0.5"、"sqrt(512)"',
    )
    registry.register(
        "get_time",
        get_time,
        "获取当前日期和时间。",
        "无参数",
    )
    registry.register(
        "rag_query",
        rag_query,
        "从项目知识库中检索相关文档。适合查找具体实现细节。",
        'question (str): 检索问题，如 "LoRA的实现代码"',
    )
    return registry
