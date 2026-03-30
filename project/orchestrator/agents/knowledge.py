"""
Knowledge Agent — RAG 检索 + 知识图谱查询
==========================================

能力:
  1. 向量检索: 从 ChromaDB 中找到最相关的文档
  2. 知识图谱: 查询实体关系 (如 "LoRA 和什么有关")
  3. 综合回答: 基于检索结果用 LLM 生成答案

本 Agent 复用了 project/api_server.py 和 knowledge_graph/ 的能力，
但封装为 LangGraph 节点可调用的函数。
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("orchestrator.knowledge")


def knowledge_agent(query: str) -> str:
    """
    Knowledge Agent 入口

    尝试顺序:
      1. RAG 向量检索 (需要 ChromaDB)
      2. 知识图谱查询 (需要图谱数据)
      3. 如果都不可用，返回提示
    """
    results = []

    # ---- RAG 检索 ----
    rag_result = _rag_search(query)
    if rag_result:
        results.append(f"[RAG 检索结果]\n{rag_result}")

    # ---- 知识图谱 ----
    kg_result = _knowledge_graph_search(query)
    if kg_result:
        results.append(f"[知识图谱]\n{kg_result}")

    if not results:
        return "知识库中未找到相关信息。"

    return "\n\n---\n\n".join(results)


def _rag_search(query: str, top_k: int = 3) -> Optional[str]:
    """RAG 向量检索"""
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        from pathlib import Path

        db_dir = Path(__file__).parent.parent.parent.parent / "data" / "project_chroma_db"
        if not db_dir.exists():
            logger.debug("向量库不存在: %s", db_dir)
            return None

        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        collection = os.getenv("COLLECTION_NAME", "project_knowledge_v2")

        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=str(db_dir),
            embedding_function=embeddings,
            collection_name=collection,
        )

        results = vector_store.similarity_search_with_score(query, k=top_k)
        if not results:
            return None

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[{i}] (相关度: {1 - score:.2f}) [{source}]\n{doc.page_content[:300]}")

        return "\n\n".join(parts)

    except Exception as e:
        logger.warning("RAG 检索失败: %s", e)
        return None


def _knowledge_graph_search(query: str) -> Optional[str]:
    """知识图谱查询 — 复用 react_agent 中的知识图谱工具"""
    try:
        from pathlib import Path
        import json

        # 尝试加载知识图谱数据
        kg_path = Path(__file__).parent.parent.parent.parent / "data" / "knowledge_graph.json"
        if not kg_path.exists():
            return None

        with open(kg_path) as f:
            graph = json.load(f)

        # 简单关键词匹配实体
        q = query.lower()
        matches = []
        for triple in graph.get("triples", []):
            subj, rel, obj = triple.get("subject", ""), triple.get("relation", ""), triple.get("object", "")
            if q in subj.lower() or q in obj.lower() or any(kw in q for kw in [subj.lower(), obj.lower()]):
                matches.append(f"{subj} --[{rel}]--> {obj}")

        return "\n".join(matches[:10]) if matches else None

    except Exception as e:
        logger.warning("知识图谱查询失败: %s", e)
        return None
