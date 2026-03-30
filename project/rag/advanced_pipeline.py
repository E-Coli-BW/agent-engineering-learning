"""
高级 RAG Pipeline — Query Rewrite + Hybrid Search + Reranker
=============================================================

标准 RAG 流程:
  用户问题
    → Query Rewrite (Multi-Query / HyDE)
    → Hybrid Retrieval (BM25 + Vector)
    → Reranker (Cross-Encoder 精排)
    → Prompt 构建
    → LLM 生成
    → Post-process

vs 简单 RAG (当前 api_server.py):
  用户问题 → Vector Search → LLM 生成

本模块实现了高级 RAG 的每个组件，可以独立使用也可以组合使用。
"""

import os
import re
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger("rag.advanced")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")


# ============================================================
# 1. Query Rewrite — 把一个问题拆成多个检索 query
# ============================================================

class QueryRewriter:
    """
    Multi-Query Rewrite: 用 LLM 把用户问题拆成多个搜索角度

    原理: 用户的原始问题可能太模糊或角度单一，
    拆成多个 query 可以提高 recall (每个 query 检索不同的文档)。

    示例:
      输入: "LoRA 和 QLoRA 有什么区别?"
      输出: [
        "LoRA 低秩适应原理",
        "QLoRA 4bit 量化微调",
        "LoRA QLoRA 对比 参数量 性能",
      ]
    """

    REWRITE_PROMPT = """你是一个搜索查询优化器。将用户问题改写为 3 个不同角度的搜索查询，
以便从知识库中检索到更全面的信息。

规则:
1. 每个查询从不同角度切入（定义、原理、对比、应用等）
2. 使用关键词形式，不要用完整句子
3. 只返回 3 个查询，每行一个，不要编号不要其他内容

用户问题: {question}
"""

    @staticmethod
    def rewrite(question: str) -> list[str]:
        """返回多个改写后的 query (包含原始 query)"""
        try:
            prompt = QueryRewriter.REWRITE_PROMPT.format(question=question)
            response = _call_ollama_chat(prompt)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            # 去掉编号前缀
            queries = [re.sub(r'^[\d\.\-\)]+\s*', '', q) for q in queries]
            queries = [q for q in queries if len(q) > 2][:3]
            # 始终包含原始 query
            return [question] + queries
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)
            return [question]


class HyDERewriter:
    """
    HyDE (Hypothetical Document Embeddings):
    让 LLM 先生成一个"假设性回答"，然后用这个回答去检索。

    原理: 假设性回答的 embedding 和真实文档更接近（比问题本身更接近）。

    示例:
      问题: "KV Cache 怎么加速推理?"
      HyDE 生成: "KV Cache 通过缓存之前的 Key 和 Value 矩阵避免重复计算..."
      → 用这段文字的 embedding 去检索
    """

    HYDE_PROMPT = """根据以下问题，生成一段简短的假设性回答（100字以内）。
不需要完全准确，只要大致描述这个概念即可。

问题: {question}
假设性回答:"""

    @staticmethod
    def generate_hypothesis(question: str) -> str:
        """生成假设性文档"""
        try:
            prompt = HyDERewriter.HYDE_PROMPT.format(question=question)
            return _call_ollama_chat(prompt)[:300]
        except Exception as e:
            logger.warning("HyDE generation failed: %s", e)
            return question


# ============================================================
# 2. Hybrid Retrieval — BM25 + Vector Search
# ============================================================

class BM25Retriever:
    """
    BM25 关键词检索 — 基于词频的传统搜索

    vs Vector Search:
      - Vector: 语义相似（"狗" 能匹配 "犬"）
      - BM25: 精确匹配（"KV Cache" 必须包含这个词）

    组合使用 (Hybrid):
      - Vector 负责语义理解
      - BM25 负责精确匹配（专有名词、代码、公式）
      - 两者结果合并去重

    本实现是简化版的 BM25（不依赖 Elasticsearch）。
    """

    def __init__(self):
        self._documents: list[dict] = []  # [{content, metadata}]
        self._index_built = False

    def add_documents(self, documents: list[dict]):
        """添加文档到 BM25 索引"""
        self._documents.extend(documents)
        self._index_built = False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 关键词检索"""
        if not self._documents:
            return []

        query_terms = _tokenize(query)
        scores = []

        for doc in self._documents:
            content = doc.get("content", "")
            doc_terms = _tokenize(content)
            score = _bm25_score(query_terms, doc_terms)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0]


def _tokenize(text: str) -> list[str]:
    """简单分词 (中英文混合)"""
    # 英文: 按空格和标点分
    # 中文: 按字分（简化版，生产环境用 jieba）
    text = text.lower()
    tokens = re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', text)
    return tokens


def _bm25_score(query_terms: list[str], doc_terms: list[str],
                k1: float = 1.5, b: float = 0.75, avg_dl: float = 200) -> float:
    """计算 BM25 分数"""
    if not doc_terms:
        return 0.0
    dl = len(doc_terms)
    score = 0.0
    doc_term_set = set(doc_terms)
    for qt in query_terms:
        if qt not in doc_term_set:
            continue
        tf = doc_terms.count(qt)
        idf = 1.0  # 简化: 不做真正的 IDF（需要全文档集）
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += idf * numerator / denominator
    return score


class HybridRetriever:
    """
    混合检索: Vector + BM25，结果用 RRF (Reciprocal Rank Fusion) 合并

    RRF 公式: score(d) = Σ 1 / (k + rank_i(d))
    k 通常取 60

    优势:
      - Vector 找到语义相关的文档
      - BM25 找到精确匹配的文档
      - RRF 合并后 recall 显著提升
    """

    @staticmethod
    def fuse(vector_results: list[dict], bm25_results: list[dict],
             top_k: int = 5, rrf_k: int = 60) -> list[dict]:
        """
        RRF 融合两路检索结果

        Args:
            vector_results: [(doc, score), ...] 向量检索结果
            bm25_results: [doc, ...] BM25 检索结果
            top_k: 返回数量
            rrf_k: RRF 参数 (默认 60)
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        # Vector 结果的 RRF 分
        for rank, item in enumerate(vector_results):
            content = item.get("content", "")
            key = content[:100]  # 用前 100 字符作为去重 key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = item

        # BM25 结果的 RRF 分
        for rank, item in enumerate(bm25_results):
            content = item.get("content", "")
            key = content[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = item

        # 按 RRF 分排序
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:top_k]]


# ============================================================
# 3. Reranker — Cross-Encoder 精排
# ============================================================

class LLMReranker:
    """
    LLM-based Reranker — 用 LLM 对检索结果重排序

    原理:
      传统 Reranker 用 Cross-Encoder 模型 (如 bge-reranker)，
      但本地环境没有这个模型。我们用 Ollama LLM 做 zero-shot reranking:
      让 LLM 判断每个文档和问题的相关性 (0-10 分)。

    vs Cross-Encoder:
      - Cross-Encoder: 快 (~10ms/doc), 需要专门模型
      - LLM Reranker: 慢 (~1s/doc), 但不需要额外模型
      - 生产环境应该用 Cross-Encoder

    本实现用 LLM 做 listwise reranking (一次性排所有文档)。
    """

    RERANK_PROMPT = """你是一个文档相关性评分器。判断以下文档和用户问题的相关性。

用户问题: {question}

文档列表:
{documents}

请对每个文档评分 (0-10, 10=完全相关)，只返回编号和分数，每行一个:
1: 分数
2: 分数
..."""

    @staticmethod
    def rerank(question: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """
        对检索结果重排序

        Args:
            question: 用户问题
            documents: 检索到的文档列表
            top_k: 返回前 K 个
        """
        if len(documents) <= 1:
            return documents[:top_k]

        # 构造文档列表字符串
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")[:200]  # 截断以控制 prompt 长度
            doc_texts.append(f"[{i}] {content}")

        prompt = LLMReranker.RERANK_PROMPT.format(
            question=question,
            documents="\n".join(doc_texts),
        )

        try:
            response = _call_ollama_chat(prompt)
            scores = _parse_rerank_scores(response, len(documents))
            # 按分数排序
            ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in ranked[:top_k]]
        except Exception as e:
            logger.warning("Reranking failed: %s, returning original order", e)
            return documents[:top_k]


def _parse_rerank_scores(response: str, num_docs: int) -> list[float]:
    """解析 LLM 返回的 rerank 分数"""
    scores = [0.0] * num_docs
    for line in response.strip().split("\n"):
        match = re.search(r'(\d+)\s*[:：]\s*([\d.]+)', line)
        if match:
            idx = int(match.group(1)) - 1
            score = float(match.group(2))
            if 0 <= idx < num_docs:
                scores[idx] = min(score, 10.0)
    return scores


# ============================================================
# 4. 完整 Advanced RAG Pipeline
# ============================================================

class AdvancedRAGPipeline:
    """
    完整的高级 RAG Pipeline

    组合:
      1. Query Rewrite (Multi-Query)
      2. Hybrid Search (Vector + BM25)
      3. Reranker (LLM-based)
      4. LLM 生成

    可通过参数控制每个步骤的开关。
    """

    def __init__(
        self,
        use_rewrite: bool = True,
        use_hybrid: bool = True,
        use_reranker: bool = False,  # 默认关闭 (LLM reranker 太慢)
        top_k: int = 5,
    ):
        self.use_rewrite = use_rewrite
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.top_k = top_k
        self.bm25 = BM25Retriever()

    def query(self, question: str, vector_store=None) -> dict:
        """
        执行完整 RAG Pipeline

        Returns: {"answer": str, "sources": list, "pipeline": dict}
        """
        pipeline_info = {"original_query": question, "steps": []}

        # ---- Step 1: Query Rewrite ----
        queries = [question]
        if self.use_rewrite:
            queries = QueryRewriter.rewrite(question)
            pipeline_info["steps"].append({
                "name": "query_rewrite",
                "queries": queries,
            })

        # ---- Step 2: Retrieval ----
        all_results = []

        # Vector Search (每个 query 都检索)
        if vector_store:
            for q in queries:
                results = vector_store.similarity_search_with_score(q, k=self.top_k)
                for doc, score in results:
                    all_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": float(score),
                        "metadata": doc.metadata,
                        "retriever": "vector",
                    })

        # BM25 Search (hybrid)
        bm25_results = []
        if self.use_hybrid and self.bm25._documents:
            for q in queries:
                bm25_hits = self.bm25.search(q, top_k=self.top_k)
                bm25_results.extend(bm25_hits)

        # ---- Step 3: Fusion ----
        if bm25_results:
            fused = HybridRetriever.fuse(all_results, bm25_results, top_k=self.top_k * 2)
            pipeline_info["steps"].append({
                "name": "hybrid_fusion",
                "vector_count": len(all_results),
                "bm25_count": len(bm25_results),
                "fused_count": len(fused),
            })
        else:
            # 去重 (多个 query 可能检索到相同文档)
            seen = set()
            fused = []
            for r in all_results:
                key = r["content"][:100]
                if key not in seen:
                    seen.add(key)
                    fused.append(r)
            fused = fused[:self.top_k * 2]

        # ---- Step 4: Rerank (可选) ----
        if self.use_reranker and len(fused) > 1:
            fused = LLMReranker.rerank(question, fused, top_k=self.top_k)
            pipeline_info["steps"].append({"name": "reranker", "reranked": len(fused)})
        else:
            fused = fused[:self.top_k]

        # ---- Step 5: Generate ----
        context = "\n\n---\n\n".join(
            f"[来源: {r.get('source', '?')}]\n{r['content']}"
            for r in fused
        )

        pipeline_info["steps"].append({
            "name": "generation",
            "context_length": len(context),
            "num_sources": len(fused),
        })
        pipeline_info["total_steps"] = len(pipeline_info["steps"])

        return {
            "context": context,
            "sources": fused,
            "pipeline": pipeline_info,
        }


# ============================================================
# 工具函数
# ============================================================

def _call_ollama_chat(prompt: str, temperature: float = 0.3) -> str:
    """调用 Ollama LLM"""
    data = json.dumps({
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return result.get("message", {}).get("content", "")
