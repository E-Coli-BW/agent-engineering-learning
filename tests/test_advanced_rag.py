"""
高级 RAG Pipeline 测试
=======================
Query Rewrite, BM25, Hybrid Fusion, Reranker
不依赖 Ollama (mock LLM 调用)
"""

from unittest.mock import patch


class TestBM25:
    def test_basic_search(self):
        from project.rag.advanced_pipeline import BM25Retriever
        bm25 = BM25Retriever()
        bm25.add_documents([
            {"content": "LoRA 是一种低秩适应方法，用于参数高效微调"},
            {"content": "Transformer 使用 Self-Attention 机制"},
            {"content": "KV Cache 缓存 Key Value 避免重复计算"},
        ])
        results = bm25.search("LoRA 微调", top_k=2)
        assert len(results) > 0
        assert "LoRA" in results[0]["content"]

    def test_empty_index(self):
        from project.rag.advanced_pipeline import BM25Retriever
        bm25 = BM25Retriever()
        results = bm25.search("test")
        assert results == []

    def test_no_match(self):
        from project.rag.advanced_pipeline import BM25Retriever
        bm25 = BM25Retriever()
        bm25.add_documents([{"content": "苹果橘子香蕉"}])
        results = bm25.search("Transformer attention")
        assert len(results) == 0


class TestHybridFusion:
    def test_rrf_merge(self):
        from project.rag.advanced_pipeline import HybridRetriever
        vector = [
            {"content": "doc A (vector rank 1)", "score": 0.9},
            {"content": "doc B (vector rank 2)", "score": 0.7},
        ]
        bm25 = [
            {"content": "doc B (vector rank 2)", "score": 0},  # B 在两个 list 都出现
            {"content": "doc C (bm25 only)", "score": 0},
        ]
        fused = HybridRetriever.fuse(vector, bm25, top_k=3)
        # doc B 应该排名最高 (两路都有)
        assert len(fused) >= 2
        assert "doc B" in fused[0]["content"]

    def test_empty_inputs(self):
        from project.rag.advanced_pipeline import HybridRetriever
        fused = HybridRetriever.fuse([], [], top_k=5)
        assert fused == []


class TestTokenize:
    def test_mixed_language(self):
        from project.rag.advanced_pipeline import _tokenize
        tokens = _tokenize("LoRA 是低秩 adaptation")
        assert "lora" in tokens
        assert "adaptation" in tokens
        assert "是" in tokens


class TestRerankerScoreParsing:
    def test_parse_scores(self):
        from project.rag.advanced_pipeline import _parse_rerank_scores
        response = "1: 8.5\n2: 3.0\n3: 9.0"
        scores = _parse_rerank_scores(response, 3)
        assert scores[0] == 8.5
        assert scores[1] == 3.0
        assert scores[2] == 9.0

    def test_parse_chinese_colon(self):
        from project.rag.advanced_pipeline import _parse_rerank_scores
        response = "1：7\n2：9"
        scores = _parse_rerank_scores(response, 2)
        assert scores[0] == 7.0
        assert scores[1] == 9.0

    def test_invalid_response(self):
        from project.rag.advanced_pipeline import _parse_rerank_scores
        scores = _parse_rerank_scores("garbage", 3)
        assert scores == [0.0, 0.0, 0.0]


class TestAdvancedPipeline:
    def test_pipeline_without_vector_store(self):
        """没有向量库时不报错"""
        from project.rag.advanced_pipeline import AdvancedRAGPipeline
        pipe = AdvancedRAGPipeline(use_rewrite=False, use_hybrid=False)
        result = pipe.query("test question", vector_store=None)
        assert result["sources"] == []
        assert result["pipeline"]["original_query"] == "test question"
