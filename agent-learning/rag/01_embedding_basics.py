"""
Level 1: Embedding 与向量检索的本质
=====================================

核心问题: RAG 的 "R" (Retrieval) 是怎么做到的？

答案: 把文本变成向量 (Embedding)，用向量距离衡量语义相似度。

原理说明:
---------
1. Embedding 是什么？
   - 把一段文本映射成一个固定长度的浮点数向量 (如 1024 维)
   - 语义相近的文本，向量之间的距离更近
   - "猫" 和 "小猫" 的向量距离 < "猫" 和 "汽车" 的向量距离

2. 为什么能用来检索？
   - 传统搜索: 关键词匹配 → "机器学习" 搜不到 "ML"
   - 向量搜索: 语义匹配 → "机器学习" 和 "ML" 的向量很近，能搜到

3. 和你在 char_transformer.py 里的 nn.Embedding 是什么关系？
   - char_transformer 里的 Embedding 是 **可训练的查找表** (token → 向量)
   - 这里的 Embedding 模型是一个 **完整的编码器** (文本 → 向量)
   - 本质一样: 都是把离散对象映射到连续向量空间
   - 区别: Embedding 模型经过了大量对比学习训练，能捕捉语义

4. 你本地的 mxbai-embed-large 模型:
   - 334M 参数的 BERT 架构编码器
   - 输出 1024 维向量
   - 完全本地运行，不需要外部 API
"""

import numpy as np
import requests
import json
from langchain_ollama import OllamaEmbeddings

EMBED_MODEL = "mxbai-embed-large"


# ============================================================
# Part 1: 裸 HTTP 调用 Embedding 模型
# ============================================================
def part1_raw_embedding():
    """
    最底层: 直接用 HTTP 调用 Ollama 的 Embedding API。
    让你看清 Embedding 到底返回了什么。
    """
    print("=" * 60)
    print("Part 1: 裸 HTTP 调用 Embedding")
    print("=" * 60)

    url = "http://localhost:11434/api/embed"
    payload = {
        "model": EMBED_MODEL,
        "input": "什么是注意力机制？"
    }

    response = requests.post(url, json=payload)
    result = response.json()

    embedding = result["embeddings"][0]
    print(f"  输入文本: '什么是注意力机制？'")
    print(f"  向量维度: {len(embedding)}")
    print(f"  前 10 个值: {[round(x, 4) for x in embedding[:10]]}")
    print(f"  向量范数: {np.linalg.norm(embedding):.4f}")
    print()
    print(f"  💡 一段文本 → {len(embedding)} 个浮点数，这就是 Embedding！")


# ============================================================
# Part 2: 语义相似度计算
# ============================================================
def part2_similarity():
    """
    Embedding 的核心应用: 计算两段文本的语义相似度。

    相似度度量:
      - 余弦相似度 (Cosine Similarity): 最常用，值域 [-1, 1]
        cos(A, B) = (A · B) / (|A| × |B|)
      - 欧氏距离 (L2 Distance): 值越小越相似
      - 点积 (Dot Product): 向量归一化后等价于余弦相似度
    """
    print("\n" + "=" * 60)
    print("Part 2: 语义相似度计算")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # ---- 构造测试文本 ----
    texts = [
        "什么是注意力机制？",           # 0: 查询
        "Attention 机制是 Transformer 的核心组件",  # 1: 语义相关
        "自注意力让模型关注序列中的重要部分",        # 2: 语义相关
        "今天天气真好，适合出去散步",                # 3: 不相关
        "我想吃火锅",                              # 4: 不相关
        "What is the attention mechanism?",        # 5: 跨语言相关
    ]

    print(f"  查询: '{texts[0]}'")
    print(f"  候选文本: {len(texts)-1} 条")
    print()

    # ---- 获取所有文本的 Embedding ----
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors)

    # ---- 计算余弦相似度 ----
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    query_vec = vectors[0]
    print(f"  相似度排名:")
    similarities = []
    for i in range(1, len(texts)):
        sim = cosine_similarity(query_vec, vectors[i])
        similarities.append((sim, texts[i]))

    # 按相似度排序
    similarities.sort(reverse=True)
    for rank, (sim, text) in enumerate(similarities, 1):
        bar = "█" * int(sim * 30)
        print(f"    {rank}. [{sim:.4f}] {bar} {text[:40]}")

    print()
    print("  💡 观察:")
    print("    - 语义相关的文本相似度高 (即使用词不同)")
    print("    - 不相关的文本相似度低")
    print("    - 跨语言也能匹配 (Embedding 捕捉了语义)")
    print("    - 这就是向量检索的基础！")


# ============================================================
# Part 3: 手写向量检索 (不用任何向量数据库)
# ============================================================
def part3_manual_search():
    """
    手写一个最简单的向量检索引擎。

    原理:
      1. 把所有文档转成向量，存在一个列表里
      2. 查询时，把 query 也转成向量
      3. 计算 query 向量和所有文档向量的相似度
      4. 返回 Top-K 最相似的文档

    这就是向量数据库 (如 ChromaDB, FAISS) 做的事情！
    只不过它们有索引优化 (如 HNSW, IVF)，速度更快。
    """
    print("\n" + "=" * 60)
    print("Part 3: 手写向量检索引擎")
    print("=" * 60)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # ---- 模拟知识库 ----
    knowledge_base = [
        "Transformer 由 Vaswani 等人在 2017 年的论文 'Attention Is All You Need' 中提出。",
        "Self-Attention 机制让模型能够关注输入序列中任意位置之间的关系。",
        "Multi-Head Attention 把注意力分成多个头，每个头学习不同的语义子空间。",
        "位置编码 (Positional Encoding) 用于向模型提供序列中 token 的位置信息。",
        "Feed-Forward Network 由两个线性层和一个激活函数组成，提供非线性变换能力。",
        "LayerNorm 对每个样本独立做归一化，稳定训练过程。",
        "残差连接 (Residual Connection) 让梯度可以跳过层直接回传，解决深层网络训练困难。",
        "BERT 是一个基于 Transformer Encoder 的双向预训练模型。",
        "GPT 是一个基于 Transformer Decoder 的自回归语言模型。",
        "LoRA 通过低秩矩阵分解，用极少参数实现大模型微调。",
        "RAG 是检索增强生成，通过检索外部知识来增强大模型的回答能力。",
        "RLHF 用人类反馈训练奖励模型，再用强化学习优化语言模型。",
    ]

    print(f"  知识库: {len(knowledge_base)} 条文档")

    # ---- Step 1: 构建索引 (把所有文档转成向量) ----
    print("  📦 构建索引 (Embedding 所有文档)...")
    doc_vectors = np.array(embeddings.embed_documents(knowledge_base))
    print(f"  索引完成: {doc_vectors.shape} (文档数 × 向量维度)")

    # ---- Step 2: 查询 ----
    queries = [
        "注意力机制是怎么工作的？",
        "如何微调大模型？",
        "什么是 RAG？",
    ]

    def search(query: str, top_k: int = 3):
        """手写的向量检索"""
        # 把 query 转成向量
        query_vec = np.array(embeddings.embed_query(query))

        # 计算和所有文档的余弦相似度
        similarities = []
        for i, doc_vec in enumerate(doc_vectors):
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((sim, i))

        # 排序取 Top-K
        similarities.sort(reverse=True)
        return similarities[:top_k]

    for query in queries:
        print(f"\n  🔍 查询: '{query}'")
        results = search(query, top_k=3)
        for rank, (sim, idx) in enumerate(results, 1):
            print(f"    {rank}. [{sim:.4f}] {knowledge_base[idx][:60]}...")

    print()
    print("  💡 关键理解:")
    print("    - 向量检索 = Embedding + 相似度计算 + 排序")
    print("    - 暴力搜索 O(n) 对小数据集够用")
    print("    - 大规模数据需要近似最近邻 (ANN) 算法: HNSW, IVF 等")
    print("    - 这就是 ChromaDB/FAISS 内部做的事情")


# ============================================================
# Part 4: ChromaDB 向量数据库
# ============================================================
def part4_chromadb():
    """
    ChromaDB: 把手写的向量检索替换成专业的向量数据库。

    ChromaDB 帮你做了:
      1. 向量存储和持久化
      2. 高效的 ANN 检索 (比暴力搜索快得多)
      3. 元数据过滤 (按标签、日期等筛选)
      4. 自动调用 Embedding 模型 (不用手动转向量)
    """
    print("\n" + "=" * 60)
    print("Part 4: ChromaDB 向量数据库")
    print("=" * 60)

    import chromadb

    # ---- 创建客户端和集合 ----
    client = chromadb.Client()  # 内存模式 (也可以持久化到磁盘)
    
    # 删除旧的同名集合（如果存在）
    try:
        client.delete_collection("demo_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="demo_knowledge",
        metadata={"description": "演示用知识库"}
    )

    # ---- 添加文档 ----
    documents = [
        "Transformer 由 Vaswani 等人在 2017 年提出，论文题为 Attention Is All You Need。",
        "Self-Attention 机制计算序列中所有位置之间的相关性权重。",
        "Multi-Head Attention 在不同的子空间中并行计算注意力。",
        "GPT 是基于 Transformer Decoder 的自回归生成模型。",
        "BERT 是基于 Transformer Encoder 的双向理解模型。",
        "LoRA 通过低秩分解大幅降低微调所需的参数量。",
        "RAG 将外部知识检索与大模型生成相结合，减少幻觉。",
        "RLHF 通过人类偏好反馈来对齐大模型的输出。",
    ]

    metadatas = [
        {"topic": "architecture", "year": 2017},
        {"topic": "attention", "year": 2017},
        {"topic": "attention", "year": 2017},
        {"topic": "model", "year": 2018},
        {"topic": "model", "year": 2018},
        {"topic": "fine-tuning", "year": 2021},
        {"topic": "application", "year": 2020},
        {"topic": "alignment", "year": 2022},
    ]

    # 用 Ollama Embedding 模型
    embeddings_model = OllamaEmbeddings(model=EMBED_MODEL)
    vectors = embeddings_model.embed_documents(documents)

    collection.add(
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))],
    )

    print(f"  ✅ 已添加 {collection.count()} 条文档到 ChromaDB")

    # ---- 查询 ----
    queries = [
        "注意力机制的原理",
        "怎么微调大模型",
    ]

    for query in queries:
        query_vec = embeddings_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
        )
        print(f"\n  🔍 查询: '{query}'")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"    {i+1}. [距离={dist:.4f}] {doc[:60]}...")

    # ---- 带元数据过滤的查询 ----
    print(f"\n  🔍 查询 + 元数据过滤 (只看 attention 相关):")
    query_vec = embeddings_model.embed_query("Transformer 的核心组件")
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        where={"topic": "attention"},  # 元数据过滤!
    )
    for i, doc in enumerate(results["documents"][0]):
        print(f"    {i+1}. {doc[:60]}...")

    print()
    print("  💡 ChromaDB vs 手写的优势:")
    print("    - 自动持久化 (不用每次重新 Embedding)")
    print("    - ANN 索引 (大数据集下比暴力搜索快 100x)")
    print("    - 元数据过滤 (按标签、日期等筛选)")
    print("    - CRUD 操作 (增删改查)")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 1: Embedding 与向量检索的本质\n")

    part1_raw_embedding()
    part2_similarity()
    part3_manual_search()
    part4_chromadb()

    print("\n" + "=" * 60)
    print("✅ Level 1 完成！")
    print()
    print("关键收获:")
    print("  1. Embedding = 文本 → 固定长度浮点向量")
    print("  2. 语义相似 → 向量距离近 (余弦相似度高)")
    print("  3. 向量检索 = Embedding + 相似度计算 + Top-K")
    print("  4. ChromaDB 封装了存储、索引、过滤等功能")
    print("  5. 这是 RAG 中 'R' (Retrieval) 的基础!")
    print()
    print("👉 下一步: python rag/02_naive_rag.py")
    print("=" * 60)
