"""
Level 4: 高级 RAG 技巧 —— 面试加分项
=======================================

基础 RAG 的问题 (面试必问!):
  1. 查询和文档的语义不匹配 (用户问法 ≠ 文档表述)
  2. 检索到的文档可能不相关 → 污染生成
  3. 单次检索可能不够，需要多步推理
  4. 无法评估 RAG 系统的好坏

本文件覆盖 4 个高级技巧:
  1. Multi-Query RAG: 改写查询，提高召回率
  2. Self-RAG: 自我判断检索结果是否足够
  3. Corrective RAG: 用 LangGraph 编排 RAG + 自纠正
  4. RAG 评估指标

这些都是面试高频考点，也是实际项目中的核心优化方向。
"""

import os
import operator
from typing import Annotated, TypedDict

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen2.5:7b"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_vector_store() -> Chroma:
    """获取或创建向量存储"""
    persist_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="project_knowledge",
        )
    else:
        print("⚠️ 向量库不存在，请先运行 rag/03_langchain_rag.py 构建索引")
        # 快速创建一个小索引
        docs = [
            Document(page_content="Transformer 使用 Self-Attention 机制计算序列元素间的关系。", metadata={"source": "demo"}),
            Document(page_content="Multi-Head Attention 在不同子空间并行计算注意力。", metadata={"source": "demo"}),
            Document(page_content="GPT 是基于 Transformer Decoder 的自回归模型。", metadata={"source": "demo"}),
            Document(page_content="RAG 通过检索外部知识来增强大模型的回答能力。", metadata={"source": "demo"}),
            Document(page_content="LoRA 通过低秩矩阵分解实现参数高效微调。", metadata={"source": "demo"}),
            Document(page_content="ReAct Agent 通过 Thought-Action-Observation 循环解决问题。", metadata={"source": "demo"}),
            Document(page_content="LangGraph 将 Agent 工作流建模为有向图。", metadata={"source": "demo"}),
            Document(page_content="Causal Mask 是一个下三角矩阵，防止自回归模型看到未来 token。", metadata={"source": "demo"}),
        ]
        return Chroma.from_documents(docs, embeddings, persist_directory=persist_dir, collection_name="project_knowledge")


# ============================================================
# 技巧 1: Multi-Query RAG
# ============================================================
def technique1_multi_query():
    """
    问题: 用户的 query 表述方式可能和文档不匹配。

    例如: 用户问 "为什么要做缩放？"
          文档写的是 "除以 sqrt(d_k) 防止梯度消失"
          → 直接检索可能找不到最相关的文档

    解决: 让 LLM 生成多个不同表述的查询，分别检索，合并结果。

    原始 query: "为什么要做缩放？"
    改写后:
      1. "Attention 机制中除以 sqrt(d_k) 的原因是什么？"
      2. "Scaled Dot-Product 中缩放因子的作用"
      3. "不做缩放会有什么问题？"

    每个改写分别检索 → 合并去重 → 更高的召回率
    """
    print("=" * 60)
    print("技巧 1: Multi-Query RAG (查询改写)")
    print("=" * 60)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.3)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # ---- Step 1: 查询改写 ----
    rewrite_prompt = ChatPromptTemplate.from_template(
        """你是一个查询改写助手。给定一个用户问题，请生成 3 个不同表述但含义相同的查询。
每个查询独占一行，不要编号，不要其他说明。

原始问题: {question}

改写后的查询:"""
    )

    original_query = "为什么 Attention 要做缩放？"
    print(f"\n  原始查询: '{original_query}'")

    response = llm.invoke(rewrite_prompt.format(question=original_query))
    alternative_queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    alternative_queries = alternative_queries[:3]  # 最多取 3 个

    print(f"  改写后:")
    for i, q in enumerate(alternative_queries, 1):
        print(f"    {i}. {q}")

    # ---- Step 2: 多查询检索 + 合并 ----
    all_queries = [original_query] + alternative_queries
    all_docs = []
    seen_contents = set()

    for query in all_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            # 去重
            content_key = doc.page_content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_docs.append(doc)

    print(f"\n  📄 合并去重后: {len(all_docs)} 个唯一片段")
    for i, doc in enumerate(all_docs[:5], 1):
        print(f"    {i}. {doc.page_content[:60]}...")

    # ---- Step 3: 生成回答 ----
    context = "\n\n".join([doc.page_content for doc in all_docs[:5]])
    answer_prompt = f"""根据以下资料回答问题。

资料:
{context}

问题: {original_query}

用中文简洁回答:"""

    answer = llm.invoke([HumanMessage(content=answer_prompt)])
    print(f"\n  🤖 回答: {answer.content[:300]}")

    print(f"""
  💡 Multi-Query 的价值:
    - 提高召回率 (不同表述命中不同文档)
    - 减少因用词不匹配导致的漏检
    - 成本: 多几次 LLM 调用 (改写) + 多几次检索
    """)


# ============================================================
# 技巧 2: Corrective RAG (用 LangGraph 编排)
# ============================================================
def technique2_corrective_rag():
    """
    Corrective RAG (CRAG): 检索后先评估质量，不好就重新检索或直接回答。

    流程:
      Query → 检索 → 评估相关性 → [相关?]
                                     ├── 是 → 用检索结果生成回答
                                     └── 否 → 提示用户或用 LLM 直接回答

    这是 Self-RAG / CRAG 论文的核心思想，用 LangGraph 来实现。
    """
    print("\n" + "=" * 60)
    print("技巧 2: Corrective RAG (LangGraph 编排)")
    print("=" * 60)

    from langgraph.graph import StateGraph, END, START

    # ---- State 定义 ----
    class RAGState(TypedDict):
        question: str
        documents: list[str]
        relevance: str          # "relevant" or "irrelevant"
        answer: str

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ---- Node 1: 检索 ----
    def retrieve_node(state: RAGState) -> dict:
        print(f"    📥 检索: '{state['question']}'")
        docs = retriever.invoke(state["question"])
        doc_texts = [doc.page_content for doc in docs]
        print(f"    📄 找到 {len(doc_texts)} 个片段")
        return {"documents": doc_texts}

    # ---- Node 2: 评估相关性 ----
    def grade_node(state: RAGState) -> dict:
        """让 LLM 判断检索结果是否和问题相关"""
        docs_text = "\n---\n".join(state["documents"][:3])
        grade_prompt = f"""判断以下检索结果是否与问题相关。

问题: {state['question']}

检索结果:
{docs_text}

只回答 "relevant" 或 "irrelevant"，不要其他文字。"""

        response = llm.invoke([HumanMessage(content=grade_prompt)])
        relevance = "relevant" if "relevant" in response.content.lower() else "irrelevant"
        print(f"    📊 相关性评估: {relevance}")
        return {"relevance": relevance}

    # ---- Node 3a: 有相关文档 → 生成回答 ----
    def generate_with_docs(state: RAGState) -> dict:
        context = "\n\n".join(state["documents"])
        prompt = f"""根据以下资料回答问题。只根据资料回答，不要编造。

资料:
{context}

问题: {state['question']}

简洁中文回答:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"    ✅ 基于检索结果生成回答")
        return {"answer": response.content}

    # ---- Node 3b: 无相关文档 → 直接回答 (带声明) ----
    def generate_without_docs(state: RAGState) -> dict:
        prompt = f"""回答以下问题。注意：知识库中没有找到直接相关的资料，请基于你的通用知识回答，并声明这不是基于特定知识库的回答。

问题: {state['question']}

简洁中文回答:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"    ⚠️ 知识库无相关内容，使用通用知识回答")
        return {"answer": f"[注: 知识库未找到相关内容] {response.content}"}

    # ---- 路由 ----
    def route_by_relevance(state: RAGState) -> str:
        return "generate_with_docs" if state["relevance"] == "relevant" else "generate_without_docs"

    # ---- 构建图 ----
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate_with_docs", generate_with_docs)
    graph.add_node("generate_without_docs", generate_without_docs)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_by_relevance, {
        "generate_with_docs": "generate_with_docs",
        "generate_without_docs": "generate_without_docs",
    })
    graph.add_edge("generate_with_docs", END)
    graph.add_edge("generate_without_docs", END)

    app = graph.compile()

    # ---- 测试 ----
    print("\n  📊 图结构:")
    print("    Query → Retrieve → Grade → [Relevant?]")
    print("                                  ├── Yes → Generate (with docs) → END")
    print("                                  └── No  → Generate (no docs)   → END")

    test_questions = [
        "什么是 Multi-Head Attention？",     # 知识库有
        "如何做模型量化？",                    # 知识库可能没有
    ]

    for q in test_questions:
        print(f"\n  {'─'*40}")
        print(f"  🧑 问题: {q}")
        result = app.invoke({"question": q})
        print(f"  🤖 回答: {result['answer'][:200]}")


# ============================================================
# 技巧 3: RAG 评估
# ============================================================
def technique3_evaluation():
    """
    RAG 系统如何评估？(面试必问!)

    三大评估维度:
      1. 检索质量 (Retrieval Quality)
         - 准确率 (Precision): 检索到的文档中有多少是相关的
         - 召回率 (Recall): 相关文档中有多少被检索到了
         - MRR (Mean Reciprocal Rank): 第一个相关结果的排名

      2. 生成质量 (Generation Quality)
         - 忠实度 (Faithfulness): 回答是否忠于检索到的文档
         - 相关性 (Relevancy): 回答是否回答了用户的问题
         - 无幻觉 (No Hallucination): 是否编造了文档中没有的信息

      3. 端到端质量 (End-to-End)
         - 用户满意度
         - 正确率 (和标准答案对比)

    常用评估框架: RAGAS, TruLens, DeepEval
    这里用 LLM-as-Judge 的方式手写评估 (面试中最实用的方法)
    """
    print("\n" + "=" * 60)
    print("技巧 3: RAG 评估 (面试必考)")
    print("=" * 60)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # ---- 模拟一组 RAG 输出 ----
    test_cases = [
        {
            "question": "什么是 Causal Mask？",
            "retrieved_docs": "Causal Mask 是一个下三角矩阵，防止自回归模型看到未来 token。上三角部分设为 -inf，softmax 后变为 0。",
            "generated_answer": "Causal Mask（因果遮罩）是一个下三角矩阵，在自回归模型（如GPT）中使用。它通过将注意力矩阵的上三角部分设置为负无穷，经过softmax后变为0，从而防止模型在生成第i个token时看到第i+1及之后的token。",
        },
        {
            "question": "LoRA 的参数量是多少？",
            "retrieved_docs": "LoRA 通过低秩矩阵分解实现参数高效微调。",
            "generated_answer": "LoRA 通过低秩分解将权重矩阵分解为两个小矩阵的乘积 W = BA，其中 B 是 d×r，A 是 r×d，参数量从 d² 降低到 2dr。通常 r=8 或 r=16，参数量仅为原始的 0.1%-1%。",
        },
    ]

    for tc in test_cases:
        print(f"\n  {'─'*40}")
        print(f"  📝 问题: {tc['question']}")

        # ---- 评估 1: 忠实度 (Faithfulness) ----
        faithfulness_prompt = f"""评估以下回答是否忠实于参考资料。回答中的每个事实是否都能在参考资料中找到依据？

参考资料: {tc['retrieved_docs']}

回答: {tc['generated_answer']}

请给出评分 (1-5) 和理由。格式:
分数: X
理由: ...
"""
        faith_result = llm.invoke([HumanMessage(content=faithfulness_prompt)])
        print(f"\n  📊 忠实度评估:")
        print(f"    {faith_result.content[:200]}")

        # ---- 评估 2: 相关性 (Relevancy) ----
        relevancy_prompt = f"""评估以下回答是否充分回答了用户的问题。

用户问题: {tc['question']}

回答: {tc['generated_answer']}

请给出评分 (1-5) 和理由。格式:
分数: X
理由: ...
"""
        rel_result = llm.invoke([HumanMessage(content=relevancy_prompt)])
        print(f"\n  📊 相关性评估:")
        print(f"    {rel_result.content[:200]}")

    print(f"""
  💡 面试答题要点:
    1. 评估维度: 检索质量 + 生成质量 + 端到端
    2. 核心指标: Faithfulness (忠实度) + Relevancy (相关性)
    3. LLM-as-Judge: 用 LLM 做评估，成本低，灵活
    4. 专业工具: RAGAS (最流行)、DeepEval、TruLens
    5. 注意: 评估模型不应该和生成模型是同一个 (防止自我偏好)
    """)


# ============================================================
# 总结: RAG 优化全景图
# ============================================================
def summary():
    print("\n" + "=" * 60)
    print("📊 RAG 优化全景图 (面试救命图)")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │                    RAG 优化方向                          │
    ├──────────────┬───────────────────────────────────────────┤
    │ 阶段         │ 优化技巧                                  │
    ├──────────────┼───────────────────────────────────────────┤
    │              │ · 选择合适的 Embedding 模型               │
    │ 索引阶段     │ · 分块策略 (大小、重叠、边界)              │
    │ (Indexing)   │ · 元数据提取 (标题、时间、类别)            │
    │              │ · 多粒度索引 (摘要索引 + 详情索引)         │
    ├──────────────┼───────────────────────────────────────────┤
    │              │ · Multi-Query: 改写查询提高召回            │
    │ 检索阶段     │ · Hybrid Search: BM25 + 向量检索           │
    │ (Retrieval)  │ · Reranking: 粗检索 + 精排序              │
    │              │ · MMR: 平衡相关性和多样性                  │
    ├──────────────┼───────────────────────────────────────────┤
    │              │ · Prompt 工程 (引用要求、格式约束)         │
    │ 生成阶段     │ · Self-RAG: 自我判断是否需要检索           │
    │ (Generation) │ · CRAG: 纠正不相关的检索结果              │
    │              │ · 引用溯源: 标注答案来自哪个文档           │
    ├──────────────┼───────────────────────────────────────────┤
    │              │ · Faithfulness: 回答是否忠于证据           │
    │ 评估阶段     │ · Relevancy: 回答是否切题                  │
    │ (Evaluation) │ · LLM-as-Judge / RAGAS                   │
    │              │ · A/B Testing: 线上对比                   │
    └──────────────┴───────────────────────────────────────────┘

    面试关键结论:
    "基础 RAG 解决了有无问题，高级 RAG 解决了好坏问题。
     优化的核心是在检索质量和生成质量之间找到平衡。"
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 4: 高级 RAG 技巧\n")

    technique1_multi_query()
    technique2_corrective_rag()
    technique3_evaluation()
    summary()

    print("\n" + "=" * 60)
    print("✅ Level 4 完成！RAG 模块全部学完！")
    print()
    print("📊 你现在的技能树:")
    print("  ✅ Transformer 原理 (multi_head_attention.py)")
    print("  ✅ 模型训练 + 可视化 (char_transformer.py)")
    print("  ✅ Agent 开发 (agent/01-04)")
    print("  ✅ RAG 系统 (rag/01-04)")
    print("  🔲 LoRA 微调 (下一阶段)")
    print("=" * 60)
