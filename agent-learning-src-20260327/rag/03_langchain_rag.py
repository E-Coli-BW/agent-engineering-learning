"""
Level 3: LangChain RAG Pipeline —— 标准工程实现
=================================================

Level 2 手写 RAG 让你理解了原理，但有几个问题:
  1. 分块策略太简单 (固定大小，不尊重语义边界)
  2. 向量检索是暴力搜索 O(n)
  3. 没有持久化 (重启就丢了)
  4. 代码耦合度高

LangChain 提供了标准化的 RAG 组件:
  - Document Loaders: 加载各种格式 (PDF, Markdown, HTML, CSV...)
  - Text Splitters: 智能分块 (递归分割，尊重边界)
  - Vector Stores: 对接各种向量数据库 (Chroma, FAISS, Pinecone...)
  - Retrievers: 统一的检索接口
  - Chains: 把检索和生成串起来

本文件用 LangChain 构建一个标准的 RAG Pipeline，
以我们项目自身的 .py 和 .md 文件作为知识源。
"""

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen2.5:7b"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Part 1: Document Loading —— 加载项目文件
# ============================================================
def load_project_documents() -> list[Document]:
    """
    加载项目中的 .py 和 .md 文件作为知识源。

    LangChain 的 Document 对象:
      - page_content: 文本内容
      - metadata: 元数据 (来源、页码、标题等)

    在真实项目中，你会用:
      - PyPDFLoader: 加载 PDF
      - UnstructuredMarkdownLoader: 加载 Markdown
      - WebBaseLoader: 加载网页
      - CSVLoader: 加载 CSV
    """
    print("📂 加载项目文档...")

    documents = []
    target_files = [
        "multi_head_attention.py",
        "char_transformer.py",
        "README.md",
        "JD_GAP_ANALYSIS.md",
        "agent/01_chat_basics.py",
        "agent/02_tool_calling.py",
        "agent/03_react_agent.py",
        "agent/04_langgraph_agent.py",
    ]

    for filename in target_files:
        filepath = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "type": "python" if filename.endswith(".py") else "markdown",
                    "chars": len(content),
                }
            ))
            print(f"  ✅ {filename} ({len(content):,} 字符)")
        else:
            print(f"  ⚠️ {filename} 不存在，跳过")

    print(f"\n  共加载 {len(documents)} 个文件")
    return documents


# ============================================================
# Part 2: Text Splitting —— 智能分块
# ============================================================
def split_documents(documents: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter 是 LangChain 最常用的分块器。

    原理 (面试会问!):
      递归地按 分隔符列表 切分文本:
        1. 先尝试按 "\\n\\n" (段落) 切分
        2. 如果块还是太大，按 "\\n" (行) 切分
        3. 如果还是太大，按 " " (空格) 切分
        4. 最后按字符切分

      这样能尽量保持语义完整性:
        段落 > 行 > 词 > 字符

    参数:
      - chunk_size: 每块的最大字符数
      - chunk_overlap: 块之间的重叠字符数 (保持上下文连贯)
      - separators: 分隔符列表 (按优先级)
    """
    print("\n📝 文本分块...")

    # Python 文件用专门的分隔符
    python_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\nclass ",        # 优先在类定义处切分
            "\ndef ",          # 其次在函数定义处切分
            "\n\n",            # 空行
            "\n",              # 换行
            " ",               # 空格
        ],
    )

    # Markdown 文件用标题分隔
    markdown_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\n## ",           # 二级标题
            "\n### ",          # 三级标题
            "\n\n",
            "\n",
            " ",
        ],
    )

    all_chunks = []
    for doc in documents:
        if doc.metadata["type"] == "python":
            chunks = python_splitter.split_documents([doc])
        else:
            chunks = markdown_splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"  分块完成: {len(documents)} 个文件 → {len(all_chunks)} 个块")
    print(f"  平均块大小: {sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} 字符")

    # 展示几个分块示例
    print(f"\n  📋 分块示例:")
    for i in range(min(3, len(all_chunks))):
        chunk = all_chunks[i]
        print(f"    块 {i+1} [{chunk.metadata['source']}]: {chunk.page_content[:80]}...")

    return all_chunks


# ============================================================
# Part 3: Vector Store —— 向量化 + 存储
# ============================================================
def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    用 ChromaDB 存储向量。

    LangChain 的 Chroma 封装做了:
      1. 自动调 Embedding 模型把文本转向量
      2. 存入 ChromaDB (可以选内存或磁盘持久化)
      3. 提供统一的检索接口

    对比 Level 1 的手写版:
      手写: 你要自己调 embed_documents + 存 numpy array
      LangChain: 一行代码搞定
    """
    print("\n📦 构建向量存储...")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 持久化到磁盘 (下次启动不用重新 Embedding)
    persist_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="project_knowledge",
    )

    print(f"  ✅ 向量存储完成，共 {vector_store._collection.count()} 个向量")
    print(f"  💾 持久化到: {persist_dir}")

    return vector_store


# ============================================================
# Part 4: RAG Chain —— 完整的检索+生成链
# ============================================================
def create_rag_chain(vector_store: Chroma):
    """
    用 LangChain LCEL (LangChain Expression Language) 构建 RAG Chain。

    LCEL 是什么？
      一种管道式的表达方式，用 | 连接各个组件:
        chain = retriever | prompt | llm | parser

      类似 Unix 管道:
        cat file | grep "error" | sort | head
    """
    print("\n🔗 构建 RAG Chain...")

    # ---- Retriever ----
    retriever = vector_store.as_retriever(
        search_type="similarity",       # 相似度搜索 (也可以用 mmr)
        search_kwargs={"k": 4},          # 返回 Top-4
    )

    # ---- Prompt Template ----
    template = """你是一个技术助手，根据提供的上下文信息回答问题。

上下文信息:
{context}

问题: {question}

要求:
1. 只根据上下文中的信息回答
2. 如果上下文中没有相关信息，明确说 "根据现有资料，我无法回答这个问题"
3. 在回答中标注信息来源文件
4. 使用中文回答，技术术语可以保留英文"""

    prompt = ChatPromptTemplate.from_template(template)

    # ---- LLM ----
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # ---- 辅助函数: 把检索到的 Document 列表格式化为字符串 ----
    def format_docs(docs: list[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[来源 {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    # ---- 组装 RAG Chain (LCEL 管道) ----
    rag_chain = (
        {
            "context": retriever | format_docs,   # 检索 + 格式化
            "question": RunnablePassthrough(),     # 原样传递问题
        }
        | prompt      # 填入模板
        | llm         # 调用 LLM
        | StrOutputParser()  # 提取字符串
    )

    print("  ✅ RAG Chain 构建完成")
    print("  📊 Pipeline: Query → Retriever → Prompt → LLM → Answer")

    return rag_chain, retriever


# ============================================================
# Part 5: 测试
# ============================================================
def test_rag(rag_chain, retriever):
    """测试 RAG 系统"""
    print("\n" + "=" * 60)
    print("🧪 RAG 系统测试")
    print("=" * 60)

    questions = [
        "为什么 Attention 要除以根号 d_k？",
        "Mini-GPT 的模型架构是什么样的？包含哪些组件？",
        "LangGraph 和 LangChain 有什么区别？",
        "ReAct Agent 的工作流程是怎样的？",
        "这个项目的 JD 差距分析结论是什么？",
    ]

    for q in questions:
        print(f"\n{'─'*50}")
        print(f"🧑 问题: {q}")

        # 先展示检索到了什么
        docs = retriever.invoke(q)
        print(f"\n  📄 检索结果 ({len(docs)} 个片段):")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            print(f"    {i}. [{source}] {doc.page_content[:60]}...")

        # RAG 回答
        answer = rag_chain.invoke(q)
        print(f"\n  🤖 RAG 回答:")
        print(f"  {answer[:400]}")


# ============================================================
# Part 6: 检索策略对比 (面试题!)
# ============================================================
def compare_retrieval_strategies(vector_store: Chroma):
    """
    面试常问: 你用过哪些检索策略？各有什么优缺点？

    1. Similarity Search: 最基本的余弦相似度搜索
    2. MMR (Maximum Marginal Relevance): 平衡相关性和多样性
       - 避免检索到多个高度重复的文档
       - 公式: score = λ * sim(q, doc) - (1-λ) * max(sim(doc, selected_docs))
    3. Similarity Score Threshold: 只返回相似度超过阈值的文档
    """
    print("\n" + "=" * 60)
    print("📊 检索策略对比 (面试重点)")
    print("=" * 60)

    query = "Transformer 的核心组件有哪些？"
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Strategy 1: 基本相似度搜索
    print(f"\n  🔍 Query: '{query}'")

    print(f"\n  📌 策略 1: Similarity Search (Top-4)")
    retriever_sim = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    docs = retriever_sim.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"    {i}. [{doc.metadata.get('source', '?')}] {doc.page_content[:50]}...")

    # Strategy 2: MMR (多样性)
    print(f"\n  📌 策略 2: MMR (Maximal Marginal Relevance)")
    retriever_mmr = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever_mmr.invoke(query)
    for i, doc in enumerate(docs, 1):
        print(f"    {i}. [{doc.metadata.get('source', '?')}] {doc.page_content[:50]}...")

    print(f"""
  💡 面试答题要点:
    - Similarity: 简单直接，但可能返回重复内容
    - MMR: 兼顾相关性和多样性，适合答案需要多角度信息的场景
    - Threshold: 控制最低质量，避免灌入不相关内容
    - Hybrid: 结合关键词搜索 (BM25) + 向量搜索，互补优势
    - Reranking: 先粗检索大量候选，再用精排模型重排序
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 3: LangChain RAG Pipeline\n")

    # Step 1: 加载文档
    documents = load_project_documents()

    # Step 2: 分块
    chunks = split_documents(documents)

    # Step 3: 向量化 + 存储
    vector_store = create_vector_store(chunks)

    # Step 4: 构建 RAG Chain
    rag_chain, retriever = create_rag_chain(vector_store)

    # Step 5: 测试
    test_rag(rag_chain, retriever)

    # Step 6: 策略对比
    compare_retrieval_strategies(vector_store)

    print("\n" + "=" * 60)
    print("✅ Level 3 完成！")
    print()
    print("关键收获:")
    print("  1. LangChain 标准化了 RAG 的每个组件")
    print("  2. RecursiveCharacterTextSplitter 按语义边界分块")
    print("  3. Chroma 提供持久化向量存储 + 高效检索")
    print("  4. LCEL 管道语法: retriever | prompt | llm | parser")
    print("  5. 不同检索策略的取舍 (Similarity vs MMR vs Hybrid)")
    print()
    print("👉 下一步: python rag/04_advanced_rag.py")
    print("=" * 60)
