"""
Level 2: 手写最简 RAG —— 不用框架，看清本质
=============================================

RAG (Retrieval-Augmented Generation) 到底在干什么？

本质上就 3 步:
  1. Retrieve: 从知识库中检索和问题相关的文档片段
  2. Augment:  把检索到的内容塞进 prompt 里
  3. Generate: 让 LLM 根据这些上下文生成回答

就这么简单！所有的 RAG 框架都是在优化这 3 步。

为什么需要 RAG？
  1. LLM 的知识有截止日期 → 无法回答最新信息
  2. LLM 会幻觉 (编造不存在的事实) → 给它证据就不容易瞎编
  3. LLM 不知道你的私有数据 → 喂给它你的文档就能回答
  4. Context Window 有限 → 不可能把所有文档都塞进去，需要先检索

RAG vs Fine-tuning (面试高频题!):
  ┌──────────────┬───────────────────┬───────────────────┐
  │              │ RAG               │ Fine-tuning       │
  ├──────────────┼───────────────────┼───────────────────┤
  │ 知识更新     │ 实时 (换文档即可)  │ 需要重新训练      │
  │ 成本         │ 低 (不改模型)      │ 高 (需要 GPU)     │
  │ 幻觉控制     │ 好 (有证据引用)    │ 一般              │
  │ 深度理解     │ 一般              │ 好 (内化了知识)    │
  │ 适用场景     │ 知识问答/文档检索  │ 风格/领域适配     │
  └──────────────┴───────────────────┴───────────────────┘
"""

import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen2.5:7b"


# ============================================================
# 1. 准备知识库 (模拟真实文档)
# ============================================================

# 用我们项目自身的知识做文档! 这样你能直观看到检索效果
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "Scaled Dot-Product Attention",
        "content": """Scaled Dot-Product Attention 是 Transformer 的核心计算单元。
计算公式为 Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V。
其中 Q 是查询矩阵，K 是键矩阵，V 是值矩阵。
除以 sqrt(d_k) 是为了防止点积过大导致 softmax 梯度消失。
当 d_k 很大时，点积的方差也会很大，softmax 的输出会接近 one-hot 分布。""",
        "source": "multi_head_attention.py",
    },
    {
        "id": "doc_2",
        "title": "Multi-Head Attention",
        "content": """Multi-Head Attention 将 Q、K、V 分成多个头，每个头独立计算注意力。
每个头的维度 d_k = d_model / num_heads。
多个头可以让模型在不同的子空间中学习不同的注意力模式。
例如某些头关注语法关系，某些头关注语义关系。
最后将所有头的输出拼接起来，再通过一个线性层 W_o 进行投影。
参数量为 4 * (d_model^2 + d_model)，包含 Q、K、V、O 四个线性层。""",
        "source": "multi_head_attention.py",
    },
    {
        "id": "doc_3",
        "title": "Causal Mask",
        "content": """Causal Mask (因果遮罩) 用于自回归模型（如 GPT），
防止模型在预测第 i 个 token 时看到第 i+1 及之后的 token。
实现方式是一个下三角矩阵，上三角部分设为 -inf，
softmax 后变为 0，从而屏蔽了未来信息。
这保证了模型只能从左到右顺序生成，每个位置只能看到之前的内容。""",
        "source": "multi_head_attention.py",
    },
    {
        "id": "doc_4",
        "title": "Mini-GPT 模型架构",
        "content": """Mini-GPT 是一个字符级语言模型，包含以下组件：
1. Token Embedding: 将字符映射为向量
2. Position Embedding: 提供位置信息
3. N 层 Transformer Block (每层包含 CausalSelfAttention + FFN + LayerNorm)
4. 使用 Pre-Norm 架构 (GPT-2 风格)，先 LayerNorm 再做 attention
5. 残差连接让梯度能跳过层直接回传
6. 最终通过 lm_head 线性层输出下一个字符的概率分布""",
        "source": "char_transformer.py",
    },
    {
        "id": "doc_5",
        "title": "训练过程",
        "content": """模型训练使用 Next Token Prediction 目标：
对于输入序列 [T, o, _, b, e]，目标序列为 [o, _, b, e, ,]。
损失函数为 CrossEntropyLoss。
使用 AdamW 优化器，学习率 3e-4。
每 100 步评估一次训练集和验证集的 loss。
使用梯度裁剪 (clip_grad_norm) 防止梯度爆炸。
训练 3000 步后，模型能生成具有莎士比亚风格的文本。""",
        "source": "char_transformer.py",
    },
    {
        "id": "doc_6",
        "title": "ReAct Agent",
        "content": """ReAct (Reasoning + Acting) 是一种 Agent 框架，
核心循环为: Thought(思考) → Action(行动) → Observation(观察)。
LLM 先分析问题需要什么信息，然后调用工具获取结果，
再根据结果继续推理或给出最终答案。
关键设计: 最大步数限制防止无限循环，错误处理确保鲁棒性。""",
        "source": "agent/03_react_agent.py",
    },
    {
        "id": "doc_7",
        "title": "LangGraph 工作流",
        "content": """LangGraph 将 Agent 工作流建模为有向图:
- Node (节点): 处理步骤，如调用 LLM、执行工具
- Edge (边): 节点间的流转路径
- State (状态): 在图中传递的共享数据
- Conditional Edge: 根据条件选择不同路径
相比手写循环，LangGraph 提供了更好的可维护性和扩展性。""",
        "source": "agent/04_langgraph_agent.py",
    },
    {
        "id": "doc_8",
        "title": "Tool Calling 原理",
        "content": """工具调用的本质: 让 LLM 输出结构化 JSON，程序解析并执行。
过程: 1) 把工具描述塞进 prompt 2) LLM 输出工具调用 JSON
3) 程序执行对应函数 4) 把结果喂回 LLM 5) LLM 生成最终回答。
LLM 本身不执行任何代码，它只是在"建议"调用哪个工具。
常见问题: 选错工具、参数错误、幻觉、执行失败。""",
        "source": "agent/02_tool_calling.py",
    },
]


# ============================================================
# 2. 文本分块 (Chunking) —— 手写实现
# ============================================================
def simple_chunk(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    最简单的文本分块: 固定大小 + 重叠。

    为什么需要分块?
      - Embedding 模型有最大长度限制 (通常 512 tokens)
      - 太长的文档检索精度下降 (信息被稀释)
      - 太短的块缺乏上下文 → 需要重叠来保持连贯性

    分块策略 (面试题):
      1. 固定大小 + 重叠 (最简单，本实现)
      2. 按句子分块 (尊重句子边界)
      3. 按段落分块 (尊重段落结构)
      4. 递归分块 (RecursiveCharacterTextSplitter，LangChain 默认)
      5. 语义分块 (按语义相似度切分，最先进)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 跳过空块
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ============================================================
# 3. 手写 RAG Pipeline
# ============================================================
class NaiveRAG:
    """
    最简单的 RAG 实现，不用任何框架。

    让你看清 RAG 的每一步到底在做什么:
      1. Index (离线): 文档 → 分块 → Embedding → 存储
      2. Retrieve (在线): Query → Embedding → 相似度搜索 → Top-K 文档
      3. Generate (在线): [Query + 检索到的文档] → LLM → 回答
    """

    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        self.llm = ChatOllama(model=CHAT_MODEL, temperature=0)

        # 存储
        self.chunks: list[dict] = []      # 原始文本块
        self.vectors: np.ndarray = None   # 向量矩阵

    def index(self, documents: list[dict]):
        """
        索引阶段 (离线处理):
          文档 → 分块 → Embedding → 存入内存
        """
        print("  📦 索引阶段:")

        # Step 1: 分块
        all_chunks = []
        for doc in documents:
            chunks = simple_chunk(doc["content"], chunk_size=200, overlap=50)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "title": doc["title"],
                    "source": doc["source"],
                    "doc_id": doc["id"],
                })
        self.chunks = all_chunks
        print(f"    分块完成: {len(documents)} 篇文档 → {len(all_chunks)} 个块")

        # Step 2: Embedding
        texts = [c["text"] for c in all_chunks]
        self.vectors = np.array(self.embeddings.embed_documents(texts))
        print(f"    Embedding 完成: 向量矩阵 {self.vectors.shape}")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        检索阶段:
          Query → Embedding → 余弦相似度 → Top-K
        """
        query_vec = np.array(self.embeddings.embed_query(query))

        # 计算相似度
        similarities = []
        for i, doc_vec in enumerate(self.vectors):
            sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((sim, i))

        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = sim
            results.append(chunk)

        return results

    def generate(self, query: str, context_docs: list[dict]) -> str:
        """
        生成阶段:
          把检索到的文档塞进 prompt，让 LLM 生成回答。

        这个 prompt 的设计是 RAG 的关键！
        """
        # 拼接检索到的上下文
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"[来源 {i}: {doc['title']} ({doc['source']})]\n{doc['text']}"
            )
        context_str = "\n\n".join(context_parts)

        # 构造 RAG prompt
        system_prompt = """你是一个严谨的技术助手。请根据提供的参考资料回答问题。

重要规则:
1. 只根据参考资料中的信息回答，不要编造
2. 如果参考资料中没有相关信息，请明确说明
3. 在回答中引用来源 (如 [来源 1])
4. 使用中文回答"""

        user_prompt = f"""参考资料:
{context_str}

问题: {query}

请根据以上参考资料回答问题。"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        return response.content

    def query(self, question: str, top_k: int = 3) -> str:
        """完整的 RAG 查询流程"""
        print(f"\n  🔍 检索中...")
        docs = self.retrieve(question, top_k=top_k)

        print(f"  📄 检索到 {len(docs)} 个相关片段:")
        for i, doc in enumerate(docs, 1):
            print(f"    {i}. [{doc['similarity']:.4f}] {doc['title']} → {doc['text'][:50]}...")

        print(f"  🤖 生成回答...")
        answer = self.generate(question, docs)

        return answer


# ============================================================
# 4. 对比: 有 RAG vs 没有 RAG
# ============================================================
def compare_with_without_rag():
    """
    对比有无 RAG 的效果差异。

    没有 RAG: LLM 只能靠自身知识回答 (可能幻觉)
    有 RAG: LLM 根据检索到的文档回答 (有据可查)
    """
    print("=" * 60)
    print("对比: 有 RAG vs 没有 RAG")
    print("=" * 60)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # 建索引
    rag = NaiveRAG()
    rag.index(DOCUMENTS)

    # 测试问题 (关于我们项目的细节，LLM 自身不可能知道)
    questions = [
        "为什么 Attention 要除以 sqrt(d_k)？",
        "Mini-GPT 使用什么架构风格？Pre-Norm 还是 Post-Norm？",
        "ReAct Agent 的核心循环是什么？",
    ]

    for q in questions:
        print(f"\n{'─'*50}")
        print(f"🧑 问题: {q}")

        # 无 RAG
        print(f"\n  ❌ 无 RAG (纯 LLM):")
        response_no_rag = llm.invoke([HumanMessage(content=q)])
        print(f"  {response_no_rag.content[:200]}...")

        # 有 RAG
        print(f"\n  ✅ 有 RAG:")
        answer_rag = rag.query(q, top_k=3)
        print(f"  {answer_rag[:300]}...")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 Level 2: 手写最简 RAG\n")

    compare_with_without_rag()

    print("\n" + "=" * 60)
    print("✅ Level 2 完成！")
    print()
    print("关键收获:")
    print("  1. RAG = Retrieve (检索) + Augment (增强) + Generate (生成)")
    print("  2. 核心是把检索到的文档塞进 LLM 的 prompt 里")
    print("  3. 分块策略影响检索精度 (块大小、重叠、切分方式)")
    print("  4. Prompt 设计是 RAG 的关键 (要求引用来源、不要编造)")
    print("  5. 有 RAG 时 LLM 能回答私有知识的问题")
    print()
    print("👉 下一步: python rag/03_langchain_rag.py")
    print("=" * 60)
