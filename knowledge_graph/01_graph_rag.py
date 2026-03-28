"""
知识图谱 + Graph RAG —— 结构化知识增强
========================================

核心问题: 向量检索只能做语义匹配，无法理解实体间的关系。
          "Transformer 的发明者是谁？" — 向量检索可能找不到。
          知识图谱可以: Transformer --发明者--> Vaswani

知识图谱 (Knowledge Graph) 是什么？
  - 用 (实体, 关系, 实体) 三元组表示知识
  - 又叫 "主-谓-宾" 或 "头实体-关系-尾实体"
  - 例: (Transformer, 提出时间, 2017), (GPT, 基于, Transformer Decoder)

Graph RAG = 知识图谱 + RAG:
  传统 RAG: 用户问题 → 向量检索 → 文档块 → LLM
  Graph RAG: 用户问题 → 图检索(关系推理) + 向量检索 → 结构化知识 + 文档 → LLM

本文件内容 (100% 本地运行，不需要 Neo4j):
  Part 1: 手写知识图谱 (理解三元组和图结构)
  Part 2: 从文本自动抽取三元组 (用本地 LLM)
  Part 3: 图上检索 (BFS/关系查询)
  Part 4: Graph RAG 实现
  Part 5: 可视化知识图谱
"""

import json
import os
import sys
from collections import defaultdict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CHAT_MODEL = "qwen2.5:7b"


# ============================================================
# Part 1: 手写知识图谱
# ============================================================
class SimpleKnowledgeGraph:
    """
    最简单的知识图谱实现：邻接表 + 三元组存储。

    不用 Neo4j，纯 Python 实现，让你看清楚图的本质。

    存储结构:
      triples: [(head, relation, tail), ...]
      adjacency: {entity: [(relation, neighbor), ...]}
    """

    def __init__(self):
        self.triples: list[tuple[str, str, str]] = []
        self.adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.entities: set[str] = set()
        self.relations: set[str] = set()

    def add_triple(self, head: str, relation: str, tail: str):
        """添加一个三元组"""
        self.triples.append((head, relation, tail))
        self.adjacency[head].append((relation, tail))
        self.adjacency[tail].append((f"~{relation}", head))  # 反向边
        self.entities.add(head)
        self.entities.add(tail)
        self.relations.add(relation)

    def get_neighbors(self, entity: str) -> list[tuple[str, str]]:
        """获取实体的所有邻居"""
        return self.adjacency.get(entity, [])

    def query_by_relation(self, head: str = None, relation: str = None, tail: str = None) -> list[tuple]:
        """按条件查询三元组"""
        results = []
        for h, r, t in self.triples:
            if head and h != head:
                continue
            if relation and r != relation:
                continue
            if tail and t != tail:
                continue
            results.append((h, r, t))
        return results

    def bfs(self, start: str, max_depth: int = 2) -> dict[str, list]:
        """BFS 遍历，获取实体周围的子图"""
        visited = {start}
        queue = [(start, 0)]
        subgraph = defaultdict(list)

        while queue:
            entity, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for relation, neighbor in self.get_neighbors(entity):
                subgraph[entity].append((relation, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return dict(subgraph)

    def to_text(self, subgraph: dict = None) -> str:
        """把图谱转为文本描述 (用于塞进 LLM prompt)"""
        triples_to_show = []
        if subgraph:
            for entity, neighbors in subgraph.items():
                for relation, neighbor in neighbors:
                    if not relation.startswith("~"):
                        triples_to_show.append((entity, relation, neighbor))
        else:
            triples_to_show = self.triples

        lines = []
        for h, r, t in triples_to_show:
            lines.append(f"  {h} --[{r}]--> {t}")
        return "\n".join(lines)

    def stats(self):
        print(f"  📊 知识图谱统计:")
        print(f"    实体数: {len(self.entities)}")
        print(f"    关系数: {len(self.relations)}")
        print(f"    三元组数: {len(self.triples)}")


def part1_build_kg():
    """手动构建知识图谱"""
    print("=" * 60)
    print("Part 1: 手动构建知识图谱")
    print("=" * 60)

    kg = SimpleKnowledgeGraph()

    # ---- 添加知识 (基于我们项目的内容) ----
    # 架构关系
    kg.add_triple("Transformer", "包含", "Self-Attention")
    kg.add_triple("Transformer", "包含", "Feed-Forward Network")
    kg.add_triple("Transformer", "包含", "Layer Normalization")
    kg.add_triple("Transformer", "包含", "Residual Connection")
    kg.add_triple("Transformer", "提出时间", "2017年")
    kg.add_triple("Transformer", "论文", "Attention Is All You Need")
    kg.add_triple("Transformer", "提出者", "Vaswani et al.")

    # Attention 细节
    kg.add_triple("Self-Attention", "计算", "QKV矩阵")
    kg.add_triple("Self-Attention", "变体", "Multi-Head Attention")
    kg.add_triple("Multi-Head Attention", "特点", "多子空间并行计算")
    kg.add_triple("Multi-Head Attention", "参数量", "4*(d_model^2+d_model)")
    kg.add_triple("Self-Attention", "缩放因子", "sqrt(d_k)")
    kg.add_triple("缩放因子", "作用", "防止梯度消失")

    # 模型变体
    kg.add_triple("GPT", "基于", "Transformer Decoder")
    kg.add_triple("BERT", "基于", "Transformer Encoder")
    kg.add_triple("GPT", "训练目标", "Next Token Prediction")
    kg.add_triple("BERT", "训练目标", "Masked Language Model")
    kg.add_triple("GPT", "使用", "Causal Mask")
    kg.add_triple("Causal Mask", "作用", "防止看到未来token")

    # 微调
    kg.add_triple("LoRA", "用于", "参数高效微调")
    kg.add_triple("LoRA", "原理", "低秩矩阵分解")
    kg.add_triple("QLoRA", "改进自", "LoRA")
    kg.add_triple("QLoRA", "特点", "4-bit量化+LoRA")
    kg.add_triple("SFT", "全称", "Supervised Fine-Tuning")
    kg.add_triple("RLHF", "用于", "模型对齐")
    kg.add_triple("DPO", "替代", "RLHF")

    # Agent
    kg.add_triple("ReAct", "循环", "Thought-Action-Observation")
    kg.add_triple("LangGraph", "建模为", "有向图")
    kg.add_triple("Agent", "核心能力", "Tool Calling")
    kg.add_triple("Agent", "框架", "LangChain")
    kg.add_triple("Agent", "框架", "LangGraph")

    # RAG
    kg.add_triple("RAG", "步骤", "Retrieve-Augment-Generate")
    kg.add_triple("RAG", "使用", "向量检索")
    kg.add_triple("RAG", "使用", "Embedding模型")
    kg.add_triple("Graph RAG", "改进自", "RAG")
    kg.add_triple("Graph RAG", "结合", "知识图谱")

    kg.stats()

    # ---- 查询示例 ----
    print(f"\n  🔍 查询: Transformer 包含什么？")
    results = kg.query_by_relation(head="Transformer", relation="包含")
    for h, r, t in results:
        print(f"    {h} --[{r}]--> {t}")

    print(f"\n  🔍 查询: 什么基于 Transformer？")
    results = kg.query_by_relation(relation="基于")
    for h, r, t in results:
        print(f"    {h} --[{r}]--> {t}")

    print(f"\n  🔍 BFS: 从 'Self-Attention' 出发，深度 2:")
    subgraph = kg.bfs("Self-Attention", max_depth=2)
    print(kg.to_text(subgraph))

    return kg


# ============================================================
# Part 2: 自动三元组抽取
# ============================================================
def part2_auto_extraction(kg: SimpleKnowledgeGraph):
    """
    用本地 LLM 从文本中自动抽取三元组。

    这是知识图谱构建的核心步骤:
      文本 → LLM 抽取 → (实体, 关系, 实体) → 存入图谱
    """
    print("\n" + "=" * 60)
    print("Part 2: 自动三元组抽取 (LLM 驱动)")
    print("=" * 60)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    extract_prompt = """从以下文本中抽取知识三元组。每个三元组格式为: (头实体, 关系, 尾实体)

要求:
1. 只抽取文本中明确提到的事实
2. 每行一个三元组
3. 格式严格: (实体1, 关系, 实体2)

文本:
{text}

三元组:"""

    test_texts = [
        "Mini-GPT 使用 Pre-Norm 架构，这是 GPT-2 采用的风格。Pre-Norm 先做 LayerNorm 再做 Attention，训练更稳定。",
        "KV Cache 通过缓存已计算的 Key 和 Value 矩阵来加速自回归推理。没有 KV Cache 时复杂度为 O(L³)，有 KV Cache 后降为 O(L²)。",
    ]

    extracted_count = 0
    for text in test_texts:
        print(f"\n  📄 文本: {text[:60]}...")
        response = llm.invoke([HumanMessage(content=extract_prompt.format(text=text))])
        print(f"  📊 抽取结果:")
        print(f"    {response.content}")

        # 解析并加入图谱
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if line.startswith("(") and line.endswith(")"):
                parts = line[1:-1].split(",")
                if len(parts) == 3:
                    h, r, t = [p.strip().strip("'\"") for p in parts]
                    kg.add_triple(h, r, t)
                    extracted_count += 1

    print(f"\n  ✅ 自动抽取了 {extracted_count} 个新三元组")
    kg.stats()

    return kg


# ============================================================
# Part 3: Graph RAG —— 图检索 + 向量检索融合
# ============================================================
def part3_graph_rag(kg: SimpleKnowledgeGraph):
    """
    Graph RAG: 结合知识图谱和向量检索。

    流程:
      1. 从用户问题中提取关键实体
      2. 在知识图谱中做子图检索 (BFS)
      3. 同时做向量检索获取文档片段
      4. 把 图谱知识 + 文档片段 一起塞给 LLM

    对比:
      纯向量 RAG: 只有文本片段，缺乏结构化关系
      纯知识图谱: 只有三元组，缺乏详细描述
      Graph RAG:  结构化关系 + 详细文本 = 最好的回答
    """
    print("\n" + "=" * 60)
    print("Part 3: Graph RAG 实现")
    print("=" * 60)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0)

    # ---- Step 1: 实体提取 ----
    def extract_entities(question: str) -> list[str]:
        prompt = f"""从以下问题中提取关键技术实体（名词/术语）。
每行一个实体，不要其他文字。

问题: {question}

实体:"""
        response = llm.invoke([HumanMessage(content=prompt)])
        entities = [e.strip().strip("-").strip() for e in response.content.strip().split("\n") if e.strip()]
        return entities[:5]

    # ---- Step 2: 图检索 ----
    def graph_retrieve(entities: list[str], kg: SimpleKnowledgeGraph) -> str:
        """从知识图谱中检索相关子图"""
        all_triples = set()
        for entity in entities:
            # 精确匹配
            subgraph = kg.bfs(entity, max_depth=2)
            for node, neighbors in subgraph.items():
                for rel, neighbor in neighbors:
                    if not rel.startswith("~"):
                        all_triples.add((node, rel, neighbor))

            # 模糊匹配 (实体名包含在图谱实体中)
            for kg_entity in kg.entities:
                if entity.lower() in kg_entity.lower() or kg_entity.lower() in entity.lower():
                    subgraph = kg.bfs(kg_entity, max_depth=1)
                    for node, neighbors in subgraph.items():
                        for rel, neighbor in neighbors:
                            if not rel.startswith("~"):
                                all_triples.add((node, rel, neighbor))

        if not all_triples:
            return "未找到相关知识图谱信息"

        lines = [f"  {h} --[{r}]--> {t}" for h, r, t in all_triples]
        return "\n".join(lines)

    # ---- Step 3: Graph RAG 回答 ----
    def graph_rag_query(question: str) -> str:
        print(f"\n  🧑 问题: {question}")

        # 提取实体
        entities = extract_entities(question)
        print(f"  🔍 提取的实体: {entities}")

        # 图检索
        graph_context = graph_retrieve(entities, kg)
        print(f"  📊 图谱知识:\n{graph_context}")

        # 构造 prompt
        prompt = f"""根据以下知识图谱信息回答问题。

知识图谱 (格式: 实体 --[关系]--> 实体):
{graph_context}

问题: {question}

要求: 根据图谱中的关系链推理回答，用中文简洁回答。"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    # ---- 测试 ----
    test_questions = [
        "Transformer 包含哪些核心组件？",
        "LoRA 和 QLoRA 是什么关系？",
        "GPT 使用了什么 Mask？这个 Mask 的作用是什么？",
    ]

    for q in test_questions:
        answer = graph_rag_query(q)
        print(f"  🤖 回答: {answer[:200]}")


# ============================================================
# Part 4: 知识图谱可视化
# ============================================================
def part4_visualize(kg: SimpleKnowledgeGraph):
    """用 matplotlib 可视化知识图谱"""
    print("\n" + "=" * 60)
    print("Part 4: 知识图谱可视化")
    print("=" * 60)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 尝试使用支持中文的字体
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

    # ---- 用简单的力导向布局 ----
    entities = list(kg.entities)
    n = len(entities)
    entity_to_idx = {e: i for i, e in enumerate(entities)}

    # 随机初始位置
    np.random.seed(42)
    pos = np.random.randn(n, 2) * 2

    # 简易力导向迭代
    for _ in range(100):
        forces = np.zeros_like(pos)

        # 斥力 (所有节点间)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = max(np.linalg.norm(diff), 0.1)
                force = diff / dist**2 * 0.5
                forces[i] += force
                forces[j] -= force

        # 引力 (有边的节点间)
        for h, r, t in kg.triples:
            if h in entity_to_idx and t in entity_to_idx:
                i, j = entity_to_idx[h], entity_to_idx[t]
                diff = pos[j] - pos[i]
                dist = max(np.linalg.norm(diff), 0.1)
                force = diff * 0.01
                forces[i] += force
                forces[j] -= force

        pos += forces * 0.1

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(16, 12))

    # 画边
    for h, r, t in kg.triples:
        if h in entity_to_idx and t in entity_to_idx:
            i, j = entity_to_idx[h], entity_to_idx[t]
            ax.annotate(
                "", xy=pos[j], xytext=pos[i],
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=0.8),
            )
            mid = (pos[i] + pos[j]) / 2
            ax.text(mid[0], mid[1], r, fontsize=5, ha="center", alpha=0.6, color="blue")

    # 画节点
    # 按实体类型着色
    colors = []
    for e in entities:
        if e in ("Transformer", "GPT", "BERT", "Mini-GPT"):
            colors.append("coral")
        elif e in ("Self-Attention", "Multi-Head Attention", "Causal Mask", "Feed-Forward Network"):
            colors.append("skyblue")
        elif e in ("LoRA", "QLoRA", "SFT", "RLHF", "DPO"):
            colors.append("lightgreen")
        elif e in ("RAG", "Graph RAG", "Agent", "ReAct", "LangGraph", "LangChain"):
            colors.append("plum")
        else:
            colors.append("lightyellow")

    for i, entity in enumerate(entities):
        ax.scatter(pos[i, 0], pos[i, 1], c=colors[i], s=200, zorder=5, edgecolors="gray")
        ax.text(pos[i, 0], pos[i, 1] + 0.15, entity, fontsize=7, ha="center", fontweight="bold")

    ax.set_title("Knowledge Graph: Deep Learning Concepts", fontsize=14)
    ax.axis("off")
    fig.tight_layout()

    output_path = os.path.join(PROJECT_ROOT, "outputs", "knowledge_graph.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  🎨 知识图谱已保存: {output_path}")


# ============================================================
# Part 5: 面试总结
# ============================================================
def part5_summary():
    print("\n" + "=" * 60)
    print("Part 5: 知识图谱面试总结")
    print("=" * 60)

    print("""
    知识图谱 vs 向量检索:
    ┌──────────────┬──────────────────────┬──────────────────────┐
    │              │ 向量检索             │ 知识图谱              │
    ├──────────────┼──────────────────────┼──────────────────────┤
    │ 表示         │ 语义向量             │ (实体,关系,实体) 三元组│
    │ 检索         │ 余弦相似度           │ 图遍历/路径查询        │
    │ 优势         │ 语义模糊匹配         │ 精确关系推理           │
    │ 劣势         │ 无法推理关系链       │ 需要构建和维护图谱     │
    │ 适用         │ 文档问答             │ 关系推理/多跳问答      │
    └──────────────┴──────────────────────┴──────────────────────┘

    Graph RAG 面试答题:
    ──────────────────
    Q: 知识图谱如何与 RAG 结合？
    A: 1. 从用户问题中提取实体
       2. 在图谱中做子图检索 (BFS/路径查询)
       3. 把图谱三元组和向量检索结果一起塞进 prompt
       4. 图谱提供结构化关系，文档提供详细描述，互补

    Q: 知识图谱相比向量检索有什么优势？
    A: 1. 能做多跳推理 (A→B→C 的关系链)
       2. 实体消歧 (同名不同义)
       3. 可解释性强 (关系路径清晰)
       4. 适合"X 和 Y 的关系是什么"类问题

    Q: 知识图谱的构建成本高吗？
    A: 是的，主要成本在:
       1. 实体识别和关系抽取 (可用 LLM 自动化)
       2. 知识验证和去重
       3. 持续更新和维护
       目前趋势: 用 LLM 做自动构建 + 人工审核
    """)


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    print("🚀 知识图谱 + Graph RAG\n")

    # Part 1: 手动构建知识图谱
    kg = part1_build_kg()

    # Part 2: LLM 自动抽取三元组
    kg = part2_auto_extraction(kg)

    # Part 3: Graph RAG
    part3_graph_rag(kg)

    # Part 4: 可视化
    part4_visualize(kg)

    # Part 5: 面试总结
    part5_summary()

    print("\n" + "=" * 60)
    print("✅ 知识图谱模块完成！")
    print("=" * 60)
