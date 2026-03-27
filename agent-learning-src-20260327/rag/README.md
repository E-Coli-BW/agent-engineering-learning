# 📚 RAG 学习路线：从原理到实战

> **100% 本地运行**，使用 Ollama qwen2.5:7b + mxbai-embed-large + ChromaDB

## 学习顺序（由浅入深）

```
Level 1: 01_embedding_basics.py    — 理解 Embedding 和向量检索的本质
Level 2: 02_naive_rag.py           — 手写最简 RAG（不用框架）
Level 3: 03_langchain_rag.py       — 用 LangChain 构建标准 RAG Pipeline
Level 4: 04_advanced_rag.py        — 高级 RAG 优化技巧
```

## 运行方式

```bash
cd agent-learning
.venv/bin/python rag/01_embedding_basics.py
.venv/bin/python rag/02_naive_rag.py
.venv/bin/python rag/03_langchain_rag.py
.venv/bin/python rag/04_advanced_rag.py
```
