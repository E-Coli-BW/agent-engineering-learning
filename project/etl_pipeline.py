"""
ETL Pipeline — 真实数据处理管线
================================

这是整个项目和"真实工程"差距最大的部分。
之前所有模块的数据都是 fake data / 硬编码，
真实项目中你需要处理:
  - PDF 文档 (扫描件、多栏排版)
  - Markdown / HTML 网页
  - 数据库记录
  - API 数据
  - CSV / Excel 表格

本模块实现一个完整的 ETL Pipeline:
  Extract:  从多种数据源加载原始数据
  Transform: 清洗 → 结构化 → 分块 → 元数据提取
  Load:     向量化 → 存入 ChromaDB

100% 本地运行，处理真实文件。
"""

import os
import re
import json
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

# ---- 配置 ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "project_chroma_db"
ETL_LOG_DIR = DATA_DIR / "etl_logs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("etl_pipeline")


# ============================================================
# 数据模型
# ============================================================
@dataclass
class RawDocument:
    """从数据源提取的原始文档"""
    content: str
    source: str                          # 文件路径或 URL
    source_type: str                     # "pdf", "markdown", "python", "html", "txt"
    metadata: dict = field(default_factory=dict)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProcessedChunk:
    """清洗、分块后的文档块"""
    content: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]


# ============================================================
# Extract: 数据源加载器
# ============================================================
class DocumentExtractor:
    """
    从多种数据源提取文档。

    真实项目中的数据源:
      - 本地文件 (PDF, MD, TXT, PY, HTML)
      - 网页 URL
      - 数据库
      - API

    面试考点:
      Q: 你们的数据源有哪些？怎么处理不同格式？
      A: 通过 Extractor 抽象层统一接口，每种格式一个 loader,
         输出统一的 RawDocument 对象。
    """

    SUPPORTED_EXTENSIONS = {".md", ".py", ".txt", ".html", ".json", ".csv"}

    def extract_file(self, filepath: str) -> Optional[RawDocument]:
        """从单个文件提取内容"""
        path = Path(filepath)

        if not path.exists():
            logger.warning(f"文件不存在: {filepath}")
            return None

        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的格式: {path.suffix}")
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="gbk")
            except Exception as e:
                logger.error(f"读取文件失败 {filepath}: {e}")
                return None

        source_type = self._detect_type(path)

        return RawDocument(
            content=content,
            source=str(path.relative_to(PROJECT_ROOT)),
            source_type=source_type,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            },
        )

    def extract_directory(self, dirpath: str, recursive: bool = True) -> list[RawDocument]:
        """递归扫描目录，提取所有支持的文件"""
        path = Path(dirpath)
        documents = []

        pattern = "**/*" if recursive else "*"
        for filepath in sorted(path.glob(pattern)):
            if filepath.is_file() and filepath.suffix in self.SUPPORTED_EXTENSIONS:
                # 跳过隐藏文件和虚拟环境
                if any(part.startswith(".") for part in filepath.parts):
                    continue
                if ".venv" in str(filepath) or "__pycache__" in str(filepath):
                    continue

                doc = self.extract_file(str(filepath))
                if doc:
                    documents.append(doc)

        logger.info(f"从 {dirpath} 提取了 {len(documents)} 个文件")
        return documents

    def extract_url(self, url: str) -> Optional[RawDocument]:
        """从 URL 提取内容 (简单实现)"""
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")

            # 简单去除 HTML 标签
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            return RawDocument(
                content=content,
                source=url,
                source_type="html",
                metadata={"url": url},
            )
        except Exception as e:
            logger.error(f"提取 URL 失败 {url}: {e}")
            return None

    @staticmethod
    def _detect_type(path: Path) -> str:
        type_map = {
            ".md": "markdown",
            ".py": "python",
            ".txt": "text",
            ".html": "html",
            ".json": "json",
            ".csv": "csv",
        }
        return type_map.get(path.suffix, "text")


# ============================================================
# Transform: 清洗 + 分块
# ============================================================
class DocumentTransformer:
    """
    数据清洗和分块。

    真实项目中的清洗需求:
      - 去除无关内容 (导航栏、页脚、广告)
      - 修复编码问题
      - 标准化格式 (统一换行符、空白)
      - 提取结构化元数据 (标题层级、代码块)
      - 去重

    面试考点:
      Q: 你们的分块策略是什么？chunk_size 怎么选？
      A: 根据 Embedding 模型的 max_tokens 和检索精度权衡:
         - chunk_size 太大 → 信息被稀释，检索不精确
         - chunk_size 太小 → 缺少上下文，LLM 无法理解
         - overlap 保持上下文连贯
         - 不同文件类型用不同分隔符 (代码按函数/类切, MD 按标题切)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_length: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # 不同类型的分块器
        self.python_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\nclass ",
                "\ndef ",
                "\n    def ",
                "\n\n",
                "\n",
                " ",
            ],
        )

        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",
                "\n### ",
                "\n#### ",
                "\n\n",
                "\n",
                " ",
            ],
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Markdown 标题层级分块
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

    def transform(self, raw_doc: RawDocument) -> list[ProcessedChunk]:
        """清洗 + 分块一个文档"""
        # Step 1: 清洗
        cleaned = self._clean(raw_doc)

        # Step 2: 分块
        chunks = self._chunk(cleaned, raw_doc.source_type)

        # Step 3: 构造 ProcessedChunk
        processed = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < self.min_chunk_length:
                continue  # 跳过太短的块

            processed.append(ProcessedChunk(
                content=chunk_text,
                source=raw_doc.source,
                chunk_index=i,
                total_chunks=len(chunks),
                metadata={
                    **raw_doc.metadata,
                    "source_type": raw_doc.source_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk_text),
                },
            ))

        logger.info(f"  {raw_doc.source}: {len(raw_doc.content)} 字符 → {len(processed)} 个块")
        return processed

    def _clean(self, raw_doc: RawDocument) -> str:
        """清洗原始文本"""
        content = raw_doc.content

        # 统一换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        # 去除多余空行 (保留最多 2 个连续空行)
        content = re.sub(r"\n{3,}", "\n\n", content)

        # 去除行尾空白
        content = "\n".join(line.rstrip() for line in content.split("\n"))

        # Python 文件: 去除大段注释块中的装饰性分隔线
        if raw_doc.source_type == "python":
            content = re.sub(r"#\s*={40,}\n", "", content)

        # Markdown: 去除 HTML 注释
        if raw_doc.source_type == "markdown":
            content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        return content.strip()

    def _chunk(self, content: str, source_type: str) -> list[str]:
        """按类型分块"""
        if source_type == "python":
            docs = self.python_splitter.create_documents([content])
        elif source_type == "markdown":
            # 先按标题结构分，再按大小细分
            header_splits = self.md_header_splitter.split_text(content)
            docs = []
            for split in header_splits:
                # split 已经是按标题分出来的 Document 了
                sub_docs = self.markdown_splitter.create_documents(
                    [split.page_content],
                    metadatas=[split.metadata],
                )
                docs.extend(sub_docs)
            if not docs:
                docs = self.markdown_splitter.create_documents([content])
        else:
            docs = self.text_splitter.create_documents([content])

        return [doc.page_content for doc in docs]


# ============================================================
# Load: 向量化 + 入库
# ============================================================
class VectorStoreLoader:
    """
    向量化并存入 ChromaDB。

    真实项目中的考虑:
      - 增量更新 (只处理新增/修改的文件)
      - 去重 (content_hash 检查)
      - 元数据索引 (支持过滤查询)
      - 批量处理 (大文件集避免 OOM)

    面试考点:
      Q: 向量库数据怎么更新？每次全量重建吗？
      A: 不，用增量更新:
         - 记录每个文档的 hash
         - 对比新文件 hash，只处理变化的
         - 删除已移除文件的向量
         - 定期全量重建保证一致性
    """

    def __init__(
        self,
        persist_dir: str = None,
        collection_name: str = "knowledge_base",
        batch_size: int = 50,
    ):
        self.persist_dir = persist_dir or str(VECTOR_DB_DIR)
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    def load(self, chunks: list[ProcessedChunk], incremental: bool = True) -> Chroma:
        """把 chunks 向量化存入 ChromaDB"""

        # 转为 LangChain Document
        documents = []
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk.content,
                metadata={
                    "source": chunk.source,
                    "content_hash": chunk.content_hash,
                    **chunk.metadata,
                },
            ))

        if incremental and os.path.exists(self.persist_dir):
            vector_store = self._incremental_load(documents)
        else:
            vector_store = self._full_load(documents)

        return vector_store

    def _full_load(self, documents: list[Document]) -> Chroma:
        """全量重建"""
        logger.info(f"全量加载 {len(documents)} 个块到 ChromaDB...")

        # 分批处理，避免一次性 embedding 太多导致超时
        vector_store = None
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    batch,
                    self.embeddings,
                    persist_directory=self.persist_dir,
                    collection_name=self.collection_name,
                )
            else:
                vector_store.add_documents(batch)
            logger.info(f"  已处理 {min(i + self.batch_size, len(documents))}/{len(documents)}")

        return vector_store

    def _incremental_load(self, documents: list[Document]) -> Chroma:
        """增量更新: 只添加新增/修改的文档"""
        logger.info("增量更新模式...")

        vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

        # 获取已有的 hash 集合
        existing = vector_store.get(include=["metadatas"])
        existing_hashes = set()
        if existing and existing["metadatas"]:
            for meta in existing["metadatas"]:
                if meta and "content_hash" in meta:
                    existing_hashes.add(meta["content_hash"])

        # 过滤出新增的
        new_docs = [
            doc for doc in documents
            if doc.metadata.get("content_hash") not in existing_hashes
        ]

        if new_docs:
            logger.info(f"  新增 {len(new_docs)} 个块 (跳过 {len(documents) - len(new_docs)} 个已有)")
            for i in range(0, len(new_docs), self.batch_size):
                batch = new_docs[i:i + self.batch_size]
                vector_store.add_documents(batch)
        else:
            logger.info("  无新增内容")

        return vector_store


# ============================================================
# ETL Pipeline: 编排 E + T + L
# ============================================================
class ETLPipeline:
    """
    完整的 ETL Pipeline，串联 Extract → Transform → Load。

    usage:
      pipeline = ETLPipeline()
      pipeline.run(sources=["/path/to/docs"], incremental=True)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        collection_name: str = "knowledge_base",
    ):
        self.extractor = DocumentExtractor()
        self.transformer = DocumentTransformer(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.loader = VectorStoreLoader(collection_name=collection_name)

    def run(
        self,
        sources: list[str],
        incremental: bool = True,
    ) -> dict:
        """
        执行完整的 ETL 流程。

        Args:
            sources: 文件路径 / 目录路径 / URL 列表
            incremental: 是否增量更新

        Returns:
            stats: 统计信息
        """
        start_time = datetime.now()
        stats = {
            "sources": len(sources),
            "files_extracted": 0,
            "chunks_created": 0,
            "chunks_loaded": 0,
            "errors": [],
        }

        # ---- Extract ----
        logger.info("=" * 50)
        logger.info("📥 Extract: 加载数据源")
        logger.info("=" * 50)

        raw_documents = []
        for source in sources:
            path = Path(source)
            if path.is_dir():
                docs = self.extractor.extract_directory(str(path))
                raw_documents.extend(docs)
            elif path.is_file():
                doc = self.extractor.extract_file(str(path))
                if doc:
                    raw_documents.append(doc)
            elif source.startswith("http"):
                doc = self.extractor.extract_url(source)
                if doc:
                    raw_documents.append(doc)
            else:
                stats["errors"].append(f"无法识别的数据源: {source}")

        stats["files_extracted"] = len(raw_documents)
        logger.info(f"提取完成: {len(raw_documents)} 个文件")

        if not raw_documents:
            logger.warning("没有提取到任何文档，退出")
            return stats

        # ---- Transform ----
        logger.info("")
        logger.info("=" * 50)
        logger.info("🔄 Transform: 清洗 + 分块")
        logger.info("=" * 50)

        all_chunks = []
        for doc in raw_documents:
            try:
                chunks = self.transformer.transform(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理失败 {doc.source}: {e}")
                stats["errors"].append(f"Transform 失败: {doc.source}: {str(e)}")

        stats["chunks_created"] = len(all_chunks)
        logger.info(f"分块完成: {len(raw_documents)} 个文件 → {len(all_chunks)} 个块")

        if not all_chunks:
            logger.warning("没有产出任何块，退出")
            return stats

        # ---- Load ----
        logger.info("")
        logger.info("=" * 50)
        logger.info("📦 Load: 向量化 + 入库")
        logger.info("=" * 50)

        try:
            vector_store = self.loader.load(all_chunks, incremental=incremental)
            count = vector_store._collection.count()
            stats["chunks_loaded"] = count
            logger.info(f"入库完成: 向量库共 {count} 条记录")
        except Exception as e:
            logger.error(f"Load 失败: {e}")
            stats["errors"].append(f"Load 失败: {str(e)}")

        # ---- 统计 ----
        elapsed = (datetime.now() - start_time).total_seconds()
        stats["elapsed_seconds"] = round(elapsed, 1)

        logger.info("")
        logger.info("=" * 50)
        logger.info("📊 ETL Pipeline 完成")
        logger.info("=" * 50)
        logger.info(f"  数据源: {stats['sources']}")
        logger.info(f"  提取文件: {stats['files_extracted']}")
        logger.info(f"  生成块数: {stats['chunks_created']}")
        logger.info(f"  入库总量: {stats['chunks_loaded']}")
        logger.info(f"  耗时: {elapsed:.1f}s")
        if stats["errors"]:
            logger.warning(f"  错误: {len(stats['errors'])} 个")
            for err in stats["errors"]:
                logger.warning(f"    - {err}")

        # 保存日志
        self._save_log(stats)

        return stats

    def _save_log(self, stats: dict):
        """保存 ETL 运行日志"""
        ETL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = ETL_LOG_DIR / f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"  日志已保存: {log_file}")


# ============================================================
# 运行: 用项目自身的真实文件跑 ETL
# ============================================================
if __name__ == "__main__":
    print("🚀 ETL Pipeline — 真实数据处理\n")

    pipeline = ETLPipeline(
        chunk_size=500,
        chunk_overlap=100,
        collection_name="project_knowledge_v2",
    )

    # 用项目自身文件作为真实数据源
    stats = pipeline.run(
        sources=[
            str(PROJECT_ROOT / "multi_head_attention.py"),
            str(PROJECT_ROOT / "char_transformer.py"),
            str(PROJECT_ROOT / "README.md"),
            str(PROJECT_ROOT / "agent"),
            str(PROJECT_ROOT / "rag"),
            str(PROJECT_ROOT / "finetune"),
            str(PROJECT_ROOT / "deploy"),
            str(PROJECT_ROOT / "knowledge_graph"),
        ],
        incremental=False,  # 首次运行用全量
    )

    # ---- 验证: 测试检索 ----
    print("\n" + "=" * 50)
    print("🔍 验证: 测试检索")
    print("=" * 50)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name="project_knowledge_v2",
    )

    test_queries = [
        "LoRA 微调的原理是什么？",
        "KV Cache 如何加速推理？",
        "知识图谱和 RAG 怎么结合？",
    ]

    for query in test_queries:
        results = vector_store.similarity_search_with_score(query, k=3)
        print(f"\n  🔍 Query: {query}")
        for doc, score in results:
            source = doc.metadata.get("source", "?")
            print(f"    [{score:.3f}] [{source}] {doc.page_content[:80]}...")

    print("\n✅ ETL Pipeline 完成！")
