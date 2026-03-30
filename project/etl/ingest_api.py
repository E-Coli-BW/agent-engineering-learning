"""
文档上传 + ETL API
===================

POST /documents/upload  — 上传文件 → 解析 → 分块 → 向量化 → 入库
GET  /documents/status  — 查看已入库的文档状态
POST /documents/ingest  — 指定本地路径批量导入

和 etl_pipeline.py 的区别:
  - etl_pipeline.py 是 CLI 工具，处理 .py/.md
  - 本模块是 API，支持 PDF/PPT/DOCX 上传，走 parsers.py
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("etl.api")

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR",
    str(Path(__file__).parent.parent.parent / "data" / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档 → 解析 → 向量化 → 入库

    支持: .pdf, .pptx, .docx, .md, .txt
    """
    from project.etl.parsers import parse_file, PARSER_MAP, LEGACY_FORMATS

    # 验证文件格式
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in LEGACY_FORMATS:
        raise HTTPException(400, LEGACY_FORMATS[suffix])
    if suffix not in PARSER_MAP:
        raise HTTPException(400, f"不支持的格式: {suffix}。支持: {list(PARSER_MAP.keys())}")

    # 保存文件
    content = await file.read()
    content_hash = hashlib.md5(content).hexdigest()[:8]
    save_path = UPLOAD_DIR / f"{content_hash}_{file.filename}"
    save_path.write_bytes(content)
    logger.info("文件已保存: %s (%d bytes)", save_path.name, len(content))

    # 解析
    start = time.time()
    docs = parse_file(save_path)

    if not docs:
        return JSONResponse({
            "status": "warning",
            "message": "文件解析成功但没有提取到内容",
            "file": file.filename,
        })

    # 向量化入库
    chunks_added = _ingest_documents(docs)
    elapsed = round(time.time() - start, 1)

    return {
        "status": "success",
        "file": file.filename,
        "format": suffix,
        "chunks_parsed": len(docs),
        "chunks_added": chunks_added,
        "elapsed_seconds": elapsed,
        "content_hash": content_hash,
    }


@router.post("/ingest")
async def ingest_from_path(body: dict):
    """
    从本地路径批量导入

    请求: {"path": "/path/to/directory", "recursive": true}
    """
    from project.etl.parsers import parse_directory

    dir_path = body.get("path", "")
    recursive = body.get("recursive", True)

    if not dir_path or not Path(dir_path).exists():
        raise HTTPException(400, f"路径不存在: {dir_path}")

    start = time.time()
    docs = parse_directory(dir_path, recursive=recursive)

    if not docs:
        return {"status": "warning", "message": "未找到可解析的文件", "path": dir_path}

    chunks_added = _ingest_documents(docs)
    elapsed = round(time.time() - start, 1)

    sources = list(set(d.metadata.get("source", "?") for d in docs))

    return {
        "status": "success",
        "path": dir_path,
        "files": len(sources),
        "file_list": sources,
        "chunks_parsed": len(docs),
        "chunks_added": chunks_added,
        "elapsed_seconds": elapsed,
    }


@router.get("/status")
async def document_status():
    """查看向量库状态"""
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma

        db_dir = str(Path(__file__).parent.parent.parent / "data" / "project_chroma_db")
        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        collection = os.getenv("COLLECTION_NAME", "project_knowledge_v2")

        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name=collection,
        )
        count = vector_store._collection.count()

        return {
            "status": "ok",
            "vector_count": count,
            "collection": collection,
            "db_dir": db_dir,
            "upload_dir": str(UPLOAD_DIR),
            "supported_formats": [".pdf", ".pptx", ".docx", ".md", ".txt", ".py"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _ingest_documents(docs: list) -> int:
    """将解析后的文档块向量化并入库"""
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document as LCDocument

        db_dir = str(Path(__file__).parent.parent.parent / "data" / "project_chroma_db")
        embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
        collection = os.getenv("COLLECTION_NAME", "project_knowledge_v2")

        embeddings = OllamaEmbeddings(model=embed_model)
        vector_store = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings,
            collection_name=collection,
        )

        # 转换为 LangChain Document，二次分块 (embedding 模型有 token 限制)
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " "],
        )

        lc_docs = []
        for doc in docs:
            content = doc.content.strip()
            if not content:
                continue
            if len(content) <= 500:
                lc_docs.append(LCDocument(page_content=content, metadata=doc.metadata))
            else:
                # 过长 chunk 二次分块
                sub_chunks = splitter.split_text(content)
                for j, chunk in enumerate(sub_chunks):
                    meta = {**doc.metadata, "sub_chunk": j + 1}
                    lc_docs.append(LCDocument(page_content=chunk, metadata=meta))

        # 批量入库
        vector_store.add_documents(lc_docs)
        logger.info("向量化完成: %d 个文档块", len(lc_docs))
        return len(lc_docs)

    except Exception as e:
        logger.error("向量化入库失败: %s", e)
        return 0
