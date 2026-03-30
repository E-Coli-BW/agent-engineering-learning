"""
RAG API Server — 可对外服务的知识库问答系统
============================================

之前的 RAG 都是"跑一下脚本看看效果"。
真实工程需要一个 API 服务，让前端 / 其他系统调用。

本模块实现:
  - FastAPI REST API (标准 HTTP 接口)
  - 完整的请求/响应模型 (Pydantic)
  - 错误处理 + 超时 + 重试
  - 流式输出 (SSE)
  - 健康检查 + 指标上报
  - CORS 支持
  - 配置管理 (环境变量)

运行:
  pip install fastapi uvicorn
  python project/api_server.py
  
  # 访问 http://localhost:8000/docs 查看 API 文档
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# ---- 配置 (通过环境变量覆盖) ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "project_chroma_db")
))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "project_knowledge_v2")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))

# ---- 日志 ----
from project.infra.logging import setup_logging
setup_logging("rag-api")
logger = logging.getLogger("rag_api")


# ============================================================
# 请求/响应模型
# ============================================================
class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, description="检索文档数", ge=1, le=20)
    temperature: float = Field(default=0.1, description="生成温度", ge=0, le=2)
    stream: bool = Field(default=False, description="是否流式输出")


class SourceDocument(BaseModel):
    """检索到的源文档"""
    content: str
    source: str
    score: float
    metadata: dict = {}


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    sources: list[SourceDocument]
    query_time_ms: int
    model: str
    retrieval_count: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    ollama: str
    vector_store: str
    vector_count: int
    models: dict
    timestamp: str


class ETLTriggerRequest(BaseModel):
    """ETL 触发请求"""
    sources: list[str] = Field(..., description="数据源路径列表")
    incremental: bool = Field(default=True, description="是否增量更新")
    chunk_size: int = Field(default=500, description="分块大小")
    chunk_overlap: int = Field(default=100, description="分块重叠")


# ============================================================
# 全局资源 (应用启动时初始化)
# ============================================================
class AppState:
    vector_store: Optional[Chroma] = None
    llm: Optional[ChatOllama] = None
    embeddings: Optional[OllamaEmbeddings] = None
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: int = 0
    start_time: Optional[datetime] = None
    # 延迟分布 (用于计算 P50/P95/P99)
    latency_samples: list = []


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时初始化资源，关闭时清理"""
    logger.info("🚀 初始化 RAG API Server...")

    try:
        state.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        state.llm = ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

        if os.path.exists(VECTOR_DB_DIR):
            state.vector_store = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=state.embeddings,
                collection_name=COLLECTION_NAME,
            )
            count = state.vector_store._collection.count()
            logger.info(f"✅ 向量库已加载: {count} 条记录")
        else:
            logger.warning(f"⚠️ 向量库不存在: {VECTOR_DB_DIR}")
            logger.warning("  请先运行 python project/etl_pipeline.py 构建索引")

        state.start_time = datetime.now()
        logger.info("✅ RAG API Server 启动完成")

    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        raise

    yield  # 应用运行中

    logger.info("🛑 RAG API Server 关闭")


# ============================================================
# FastAPI 应用
# ============================================================
app = FastAPI(
    title="RAG Knowledge Base API",
    description="基于本地 Ollama + ChromaDB 的知识库问答 API",
    version="1.0.0",
    lifespan=lifespan,
)

# 统一错误处理
from project.infra.errors import register_error_handlers
register_error_handlers(app)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- 请求计时 + 追踪 ID 中间件 ----
@app.middleware("http")
async def add_timing_and_trace(request: Request, call_next):
    import uuid as _uuid
    start = time.time()

    # 从 Gateway 继承 X-Request-Id，或自己生成
    request_id = request.headers.get("X-Request-Id", _uuid.uuid4().hex[:8])

    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000)

    response.headers["X-Response-Time-Ms"] = str(elapsed)
    response.headers["X-Request-Id"] = request_id

    # 收集延迟样本 (只保留最近 1000 个，用于 P50/P95/P99)
    state.latency_samples.append(elapsed)
    if len(state.latency_samples) > 1000:
        state.latency_samples = state.latency_samples[-1000:]

    logger.info("[%s] %s %s → %s (%dms)",
                request_id, request.method, request.url.path,
                response.status_code, elapsed)
    return response


# ============================================================
# API 路由
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点。

    真实项目中的健康检查:
      - 检查 Ollama 是否可达
      - 检查向量库是否可用
      - 返回关键指标
      - K8s / 负载均衡器用这个做存活检测
    """
    import requests as req

    # 检查 Ollama
    ollama_status = "unknown"
    models = {}
    try:
        resp = req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            ollama_status = "healthy"
            for m in resp.json().get("models", []):
                models[m["name"]] = f"{m['size'] / 1e9:.1f}GB"
        else:
            ollama_status = f"unhealthy (HTTP {resp.status_code})"
    except Exception as e:
        ollama_status = f"unreachable ({str(e)[:50]})"

    # 检查向量库
    vector_status = "not_loaded"
    vector_count = 0
    if state.vector_store:
        try:
            vector_count = state.vector_store._collection.count()
            vector_status = "healthy"
        except Exception:
            vector_status = "error"

    return HealthResponse(
        status="ok" if ollama_status == "healthy" and vector_status == "healthy" else "degraded",
        ollama=ollama_status,
        vector_store=vector_status,
        vector_count=vector_count,
        models=models,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    知识库问答。

    流程:
      1. 向量检索 Top-K 文档
      2. 构造 prompt (系统提示 + 上下文 + 问题)
      3. 调用 LLM 生成回答
      4. 返回回答 + 来源文档

    错误处理:
      - 向量库不可用 → 503
      - LLM 超时 → 504 + 重试
      - 未知错误 → 500
    """
    start_time = time.time()
    state.request_count += 1

    if not state.vector_store:
        raise HTTPException(
            status_code=503,
            detail="向量库未初始化，请先运行 ETL Pipeline 构建索引",
        )

    try:
        # ---- Step 1: 检索 ----
        results = state.vector_store.similarity_search_with_score(
            request.question,
            k=request.top_k,
        )

        sources = []
        context_parts = []
        for doc, score in results:
            sources.append(SourceDocument(
                content=doc.page_content[:500],
                source=doc.metadata.get("source", "unknown"),
                score=round(float(score), 4),
                metadata={k: v for k, v in doc.metadata.items()
                          if k in ("source_type", "chunk_index", "filename")},
            ))
            context_parts.append(
                f"[来源: {doc.metadata.get('source', '?')}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # ---- Step 2: 生成 (带重试) ----
        system_prompt = """你是一个技术知识库助手。根据提供的上下文信息回答问题。

规则:
1. 只根据上下文信息回答，不要编造
2. 如果上下文中没有相关信息，明确说明
3. 引用信息来源
4. 用中文回答，技术术语保留英文"""

        user_prompt = f"""上下文信息:
{context}

问题: {request.question}

请根据上下文回答:"""

        answer = None
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                llm = ChatOllama(
                    model=CHAT_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    temperature=request.temperature,
                )
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                answer = response.content
                break
            except Exception as e:
                last_error = e
                logger.warning(f"LLM 调用失败 (尝试 {attempt + 1}/{MAX_RETRIES + 1}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(1)

        if answer is None:
            state.error_count += 1
            raise HTTPException(
                status_code=504,
                detail=f"LLM 调用超时/失败 (已重试 {MAX_RETRIES} 次): {last_error}",
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        state.total_latency_ms += elapsed_ms

        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time_ms=elapsed_ms,
            model=CHAT_MODEL,
            retrieval_count=len(sources),
        )

    except HTTPException:
        raise
    except Exception as e:
        state.error_count += 1
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式输出版本 (Server-Sent Events)。

    前端可以用 EventSource 接收:
      const es = new EventSource('/query/stream');
      es.onmessage = (e) => console.log(e.data);

    面试考点:
      Q: 为什么要用流式输出？
      A: 减少用户等待感。Token 一个个蹦出来 (TTFT < 1s),
         而不是等全部生成完才返回 (可能 10s+)。
    """
    if not state.vector_store:
        raise HTTPException(status_code=503, detail="向量库未初始化")

    results = state.vector_store.similarity_search_with_score(
        request.question,
        k=request.top_k,
    )

    context_parts = []
    for doc, score in results:
        context_parts.append(doc.page_content)
    context = "\n\n---\n\n".join(context_parts)

    async def generate():
        llm = ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=request.temperature,
        )

        system_prompt = "你是一个技术知识库助手。根据上下文信息用中文回答问题。只根据上下文回答，不要编造。"
        user_prompt = f"上下文:\n{context}\n\n问题: {request.question}"

        import json
        for chunk in llm.stream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]):
            yield f"data: {json.dumps({'token': chunk.content}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/etl/run")
async def trigger_etl(request: ETLTriggerRequest):
    """
    触发 ETL Pipeline。

    真实项目中:
      - 定时任务触发 (cron / Airflow)
      - webhook 触发 (文档更新时)
      - 手动触发 (管理后台)
    """
    from project.etl_pipeline import ETLPipeline

    try:
        pipeline = ETLPipeline(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            collection_name=COLLECTION_NAME,
        )
        stats = pipeline.run(
            sources=request.sources,
            incremental=request.incremental,
        )

        # 重新加载向量库
        state.vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=state.embeddings,
            collection_name=COLLECTION_NAME,
        )

        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETL 失败: {str(e)}")


@app.get("/metrics")
async def metrics(request: Request):
    """
    指标端点。

    支持两种格式:
      - Accept: text/plain → Prometheus 文本格式 (Prometheus 采集用)
      - 其他 → JSON 格式 (前端 Dashboard / 人工查看)
    """
    uptime = (datetime.now() - state.start_time).total_seconds() if state.start_time else 0
    avg_latency = state.total_latency_ms // state.request_count if state.request_count > 0 else 0
    vector_count = state.vector_store._collection.count() if state.vector_store else 0

    # 计算 P50/P95/P99
    p50 = p95 = p99 = 0
    samples = sorted(state.latency_samples) if state.latency_samples else []
    if samples:
        p50 = samples[int(len(samples) * 0.5)]
        p95 = samples[int(len(samples) * 0.95)]
        p99 = samples[min(int(len(samples) * 0.99), len(samples) - 1)]

    accept = request.headers.get("accept", "")

    # Prometheus 文本格式
    if "text/plain" in accept or "text/plain" in request.query_params.get("format", ""):
        from fastapi.responses import PlainTextResponse
        lines = [
            "# HELP rag_uptime_seconds RAG API server uptime in seconds",
            "# TYPE rag_uptime_seconds gauge",
            f"rag_uptime_seconds {uptime:.1f}",
            "",
            "# HELP rag_requests_total Total number of RAG requests",
            "# TYPE rag_requests_total counter",
            f'rag_requests_total{{status="success"}} {state.request_count - state.error_count}',
            f'rag_requests_total{{status="error"}} {state.error_count}',
            "",
            "# HELP rag_request_duration_ms RAG request latency in milliseconds",
            "# TYPE rag_request_duration_ms summary",
            f'rag_request_duration_ms{{quantile="0.5"}} {p50}',
            f'rag_request_duration_ms{{quantile="0.95"}} {p95}',
            f'rag_request_duration_ms{{quantile="0.99"}} {p99}',
            f"rag_request_duration_ms_avg {avg_latency}",
            "",
            "# HELP rag_vector_count Number of vectors in the store",
            "# TYPE rag_vector_count gauge",
            f"rag_vector_count {vector_count}",
            "",
            "# HELP rag_error_rate Ratio of failed requests",
            "# TYPE rag_error_rate gauge",
            f"rag_error_rate {state.error_count / max(state.request_count, 1):.4f}",
        ]
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # JSON 格式 (默认)
    return {
        "uptime_seconds": round(uptime),
        "total_requests": state.request_count,
        "total_errors": state.error_count,
        "error_rate": round(state.error_count / max(state.request_count, 1), 4),
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "vector_count": vector_count,
    }


# ============================================================
# 聊天记忆 API
# ============================================================

@app.post("/chat/history")
async def add_chat_message(request: Request):
    """保存一条聊天消息"""
    from project.infra.memory import get_chat_memory
    body = await request.json()
    session_id = body.get("session_id", "default")
    role = body.get("role", "user")
    content = body.get("content", "")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    memory = get_chat_memory()
    memory.add(session_id, role, content)
    return {"status": "ok"}


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, max_turns: int = 0):
    """获取聊天历史"""
    from project.infra.memory import get_chat_memory
    memory = get_chat_memory()
    return {"session_id": session_id, "messages": memory.get_history(session_id, max_turns)}


@app.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清除聊天历史"""
    from project.infra.memory import get_chat_memory
    memory = get_chat_memory()
    memory.clear(session_id)
    return {"status": "ok"}


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import uvicorn

    print("🚀 启动 RAG API Server")
    print(f"   Chat Model: {CHAT_MODEL}")
    print(f"   Embed Model: {EMBED_MODEL}")
    print(f"   Vector DB: {VECTOR_DB_DIR}")
    print(f"   API Docs: http://localhost:8000/docs")
    print()

    uvicorn.run(
        "project.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
