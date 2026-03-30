"""
统一错误模型
=============

所有后端 API 统一返回的错误格式:
{
    "error": {
        "code": "VECTOR_STORE_NOT_READY",
        "message": "向量库未初始化，请先运行 ETL Pipeline",
        "request_id": "abc12345",
        "timestamp": "2026-03-30T14:00:00"
    }
}

用法:
  from project.infra.errors import AppError, error_response, register_error_handlers

  # 在路由中抛出
  raise AppError("VECTOR_STORE_NOT_READY", "向量库未初始化", status_code=503)

  # 在 FastAPI app 上注册全局处理器
  register_error_handlers(app)
"""

from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """统一业务异常"""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def error_response(code: str, message: str, status_code: int = 400, request_id: str = "") -> JSONResponse:
    """构造统一错误响应"""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """注册全局错误处理器"""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        request_id = request.headers.get("X-Request-Id", "")
        return error_response(exc.code, exc.message, exc.status_code, request_id)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        request_id = request.headers.get("X-Request-Id", "")
        return error_response(
            "INTERNAL_ERROR",
            f"内部错误: {str(exc)[:200]}",
            500,
            request_id,
        )
