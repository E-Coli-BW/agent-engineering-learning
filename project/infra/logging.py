"""
结构化日志
===========

统一日志格式为 JSON，方便 ELK/Loki 采集和分析。

用法:
  from project.infra.logging import setup_logging
  setup_logging("rag-api")

  import logging
  logger = logging.getLogger("rag-api")
  logger.info("query processed", extra={"request_id": "abc", "latency_ms": 42})

输出 (JSON):
  {"timestamp":"2026-03-30T14:00:00","level":"INFO","service":"rag-api",
   "message":"query processed","request_id":"abc","latency_ms":42}

环境变量:
  LOG_FORMAT=json  → JSON 格式 (生产, 默认)
  LOG_FORMAT=text  → 传统文本格式 (本地开发)
  LOG_LEVEL=DEBUG  → 日志级别
"""

import os
import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """JSON 日志格式化器"""

    def __init__(self, service: str = "unknown"):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并 extra 字段 (request_id, latency_ms 等)
        for key in ("request_id", "latency_ms", "task_id", "skill", "route",
                     "method", "path", "status_code", "client_ip", "user_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        # 异常信息
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """传统文本格式 (本地开发友好)"""

    def __init__(self, service: str = "unknown"):
        super().__init__(
            fmt=f"%(asctime)s [{service}] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def setup_logging(service: str = "app") -> None:
    """
    初始化结构化日志。

    根据 LOG_FORMAT 环境变量选择格式:
      - json (默认): 生产环境, ELK/Loki 采集
      - text: 本地开发, 人眼可读
    """
    log_format = os.getenv("LOG_FORMAT", "json").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # 清除已有 handler (避免重复)
    root.handlers.clear()

    handler = logging.StreamHandler()
    if log_format == "text":
        handler.setFormatter(TextFormatter(service))
    else:
        handler.setFormatter(JsonFormatter(service))

    root.addHandler(handler)

    # 降低第三方库噪音
    for noisy in ("urllib3", "httpcore", "chromadb", "httpx", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
