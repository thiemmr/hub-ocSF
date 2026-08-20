"""日志模块：统一控制台输出，级别由 .env 的 LOG_LEVEL 控制。"""
from __future__ import annotations

import logging
import os
import sys

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    # 优先读 .env 中的 LOG_LEVEL，读不到则用环境变量，再读不到则默认 INFO。
    # 避免在导入阶段强制依赖完整配置（如 API Key），保证模块可独立导入。
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
