"""配置模块：读取 .env，构建 LLM 客户端。"""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从 .env 读取的配置，缺失字段会在启动时报错。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    request_timeout: int = 60
    max_sub_agents: int = 5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """单例配置，避免重复读取 .env。"""
    return Settings()


@lru_cache
def get_llm() -> AsyncOpenAI:
    """构建指向 DeepSeek 的异步 OpenAI 兼容客户端。"""
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.deepseek_api_key,
        base_url=s.deepseek_base_url,
        timeout=s.request_timeout,
    )
