"""
LLM 客户端抽象层

支持 OpenAI 兼容接口（包括 DeepSeek、Moonshot 等），
统一封装 chat_completion，供 SkillGenerator 和 AgentHarness 使用。
"""

from __future__ import annotations

import os
from typing import Any, Optional


class LLMClient:
    """LLM 客户端封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        # 懒加载 openai 客户端
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "请安装 openai: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """调用 LLM chat completion，返回字符串内容。"""
        client = self._get_client()
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        return content or ""

    def structured_chat(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        temperature: float = 0.5,
    ) -> dict[str, Any]:
        """调用 LLM 并返回结构化 JSON 输出。"""
        import json

        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            },
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
