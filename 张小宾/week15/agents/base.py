"""BaseAgent：统一的 LLM 调用封装，主/子 Agent 共用。"""
from __future__ import annotations

from openai import AsyncOpenAI

from config import get_llm, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent:
    """所有 Agent 的基类，封装 system prompt 与 LLM 调用。"""

    def __init__(self, system_prompt: str, role: str = "agent") -> None:
        self.llm: AsyncOpenAI = get_llm()
        self.settings = get_settings()
        self.system_prompt = system_prompt
        self.role = role

    async def chat(self, user_msg: str, json_mode: bool = False) -> str:
        """调用 DeepSeek，返回文本内容。

        Args:
            user_msg: 用户消息内容
            json_mode: 是否强制 JSON 输出（用于结构化解析）
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        logger.debug("[%s] 调用 LLM, json_mode=%s", self.role, json_mode)
        resp = await self.llm.chat.completions.create(
            model=self.settings.deepseek_model,
            messages=messages,
            response_format={"type": "json_object"} if json_mode else None,
        )
        content = resp.choices[0].message.content or ""
        logger.debug("[%s] LLM 返回 %d 字符", self.role, len(content))
        return content
