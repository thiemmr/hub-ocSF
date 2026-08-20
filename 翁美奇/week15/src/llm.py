"""LLM 客户端共享配置（OpenAI 兼容接口，默认 deepseek-v4-flash）。

主 Agent 与子 Agent 共用此客户端，保证模型/密钥/地址统一配置。
"""
import os

from openai import OpenAI
from dotenv import load_dotenv

# ========== 配置 ==========
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

load_dotenv()

LLM_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def chat(messages, tools=None, temperature=0.2, stop=None):
    """统一封装的 chat completion 调用。

    stop: 停止序列列表（如 ["Observation:"]），让 LLM 输出到此处即停。
    """
    kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    if stop:
        kwargs["stop"] = stop
    return client.chat.completions.create(**kwargs)
