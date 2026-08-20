"""程序入口：读取用户输入，启动主 Agent。"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

# 先加载 .env，让 LOG_LEVEL 等环境变量对 logger 与配置模块都可见
load_dotenv()

from agents.main_agent import MainAgent
from utils.logger import get_logger

logger = get_logger(__name__)


async def main() -> None:
    if len(sys.argv) > 1:
        # 支持命令行直接传任务：python main.py "你的任务"
        user_task = " ".join(sys.argv[1:])
    else:
        user_task = input("请输入任务: ").strip()

    if not user_task:
        print("任务为空，退出。")
        return

    logger.info("开始处理任务: %s", user_task)
    try:
        result = await MainAgent().run(user_task)
    except Exception as e:  # noqa: BLE001
        # 配置缺失（如未设置 DEEPSEEK_API_KEY）等错误，给出友好提示
        if "deepseek_api_key" in str(e).lower() or "field required" in str(e).lower():
            print("\n[配置错误] 未检测到有效的 .env 配置。")
            print("请执行: cp .env.example .env")
            print("然后在 .env 中填入真实的 DEEPSEEK_API_KEY。")
            sys.exit(1)
        raise
    print("\n===== 最终结果 =====\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
