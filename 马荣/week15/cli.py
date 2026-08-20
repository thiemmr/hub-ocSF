from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .openai_backend import OpenAIHTTPBackend, RawResponsesClient
from .orchestrator import HierarchicalAgentSystem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="主 Agent 拆解任务、并行调用子 Agent，并整合最终回答。"
    )
    parser.add_argument("task", nargs="+", help="要处理的复杂任务")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI 模型名（默认读取 OPENAI_MODEL）",
    )
    parser.add_argument("--concurrency", type=int, default=4, help="最大并发子 Agent 数")
    parser.add_argument("--attempts", type=int, default=2, help="每个子任务最大尝试次数")
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：请先设置 OPENAI_API_KEY 环境变量。", file=sys.stderr)
        return 2

    client = RawResponsesClient(os.environ["OPENAI_API_KEY"])
    backend = OpenAIHTTPBackend(client.create, model=args.model)
    system = HierarchicalAgentSystem(
        backend,
        max_concurrency=args.concurrency,
        max_attempts=args.attempts,
        on_event=lambda _kind, message: print(f"→ {message}", file=sys.stderr),
    )
    report = await system.run(" ".join(args.task))
    print(report.final_answer)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
