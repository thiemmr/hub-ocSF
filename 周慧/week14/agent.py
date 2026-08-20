"""
AI 热点日报生成 Agent（基于 Function Calling）

使用 DeepSeek V4 Flash API。所有工作流、分类规则、日期控制、文档模板均由
skill/SKILL.md 定义（单一数据源），本文件运行时加载该 Skill 作为 System Prompt，
不在此重复维护这些内容。

Agent 仅负责：
  1. 加载 SKILL.md 作为 System Prompt
  2. 注入当前日期（今天 + 昨天）到用户消息
  3. 通过 Function Calling 调度 web_search / web_fetch / write_markdown 三个工具
  4. 打印执行过程并交付结果

使用方式：
  python agent.py                              # 默认生成今天+昨天的 AI 热点日报
  python agent.py --topic "最近 AI 大模型进展"   # 自定义主题
  python agent.py --output-dir ./outputs       # 自定义输出目录
  python agent.py --max-steps 20              # 调整最大推理步数

环境变量：
  DEEPSEEK_API_KEY  必填
  DEEPSEEK_URL      可选，默认 https://api.deepseek.com
  AGENT_MODEL       可选，默认 deepseek-v4-flash
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Generator

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── 配置 ─────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com")
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Skill 定义文件（单一数据源：工作流、分类规则、日期控制、文档模板均在此维护）
SKILL_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "skill", "SKILL.md")

# Agent 特有的工具绑定说明（不属于 Skill 内容，仅声明本 Agent 可用的工具）
AGENT_TOOLS_NOTE = """

## 本 Agent 可用工具

你只能使用以下三个工具完成上述 Skill 定义的工作流：
- `web_search`：多引擎网页搜索（搜索词必须带日期，按 Skill 第二步执行）
- `web_fetch`：抓取网页正文（用于深入阅读重要文章，须核对发表日期）
- `write_markdown`：将整理好的日报写入 Markdown 文件（按 Skill 第四步执行）

工具调用顺序：搜索 →（按需）阅读 → 整理 → 写入文档 → 交付要点。
每次只做一个动作。
"""


def _load_skill_prompt(skill_path: str = SKILL_PATH) -> str:
    """
    从 SKILL.md 加载 Skill 定义作为 System Prompt。

    Skill 是唯一数据源：工作流、分类规则、日期控制、文档模板均由 SKILL.md 维护，
    本文件不重复定义这些内容。此函数读取 SKILL.md，剥离 YAML frontmatter，
    返回正文，并追加 Agent 特有的工具绑定说明。

    Args:
        skill_path: SKILL.md 文件路径

    Returns:
        完整的 System Prompt 字符串（Skill 正文 + 工具说明）
    """
    with open(skill_path, encoding="utf-8") as f:
        raw = f.read()

    # 剥离 YAML frontmatter（开头的 --- ... --- 块）
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            skill_body = parts[2].strip()
        else:
            skill_body = raw
    else:
        skill_body = raw.strip()

    return skill_body + "\n" + AGENT_TOOLS_NOTE


def _get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "请设置环境变量 DEEPSEEK_API_KEY，例如：\n"
            "  export DEEPSEEK_API_KEY='sk-你的密钥'"
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_URL)


# ── Agent 实现 ───────────────────────────────────────────────────────────────

class AIWeeklyNewsAgent:
    """
    AI 热点日报生成 Agent

    使用 Function Calling 模式：
    1. 加载 SKILL.md 作为 System Prompt（单一数据源）
    2. LLM 自主决定何时调用哪个工具
    3. 工具返回结果后，LLM 继续推理或给出最终答案
    4. 最多执行 max_steps 步
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_steps: int = 20):
        from tools import TOOLS_MAP, TOOLS_SCHEMA

        self.model = model
        self.max_steps = max_steps
        self.tools_map = TOOLS_MAP
        self.tools_schema = TOOLS_SCHEMA
        self.client = _get_client()
        self.messages = [
            {"role": "system", "content": _load_skill_prompt()},
        ]
        self.search_count = 0
        self.fetch_count = 0

    def run(self, topic: str) -> Generator[dict, None, None]:
        """
        运行 Agent 生成 AI 热点日报

        Args:
            topic: 用户的主题请求，如"今天和昨天 AI 圈有什么热门事件"

        Yields:
            每一步的执行结果字典
        """
        self.messages.append({"role": "user", "content": topic})

        today = datetime.now()
        yesterday = datetime.fromtimestamp(today.timestamp() - 86400)
        today_str = today.strftime("%Y年%m月%d日")
        yesterday_str = yesterday.strftime("%Y年%m月%d日")
        today_en = today.strftime("%B %d %Y")
        yesterday_en = yesterday.strftime("%B %d %Y")

        # 在用户消息中注入当前日期信息，强制限定检索范围为今天和昨天
        self.messages[-1]["content"] += (
            f"\n\n当前日期：{today_str}（{today.strftime('%Y-%m-%d')}）。"
            f"请严格只检索「今天 {today_str}」和「昨天 {yesterday_str}」两天的信息。"
            f"搜索时使用以下日期关键词："
            f"中文搜索用 '{today_str}' 和 '{yesterday_str}'，"
            f"英文搜索用 '{today_en}' 和 '{yesterday_en}'。"
            f"超出这两天范围的信息一律不收录。"
            f"输出文件名格式：AI_Daily_News_{today.strftime('%Y%m%d')}.md"
        )

        yield {
            "step": 0,
            "type": "info",
            "message": f"🤖 AI 热点日报 Agent 启动，模型: {self.model}",
        }

        for step in range(1, self.max_steps + 1):
            # 调用 LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools_schema,
                    tool_choice="auto",
                    temperature=0.1,
                )
            except Exception as e:
                yield {
                    "step": step,
                    "type": "error",
                    "message": f"LLM 调用失败: {e}",
                }
                break

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 检查是否有 tool_calls
            if finish_reason == "stop" or not msg.tool_calls:
                # LLM 直接给出最终答案
                self.messages.append(
                    {"role": "assistant", "content": msg.content or ""}
                )
                yield {
                    "step": step,
                    "type": "final",
                    "answer": msg.content or "（模型返回空内容）",
                }
                return

            # 处理 tool_calls
            self.messages.append(
                {
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls or []
                    ],
                }
            )

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # 统计调用次数
                if tool_name == "web_search":
                    self.search_count += 1
                elif tool_name == "web_fetch":
                    self.fetch_count += 1

                # 执行工具
                tool_fn = self.tools_map.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具: {tool_name}"
                else:
                    logger.info(
                        f"[Step {step}] 调用 {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})"
                    )
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"
                    except Exception as e:
                        observation = f"工具执行出错: {e}"

                # 生成步骤结果
                step_result = {
                    "step": step,
                    "type": "action",
                    "tool": tool_name,
                    "args": tool_args,
                    "observation_summary": (
                        observation[:200] + "..."
                        if len(observation) > 200
                        else observation
                    ),
                }
                yield step_result

                # 将工具结果加入消息历史
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(observation),
                    }
                )

        # 达到最大步数
        yield {
            "step": self.max_steps + 1,
            "type": "max_steps",
            "message": f"⚠️ 已达最大步数 {self.max_steps}，搜索 {self.search_count} 次，抓取 {self.fetch_count} 次",
        }


# ── CLI 输出 ─────────────────────────────────────────────────────────────────

COLORS = {
    "info": "\033[36m",
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def run_and_print(
    topic: str,
    max_steps: int = 20,
    output_dir: str = "",
):
    """
    运行 Agent 并打印彩色输出

    output_dir 会注入到用户消息中，引导 LLM 调用 write_markdown 时使用该目录。
    """
    print(f"\n{'=' * 70}")
    print(_c("bold", "🤖 AI 热点日报生成 Agent"))
    print(f"   模型: {DEFAULT_MODEL}")
    print(f"   主题: {topic}")
    print(f"   输出目录: {output_dir or '(当前目录)'}")
    print(f"   最大步数: {max_steps}")
    print("=" * 70)

    # 将输出目录注入主题，引导 LLM 在调用 write_markdown 时使用
    effective_topic = topic
    if output_dir:
        effective_topic += f"\n\n请将最终生成的 Markdown 文件写入目录：{output_dir}"

    agent = AIWeeklyNewsAgent(max_steps=max_steps)
    start = time.time()

    last_result = None
    for step_data in agent.run(effective_topic):
        stype = step_data["type"]

        if stype == "info":
            print(f"\n{_c('info', step_data['message'])}")

        elif stype == "action":
            tool = step_data["tool"]
            args_str = json.dumps(step_data["args"], ensure_ascii=False)
            summary = step_data["observation_summary"]

            tool_emoji = {"web_search": "🔍", "web_fetch": "📄", "write_markdown": "📝"}.get(
                tool, "🔧"
            )

            print(f"\n{_c('action', f'[Step {step_data['step']}] {tool_emoji} {tool}')}")
            print(_c("action", f"   参数: {args_str[:150]}"))
            print(_c("obs", f"   结果: {summary}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─' * 70}")
            print(_c("final", "✅ Agent 完成！"))
            print(f"\n{step_data['answer']}")
            print(f"\n{'─' * 70}")
            print(
                _c(
                    "final",
                    f"📊 统计: 搜索 {agent.search_count} 次，抓取 {agent.fetch_count} 次，耗时 {elapsed:.1f}s",
                )
            )
            last_result = step_data

        elif stype == "error":
            print(_c("error", f"\n❌ 错误: {step_data['message']}"))

        elif stype == "max_steps":
            elapsed = time.time() - start
            print(_c("error", f"\n⚠️  {step_data['message']} (耗时 {elapsed:.1f}s)"))

    return last_result


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI 热点日报生成 Agent - 检索今天和昨天的 AI 热门事件，按国内/国外/AI技术三大类各取热度前三",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成今天+昨天的 AI 热点日报
  python agent.py

  # 自定义主题
  python agent.py --topic "最近 AI 大模型有什么新发布"

  # 指定输出目录
  python agent.py --output-dir ./my_reports

  # 调整最大推理步数
  python agent.py --max-steps 30
        """,
    )
    parser.add_argument(
        "--topic",
        default="请帮我搜索今天和昨天 AI 圈发生了哪些热门事件，按国内、国外、AI技术三大类各取热度前三",
        help="日报主题描述",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Agent 最大推理步数（默认 20）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Markdown 输出目录（默认当前目录）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，仅输出最终结果",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    run_and_print(
        topic=args.topic,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
