"""主 Agent 的 Aggregator 提示词：负责汇总子 Agent 的执行结果。"""
from __future__ import annotations

import json


def build_prompt(user_task: str, results: list[dict]) -> str:
    return f"""请根据以下子任务的执行结果，汇总出一份针对用户任务的最终答复。

要求：
1. 整合所有子结果，去除冗余，保留有效信息。
2. 如有子任务失败，请说明影响并尽量给出补救建议。
3. 输出结构清晰，可使用 Markdown 标题/列表。
4. 不要复述原始 JSON，直接给出最终交付内容。

用户任务：
{user_task}

各子任务执行结果（JSON）：
{json.dumps(results, ensure_ascii=False, indent=2)}
"""
