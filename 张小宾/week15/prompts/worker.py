"""子 Agent 的 Worker 提示词：按角色执行单个子任务。"""
from __future__ import annotations

_ROLE_DESCRIPTIONS = {
    "researcher": "你是一名调研专家，擅长检索信息、整理资料、归纳要点。",
    "coder": "你是一名工程师，擅长编写清晰、可运行、有注释的代码。",
    "writer": "你是一名文案撰写者，擅长结构化表达，输出清晰易读的文字。",
    "reviewer": "你是一名审查者，擅长发现问题、校验逻辑、给出改进建议。",
}

_DEFAULT = "你是一名通用任务执行者，按要求认真完成给定任务。"


def build_prompt(role: str) -> str:
    desc = _ROLE_DESCRIPTIONS.get(role, _DEFAULT)
    return (
        f"{desc}\n\n"
        "要求：\n"
        "1. 严格围绕给定任务作答，不要扩展无关内容。\n"
        "2. 输出结构化、条理清晰。\n"
        "3. 如果任务需要代码，请给出可直接运行的代码块。\n"
        "4. 不要编造事实，不确定时明确说明。"
    )
