"""数据结构：主 Agent 与子 Agent 之间的结构化通信协议。"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """主 Agent 拆解出的单个子任务。"""

    id: str = Field(description="子任务唯一标识，如 t1/t2")
    description: str = Field(description="子任务的具体描述与上下文")
    agent_role: str = Field(description="执行角色，如 researcher / coder / writer / reviewer")


class Plan(BaseModel):
    """主 Agent 生成的执行计划。"""

    subtasks: list[SubTask] = Field(default_factory=list, description="拆解出的子任务列表")


class SubResult(BaseModel):
    """子 Agent 执行结果。"""

    task_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
