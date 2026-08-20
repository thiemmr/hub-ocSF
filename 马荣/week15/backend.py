from __future__ import annotations

from typing import Protocol

from .models import Subtask, SubtaskResult, TaskPlan


class AgentBackend(Protocol):
    """Boundary between deterministic orchestration and an agent runtime."""

    async def create_plan(self, task: str) -> TaskPlan: ...

    async def execute_subtask(
        self,
        original_task: str,
        subtask: Subtask,
        dependency_results: list[SubtaskResult],
    ) -> str: ...

    async def synthesize(
        self,
        original_task: str,
        plan: TaskPlan,
        results: list[SubtaskResult],
    ) -> str: ...

