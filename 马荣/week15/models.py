from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Subtask(BaseModel):
    """主 Agent 拆解出的一个边界明确的子任务。"""

    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=120)
    instructions: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    """包含依赖关系、可按批次并行执行的任务计划。"""

    reasoning_summary: str = Field(
        description="简要说明任务拆解依据，不包含隐藏的思维链。"
    )
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)
    synthesis_guidance: str = Field(
        description="说明最终回答必须包含什么，以及如何协调各子任务结果。"
    )

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskPlan":
        # 子任务 ID 是依赖引用的主键，因此必须唯一。
        ids = [task.id for task in self.subtasks]
        if len(ids) != len(set(ids)):
            raise ValueError("subtask ids must be unique")

        known = set(ids)
        for task in self.subtasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(f"{task.id} has unknown dependencies: {sorted(unknown)}")
            if task.id in task.depends_on:
                raise ValueError(f"{task.id} cannot depend on itself")

        # 使用深度优先搜索检测 DAG 中是否存在循环依赖。
        visited: set[str] = set()
        visiting: set[str] = set()
        dependency_map = {task.id: task.depends_on for task in self.subtasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("subtask dependency graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependency_map[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)
        return self


class SubtaskResult(BaseModel):
    id: str
    title: str
    output: str
    success: bool = True
    attempts: int = 1


class RunReport(BaseModel):
    task: str
    plan: TaskPlan
    subtask_results: list[SubtaskResult]
    final_answer: str
