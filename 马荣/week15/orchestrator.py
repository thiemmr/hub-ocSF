from __future__ import annotations

import asyncio
from collections.abc import Callable

from .backend import AgentBackend
from .models import RunReport, Subtask, SubtaskResult

EventHandler = Callable[[str, str], None]


class HierarchicalAgentSystem:
    """将复杂任务拆成 DAG，并行执行子任务，然后整合最终结果。"""

    def __init__(
        self,
        backend: AgentBackend,
        *,
        max_concurrency: int = 4,
        max_attempts: int = 2,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于或等于 1")
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于或等于 1")
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.max_attempts = max_attempts
        self.on_event = on_event or (lambda _kind, _message: None)

    async def run(self, task: str) -> RunReport:
        if not task.strip():
            raise ValueError("任务内容不能为空")

        # 第一阶段：主 Agent 将自然语言任务转换成经过校验的任务 DAG。
        self._emit("planning", "主 Agent 正在拆解任务")
        plan = await self.backend.create_plan(task)
        self._emit("planned", f"已生成 {len(plan.subtasks)} 个子任务")

        pending = {item.id: item for item in plan.subtasks}
        completed: dict[str, SubtaskResult] = {}

        # 信号量限制同时运行的子 Agent 数量，避免瞬间触发过多 API 请求。
        semaphore = asyncio.Semaphore(self.max_concurrency)

        while pending:
            # 每一轮只挑选“所有依赖都已完成”的子任务；同一轮任务可以并发执行。
            ready = [
                item
                for item in pending.values()
                if all(dependency in completed for dependency in item.depends_on)
            ]

            # TaskPlan 已经检查过循环依赖；进入这里通常意味着编排器状态异常。
            if not ready:
                raise RuntimeError("没有可执行的子任务，请检查任务依赖关系")

            # asyncio.gather 会并行等待本轮所有子 Agent，而非逐个串行执行。
            wave_results = await asyncio.gather(
                *[
                    self._run_one(task, item, completed, semaphore)
                    for item in ready
                ]
            )
            for result in wave_results:
                completed[result.id] = result
                pending.pop(result.id)

        # 恢复为规划时的任务顺序，让最终汇总输入稳定、测试结果可复现。
        ordered_results = [completed[item.id] for item in plan.subtasks]

        # 第三阶段：主 Agent 读取计划和全部子结果，生成唯一的最终回答。
        self._emit("synthesizing", "主 Agent 正在整合子 Agent 结果")
        final_answer = await self.backend.synthesize(task, plan, ordered_results)
        self._emit("completed", "最终回答已生成")
        return RunReport(
            task=task,
            plan=plan,
            subtask_results=ordered_results,
            final_answer=final_answer,
        )

    async def _run_one(
        self,
        original_task: str,
        subtask: Subtask,
        completed: dict[str, SubtaskResult],
        semaphore: asyncio.Semaphore,
    ) -> SubtaskResult:
        # 子 Agent 只获取显式依赖结果，避免把无关子任务全部塞进上下文。
        dependencies = [completed[item] for item in subtask.depends_on]
        last_error: Exception | None = None

        async with semaphore:
            for attempt in range(1, self.max_attempts + 1):
                self._emit(
                    "worker_started",
                    f"[{subtask.id}] {subtask.title}（第 {attempt} 次尝试）",
                )
                try:
                    output = await self.backend.execute_subtask(
                        original_task, subtask, dependencies
                    )
                    self._emit("worker_completed", f"[{subtask.id}] 已完成")
                    return SubtaskResult(
                        id=subtask.id,
                        title=subtask.title,
                        output=output,
                        attempts=attempt,
                    )
                except Exception as exc:
                    # 单个子 Agent 失败只触发自身重试，不取消同一轮中的其他任务。
                    last_error = exc
                    self._emit("worker_retry", f"[{subtask.id}] 失败：{exc}")

        # 重试耗尽后返回失败结果，让主 Agent 决定如何在最终回答中披露或降级。
        return SubtaskResult(
            id=subtask.id,
            title=subtask.title,
            output=f"子 Agent 在 {self.max_attempts} 次尝试后仍失败：{last_error}",
            success=False,
            attempts=self.max_attempts,
        )

    def _emit(self, kind: str, message: str) -> None:
        """向 CLI 或其他调用方发送执行进度事件。"""

        self.on_event(kind, message)

