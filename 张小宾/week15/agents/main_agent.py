"""MainAgent：编排核心，负责拆解任务、并发分发、汇总结果。"""
from __future__ import annotations

import asyncio
import json

from agents.base import BaseAgent
from agents.sub_agent import SubAgent
from prompts.aggregator import build_prompt as build_aggregator_prompt
from prompts.planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from prompts.planner import build_plan_prompt
from schemas import Plan, SubResult, SubTask
from utils.logger import get_logger

logger = get_logger(__name__)


class MainAgent(BaseAgent):
    """主 Agent（Orchestrator）：plan → dispatch → aggregate。"""

    def __init__(self) -> None:
        super().__init__(system_prompt=PLANNER_SYSTEM_PROMPT, role="orchestrator")

    async def run(self, user_task: str) -> str:
        # 1. 规划：拆解任务
        plan = await self._plan(user_task)
        if not plan.subtasks:
            logger.warning("未拆解出任何子任务，直接返回")
            return "未能生成可执行的子任务计划。"

        logger.info("拆解出 %d 个子任务", len(plan.subtasks))
        for sub in plan.subtasks:
            logger.info("  - %s [%s]: %s", sub.id, sub.agent_role, sub.description[:60])

        # 2. 分发：并发执行子任务
        results = await self._dispatch(plan.subtasks)

        # 3. 汇总
        return await self._aggregate(user_task, results)

    async def _plan(self, user_task: str) -> Plan:
        raw = await self.chat(build_plan_prompt(user_task), json_mode=True)
        try:
            return Plan(**json.loads(raw))
        except Exception as e:  # noqa: BLE001
            logger.error("Plan 解析失败: %s\n原始输出: %s", e, raw)
            return Plan()

    async def _dispatch(self, subtasks: list[SubTask]) -> list[SubResult]:
        sem = asyncio.Semaphore(self.settings.max_sub_agents)

        async def _run(sub: SubTask) -> SubResult:
            async with sem:
                return await SubAgent(sub.agent_role).run(sub)

        return await asyncio.gather(*[_run(s) for s in subtasks])

    async def _aggregate(self, user_task: str, results: list[SubResult]) -> str:
        prompt = build_aggregator_prompt(
            user_task, [r.model_dump() for r in results]
        )
        return await self.chat(prompt)
