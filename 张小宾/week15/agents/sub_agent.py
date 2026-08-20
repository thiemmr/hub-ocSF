"""SubAgent：执行单个子任务，返回结构化结果。"""
from __future__ import annotations

from agents.base import BaseAgent
from prompts.worker import build_prompt
from schemas import SubResult, SubTask
from utils.logger import get_logger

logger = get_logger(__name__)


class SubAgent(BaseAgent):
    """子 Agent，按指定角色执行单个子任务。"""

    def __init__(self, role: str) -> None:
        super().__init__(system_prompt=build_prompt(role), role=role)

    async def run(self, task: SubTask) -> SubResult:
        logger.info("子任务开始 [%s] %s: %s", task.id, task.agent_role, task.description[:60])
        try:
            output = await self.chat(task.description)
            logger.info("子任务完成 [%s]", task.id)
            return SubResult(task_id=task.id, success=True, output=output)
        except Exception as e:  # noqa: BLE001
            logger.exception("子任务失败 [%s]", task.id)
            return SubResult(task_id=task.id, success=False, output="", error=str(e))
