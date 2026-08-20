from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any

from .models import Subtask, SubtaskResult, TaskPlan


PLANNER_INSTRUCTIONS = """
你是一个分层多 Agent 系统的主 Agent，负责规划和任务拆解。请把用户的复杂任务拆分为
1 到 8 个边界清晰、互不重叠的子任务。优先设计相互独立的子任务，以便系统并行执行；
只有当某个子任务确实需要前置结果时，才添加依赖关系。每个子任务必须包含足够的上下文、
明确的执行要求和具体的交付物。你只负责任务规划，不要直接解决任务。仅返回符合指定格式
的结构化计划。reasoning_summary 只需简要说明拆解依据，不得输出隐藏的思维链。
""".strip()


WORKER_INSTRUCTIONS = """
你是负责执行专项工作的子 Agent。请只完成分配给你的、边界明确的子任务，不要扩大任务
范围。回答应具体、准确，并以事实和证据为导向。输出一份上下文完整、可独立理解的结果，
供主 Agent 后续汇总。若输入中包含前置子任务结果，请合理使用，但不要重复堆砌其内容。
""".strip()


SYNTHESIZER_INSTRUCTIONS = """
你是负责生成最终答复的主 Agent。请根据用户的原始任务和所有子 Agent 的结果，整合出一份
连贯、准确、可直接交付给用户的回答。你需要解决结果之间的冲突、删除重复内容，并严格遵守
用户的原始要求。若有关键子任务失败或存在重要不确定性，应在最终回答中明确说明。除非内部
执行过程对用户确有帮助，否则不要描述系统内部的调度细节。不得简单拼接子 Agent 的输出。
""".strip()


class ResponsesAPIError(RuntimeError):
    """原生 Responses API 请求失败或响应无效时抛出的异常。"""


Transport = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """将 Pydantic Schema 转换为 Responses API 支持的严格 JSON Schema。"""

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            # Pydantic 会给带默认值的字段生成 default，但严格模式不接受该关键字。
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                # 严格模式要求对象的所有属性都列入 required，且禁止额外字段。
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    copied = json.loads(json.dumps(schema))
    visit(copied)
    return copied


def _extract_output_text(response: dict[str, Any]) -> str:
    """从 Responses API 原始 JSON 响应中提取模型输出文本。"""

    if response.get("error"):
        error = response["error"]
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise ResponsesAPIError(message)
    if response.get("status") not in (None, "completed"):
        details = response.get("incomplete_details") or response.get("status")
        raise ResponsesAPIError(f"response did not complete: {details}")

    # 一个响应可能包含多个 output item 或多个文本片段，需要按返回顺序合并。
    texts: list[str] = []
    refusals: list[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif part.get("type") == "refusal":
                refusals.append(str(part.get("refusal", "request refused")))

    if texts:
        return "\n".join(texts)
    if refusals:
        raise ResponsesAPIError("; ".join(refusals))
    raise ResponsesAPIError("response contains no output_text")


class RawResponsesClient:
    """仅使用 Python 标准库实现的轻量异步 Responses API 客户端。"""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError("api_key cannot be empty")
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/responses"
        self.timeout = timeout
        self.max_retries = max_retries

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        # urllib 是同步接口，放入工作线程，避免阻塞 asyncio 事件循环和其他子 Agent。
        return await asyncio.to_thread(self._create_sync, payload)

    def _create_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "hierarchical-agent-system/0.1.0",
            },
        )

        # 仅对限流、服务端错误和临时网络错误重试；客户端参数错误直接返回。
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as raw:
                    return json.loads(raw.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.max_retries:
                    raise ResponsesAPIError(
                        f"Responses API HTTP {exc.code}: {error_body}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                # 优先遵守服务端 Retry-After，否则使用指数退避并加入随机抖动。
                delay = float(retry_after) if retry_after else 2**attempt + random.random()
                time.sleep(min(delay, 30.0))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise ResponsesAPIError(f"Responses API network error: {exc}") from exc
                time.sleep(min(2**attempt + random.random(), 30.0))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResponsesAPIError("Responses API returned invalid JSON") from exc

        raise AssertionError("unreachable")


class OpenAIHTTPBackend:
    """直接基于 REST Responses API 实现的 Agent 后端。"""

    def __init__(self, transport: Transport, *, model: str = "gpt-5-mini") -> None:
        self.transport = transport
        self.model = model

    async def _respond(
        self,
        *,
        instructions: str,
        input_text: str,
        text_format: dict[str, Any] | None = None,
    ) -> str:
        # 三种 Agent 角色共用同一个原生 HTTP 调用入口，区别由 instructions 决定。
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "store": False,
        }
        if text_format is not None:
            # 规划阶段通过 JSON Schema 强制模型返回可验证的任务 DAG。
            payload["text"] = {"format": text_format}
        response = await self.transport(payload)
        return _extract_output_text(response)

    async def create_plan(self, task: str) -> TaskPlan:
        # 主 Agent 第一次调用模型：只拆解任务，不生成最终答案。
        output = await self._respond(
            instructions=PLANNER_INSTRUCTIONS,
            input_text=task,
            text_format={
                "type": "json_schema",
                "name": "task_plan",
                "strict": True,
                "schema": _strict_schema(TaskPlan.model_json_schema()),
            },
        )
        try:
            return TaskPlan.model_validate_json(output)
        except ValueError as exc:
            raise ResponsesAPIError(f"invalid task plan: {exc}") from exc

    async def execute_subtask(
        self,
        original_task: str,
        subtask: Subtask,
        dependency_results: list[SubtaskResult],
    ) -> str:
        # 每个子 Agent 只收到原始任务、自己的指令及其直接依赖结果，减少上下文干扰。
        payload = {
            "original_task": original_task,
            "assigned_subtask": subtask.model_dump(),
            "dependency_results": [item.model_dump() for item in dependency_results],
        }
        return await self._respond(
            instructions=WORKER_INSTRUCTIONS,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
        )

    async def synthesize(
        self,
        original_task: str,
        plan: TaskPlan,
        results: list[SubtaskResult],
    ) -> str:
        # 所有子任务结束后，再由主 Agent 做冲突消解、去重并生成最终回答。
        payload = {
            "original_task": original_task,
            "synthesis_guidance": plan.synthesis_guidance,
            "plan": plan.model_dump(),
            "worker_results": [item.model_dump() for item in results],
        }
        return await self._respond(
            instructions=SYNTHESIZER_INSTRUCTIONS,
            input_text=json.dumps(payload, ensure_ascii=False, indent=2),
        )
