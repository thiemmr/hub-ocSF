"""
Agent Harness 主循环

核心职责：
1. 接收用户输入
2. 通过 SkillManager 渐进式加载匹配的技能
3. 将系统提示 + 技能索引 + 已加载技能详情 + 历史对话组装成 LLM 上下文
4. 调用 LLM 生成回复
5. 维护对话历史
6. 处理特殊命令（生成技能、列出技能等）
"""

from __future__ import annotations

from typing import Any

from harness.llm import LLMClient
from harness.manager import SkillManager
from skills.schema import Skill


class AgentHarness:
    """Harness Agent —— 支持渐进式 Skill 加载的 LLM Agent。"""

    def __init__(
        self,
        llm_client: LLMClient,
        skill_manager: SkillManager,
        system_prompt: str = "",
        max_history: int = 20,
    ):
        self.llm = llm_client
        self.manager = skill_manager
        self.system_prompt = system_prompt
        self.max_history = max_history
        self._history: list[dict[str, str]] = []

    # ── 公开 API ──

    def run_once(self, user_query: str) -> str:
        """处理一次用户请求，返回 Agent 回复。

        内部流程：
          1. 渐进式加载匹配技能
          2. 组装系统提示 + 索引 + 技能详情
          3. 调用 LLM
          4. 更新历史
        """
        # Step 1: 渐进式加载
        matched_skills = self.manager.load_for_query(user_query)

        # Step 2: 构建消息
        messages = self._build_messages(user_query, matched_skills)
        # Step 3: LLM 调用
        response = self.llm.chat(messages, temperature=0.7)
        # Step 4: 更新历史
        self._history.append({"role": "user", "content": user_query})
        self._history.append({"role": "assistant", "content": response})
        self._trim_history()

        return response

    def generate_skill(
        self,
        skill_name: str,
        skill_description: str,
        reflection: bool = True,
    ) -> str:
        """生成新技能并保存到技能存储。

        返回技能保存路径。
        """
        from harness.generator import SkillGenerator

        generator = SkillGenerator(self.llm)

        if reflection:
            skill = generator.generate_with_reflection(
                skill_name, skill_description
            )
        else:
            skill = generator.generate(skill_name, skill_description)

        path = self.manager.add_skill(skill)
        return path

    def list_skills(self) -> str:
        """列出所有可用技能的索引摘要。"""
        return self.manager.get_index_summary()

    def remove_skill(self, skill_name: str) -> bool:
        """删除技能。"""
        return self.manager.remove_skill(skill_name)

    def clear_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()
        self.manager.clear_active_skills()

    # ── 内部方法 ──

    def _build_messages(
        self, user_query: str, active_skills: list[Skill]
    ) -> list[dict[str, str]]:
        """组装发送给 LLM 的完整消息列表。"""
        messages: list[dict[str, str]] = []
        # 系统提示
        sys_content = self._build_system_content(active_skills)
        messages.append({"role": "system", "content": sys_content})

        # 历史对话
        messages.extend(self._history)

        # 当前请求
        messages.append({"role": "user", "content": user_query})

        return messages

    def _build_system_content(self, active_skills: list[Skill]) -> str:
        """构建系统提示内容，包含基础人设 + 技能索引 + 已加载技能详情。"""
        parts: list[str] = []

        # 1. 基础人设
        if self.system_prompt:
            parts.append(self.system_prompt)
        else:
            parts.append(
                "你是一个强大的 Harness Agent，能够根据用户需求加载和使用技能。"
            )

        # 2. 技能索引（始终注入 —— 轻量级）
        index_summary = self.manager.get_index_summary()
        if index_summary != "（暂无可用技能）":
            parts.append("")
            parts.append(index_summary)

        # 3. 已加载技能详情（仅当匹配到技能时注入）
        if active_skills:
            parts.append("")
            parts.append("## 已加载技能详情")
            parts.append("以下技能与用户当前请求相关，请严格按照其指令执行：")
            for skill in active_skills:
                parts.append("")
                parts.append(skill.to_full_prompt())

        return "\n".join(parts)

    def _trim_history(self) -> None:
        """裁剪历史记录，保留最近 N 轮（user + assistant = 2 条为一轮）。"""
        max_messages = self.max_history * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
