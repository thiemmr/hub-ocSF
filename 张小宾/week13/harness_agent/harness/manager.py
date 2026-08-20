"""
Skill 管理器

负责：
1. 维护轻量级索引（元数据）
2. 实现匹配策略（关键词 / LLM 意图匹配）
3. 触发渐进式加载（仅加载匹配到的技能的详情）
4. 提供 CRUD 接口
"""

from __future__ import annotations

from typing import Optional

from skills.loader import SkillLoader
from skills.schema import Skill, SkillMetadata


class SkillManager:
    """技能管理器：索引 + 匹配 + 渐进式加载的统一入口。"""

    def __init__(
        self,
        loader: SkillLoader,
        match_strategy: str = "keyword",
        max_active_skills: int = 3,
    ):
        self.loader = loader
        self.match_strategy = match_strategy
        self.max_active_skills = max_active_skills

        # 运行时缓存：当前已加载的完整技能
        self._active_skills: dict[str, Skill] = {}

    # ── 索引 ──

    def list_index(self) -> list[SkillMetadata]:
        """获取所有技能的轻量级索引。"""
        return self.loader.load_index()

    def get_index_summary(self) -> str:
        """生成索引摘要文本，用于注入 Agent 上下文。"""
        metas = self.list_index()
        if not metas:
            return "（暂无可用技能）"
        lines = ["## 可用技能索引"]
        for meta in metas:
            lines.append(meta.to_index_line())
        return "\n".join(lines)

    # ── 匹配 ──

    def match_skills(self, query: str) -> list[SkillMetadata]:
        """根据用户 query 匹配相关技能。

        支持两种策略：
          - keyword: 触发词匹配（快速、无 LLM 调用）
          - llm:     LLM 意图匹配（精准、需一次 LLM 调用）
        """
        metas = self.list_index()
        if not metas:
            return []

        if self.match_strategy == "keyword":
            return self._match_by_keyword(query, metas)

        # 默认回退到 keyword
        return self._match_by_keyword(query, metas)

    def _match_by_keyword(
        self, query: str, metas: list[SkillMetadata]
    ) -> list[SkillMetadata]:
        """关键词匹配：query 包含任意 trigger 则命中。"""
        query_lower = query.lower()
        matched: list[SkillMetadata] = []
        for meta in metas:
            for trigger in meta.triggers:
                if trigger.lower() in query_lower:
                    matched.append(meta)
                    break
        return matched

    # ── 渐进式加载 ──

    def load_for_query(self, query: str) -> list[Skill]:
        """渐进式加载入口。

        流程：
          1. 用 query 匹配相关技能的元数据
          2. 对匹配到的技能，加载完整详情
          3. 返回已加载的完整技能列表
        """
        matched_metas = self.match_skills(query)

        # 限制最大加载数量
        matched_metas = matched_metas[: self.max_active_skills]

        skill_names = [m.name for m in matched_metas]
        skills = self.loader.load_details(skill_names)

        # 更新运行时缓存
        for skill in skills:
            self._active_skills[skill.name] = skill

        return skills

    def get_active_skills(self) -> list[Skill]:
        """获取当前已加载到内存中的完整技能。"""
        return list(self._active_skills.values())

    def clear_active_skills(self) -> None:
        """清空运行时缓存。"""
        self._active_skills.clear()

    # ── CRUD ──

    def add_skill(self, skill: Skill) -> str:
        """添加/保存技能。"""
        path = self.loader.save_skill(skill)
        self._active_skills[skill.name] = skill
        return path

    def remove_skill(self, skill_name: str) -> bool:
        """删除技能。"""
        ok = self.loader.delete_skill(skill_name)
        if ok:
            self._active_skills.pop(skill_name, None)
        return ok

    def get_skill(self, skill_name: str) -> Optional[Skill]:
        """获取技能（先从运行时缓存，再从文件加载）。"""
        if skill_name in self._active_skills:
            return self._active_skills[skill_name]
        skill = self.loader.load_detail(skill_name)
        if skill:
            self._active_skills[skill_name] = skill
        return skill
