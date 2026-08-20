"""
Skill"""
Skill 管理器

编排渐进式加载的完整流程：
  1. 获取技能索引（Phase 0 + 1）
  2. 根据用户查询匹配相关"""
Skill 管理器

编排渐进式加载的完整流程：
  1. 获取技能索引（Phase 0 + 1）
  2. 根据用户查询匹配相关技能"""
Skill 管理器

编排渐进式加载的完整流程：
  1. 获取技能索引（Phase 0 + 1）
  2. 根据用户查询匹配相关技能
  3. 仅加载匹配技能的完整详情（Phase 2）

支持两种匹配策略：
  - keyword: 基于触发词和描述的关键词匹配（快速、无需 LLM）
  - llm:     使用 LLM 进行语义匹配（更准确、需"""
Skill 管理器

编排渐进式加载的完整流程：
  1. 获取技能索引（Phase 0 + 1）
  2. 根据用户查询匹配相关技能
  3. 仅加载匹配技能的完整详情（Phase 2）

支持两种匹配策略：
  - keyword: 基于触发词和描述的关键词匹配（快速、无需 LLM）
  - llm:     使用 LLM 进行语义匹配（更准确、需额外调用）
"""

from __future__ import annotations

from typing import Optional

from ..harness.llm import LLMClient
from .loader import SkillLoader
from .schema import Skill, SkillMetadata


class SkillManager:
    """技能管理器 —— 渐进式加载的编排中心。"""

    def __init__(
        self,
        loader: SkillLoader,
        llm_client: Optional[LLMClient] = None,
        match_strategy: str = "keyword",
        max_active_skills: int = 3,
    ):
        self.loader = loader
        self