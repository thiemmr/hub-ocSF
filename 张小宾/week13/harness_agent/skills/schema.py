"""
Skill 数据模型定义

定义了渐进式加载所需的两个层级：
  - SkillMetadata: 轻量级元数据（索引阶段加载，仅含 name/description/triggers）
  - Skill:         完整技能（详情阶段加载，含 detail 正文内容）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillMetadata:
    """技能元数据 —— 渐进式加载第一阶段（索引阶段）的数据。

    仅包含让 Agent 判断"是否需要此技能"的最小信息，
    不会把完整正文塞入上下文，从而节省 token。
    """

    name: str
    """技能唯一标识符，同时也是技能目录名。"""

    description: str
    """简洁描述（建议 <200 字符），说明技能做什么以及何时触发。"""

    triggers: list[str] = field(default_factory=list)
    """触发关键词列表，用于关键词匹配阶段快速筛选。"""

    version: str = "1.0.0"
    """技能版本号。"""

    tags: list[str] = field(default_factory=list)
    """标签，用于分类和检索。"""

    def to_index_line(self) -> str:
        """生成索引摘要行，用于注入 Agent 上下文。"""
        trigger_str = ", ".join(self.triggers) if self.triggers else "无"
        return f"- [{self.name}] {self.description} (触发词: {trigger_str})"


@dataclass
class Skill:
    """完整技能 —— 渐进式加载第二阶段（详情阶段）的数据。

    在元数据基础上增加了 detail（Markdown 正文），
    只有当 Agent 判定某技能与当前请求相关时才会加载。
    """

    metadata: SkillMetadata
    """技能元数据。"""

    detail: str
    """技能正文（Markdown 格式），包含详细指令、示例等。"""

    file_path: Optional[str] = None
    """SKILL.md 文件路径（内部使用）。"""

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    def to_full_prompt(self) -> str:
        """生成完整技能提示词，注入 Agent 上下文。"""
        lines = [
            f"# 技能: {self.metadata.name}",
            f"",
            f"**描述:** {self.metadata.description}",
            f"",
            self.detail,
        ]
        return "\n".join(lines)
