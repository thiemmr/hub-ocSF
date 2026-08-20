"""
渐进式 Skill 加载器

实现三阶段渐进加载：
  Phase 0 (索引扫描): 扫描技能目录，发现所有可用技能
  Phase 1 (元数据加载): 解析每个技能的 frontmatter，构建轻量级索引
  Phase 2 (详情加载):   仅对匹配到的技能加载完整 Markdown 正文

核心思想：始终只在上下文中保留元数据索引，正文按需加载。
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import yaml

from .schema import Skill, SkillMetadata


# ── SKILL.md 解析 ──────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(?P<frontmatter>.*?)\n---\s*\n(?P<body>.*)$",
    re.DOTALL,
)


def parse_skill_md(content: str, file_path: Optional[str] = None) -> Skill:
    """解析 SKILL.md 文件内容，返回 Skill 对象。

    文件格式：
        ---
        name: "skill-name"
        description: "简洁描述"
        triggers: ["关键词1", "关键词2"]
        version: "1.0.0"
        tags: ["tag1"]
        ---
        # 技能正文（Markdown）
        详细指令...
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError(
            f"无法解析 SKILL.md（缺少 frontmatter）: {file_path or '未知路径'}"
        )

    frontmatter_text = match.group("frontmatter")
    body_text = match.group("body").strip()

    try:
        meta_dict = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as e:
        raise ValueError(f"frontmatter YAML 解析失败: {e}") from e

    if not isinstance(meta_dict, dict):
        raise ValueError("frontmatter 必须是一个 YAML 字典")

    name = meta_dict.get("name", "")
    if not name:
        raise ValueError("技能缺少 name 字段")

    description = meta_dict.get("description", "")
    triggers = meta_dict.get("triggers", []) or []
    version = meta_dict.get("version", "1.0.0")
    tags = meta_dict.get("tags", []) or []

    metadata = SkillMetadata(
        name=name,
        description=str(description),
        triggers=[str(t) for t in triggers],
        version=str(version),
        tags=[str(t) for t in tags],
    )

    return Skill(metadata=metadata, detail=body_text, file_path=file_path)


def serialize_skill_md(skill: Skill) -> str:
    """将 Skill 对象序列化为 SKILL.md 格式字符串。"""
    frontmatter = {
        "name": skill.metadata.name,
        "description": skill.metadata.description,
        "triggers": skill.metadata.triggers,
        "version": skill.metadata.version,
        "tags": skill.metadata.tags,
    }
    fm_text = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)
    return f"---\n{fm_text}---\n\n{skill.detail}\n"


# ── 渐进式加载器 ───────────────────────────────────────────────


class SkillLoader:
    """渐进式技能加载器。

    所有文件 I/O 都集中在此类，上层 Manager 只调用其接口。
    内置缓存：索引扫描结果带 TTL，详情按技能名缓存。
    """

    def __init__(self, skill_store_dir: str, index_cache_ttl: int = 300):
        self.skill_store_dir = os.path.abspath(skill_store_dir)
        self.index_cache_ttl = index_cache_ttl

        # 缓存
        self._index_cache: list[SkillMetadata] = []
        self._index_cache_time: float = 0
        self._detail_cache: dict[str, Skill] = {}

    # ── Phase 0 + Phase 1: 索引扫描 + 元数据加载 ──

    def load_index(self, force_refresh: bool = False) -> list[SkillMetadata]:
        """扫描技能目录，返回所有技能的元数据列表。

        如果缓存未过期且未强制刷新，直接返回缓存结果。
        """
        now = time.time()
        if (
            not force_refresh
            and self._index_cache
            and (now - self._index_cache_time) < self.index_cache_ttl
        ):
            return list(self._index_cache)

        metadatas: list[SkillMetadata] = []

        if not os.path.isdir(self.skill_store_dir):
            os.makedirs(self.skill_store_dir, exist_ok=True)
            self._index_cache = metadatas
            self._index_cache_time = now
            return metadatas

        for entry in sorted(os.listdir(self.skill_store_dir)):
            skill_dir = os.path.join(self.skill_store_dir, entry)
            if not os.path.isdir(skill_dir):
                continue

            skill_md_path = os.path.join(skill_dir, "SKILL.md")
            if not os.path.isfile(skill_md_path):
                continue

            try:
                with open(skill_md_path, "r", encoding="utf-8") as f:
                    content = f.read()
                skill = parse_skill_md(content, skill_md_path)
                metadatas.append(skill.metadata)
            except (ValueError, OSError) as e:
                print(f"[Loader] 跳过无效技能 '{entry}': {e}")

        # 更新缓存
        self._index_cache = metadatas
        self._index_cache_time = now
        # 清空详情缓存（索引可能已变化）
        self._detail_cache.clear()

        return list(metadatas)

    # ── Phase 2: 详情加载 ──

    def load_detail(self, skill_name: str) -> Optional[Skill]:
        """加载指定技能的完整内容（含正文）。

        先查缓存，未命中则从文件系统读取。
        """
        if skill_name in self._detail_cache:
            return self._detail_cache[skill_name]

        skill_md_path = os.path.join(
            self.skill_store_dir, skill_name, "SKILL.md"
        )
        if not os.path.isfile(skill_md_path):
            return None

        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            skill = parse_skill_md(content, skill_md_path)
        except (ValueError, OSError) as e:
            print(f"[Loader] 加载技能 '{skill_name}' 失败: {e}")
            return None

        self._detail_cache[skill_name] = skill
        return skill

    def load_details(self, skill_names: list[str]) -> list[Skill]:
        """批量加载多个技能的详情。"""
        results: list[Skill] = []
        for name in skill_names:
            skill = self.load_detail(name)
            if skill is not None:
                results.append(skill)
        return results

    # ── 写入 ──

    def save_skill(self, skill: Skill) -> str:
        """将技能保存到技能存储目录。

        返回保存的 SKILL.md 路径。
        """
        skill_dir = os.path.join(self.skill_store_dir, skill.metadata.name)
        os.makedirs(skill_dir, exist_ok=True)

        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        content = serialize_skill_md(skill)

        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 更新缓存
        self._detail_cache[skill.metadata.name] = skill
        self.load_index(force_refresh=True)

        return skill_md_path

    def delete_skill(self, skill_name: str) -> bool:
        """删除技能目录。"""
        import shutil

        skill_dir = os.path.join(self.skill_store_dir, skill_name)
        if not os.path.isdir(skill_dir):
            return False

        shutil.rmtree(skill_dir)
        self._detail_cache.pop(skill_name, None)
        self.load_index(force_refresh=True)
        return True

    def skill_exists(self, skill_name: str) -> bool:
        """检查技能是否存在。"""
        skill_md_path = os.path.join(
            self.skill_store_dir, skill_name, "SKILL.md"
        )
        return os.path.isfile(skill_md_path)
