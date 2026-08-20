"""
Skill 生成器

通过 LLM 自动生成 SKILL.md 内容，
接受用户描述 + 技能名称，返回完整的 Skill 对象。
"""

from __future__ import annotations

from typing import Any

from harness.llm import LLMClient
from skills.schema import Skill, SkillMetadata


class SkillGenerator:
    """技能生成器：利用 LLM 将自然语言需求转化为标准 SKILL。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    # ── Prompt 模板 ──

    _SKILL_GENERATION_PROMPT = """你是一位专业的 AI 技能工程师。

请根据用户提供的【技能名称】和【技能描述】，生成一个完整的技能定义。

技能定义包含两个部分：
1. Frontmatter（YAML）：包含 name, description, triggers, version, tags
2. Detail（Markdown）：包含技能的详细指令、使用示例、注意事项等

要求：
- name 必须与技能名称一致
- description 需简洁，说明"做什么"和"何时触发"（<200字符）
- triggers 是触发关键词列表，用于匹配用户 query
- detail 需详细、具体，让 LLM 拿到后能独立执行该技能
- 全部使用中文

请直接输出 frontmatter + detail 的完整 SKILL.md 内容。

格式示例：
---
name: "skill-name"
description: "描述。Invoke when..."
triggers:
  - "关键词1"
  - "关键词2"
version: "1.0.0"
tags:
  - "标签1"
---

# 技能正文

## 说明
...

## 步骤
1. ...
2. ...

## 示例
...

---

现在请为以下需求生成技能：

技能名称: {skill_name}
技能描述: {skill_description}
"""

    def generate(
        self,
        skill_name: str,
        skill_description: str,
    ) -> Skill:
        """生成技能。

        返回 Skill 对象，包含完整的 metadata 和 detail。
        """
        prompt = self._SKILL_GENERATION_PROMPT.format(
            skill_name=skill_name,
            skill_description=skill_description,
        )

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的 AI 技能定义生成专家。",
            },
            {"role": "user", "content": prompt},
        ]

        content = self.llm.chat(messages, temperature=0.7)

        # 解析生成的 SKILL.md
        from skills.loader import parse_skill_md

        skill = parse_skill_md(content)

        # 强制覆盖 name
        skill.metadata.name = skill_name

        return skill

    def generate_with_reflection(
        self,
        skill_name: str,
        skill_description: str,
        max_iterations: int = 2,
    ) -> Skill:
        """带反思的生成：生成 -> 评估 -> 改进。

        多轮迭代提升质量。
        """
        skill = self.generate(skill_name, skill_description)

        for i in range(max_iterations):
            feedback = self._evaluate_skill(skill, skill_description)
            if feedback.get("satisfactory", False):
                break

            improvement = feedback.get("improvement", "")
            if not improvement:
                break

            # 改进
            skill = self._improve_skill(skill, improvement)

        return skill

    def _evaluate_skill(
        self, skill: Skill, original_description: str
    ) -> dict[str, Any]:
        """评估技能质量。"""
        prompt = f"""请评估以下技能定义的质量。

原始需求: {original_description}

技能名: {skill.name}
描述: {skill.description}
触发词: {skill.metadata.triggers}
正文:
---
{skill.detail}
---

请判断：
1. 正文是否足够详细，能让 LLM 独立执行该技能？
2. 触发词是否覆盖了需求的核心场景？
3. 描述是否准确传达了"做什么"和"何时触发"？

输出 JSON:
{{
  "satisfactory": true/false,
  "score": 1-10,
  "issues": ["问题1", "问题2"],
  "improvement": "改进建议（如果没有问题则留空）"
}}
"""
        import json

        messages = [
            {
                "role": "system",
                "content": "你是一个严格的技能质量评估专家。",
            },
            {"role": "user", "content": prompt},
        ]

        content = self.llm.chat(messages, temperature=0.3)

        # 尝试解析 JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "satisfactory": True,
                "score": 7,
                "issues": [],
                "improvement": "",
            }

    def _improve_skill(self, skill: Skill, improvement: str) -> Skill:
        """根据改进建议优化技能。"""
        prompt = f"""请根据改进建议优化以下技能定义。

当前技能:
---
{skill.to_full_prompt()}
---

改进建议:
{improvement}

请输出优化后的完整 SKILL.md 内容。
"""
        from skills.loader import parse_skill_md

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的 AI 技能定义优化专家。",
            },
            {"role": "user", "content": prompt},
        ]

        content = self.llm.chat(messages, temperature=0.7)
        improved = parse_skill_md(content)
        improved.metadata.name = skill.name
        return improved
