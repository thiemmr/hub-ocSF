# Harness Agent

一个支持**渐进式 Skill 加载**的 Python Agent 框架。

## 核心特性

- **Skill 生成与加载**：支持通过 LLM 自动生成技能定义，以及从文件系统加载技能
- **渐进式加载**：三阶段加载模式 —— 索引扫描 → 元数据匹配 → 详情按需加载，避免 token 浪费
- **OpenAI 兼容**：支持任意 OpenAI 兼容接口（OpenAI、DeepSeek、Moonshot 等）
- **交互式 CLI**：提供 Rich 风格的命令行交互体验

## 项目结构

```
harness_agent/
├── main.py                    # CLI 入口
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖
├── README.md                 # 本文档
├── harness/                  # Agent 核心逻辑
│   ├── llm.py               # LLM 客户端封装
│   ├── manager.py           # Skill 管理器（索引 + 匹配 + 渐进加载）
│   ├── generator.py         # Skill 生成器（LLM 生成 SKILL.md）
│   └── agent.py             # Agent Harness 主循环
├── skills/                   # Skill 基础设施
│   ├── schema.py            # Skill 数据模型
│   └── loader.py            # 渐进式 Skill 加载器
└── skills_store/             # 技能存储目录
    ├── code-reviewer/        # 示例：代码审查技能
    │   └── SKILL.md
    └── tech-research/        # 示例：技术调研技能
        └── SKILL.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `config.yaml`，填入你的 LLM API 配置：

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-your-api-key"  # 或设置环境变量 OPENAI_API_KEY
  model: "gpt-4o-mini"
```

### 3. 运行交互式模式

```bash
python main.py
```

### 4. 单次查询模式

```bash
python main.py "帮我审查这段代码"
```

## 交互式命令

| 命令 | 说明 |
|------|------|
| `/generate <name> <desc>` | 通过 LLM 生成新技能并保存 |
| `/list` | 列出所有可用技能 |
| `/remove <name>` | 删除技能 |
| `/clear` | 清空对话历史 |
| `/exit` | 退出 |

## 渐进式加载原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                        渐进式 Skill 加载流程                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Phase 0: 索引扫描      Phase 1: 元数据匹配      Phase 2: 详情加载 │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐     │
│   │ 扫描目录     │      │ 关键词/LLM   │      │ 加载 Markdown│     │
│   │ 发现 SKILL.md│ ───> │ 匹配 triggers│ ───> │ 正文 (detail)│     │
│   │              │      │              │      │              │     │
│   │ 输出:        │      │ 输出:        │      │ 输出:        │     │
│   │ SkillMetadata│      │ 相关技能列表 │      │ Skill (完整) │     │
│   │ (轻量级)     │      │ (仍轻量级)   │      │ (按需加载)   │     │
│   └──────────────┘      └──────────────┘      └──────────────┘     │
│                                                                     │
│   关键设计:                                                         │
│   - 索引始终轻量（只含 name + description + triggers）              │
│   - 正文仅当技能被匹配时才加载                                      │
│   - max_active_skills 限制同时加载数量（默认 3）                    │
│   - 索引和详情都有缓存机制                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Skill 格式

每个技能是一个目录，包含一个 `SKILL.md` 文件：

```markdown
---
name: "skill-name"
description: "简短描述，说明做什么和何时触发"
triggers:
  - "关键词1"
  - "关键词2"
version: "1.0.0"
tags:
  - "标签1"
---

# 技能标题

## 说明
详细指令...

## 步骤
1. ...
2. ...
```

## 架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                            User Input                               │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                        AgentHarness.run_once()                      │
│  1. manager.load_for_query(query)  ──> 渐进式加载匹配的技能          │
│  2. _build_messages()  ──> 组装 system + index + skills + history  │
│  3. llm.chat()  ──> 调用 LLM 生成回复                               │
│  4. 更新历史                                                        │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ SkillIndex│   │Matched    │   │  LLM      │
        │ (元数据)   │   │Skills     │   │  Client   │
        │           │   │ (详情)     │   │           │
        └───────────┘   └───────────┘   └───────────┘
                │               │               │
                └───────┬───────┘               │
                        ▼                       │
                ┌───────────────┐               │
                │ SkillLoader   │               │
                │               │               │
                │ - load_index()│               │
                │ - load_detail()│              │
                │ - save_skill()│               │
                │ - delete_skill()│             │
                └───────────────┘               │
                                                │
                                ┌───────────────┘
                                ▼
                        ┌───────────────┐
                        │ SkillGenerator│
                        │               │
                        │ - generate()  │
                        │ - reflection  │
                        └───────────────┘
```
