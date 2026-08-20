#!/usr/bin/env python3
"""
Harness Agent CLI 入口

用法：
    python main.py                          # 交互式模式
    python main.py --config config.yaml     # 指定配置文件
    python main.py "用户请求"               # 单次模式

命令（交互式）：
    /generate <name> <description>  生成新技能
    /list                           列出所有技能
    /remove <name>                  删除技能
    /clear                          清空对话历史
    /exit                           退出
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from harness.agent import AgentHarness
from harness.llm import LLMClient
from harness.manager import SkillManager
from skills.loader import SkillLoader


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def create_agent(config: dict) -> AgentHarness:
    """根据配置创建 AgentHarness 实例。"""
    llm_cfg = config.get("llm", {})
    llm_client = LLMClient(
        api_key=llm_cfg.get("api_key", os.getenv("OPENAI_API_KEY", "")),
        base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
        model=llm_cfg.get("model", "gpt-4o-mini"),
    )

    progressive_cfg = config.get("progressive_loading", {})
    skill_store_dir = config.get("skill_store_dir", "skills_store")
    skill_store_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), skill_store_dir
    )

    loader = SkillLoader(
        skill_store_dir=skill_store_dir,
        index_cache_ttl=progressive_cfg.get("index_cache_ttl", 300),
    )

    manager = SkillManager(
        loader=loader,
        match_strategy=progressive_cfg.get("match_strategy", "keyword"),
        max_active_skills=progressive_cfg.get("max_active_skills", 3),
    )

    agent_cfg = config.get("agent", {})
    return AgentHarness(
        llm_client=llm_client,
        skill_manager=manager,
        system_prompt=agent_cfg.get("system_prompt", ""),
        max_history=agent_cfg.get("max_history", 20),
    )


def interactive_mode(agent: AgentHarness, console: Console) -> None:
    """交互式对话循环。"""
    console.print(
        Panel.fit(
            "[bold green]Harness Agent[/bold green]\n"
            "渐进式 Skill 加载 Agent\n\n"
            "命令：/generate <名称> <描述>  生成技能\n"
            "      /list                     列出技能\n"
            "      /remove <名称>            删除技能\n"
            "      /clear                    清空历史\n"
            "      /exit                     退出",
            title="欢迎",
            border_style="green",
        )
    )

    while True:
        try:
            user_input = console.input("[bold blue]>>> [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd == "exit":
                console.print("[dim]再见！[/dim]")
                break

            elif cmd == "clear":
                agent.clear_history()
                console.print("[dim]对话历史已清空。[/dim]")
                continue

            elif cmd == "list":
                summary = agent.list_skills()
                console.print(Panel(summary, title="技能列表", border_style="cyan"))
                continue

            elif cmd == "remove":
                if len(parts) < 2:
                    console.print(
                        "[red]用法: /remove <技能名称>[/red]"
                    )
                    continue
                skill_name = parts[1]
                ok = agent.remove_skill(skill_name)
                if ok:
                    console.print(
                        f"[green]技能 '{skill_name}' 已删除。[/green]"
                    )
                else:
                    console.print(
                        f"[red]技能 '{skill_name}' 不存在。[/red]"
                    )
                continue

            elif cmd == "generate":
                if len(parts) < 3:
                    console.print(
                        "[red]用法: /generate <名称> <描述>[/red]"
                    )
                    continue
                skill_name = parts[1]
                skill_desc = parts[2]
                console.print(
                    f"[dim]正在生成技能 '{skill_name}'...[/dim]"
                )
                try:
                    path = agent.generate_skill(skill_name, skill_desc)
                    console.print(
                        f"[green]技能已生成并保存到: {path}[/green]"
                    )
                except Exception as e:
                    console.print(f"[red]生成失败: {e}[/red]")
                continue

            else:
                console.print(f"[red]未知命令: /{cmd}[/red]")
                continue

        # 普通对话
        try:
            with console.status("[dim]思考中...[/dim]", spinner="dots"):
                response = agent.run_once(user_input)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness Agent CLI")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="单次查询模式（不进入交互式）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    agent = create_agent(config)
    console = Console()

    if args.query:
        # 单次模式
        try:
            with console.status("[dim]思考中...[/dim]", spinner="dots"):
                response = agent.run_once(args.query)
            console.print(Markdown(response))
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            sys.exit(1)
    else:
        # 交互式模式
        interactive_mode(agent, console)


if __name__ == "__main__":
    main()
