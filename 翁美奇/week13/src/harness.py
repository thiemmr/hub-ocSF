"""
渐进式加载执行skills的harness

启动 Harness
  |
扫描 skills/*/SKILL.md
  |
解析 YAML Frontmatter, 构建技能注册表 (name, desc, params, 模块路径)
  |
对外提供技能列表 / 执行接口
  |
用户请求执行技能 "foo"
  |
从注册表获取 "foo" 的模块路径
  |
动态 import(路径), 获取执行函数
  |
校验参数, 执行, 返回结果
"""

from pathlib import Path
import importlib
import re
import subprocess
import sys
import shutil
import argparse
import json

ROOT_PATH = Path(__file__).parent.parent
SKILLS_DIR = ROOT_PATH / "skills"


def _flush_block(metadata, current_key, current_mode, current_lines):
    """Flush block scalar lines into metadata, return reset state."""
    if current_key is None:
        return current_key, current_mode, current_lines
    raw = "\n".join(current_lines)
    if current_mode in (">", ">-"):
        metadata[current_key] = re.sub(r"\s*\n\s*", " ", raw).strip()
    elif current_mode in ("|", "|-"):
        metadata[current_key] = raw.strip()
    else:
        metadata[current_key] = raw.strip()
    return None, "", []


def parse_frontmatter(content):
    """Parse YAML frontmatter, return (metadata, body)."""
    if not content.startswith("---"):
        return {}, content.strip()

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content.strip()

    yaml_text = parts[1].strip()
    body = parts[2].strip()

    metadata = {}
    current_key = None
    current_mode = ""
    current_lines = []

    for line in yaml_text.split("\n"):
        stripped = line.strip()

        if not stripped:
            current_key, current_mode, current_lines = _flush_block(
                metadata, current_key, current_mode, current_lines
            )
            continue

        match = re.match(r'^([A-Za-z_][\w-]*)\s*:\s*(.*)', line)
        if match:
            current_key, current_mode, current_lines = _flush_block(
                metadata, current_key, current_mode, current_lines
            )
            key = match.group(1)
            value = match.group(2).strip()

            if value in (">", "|-", ">-", "|"):
                current_key = key
                current_mode = value
                current_lines = []
            elif value:
                # Strip surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                metadata[key] = value
            else:
                current_key = key
                current_mode = ""
                current_lines = []
        elif current_key is not None:
            indent = len(line) - len(line.lstrip())
            if indent > 0 or stripped.startswith((" ", "\t")):
                current_lines.append(stripped)

    _flush_block(metadata, current_key, current_mode, current_lines)
    return metadata, body


def parse_skill_file(skill_file):
    """Parse a SKILL.md file, return (metadata, body)."""
    with open(skill_file, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_frontmatter(content)


def build_skill_registry():
    """Scan skills/ dir and build a registry of all skills."""
    skill_registry = {}
    for skill_file in SKILLS_DIR.glob("**/SKILL.md"):
        metadata, content = parse_skill_file(skill_file)
        skill_name = metadata.get("name")
        if skill_name:
            skill_registry[skill_name] = {
                "metadata": metadata,
                "content": content,
                "file_path": str(skill_file),
                "skill_dir": str(skill_file.parent),
            }
        else:
            print(f"[WARN] skill file {skill_file} missing name field")
    return skill_registry


def list_skills(skill_registry):
    """Return a list of skill summaries."""
    skills = []
    for name, info in sorted(skill_registry.items()):
        skills.append({
            "name": name,
            "description": info["metadata"].get("description", ""),
            "skill_dir": info["skill_dir"],
        })
    return skills


def get_skill_info(skill_registry, skill_name):
    """Get full info for a single skill."""
    return skill_registry.get(skill_name)


def execute_skill(skill_registry, skill_name, args=None):
    """
    Execute a skill by name.

    Strategy (by priority):
      1. If skill_dir/main.py exists -> import and call execute(args)
      2. If scripts/*.py exists -> run as subprocess
      3. If scripts/*.ts exists -> run with bun / npx
    """
    if skill_name not in skill_registry:
        return {"success": False, "error": f"skill not found: {skill_name}"}

    info = skill_registry[skill_name]
    skill_dir = Path(info["skill_dir"])
    args = args or []

    # Strategy 1: import main.py
    main_py = skill_dir / "main.py"
    if main_py.exists():
        try:
            module = importlib.import_module(f"skills.{skill_name}.main")
            if hasattr(module, "execute"):
                result = module.execute(args)
                return {
                    "success": True,
                    "mode": "import",
                    "result": result,
                    "skill_dir": str(skill_dir),
                    "invoked_script": "main.py",
                }
        except Exception as e:
            return {"success": False, "error": str(e), "mode": "import_fallback"}

    # Strategy 2 & 3: find scripts
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.exists():
        py_scripts = sorted(scripts_dir.glob("*.py"))
        ts_scripts = sorted(scripts_dir.glob("*.ts"))

        if py_scripts:
            script = py_scripts[0]
            cmd = [sys.executable, str(script)] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(ROOT_PATH)
            )
            return {
                "success": result.returncode == 0,
                "mode": "subprocess",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "skill_dir": str(skill_dir),
                "invoked_script": f"scripts/{script.name}",
                "returncode": result.returncode,
            }

        if ts_scripts:
            script = ts_scripts[0]
            bun = shutil.which("bun")
            if bun:
                cmd = [bun, str(script)] + args
            else:
                cmd = ["npx", "-y", "bun", str(script)] + args
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(ROOT_PATH),
                timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "mode": "subprocess",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "skill_dir": str(skill_dir),
                "invoked_script": f"scripts/{script.name}",
                "returncode": result.returncode,
            }

    return {
        "success": False,
        "error": f"skill '{skill_name}' has no executable script",
        "skill_dir": str(skill_dir),
    }


def initialize_skills(skill_registry):
    """Initialize all skills that support import-based init."""
    for skill_name, info in skill_registry.items():
        metadata = info["metadata"]
        content = info["content"]
        try:
            module = importlib.import_module(f"skills.{skill_name}.main")
            if hasattr(module, "initialize"):
                module.initialize(metadata, content)
            else:
                print(f"[INFO] skill '{skill_name}' has no initialize function")
        except ModuleNotFoundError:
            print(f"[INFO] skill '{skill_name}' has no importable module")
        except Exception as e:
            print(f"[WARN] skill '{skill_name}' init failed: {e}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Progressive skill loader and executor Harness"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all registered skills"
    )
    parser.add_argument(
        "--info", "-i", metavar="SKILL",
        help="Show details for a specific skill"
    )
    parser.add_argument(
        "--run", "-r", metavar="SKILL",
        help="Execute a skill (pass extra args after)"
    )
    parser.add_argument(
        "--init", action="store_true",
        help="Initialize all skill modules"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format"
    )

    known_args, skill_args = parser.parse_known_args()
    registry = build_skill_registry()

    if known_args.list:
        skills = list_skills(registry)
        if known_args.json:
            print(json.dumps(skills, ensure_ascii=False, indent=2))
        else:
            if not skills:
                print("No skills available.")
            else:
                print(f"Registered {len(skills)} skills:\n")
                for s in skills:
                    desc = s["description"][:100]
                    if len(s["description"]) > 100:
                        desc += "..."
                    print(f"  [{s['name']}]")
                    print(f"    {desc}")
                    print()

    elif known_args.info:
        info = get_skill_info(registry, known_args.info)
        if info:
            if known_args.json:
                print(json.dumps({
                    "name": info["metadata"].get("name"),
                    "description": info["metadata"].get("description"),
                    "skill_dir": info["skill_dir"],
                    "metadata": info["metadata"],
                    "content": info["content"],
                }, ensure_ascii=False, indent=2))
            else:
                print(f"Skill: {info['metadata'].get('name')}")
                print(f"Desc:  {info['metadata'].get('description', '')}")
                print(f"Dir:   {info['skill_dir']}")
                print(f"File:  {info['file_path']}")
                print(f"\n--- Content ---")
                print(info["content"][:500])
                if len(info["content"]) > 500:
                    print("...")
        else:
            print(f"Skill not found: {known_args.info}")

    elif known_args.run:
        result = execute_skill(registry, known_args.run, skill_args)
        if known_args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            if result.get("success"):
                print(f"OK: skill '{known_args.run}' executed successfully")
                if result.get("stdout"):
                    print(result["stdout"])
                if result.get("result") is not None:
                    print(result["result"])
            else:
                print(f"FAIL: skill '{known_args.run}' failed")
                print(f"  error: {result.get('error', 'unknown')}")
                if result.get("stderr"):
                    print(f"  stderr: {result['stderr']}")

    elif known_args.init:
        print("Initializing all skill modules...")
        initialize_skills(registry)
        print("Done.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
