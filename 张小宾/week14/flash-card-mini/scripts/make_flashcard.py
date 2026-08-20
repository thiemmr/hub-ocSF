"""英语单词 Flash Card 生成器。

用法:
  python make_flashcard.py                          # 默认处理 ./data 目录下所有 JSON
  python make_flashcard.py data/<word>.json         # 处理单个 JSON 文件
  python make_flashcard.py data/                    # 处理目录下所有 JSON
  python make_flashcard.py data/<word>.json -o out/ # 指定输出目录（自动用 <word>.html 命名）
  python make_flashcard.py -o ./out                 # 默认 data/ 输入 + 指定输出目录
"""
from __future__ import annotations

import argparse, json, html, sys
from pathlib import Path
from typing import Optional, List

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{word} - Flash Card</title>
<style>
:root{{--bg:#f5f7fb;--ink:#1f2937;--muted:#6b7280;--accent:#4f46e5;--soft:#eef2ff;--border:#e5e7eb}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Roboto,sans-serif;background:var(--bg);color:var(--ink);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}}
.card{{width:100%;max-width:720px;background:#fff;border-radius:20px;box-shadow:0 10px 30px rgba(17,24,39,.08);overflow:hidden}}
.header{{padding:32px 36px 24px;background:linear-gradient(135deg,var(--accent) 0%,#7c3aed 100%);color:#fff}}
.word{{margin:0;font-size:44px;font-weight:700;letter-spacing:-.5px}}
.phonetic{{margin-top:8px;font-size:18px;opacity:.92;font-style:italic}}
.body{{padding:28px 36px 36px}}
.definition{{font-size:20px;line-height:1.6;padding:14px 16px;background:var(--soft);border-left:4px solid var(--accent);border-radius:8px}}
.pos{{color:var(--accent);font-weight:600;margin-right:6px}}
h2{{margin:28px 0 14px;font-size:16px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px}}
.synonyms{{display:flex;flex-wrap:wrap;gap:10px}}
.tag{{padding:6px 14px;background:var(--soft);color:var(--accent);border-radius:999px;font-size:14px;font-weight:500}}
.examples{{list-style:none;padding:0;margin:0}}
.examples li{{padding:14px 16px;margin-bottom:10px;background:#fafafa;border:1px solid var(--border);border-radius:10px}}
.en{{font-size:17px;line-height:1.55}}
.zh{{margin-top:6px;font-size:14px;color:var(--muted);line-height:1.55}}
.footer{{margin-top:28px;padding-top:16px;border-top:1px dashed var(--border);font-size:12px;color:var(--muted);text-align:center}}
</style>
</head>
<body>
<div class="card">
<div class="header"><h1 class="word">{word}</h1><div class="phonetic">{phonetic}</div></div>
<div class="body">
<div class="definition"><span class="pos">{pos}</span>{definition}</div>
<h2>近义词</h2>
<div class="synonyms">{synonyms_html}</div>
<h2>例句</h2>
<ul class="examples">{examples_html}</ul>
<div class="footer">Flash Card · 学一个词，记一组词</div>
</div>
</div>
</body>
</html>
"""


def build(d):
    esc = html.escape
    syns = "\n".join(f'<span class="tag">{esc(s)}</span>' for s in d.get("synonyms", []))
    exs = (list(d.get("examples", [])[:3]) + [{}] * (3 - len(d.get("examples", []))))[:3]
    items = "\n".join(
        f'<li><div class="en">{esc(e.get("en", "") or "（待补充例句）")}</div>'
        f'<div class="zh">{esc(e.get("zh", "") or "（待补充翻译）")}</div></li>'
        for e in exs
    )
    return TEMPLATE.format(
        word=esc(d["word"]), phonetic=esc(d.get("phonetic", "")),
        pos=esc(d.get("pos", "")), definition=esc(d.get("definition", "")),
        synonyms_html=syns, examples_html=items,
    )


def resolve_output_path(output_arg: Optional[str], word: str) -> Path:
    """根据 -o 参数和单词名计算最终输出路径。

    - 未指定 -o: 当前目录/<word>.html
    - -o 是已存在的目录 或 以 '/' 结尾: 该目录/<word>.html
    - 其他情况: 视为完整文件路径
    """
    if not output_arg:
        return Path.cwd() / f"{word}.html"
    out = Path(output_arg)
    # 如果路径已存在且是目录，或显式以分隔符结尾（用户意图是目录），放入 <word>.html
    if out.is_dir() or output_arg.endswith(("/", "\\")):
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{word}.html"
    # 否则视为文件路径：如果父目录不存在则创建（方便写 ./out/xxx.html）
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def collect_json_files(input_path: Path) -> List[Path]:
    """输入可以是单个 JSON 文件或目录；返回待处理的 JSON 文件列表。"""
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            print(f"[警告] 跳过非 JSON 文件: {input_path}", file=sys.stderr)
            return []
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() == ".json")
        return files
    print(f"错误: 输入路径不存在: {input_path}", file=sys.stderr)
    sys.exit(2)


def process_one(json_path: Path, output_arg: Optional[str]) -> Optional[Path]:
    try:
        d = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[警告] {json_path.name} JSON 解析失败: {e}，已跳过", file=sys.stderr)
        return None
    if not isinstance(d, dict) or "word" not in d:
        # 目录里可能夹杂非单词卡 JSON（如 eval_set.json），静默跳过不打断批量处理
        return None
    out = resolve_output_path(output_arg, d["word"])
    out.write_text(build(d), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(
        description="生成英语单词 Flash Card HTML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  %(prog)s                        # 处理 ./data 目录下所有 JSON
  %(prog)s resilient.json         # 处理单个文件
  %(prog)s data/                  # 处理目录
  %(prog)s -o ./out               # 输入默认 data/，输出到 out/ 目录
  %(prog)s w.json -o cards/my.html # 输出到指定文件路径""",
    )
    p.add_argument(
        "data", nargs="?", default="./data",
        help="输入 JSON 文件或目录（默认: ./data）",
    )
    p.add_argument(
        "-o", "--output",
        help="输出路径：文件路径，或目录路径（目录内自动用 <word>.html 命名）",
    )
    a = p.parse_args()

    input_path = Path(a.data)
    files = collect_json_files(input_path)

    if not files:
        print(
            f"错误: 在 {input_path} 中未找到任何 .json 文件。\n"
            f"       请先准备单词 JSON（参考 SKILL.md 的格式），或显式指定输入路径。",
            file=sys.stderr,
        )
        sys.exit(1)

    generated = 0
    for fp in files:
        out = process_one(fp, a.output)
        if out is not None:
            print(f"已生成: {out}")
            generated += 1

    if generated == 0:
        print(
            f"提示: 在 {input_path} 下找到 {len(files)} 个 .json 文件，"
            f"但没有符合单词卡格式（含 'word' 字段）的。请参考 SKILL.md。",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
