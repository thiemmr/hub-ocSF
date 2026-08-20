#!/usr/bin/env python3
"""Create a standalone Hot 100 practice page from the bundled template."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_TITLE = "Hot 100 随机练习"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone LeetCode Hot 100 practice page."
    )
    parser.add_argument("output_dir", type=Path, help="Directory for index.html")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Page title")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing index.html while preserving other files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = Path(__file__).resolve().parent.parent / "assets" / "template" / "index.html"
    if not template.is_file():
        raise SystemExit(f"Template not found: {template}")

    output_dir = args.output_dir.expanduser().resolve()
    output_file = output_dir / "index.html"
    if output_file.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite {output_file}. Pass --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    html = template.read_text(encoding="utf-8")
    html = html.replace("Hot 100 随机练习", args.title)
    output_file.write_text(html, encoding="utf-8")
    print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
