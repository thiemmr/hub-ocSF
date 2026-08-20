"""
AI 周报生成 Agent 工具集

提供 3 个工具供 Agent 调用：
1. web_search  - 多引擎网页搜索（DuckDuckGo / 备用 Google）
2. web_fetch   - 抓取网页正文内容
3. write_markdown - 将整理好的内容写入 Markdown 文件

使用方式：
    from tools import TOOLS_MAP, TOOLS_SCHEMA
    results = TOOLS_MAP["web_search"](query="AI 最新进展 2026年8月", num=10)

依赖：
    pip install duckduckgo-search httpx beautifulsoup4
"""

import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

TIMEOUT = 15
MAX_CONTENT_LEN = 8000

SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


# ── 1. Web Search ────────────────────────────────────────────────────────────

def tool_web_search(query: str, num: int = 10, lang: str = "zh") -> str:
    """
    多引擎网页搜索，按优先级尝试：ddgs 库 → DuckDuckGo HTML → Bing HTML → Google HTML。
    返回标题+摘要+URL 的格式化文本。

    Args:
        query: 搜索关键词
        num:   期望返回的结果数（最多 10）
        lang:  语言偏好，zh=中文，en=英文

    Returns:
        格式化的搜索结果文本，每条包含标题、摘要、来源URL
    """
    results = []

    # 方案 A：ddgs 库（最新版 DuckDuckGo 封装）
    try:
        results = _search_via_ddgs(query, num)
    except Exception as e:
        logger.warning(f"ddgs 搜索失败: {e}")

    # 方案 B：DuckDuckGo HTML 直接请求
    if not results:
        try:
            results = _search_via_ddg_html(query, num, lang)
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML 搜索失败: {e}")

    # 方案 C：Bing HTML
    if not results:
        try:
            results = _search_via_bing(query, num, lang)
        except Exception as e:
            logger.warning(f"Bing 搜索失败: {e}")

    # 方案 D：Google HTML（最后兜底）
    if not results:
        try:
            results = _search_via_google(query, num, lang)
        except Exception as e:
            logger.error(f"所有搜索引擎均失败: {e}")
            return f"搜索 '{query}' 失败：所有搜索引擎均不可用"

    if not results:
        return f"搜索 '{query}' 未找到任何结果"

    lines = [f'搜索词: "{query}"  共找到 {len(results)} 条结果:\n']
    for i, item in enumerate(results, 1):
        title = item.get("title", "无标题")
        snippet = item.get("snippet", "")[:300]
        url = item.get("url", "")
        lines.append(f"[{i}] {title}\n    {snippet}\n    {url}\n")

    return "\n".join(lines)


def _search_via_ddgs(query: str, num: int) -> list[dict]:
    """使用 ddgs 库（DuckDuckGo 新版封装）"""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for i, item in enumerate(
            ddgs.text(query, max_results=min(num, 10), region="wt-wt")
        ):
            if i >= num:
                break
            results.append(
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("body", ""),
                    "url": item.get("href", ""),
                }
            )
    return results


def _search_via_ddg_html(query: str, num: int, lang: str) -> list[dict]:
    """通过 DuckDuckGo HTML 版搜索"""
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query, "kl": "cn-zh" if lang == "zh" else "us-en"}
    results = []

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=SEARCH_HEADERS) as client:
        resp = client.post(url, data=data)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for result in soup.select(".result"):
        title_el = result.select_one(".result__title a, .result__a")
        snippet_el = result.select_one(".result__snippet")
        if title_el:
            results.append(
                {
                    "title": title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "url": title_el.get("href", ""),
                }
            )
        if len(results) >= num:
            break

    return results


def _search_via_bing(query: str, num: int, lang: str) -> list[dict]:
    """通过 Bing 搜索"""
    url = "https://www.bing.com/search"
    params = {"q": query, "count": min(num, 10), "setlang": "zh-CN" if lang == "zh" else "en"}
    results = []

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=SEARCH_HEADERS) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for li in soup.select("li.b_algo"):
        title_el = li.select_one("h2 a")
        snippet_el = li.select_one(".b_caption p, p")
        if title_el:
            results.append(
                {
                    "title": title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "url": title_el.get("href", ""),
                }
            )
        if len(results) >= num:
            break

    return results


def _search_via_google(query: str, num: int, lang: str) -> list[dict]:
    """通过 Google 搜索（最终兜底）"""
    url = "https://www.google.com/search"
    params = {
        "q": query,
        "num": min(num, 10),
        "hl": "zh-CN" if lang == "zh" else "en",
    }
    results = []

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=SEARCH_HEADERS) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for g in soup.select("div.g"):
        title_el = g.select_one("h3")
        link_el = g.select_one("a")
        snippet_el = g.select_one("div.VwiC3b, span.aCOpRe")
        if link_el and title_el:
            results.append(
                {
                    "title": title_el.get_text(strip=True),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    "url": link_el.get("href", ""),
                }
            )
        if len(results) >= num:
            break

    return results


# ── 2. Web Fetch ─────────────────────────────────────────────────────────────

def tool_web_fetch(url: str) -> str:
    """
    抓取指定 URL 的网页正文内容，提取纯文本。
    自动过滤导航、广告、脚本等噪音。

    Args:
        url: 目标网页 URL

    Returns:
        网页正文的纯文本内容（最多 8000 字符）
    """
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, headers=SEARCH_HEADERS)
            resp.raise_for_status()
    except Exception as e:
        return f"抓取失败 ({url}): {e}"

    soup = BeautifulSoup(resp.text, "html.parser")

    # 移除无关标签
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # 尝试找到正文容器
    article = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
        or soup.find("main")
    )
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if len(text) > MAX_CONTENT_LEN:
        text = text[:MAX_CONTENT_LEN] + "\n\n... (内容已截断)"

    source_note = f"[来源: {url}]\n\n"
    return source_note + text


# ── 3. Write Markdown ────────────────────────────────────────────────────────

def tool_write_markdown(content: str, filename: str, output_dir: str = "") -> str:
    """
    将 AI 整理好的内容写入 Markdown 文件。

    Args:
        content:   完整的 Markdown 文本
        filename:  文件名（不含路径，如 AI_Weekly_News.md）
        output_dir: 输出目录（可选，为空时使用当前目录）

    Returns:
        写入成功的文件路径
    """
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path.cwd()

    # 确保文件名以 .md 结尾
    if not filename.endswith(".md"):
        filename += ".md"

    filepath = out_path / filename

    # 添加头部元信息
    header = (
        "<!--\n"
        f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  生成工具: AI Weekly News Agent (deepseek-v4-flash)\n"
        "-->\n\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(content)

    size_kb = len(content.encode("utf-8")) / 1024
    return f"文件已写入: {filepath}  (约 {size_kb:.1f} KB)"


# ── 工具注册表 ────────────────────────────────────────────────────────────────

TOOLS_MAP: dict[str, Any] = {
    "web_search": tool_web_search,
    "web_fetch": tool_web_fetch,
    "write_markdown": tool_write_markdown,
}

# Function Calling 版 JSON Schema
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索引擎，输入关键词返回标题+摘要+URL的搜索结果。用于查找AI新闻、产品发布、技术突破等信息。支持中英文搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如'AI 最新进展 2026年8月'",
                    },
                    "num": {
                        "type": "integer",
                        "description": "返回结果数，默认10，最多10",
                        "default": 10,
                    },
                    "lang": {
                        "type": "string",
                        "description": "搜索语言，zh=中文，en=英文",
                        "enum": ["zh", "en"],
                        "default": "zh",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "抓取指定URL的网页正文内容，提取纯文本。用于深入阅读搜索结果中的文章。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标网页的完整URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_markdown",
            "description": "将整理好的AI周报内容写入Markdown文件。在完成所有搜索和整理后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "完整的Markdown格式周报内容",
                    },
                    "filename": {
                        "type": "string",
                        "description": "文件名，如 AI_Weekly_News_20260806.md",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "输出目录路径，可选",
                        "default": "",
                    },
                },
                "required": ["content", "filename"],
            },
        },
    },
]
