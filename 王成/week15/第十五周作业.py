"""
主 Agent + 并行 Subagent 编排

职责：
  1. 接收磁盘扫描结果（file_scanner.scan_folder 产出）
  2. 为每个子文件夹派一个 subagent，并行读取 Top3 文件并生成内容摘要
  3. 收齐所有结果后汇总返回，全程通过进度回调汇报进度

并行核心：
  用 ThreadPoolExecutor 并行跑多个 subagent，
  用 max_workers 限制最大并发（防止子文件夹过多时触发 LLM 限流）。

进度回调：
  on_progress(done, total, message) 会在每个文件摘要完成时触发，
  用于桌面 UI 更新进度条。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from src.llm_client import llm_chat
from src.file_scanner import format_size, TEXT_EXTENSIONS

logger = logging.getLogger(__name__)

# 单个文件最多读取的字符数（防止超大文件撑爆上下文）
MAX_READ_CHARS = 8000

# subagent 摘要的系统提示词
SUB_SYSTEM = """你是文件内容摘要助手。用户会给你一个文件的内容（或无法读取的说明），
请你用简洁的中文总结这个文件的主要内容。

输出要求：
- 用 2~4 句话概括文件的核心内容、用途或主题
- 如果是代码文件，说明它的功能和主要逻辑
- 如果内容显示"无法读取"，直接说明无法摘要的原因
- 只输出摘要本身，不要加"摘要："之类的前缀，不要加多余解释"""


def _read_file(file_path: str) -> str:
    """读取一个文本文件的内容，返回可读文本或"无法摘要"的说明。

    尝试多种编码读取，优先 UTF-8，失败则尝试 GBK。
    若文件是二进制（含大量不可见控制字符）或过大，返回原因说明。
    """
    from pathlib import Path
    p = Path(file_path)
    try:
        size = p.stat().st_size
    except OSError:
        return "无法摘要：文件不存在或无权访问"

    # 二进制检测 + 读取
    try:
        raw = p.read_bytes()
    except OSError:
        return "无法摘要：文件读取失败"

    # 二进制检测：如果前 1024 字节里出现了空字节，判为二进制
    head = raw[:1024]
    if b"\x00" in head:
        return "无法摘要：二进制文件，不含可读文本"

    # 尝试多种编码解码
    text = None
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if text is None:
        return "无法摘要：文件编码无法识别"

    # 截断到最大字符数
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n...(内容过长，已截断)"
    return text


def _make_result(file_info: dict, summary: str) -> dict:
    """统一构造单个文件的摘要结果，包含文件名、完整路径、相对路径、类型、大小。"""
    return {
        "name": file_info["name"],
        "path": file_info.get("path", ""),          # 完整绝对路径
        "rel_path": file_info.get("rel_path", ""),  # 相对该子文件夹的路径
        "ext": file_info.get("ext", ""),            # 扩展名（文件类型）
        "size": file_info.get("size", 0),
        "size_human": file_info.get("size_human", ""),
        "summary": summary,
    }


def _summarize_one_file(file_info: dict) -> dict:
    """对单个文件生成摘要，返回 {name, path, rel_path, ext, size, size_human, summary}。"""
    if not file_info.get("is_text"):
        return _make_result(
            file_info, "无法摘要：非文本类型文件（如图片、视频、二进制）")

    content = _read_file(file_info["path"])
    if content.startswith("无法摘要"):
        return _make_result(file_info, content)

    try:
        summary = llm_chat(SUB_SYSTEM, content, temperature=0.0,
                           max_tokens=300).strip()
    except Exception as e:
        summary = f"无法摘要：LLM 调用失败({str(e)[:80]})"

    return _make_result(file_info, summary)


def run_analysis(scan_result: dict,
                 on_progress: Optional[Callable] = None,
                 max_workers: int = 8) -> dict:
    """对磁盘扫描结果执行并行摘要，返回完整分析结果。

    参数：
      scan_result : file_scanner.scan_folder 的返回结果
      on_progress  : 进度回调，签名 (done, total, message)
                     done 已完成文件数，total 总文件数，message 进度文字
      max_workers  : 最大并发 subagent 数（限制 LLM 并发，防止限流）

    返回：
      在 scan_result 基础上，为每个子文件夹的 top_files 增加 summary 字段。
    """
    subfolders = scan_result.get("subfolders", [])

    # 没有子文件夹就直接返回
    if not subfolders:
        return scan_result

    # 统计总文件数（用于进度条）
    total_files = sum(len(sf["top_files"]) for sf in subfolders)
    done_counter = {"done": 0}  # 用可变容器跨线程计数

    def _on_file_done(subfolder_name: str, file_result: dict):
        """单个文件摘要完成时触发，更新进度。"""
        done_counter["done"] += 1
        if on_progress:
            on_progress(
                done_counter["done"],
                total_files,
                f"正在分析 {subfolder_name} 的 {file_result['name']} ...",
            )

    # 并行处理每个子文件夹，用 max_workers 限制并发
    results = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(subfolders))) as pool:
        futs = {
            pool.submit(_summarize_subfolder, sf, _on_file_done): sf["name"]
            for sf in subfolders
        }
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                logger.error(f"子文件夹 {name} 摘要失败: {e}")
                results[name] = {
                    "name": name,
                    "size_human": "未知",
                    "top_files": [],
                    "error": str(e),
                }

    # 保持原有的排序顺序（按大小降序）
    ordered = []
    for sf in subfolders:
        if sf["name"] in results:
            ordered.append(results[sf["name"]])
    scan_result["subfolders"] = ordered

    return scan_result


def _summarize_subfolder(subfolder: dict,
                         on_file_done: Optional[Callable] = None) -> dict:
    """处理单个子文件夹：对它的 Top3 文件依次生成摘要。"""
    results = []
    for f in subfolder["top_files"]:
        r = _summarize_one_file(f)
        results.append(r)
        if on_file_done:
            on_file_done(subfolder["name"], r)

    return {
        "name": subfolder["name"],
        "path": subfolder["path"],
        "size": subfolder["size"],
        "size_human": subfolder["size_human"],
        "top_files": results,
    }


"""
磁盘扫描模块（纯 Python，不调用 LLM）

职责：
  1. 计算文件夹总大小
  2. 列出所有直接子文件夹，以及各自的大小
  3. 在每个子文件夹内按文件大小排序，取 Top3 文件

设计要点：
  - 全部用标准库 os / pathlib，零额外依赖
  - 按"直接子文件夹"派发分析单元，但每个子文件夹会递归扫描整个子树
  - 返回结构化数据，供 agents.py 派发 subagent 做内容摘要
"""

import os
from pathlib import Path

# 可读文本文件的扩展名白名单（这些类型会尝试做内容摘要）
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".java", ".c", ".cpp",
    ".h", ".hpp", ".go", ".rs", ".json", ".yaml", ".yml", ".xml", ".html",
    ".htm", ".css", ".csv", ".tsv", ".log", ".ini", ".conf", ".sql", ".sh",
    ".bat", ".ps1", ".toml", ".cfg", ".properties", ".rst", ".tex",
}


def _dir_size(path: Path) -> int:
    """递归计算一个目录的总大小（字节）。

    用 os.walk 遍历目录树，累加所有文件的大小。
    遇到无法访问的文件（权限/软链接等）跳过，不报错。
    """
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    # 文件可能被删除或无权访问，跳过
                    continue
    except OSError:
        pass
    return total


def format_size(num_bytes: int) -> str:
    """把字节数格式化成人类可读的字符串，如 "1.5 MB"。

    依次尝试 GB / MB / KB，保留 1 位小数。
    """
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.1f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def scan_folder(root_path: str, top_n: int = 3) -> dict:
    """扫描一个文件夹，返回结构化结果。

    返回结构：
    {
      "root": "/path/to/folder",
      "total_size": 123456789,           # 总大小（字节）
      "total_size_human": "117.7 MB",    # 人类可读的总大小
      "subfolders": [                    # 直接子文件夹列表
        {
          "name": "subdir1",             # 子文件夹名
          "path": ".../subdir1",         # 完整路径
          "size": 123456,                # 该子文件夹大小（字节）
          "size_human": "120.6 KB",
          "top_files": [                 # 该子文件夹内 Top3 文件
            {
              "name": "a.txt",
              "path": ".../a.txt",
              "ext": ".txt",             # 扩展名（小写）
              "size": 10000,
              "size_human": "9.8 KB",
              "is_text": True,           # 是否是文本文件（可尝试摘要）
            },
            ...
          ],
        },
        ...
      ],
    }

    参数：
      root_path : 用户提供的文件夹路径
      top_n     : 每个子文件夹取前几大的文件，默认 3
    """
    root = Path(root_path).resolve()
    result = {
        "root": str(root),
        "total_size": _dir_size(root),
        "subfolders": [],
    }
    result["total_size_human"] = format_size(result["total_size"])

    # 只处理直接子文件夹（一层）
    if not root.exists() or not root.is_dir():
        return result

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        # 统计该子文件夹大小
        sub_size = _dir_size(child)
        sub_entry = {
            "name": child.name,
            "path": str(child),
            "size": sub_size,
            "size_human": format_size(sub_size),
            "top_files": [],
        }

        # 递归收集该子文件夹内的所有文件（遍历整个子树，含嵌套子目录）
        files = []
        for dirpath, _dirnames, filenames in os.walk(child):
            for fn in filenames:
                fp = Path(dirpath) / fn
                try:
                    fsize = fp.stat().st_size
                except OSError:
                    continue
                ext = fp.suffix.lower()
                files.append({
                    "name": fn,
                    "path": str(fp),
                    # 相对该子文件夹的路径，方便看出文件在哪一层
                    "rel_path": str(fp.relative_to(child)),
                    "ext": ext,
                    "size": fsize,
                    "size_human": format_size(fsize),
                    "is_text": ext in TEXT_EXTENSIONS,
                })

        # 按大小降序排序，取 Top N（整个子文件夹子树里最大的 N 个文件）
        files.sort(key=lambda x: x["size"], reverse=True)
        sub_entry["top_files"] = files[:top_n]

        result["subfolders"].append(sub_entry)

    # 子文件夹也按大小降序排，大的在前
    result["subfolders"].sort(key=lambda x: x["size"], reverse=True)

    return result


"""
极简 DeepSeek LLM 客户端

说明：
  - 使用 DeepSeek 的 deepseek-chat 模型（OpenAI 兼容接口）
  - 单例模式：全局只创建一个客户端，避免重复初始化
  - 带重试：网络抖动时自动重试，指数退避（1秒、2秒、4秒...）

依赖：pip install openai
环境变量：DEEPSEEK_API_KEY
"""

import os
import time
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

# DeepSeek 的接口地址（OpenAI 兼容）
DEEPSEEK_URL = "https://api.deepseek.com"
# 使用的模型名
DEEPSEEK_MODEL = "deepseek-chat"

# 全局客户端（懒加载单例：第一次调用时才真正创建）
_client = None


def get_client() -> OpenAI:
    """获取或创建 DeepSeek 客户端（单例模式）。

    第一次调用时从环境变量 DEEPSEEK_API_KEY 读取密钥并创建客户端，
    之后每次都复用同一个客户端，避免重复初始化开销。
    """
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 1024, retries: int = 3) -> str:
    """单轮 LLM 对话，返回模型回复文本。

    参数：
      system      : 系统提示词（定义 AI 角色和行为）
      user        : 用户提示词（具体要 AI 做的事）
      temperature : 温度，0.0 表示最确定（不要随机性）
      max_tokens  : 最大生成 token 数（限制回复长度）
      retries     : 失败重试次数
    """
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            # 最后一次尝试仍失败就抛出异常，不再重试
            if attempt == retries - 1:
                raise
            # 指数退避：第 1 次失败等 1 秒，第 2 次等 2 秒，第 3 次等 4 秒
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 调用失败，重试({attempt + 1}): {str(e)[:80]}")
