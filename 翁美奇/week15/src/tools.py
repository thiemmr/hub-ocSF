"""
数值计算工具 + RAG 检索工具
- calc_profit_margin : 毛利率 = (收入 - 成本) / 收入 * 100%
- calc_asset_turnover : 资产周转率 = 收入 / 总资产
- rag_search : 从 FAISS 向量库检索财报文本
"""
import os
import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env（与 src/ 同级）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ========== 数值计算工具 ==========

def calc_profit_margin(revenue: float, cost: float) -> dict:
    """毛利率 = (收入 - 成本) / 收入 * 100%"""
    if revenue == 0:
        return {"error": "revenue 不能为 0"}
    value = (revenue - cost) / revenue * 100
    return {
        "indicator": "毛利率",
        "value": round(value, 2),
        "unit": "%",
        "formula": "(revenue - cost) / revenue * 100%",
    }


def calc_asset_turnover(revenue: float, total_assets: float) -> dict:
    """资产周转率 = 收入 / 总资产"""
    if total_assets == 0:
        return {"error": "total_assets 不能为 0"}
    value = revenue / total_assets
    return {
        "indicator": "资产周转率",
        "value": round(value, 4),
        "formula": "revenue / total_assets",
    }


# 数值工具的 function-calling schema（供数值计算专家子 Agent 使用）
NUMERIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc_profit_margin",
            "description": "计算毛利率 = (营业收入 - 营业成本) / 营业收入 * 100%",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number", "description": "营业收入"},
                    "cost": {"type": "number", "description": "营业成本"},
                },
                "required": ["revenue", "cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc_asset_turnover",
            "description": "计算资产周转率 = 营业收入 / 总资产",
            "parameters": {
                "type": "object",
                "properties": {
                    "revenue": {"type": "number", "description": "营业收入"},
                    "total_assets": {"type": "number", "description": "总资产"},
                },
                "required": ["revenue", "total_assets"],
            },
        },
    },
]

# 工具名 -> 函数映射
NUMERIC_TOOL_MAP = {
    "calc_profit_margin": calc_profit_margin,
    "calc_asset_turnover": calc_asset_turnover,
}


# ========== RAG 检索工具（原生 faiss + openai SDK）==========

_FAISS_INDEX = None
_FAISS_META = None
_EMB_CLIENT = None

# embedding 配置：用 openai SDK 调 DashScope 兼容端点（与建库时一致：text-embedding-v3，1024 维）
_EMB_BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()
_EMB_API_KEY = (
    os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
).strip()
_EMB_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3").strip()


def _load_vectorstore():
    """加载原生 FAISS 索引 + 元数据（vectorstore/faiss_index.bin + faiss_meta.json）"""
    global _FAISS_INDEX, _FAISS_META, _EMB_CLIENT
    if _FAISS_INDEX is None:
        base = Path(__file__).resolve().parent.parent / "vectorstore"
        _FAISS_INDEX = faiss.read_index(str(base / "faiss_index.bin"))
        with open(base / "faiss_meta.json", encoding="utf-8") as f:
            _FAISS_META = json.load(f)
        _EMB_CLIENT = OpenAI(api_key=_EMB_API_KEY, base_url=_EMB_BASE_URL)
    return _FAISS_INDEX, _FAISS_META, _EMB_CLIENT


def rag_search(query: str, k: int = 4) -> dict:
    """从财报向量库检索与 query 最相关的 k 条文本片段。

    流程：openai SDK 生成 query 向量 → 原生 faiss 检索 → 从 faiss_meta.json 取元数据。
    """
    idx, meta, emb_client = _load_vectorstore()
    # 1. query 向量化（DashScope 兼容端点，默认 1024 维与索引对齐）
    resp = emb_client.embeddings.create(model=_EMB_MODEL, input=query)
    qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    # 2. faiss 检索
    D, I = idx.search(qvec, k)
    # 3. 组装结果
    results = []
    for dist, i in zip(D[0], I[0]):
        if i < 0 or i >= len(meta):
            continue
        m = meta[i]
        results.append(
            {
                "content": m.get("content", ""),
                "stock_code": m.get("stock_code"),
                "year": m.get("year"),
                "section": m.get("section"),
                "score": float(dist),
            }
        )
    return {"query": query, "count": len(results), "results": results}
