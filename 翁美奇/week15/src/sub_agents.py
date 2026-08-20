"""子 Agent（执行者）

- numeric_expert：数值计算专家，function calling 调用 calc 工具，不带 RAG
- rag_expert     ：RAG 检索专家，向量检索 + 摘要，不带数学工具
"""
import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from llm import chat
from tools import NUMERIC_TOOLS, NUMERIC_TOOL_MAP, rag_search


# ========== 子 Agent A：数值计算专家 ==========
def numeric_expert(task: str) -> str:
    """通过 function calling 调用数学工具完成计算任务"""
    messages = [
        {
            "role": "system",
            "content": "你是数值计算专家，只能调用提供的数学工具完成计算，不要臆造数据。"
            "若缺少必要数值，请明确说明需要哪些输入。最后给出指标结果与所用公式。",
        },
        {"role": "user", "content": task},
    ]
    for _ in range(5):  # 最多 5 次工具调用
        resp = chat(messages, tools=NUMERIC_TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "（数值计算专家未返回结果）"
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = NUMERIC_TOOL_MAP[tc.function.name](**args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    return "数值计算专家达到最大工具调用次数。"


# ========== 子 Agent B：RAG 检索专家 ==========
def rag_expert(query: str) -> str:
    """向量检索财报片段 + LLM 摘要"""
    retrieved = rag_search(query, k=4)
    context = "\n\n".join(
        f"[{r['stock_code']} {r['year']}] {r['content']}"
        for r in retrieved["results"]
    )
    sources = ", ".join(
        f"{r['stock_code']}/{r['year']}" for r in retrieved["results"]
    )
    messages = [
        {
            "role": "system",
            "content": "你是 RAG 检索专家，只能依据检索到的财报文本回答问题，不要编造库外信息。"
            "若检索内容与问题无关，请说明。",
        },
        {"role": "user", "content": f"问题：{query}\n\n检索到的财报片段：\n{context}"},
    ]
    resp = chat(messages)
    answer = resp.choices[0].message.content
    return f"{answer}\n[来源: {sources}]"


# ========== 子 Agent 调度表 ==========
SUB_AGENT_MAP = {
    "numeric_expert": numeric_expert,
    "rag_expert": rag_expert,
}


def exec_subagent(name: str, action_input: str) -> str:
    """按名称执行单个子 Agent（不含计时）"""
    fn = SUB_AGENT_MAP.get(name)
    if fn is None:
        return f"未知 Action: {name}，可选: {list(SUB_AGENT_MAP)}"
    return fn(action_input)
