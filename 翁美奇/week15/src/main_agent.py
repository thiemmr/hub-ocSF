"""主 Agent（协调者）：ReAct 循环 + 并行扇出调度

架构：
  主 Agent（ReAct：推理→行动→观察→汇总）
    ├─ 子 Agent A：数值计算专家（见 sub_agents.py）
    └─ 子 Agent B：RAG 检索专家（见 sub_agents.py）

核心优势：主 Agent 可在一次行动中扇出多个子 Agent，用 ThreadPoolExecutor 并行执行。
性能量化：wall_clock（并行墙钟）vs serial_sum（各子 Agent 时长之和 = 串行基线）。
A/B 对比：parallel=False 时退化为 for 循环，作为串行基线。
"""
import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))

from llm import chat
from sub_agents import exec_subagent

from dotenv import load_dotenv

MAX_REACT_STEPS = 3  # ReAct 最大行动轮次


# ========== 行动解析：支持并行扇出（多个 Action N）==========
def _parse_actions(text: str):
    """从主 Agent 输出中解析出 [(name, input), ...]。

    支持两种写法：
      单个：Action: <name>  /  Action Input: <input>
      多个：Action 1: <name> / Action Input 1: <input>  （并行扇出）
    """
    names, inputs = {}, {}
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"Action\s*Input\s*(\d*)\s*:\s*(.+)", s)
        if m:
            inputs[m.group(1) or "1"] = m.group(2).strip()
            continue
        m = re.match(r"Action\s*(\d*)\s*:\s*(.+)", s)
        if m:
            names[m.group(1) or "1"] = m.group(2).strip()
    return [(names[n], inputs[n]) for n in names if n in inputs]


# ========== 并行 / 串行执行器（含计时）==========
def _run_actions(actions, parallel=True):
    """执行一批子 Agent 调用，返回 (results, wall_clock, serial_sum)。

    - parallel=True 且 action 数>1：ThreadPoolExecutor 并行，wall_clock≈最慢者耗时
    - 否则：for 循环串行，wall_clock≈各耗时之和
    - serial_sum 始终 = 各子 Agent 单独耗时之和（串行基线）
    """
    def _one(idx, name, action_input):
        t = time.time()
        res = exec_subagent(name, action_input)
        return idx, name, action_input, res, time.time() - t

    results = []
    if parallel and len(actions) > 1:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(actions)) as ex:
            futs = [
                ex.submit(_one, i, n, p) for i, (n, p) in enumerate(actions)
            ]
            for f in futs:
                results.append(f.result())
        wall = time.time() - t0
    else:
        t0 = time.time()
        for i, (n, p) in enumerate(actions):
            results.append(_one(i, n, p))
        wall = time.time() - t0

    results.sort(key=lambda x: x[0])  # 按提交顺序还原
    serial_sum = sum(r[4] for r in results)
    return results, wall, serial_sum


def _print_stats(mode, total_wall, total_serial):
    print("\n===== 性能对比 =====")
    print(f"执行模式 mode         : {mode}")
    print(f"总墙钟时间 wall_clock : {total_wall:.2f}s")
    print(f"串行基线 serial_sum   : {total_serial:.2f}s")
    if total_wall > 0:
        print(f"加速比 speedup        : {total_serial / total_wall:.2f}x")


# ========== 主 Agent：ReAct 循环 ==========
REACT_SYSTEM = """你是一个财务分析主管（主 Agent），使用 ReAct（推理-行动-观察）循环工作。
你有两个专家子 Agent 可调用：
1. numeric_expert：数值计算专家，计算毛利率、资产周转率等。输入是具体的计算任务描述（必须含所需数值）。
2. rag_expert：RAG 检索专家，从财报向量库检索附注与文本语义。输入是检索查询文本。

【并行扇出】每一步你可以输出一个或多个 Action。当多个子任务相互独立时，应一次性输出多个 Action，
它们会被【并行执行】——这是本系统的核心优势（例如：同时算两家公司的指标、或同时算指标 + 检索附注）。

每一步按以下格式输出（一次可含多个 Action，或一个 Final Answer）：

Thought: <推理：还缺什么信息、这一步要做哪几件事、它们是否可并行>
Action 1: <numeric_expert 或 rag_expert>
Action Input 1: <交给子 Agent 1 的输入>
Action 2: <numeric_expert 或 rag_expert>
Action Input 2: <交给子 Agent 2 的输入>
...（依需继续）

【重要】输出完 Action Input 后必须立即停止，不要自己编写 Observation！
系统会自动执行这些子 Agent 并把真实结果作为 Observation 返回给你。
你绝对不能自己生成 Observation 行，否则结果会作废。

当你已收集到足够信息可以回答用户时，输出：
Thought: <推理摘要>
Final Answer: <给用户的最终回答，需包含：结论、数值依据、文本依据>

规则：
- 不要编造数值，所有数值必须来自 numeric_expert 的计算结果。
- 文本依据必须来自 rag_expert 的检索结果。
- 绝不要自己写 Observation，那是由系统注入的。
- 最多进行 {max_steps} 步行动。"""


def react_run(question: str, parallel: bool = True) -> str:
    """主 Agent 的 ReAct 循环：推理 → 行动（可并行扇出）→ 观察 → 汇总。

    parallel=True  ：子 Agent 并行执行（默认，凸显优势）
    parallel=False ：退化为 for 循环串行执行（A/B 对比基线）
    """
    mode = "并行 (ThreadPoolExecutor)" if parallel else "串行 (for 循环基线)"
    messages = [
        {"role": "system", "content": REACT_SYSTEM.format(max_steps=MAX_REACT_STEPS)},
        {"role": "user", "content": question},
    ]
    trace = []  # 推理链日志（可解释性）
    total_wall, total_serial = 0.0, 0.0

    for step in range(1, MAX_REACT_STEPS + 1):
        # stop 序列：LLM 想写 "Observation:" 时立即截断，强制交还控制权由系统执行子 Agent
        resp = chat(messages, temperature=0.3, stop=["Observation:"])
        text = resp.choices[0].message.content.strip()
        trace.append(text)

        # 终止条件：已得到最终答案
        if "Final Answer:" in text:
            final = text.split("Final Answer:", 1)[1].strip()
            print("\n===== 推理摘要（CoT）=====")
            print("\n\n".join(trace))
            _print_stats(mode, total_wall, total_serial)
            return final

        # 行动阶段：解析（可能多个）Action
        actions = _parse_actions(text)
        if not actions:
            _print_stats(mode, total_wall, total_serial)
            return text  # 未按格式输出，直接作为最终回答

        names_preview = ", ".join(n for n, _ in actions)
        print(f"[Step {step}] 扇出 {len(actions)} 个子Agent: {names_preview}")

        # 观察阶段：并行 / 串行执行 + 计时
        results, wall, serial_sum = _run_actions(actions, parallel=parallel)
        total_wall += wall
        total_serial += serial_sum
        print(
            f"[Step {step}] wall_clock={wall:.2f}s | "
            f"serial_sum={serial_sum:.2f}s"
        )

        # 把各子 Agent 的 Observation 拼回上下文
        obs_lines = []
        for idx, name, inp, res, dur in results:
            print(f"  - {name} ({dur:.2f}s): {res[:80]}")
            obs_lines.append(f"Observation {idx + 1} ({name}): {res}")
        observation = "\n".join(obs_lines)

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    _print_stats(mode, total_wall, total_serial)
    return "已达最大推理轮次，未能给出最终答案。推理链：\n" + "\n".join(trace)


# ========== 入口 ==========
def _parse_cli_args():
    args = sys.argv[1:]
    parallel = True
    compare = False
    if "--serial" in args:
        parallel = False
    if "--compare" in args:
        compare = True
    return parallel, compare


if __name__ == "__main__":
    parallel, compare = _parse_cli_args()
    q = input("请输入财务问题：").strip()
    if not q:
        sys.exit(0)

    if compare:
        # A/B 对比：同一问题分别跑串行与并行
        print("\n################ A: 串行基线 ################")
        ans_s = react_run(q, parallel=False)
        print("\n最终回答(串行):\n", ans_s)

        print("\n################ B: 并行模式 ################")
        ans_p = react_run(q, parallel=True)
        print("\n最终回答(并行):\n", ans_p)
    else:
        print(f"\n===== 最终回答（{'并行' if parallel else '串行'}）=====")
        print(react_run(q, parallel=parallel))
