import os, time, json, logging, uuid, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

RESULT_DIR = Path(__file__).parent.parent / "result"

# 关键词自动展开：当用户只输入短技术词（如 "skill"、"RAG"）时，
# 自动把问题补充为 AI/技术语境下的完整技术调研，并按 4 个标准维度派发子课题。
# 避免因用户输入太简短而触发"单一事实问题 → web_search 一次"的降级分支。
_KEYWORD_HINTS = (
    "技术", "算法", "模型", "框架", "系统", "架构", "协议",
    "RAG", "LLM", "AI", "Agent", "Vector", "Embedding",
    "Transformer", "Attention", "Diffusion", "GAN",
)

def _is_short_keyword(question: str) -> bool:
    """启发式判断是否是短关键词输入（需要自动展开）。
    - 纯关键词特征：字数≤12 且不含问号/怎么/为什么/请调研/介绍一下 等明显问句或指令词
    - 或命中 _KEYWORD_HINTS 且总字数≤20
    """
    q = question.strip()
    if not q:
        return False
    # 明确有调研/介绍指令的，不走自动展开（交给主 agent 正常解析即可）
    directive_kw = ("调研", "介绍", "说明", "讲解", "分析", "报告", "对比",
                    "？", "?", "怎么", "为什么", "是什么", "请", "请教")
    if any(k in q for k in directive_kw):
        return False
    # 短词命中：字数少 + 不含标点或只含中英文名词
    if len(q) <= 12:
        return True
    # 技术关键词命中且长度不长
    if any(k.lower() in q.lower() for k in _KEYWORD_HINTS) and len(q) <= 20:
        return True
    return False


def _build_subtopics_for_keyword(keyword: str) -> tuple[str, str]:
    """针对短关键词构造完整提问 + 4 个标准维度子课题。
    返回 (normalized_question, pipe_separated_subtopics)。"""
    kw = keyword.strip()
    # 若用户没有"AI"、"技术"等语境，补一个技术语境前缀（避免 skill 搜到职业技能类内容）
    contextual = f"AI 领域中 {kw} 相关技术" if not any(
        h.lower() in kw.lower() for h in ("AI", "LLM", "技术", "算法", "模型", "框架",
                                          "系统", "架构", "RAG", "Agent")) else kw

    question = f"请技术调研 {contextual}：包含背景、原理、应用场景"
    subtopics = (
        f"{contextual} 的定义、别名与发展历史（背景角度）",
        f"{contextual} 的核心原理、算法/架构与关键组件（原理角度，尽量深入）",
        f"{contextual} 的典型应用场景、行业案例与落地实践（应用角度，含例子）",
        f"{contextual} 与同类技术的优缺点对比、最新进展与趋势（对比角度）",
    )
    return question, " | ".join(subtopics)

MAIN_SYSTEM = """你是技术调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要用户输入的是技术名词/技术话题（哪怕只是一两个单词，如 "skill"、"RAG"、"图工程"），
  都应当作技术调研需求处理：先补全语境（如"AI 领域的 XX 技术"），
  再用 dispatch_subagents 派发 4 个标准维度的子调研员并行搜索，绝对不能只用一次 web_search。
- 只要问题涉及某技术的调研（原理、背景、应用等），必须用 dispatch_subagents 派发多个子调研员
  从不同角度/范围并行搜索，不要自己串行 web_search 多次。
  建议派发维度（按需要选 3-4 个）：
    1. 技术名称、别名与发展历史（背景角度）
    2. 技术核心原理、算法架构、工作机制（原理角度）
    3. 技术典型应用场景、行业案例、落地实践（应用角度）
    4. 技术优缺点对比、与同类技术的差异（对比角度）
    5. 技术最新进展、生态与未来趋势（趋势角度）
  示例："请调研 RAG 技术" → Action: dispatch_subagents
        Action Input: RAG 技术的定义、别名与发展历史 | RAG 技术核心原理与架构详解 | RAG 典型应用场景与落地案例 | RAG 与同类技术（Fine-tuning、向量检索）对比
- 只有单一事实问题（如"RAG 全称是什么"）才直接 web_search
- 拿到子调研结果后，调用汇总去重能力，综合输出一份完整、无重复的技术调研报告

【汇总输出结构要求】
Final Answer 必须严格按以下结构组织，以**教师讲解**的口吻输出（而非报告式罗列），注意去重合并：

## 技术名称
（正式名称 + 别名，一句话带过）

## 技术背景
（2-3句话简述该技术为什么会出现、解决什么痛点。不要长篇发展史，点到为止）

## 技术原理
（像老师讲课一样循序渐进地讲解核心原理：先讲整体思路，再拆解关键组件，
  用通俗类比帮助理解，让读者明白"为什么这样设计"。这是最重要的板块，尽量深入）

## 应用场景
（结合具体例子讲解什么场景该用、什么场景不该用，让读者有实操判断力）

## 缺点与对比
（客观讲解该技术的不足之处：性能瓶颈、适用边界、已知缺陷等；
  并与同类技术做横向对比，让读者明白"什么时候该选它、什么时候该选别的"）

## 总结与展望
（一句话总结核心价值，简述未来方向）

风格要求：用"讲解"而非"罗列"的语气，如"这项技术的核心思想是…""你可以把它理解为…""关键在于…"。
避免堆砌来源编号，重在让读者听懂。多个子调研员的重复信息合并为一条最完整的。"""


SUBAGENT_SYSTEM = """你是技术调研员，负责就一个指定子课题（如某技术的背景/原理/应用/对比）进行联网搜索调研。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步

【输出结构要求】
Final Answer 必须严格按以下结构组织（便于主 agent 后续汇总去重），以教师讲解的口吻输出：

## 技术名称
（子课题涉及的技术正式名称，如有别名也列出，一句话带过）

## 技术背景
（如果子课题涉及背景：2-3句话简述该技术为什么会出现、解决什么痛点，不要长篇发展史；
  否则写"本子课题不覆盖此板块"）

## 技术原理
（如果子课题涉及原理：像老师讲课一样循序渐进地讲解——先讲整体思路，再拆解关键组件，
  用通俗类比帮助理解，让读者明白"为什么这样设计"。尽量深入，这是最重要的板块；
  否则写"本子课题不覆盖此板块"）

## 应用场景
（如果子课题涉及应用：结合具体例子讲解什么场景该用、什么场景不该用；
  否则写"本子课题不覆盖此板块"）

## 缺点与对比
（如果子课题涉及对比/缺点：客观讲解该技术的不足之处（性能瓶颈、适用边界、已知缺陷），
  并与同类技术做横向对比；否则写"本子课题不覆盖此板块"）

风格要求：用"讲解"而非"罗列"的语气，如"核心思想是…""你可以理解为…""关键在于…"。
如果子课题是"趋势"类，可在末尾补充趋势内容，但仍需先填好上面五个板块。"""


DEDUP_SYSTEM = """你是技术调研汇总去重专家，也是一位善于讲解的技术教师。

你的任务：把多个子调研员从不同角度搜索到的技术调研结果，去重合并为一份完整的技术科普文章。

【输出结构（严格遵守，6 个板块，缺的写"暂无"）】
## 技术名称
（正式名称 + 别名，一句话带过）

## 技术背景
（2-3句话简述该技术为什么会出现、解决什么痛点。不要长篇发展史，点到为止）

## 技术原理
（最重要的板块。像老师讲课一样循序渐进：先讲整体思路，再拆解关键组件，
  用通俗类比帮助理解，让读者明白"为什么这样设计"。尽量深入透彻）

## 应用场景
（结合具体例子讲解什么场景该用、什么场景不该用，让读者有实操判断力）

## 缺点与对比
（客观讲解该技术的不足之处：性能瓶颈、适用边界、已知缺陷等；
  并与同类技术做横向对比，让读者明白"什么时候该选它、什么时候该选别的"）

## 总结与展望
（一句话总结核心价值，简述未来方向）

【风格要求】
- 用"讲解"语气，如"这项技术的核心思想是…""你可以把它理解为…""关键在于…"
- 不要堆砌来源编号，重在让读者听懂
- 不要写成调研报告的罗列体，要像一篇技术科普文章

【去重合并原则】
1. 多个子 agent 都提到的同一事实，只保留最完整的一条，不要重复叙述。
2. 互补信息合并（如一个讲了架构，一个讲了算法细节，合并为完整讲解）。
3. 矛盾表述并列标注"不同来源存在差异：X vs Y"。

直接输出 6 个板块的报告内容，不要输出 Thought/Action 等任何其他内容。"""


def _save_report_tool(action_input: str, shared_state: dict = None) -> str:
    """save_report 工具：将技术调研报告保存为 markdown 文件。
    action_input 格式: "文件名|报告内容"（以第一个 | 分隔，文件名不含扩展名）
    若未提供文件名，则从 shared_state["question"] 推导。
    文件保存到 RESULT_DIR 目录，自动加 .md 扩展名。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if "|" in action_input:
        filename, content = action_input.split("|", 1)
        filename = filename.strip()
    else:
        # 没有文件名分隔符，从 question 推导文件名
        content = action_input
        q = (shared_state or {}).get("question", "技术调研报告")
        filename = q[:30]

    # 清理文件名中的非法字符
    safe_name = re.sub(r'[\\/:*?"<>|]', "", filename).strip()
    if not safe_name:
        safe_name = "技术调研报告"
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    path = RESULT_DIR / safe_name
    path.write_text(content.strip(), encoding="utf-8")
    logger.info(f"技术调研报告已保存: {path}")
    return f"报告已保存至: {path}"


def _deduplicate_summarize(sub_results: dict, question: str = "",
                            on_step: Callable = None) -> str:
    """汇总去重：将多个子 agent 的技术调研结果去重合并为结构化报告，
    并保存为 markdown 文件。

    直接调用 LLM 生成报告（不走 ReAct 循环），然后手动保存 md 文件。
    避免 ReAct 循环中 LLM 把 token 浪费在 Thought 上导致报告内容缺失。

    sub_results: {sid: {"subtopic":.., "final_answer":..}} 或 {sid: (topic, res_dict)}
    question: 原始提问（用于推导报告文件名）
    on_step: 可选步骤回调（SSE 流式用）
    返回去重合并后的报告文本。"""
    from llm_client import llm_chat

    # ── 1. 收集所有子 agent 的原始文本 ──
    parts = []
    for sid, data in sub_results.items():
        if isinstance(data, tuple):
            topic, val = data[0], data[1]
            final_answer = val if isinstance(val, str) else val.get("final_answer", "")
        elif isinstance(data, dict):
            topic = data.get("subtopic", sid)
            final_answer = data.get("final_answer", "")
        else:
            topic, final_answer = sid, str(data)
        parts.append(f"=== 子课题: {topic} (子agent={sid}) ===\n{final_answer}")
    raw = "\n\n".join(parts)
    if not raw.strip():
        return "无子 agent 结果，无法生成报告。"

    # ── 2. 直接调 LLM 生成去重报告 ──
    if on_step:
        on_step({"idx": 0, "agent": "dedup", "thought": "正在汇总去重并生成教师讲解式报告…",
                 "action": "llm_dedup", "action_input": None, "observation": None,
                 "final": False})

    try:
        report = llm_chat(
            DEDUP_SYSTEM,
            f"请汇总去重以下技术调研结果，按上述结构输出完整报告：\n\n{raw[:12000]}",
            temperature=0.0, max_tokens=2048,
        )
        # 清理可能的 Thought/Action 前缀残留
        if "## 技术名称" in report:
            report = report[report.index("## 技术名称"):]
    except Exception as e:
        logger.warning(f"去重汇总 LLM 调用失败，退回原始拼接: {e}")
        report = "（自动去重失败，以下为原始拼接）\n\n" + raw

    report = report.strip()

    # ── 3. 保存为 md 文件 ──
    if on_step:
        on_step({"idx": 1, "agent": "dedup", "thought": "报告生成完成，正在保存为 md 文件…",
                 "action": "save_report", "action_input": question[:30],
                 "observation": None, "final": False})

    try:
        saved_path = _save_report_tool(
            f"{question[:30]}|{report}",
            shared_state={"question": question},
        )
        if on_step:
            on_step({"idx": 1, "agent": "dedup", "thought": "报告已保存",
                     "action": "save_report", "action_input": question[:30],
                     "observation": saved_path, "done": True, "final": True})
    except Exception as e:
        logger.warning(f"保存报告失败: {e}")

    return report


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)",
            system_prompt=SUBAGENT_SYSTEM)   # ← 子 agent 用专属技术调研 prompt
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次技术调研。返回 {final_answer, main_trace, subagents, parallel_stats}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。

    【短关键词自动展开】
    若用户只输入短技术关键词（如 "skill"、"RAG"），会自动补充为 AI 语境下的技术调研，
    并按 4 个标准维度（背景/原理/应用/对比）直接派发子 agent，
    不依赖主 agent 的判断，避免因提问太短被降级为单搜。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}
    final_answer = ""

    # ── 1. 短关键词自动展开：直接派发标准 4 维度子课题，不进主 agent ──
    if _is_short_keyword(question):
        normalized_q, subtopics_str = _build_subtopics_for_keyword(question)
        logger.info(f"短关键词 '{question}' 自动展开为调研: {normalized_q}")
        # 发出主 agent 的一步"派发"step（保持前端拓扑/事件流一致）
        synthetic_step = {"idx": 0, "agent": "main",
                          "thought": f"用户输入为技术关键词，自动补充语境为『{normalized_q}』，"
                                     f"按 4 个标准维度派发子调研员并行搜索",
                          "action": "dispatch_subagents",
                          "action_input": subtopics_str,
                          "observation": None, "final": False}
        if on_main_step:
            on_main_step(synthetic_step)

        # 直接派发子 agent
        _dispatch_subagents(subtopics_str, shared_state=shared_state,
                            on_subagent_step=on_subagent_step,
                            on_subagent_done=on_subagent_done,
                            on_dispatch=on_dispatch,
                            serial=serial)

        synthetic_step["observation"] = (
            f"已自动派发 {len(shared_state['subagents'])} 个子调研员并行完成调研"
        )
        synthetic_step["done"] = True
        if on_main_step:
            on_main_step(synthetic_step)

        # 走去重汇总
        subagents_data = shared_state["subagents"]
        if subagents_data:
            try:
                final_answer = _deduplicate_summarize(subagents_data,
                                                       question=normalized_q)
            except Exception as e:
                logger.warning(f"短关键词分支去重汇总失败: {e}")
                final_answer = ""
        if not final_answer:
            final_answer = "子 agent 未返回结果"

        return {
            "final_answer": final_answer,
            "main_trace": [synthetic_step],  # 合成的一步派发 trace
            "subagents": shared_state["subagents"],
            "parallel_stats": shared_state["parallel_stats"],
            "dispatches": shared_state["dispatches"],
        }

    # ── 2. 常规分支：完整调研指令交给主 agent 判断派发 ──

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        # dispatch 事件由 _dispatch_subagents 用真实 subagent id 发出
        # （不能在这里预生成 id，否则和 subagent_step 的 id 对不上）
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,   # ← 传主 agent 的派发引导 prompt
    )
    # 把 shared_state 注入主 agent run
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)

    # ── 新增：主 agent 收齐子结果后，走去重汇总 agent（含 save_report 工具）──
    subagents_data = shared_state["subagents"]
    if subagents_data:
        try:
            deduplicated = _deduplicate_summarize(subagents_data, question=question)
            # 如果去重结果有效且更长（不是失败兜底），优先使用
            if deduplicated and len(deduplicated) > 200:
                final_answer = deduplicated
            else:
                final_answer = result["final_answer"]
        except Exception as e:
            logger.warning(f"收尾去重汇总失败，沿用主 agent 原始输出: {e}")
            final_answer = result["final_answer"]
    else:
        final_answer = result["final_answer"]

    return {
        "final_answer": final_answer,
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "请技术调研 RAG（检索增强生成）技术：包含背景、原理、应用场景"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n去重汇总后的报告（前 500 字）:\n{r['final_answer'][:500]}")
