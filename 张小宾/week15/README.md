# 多 Agent 系统（Orchestrator-Worker）

基于 DeepSeek v4 的多 Agent 编排系统：1 个主 Agent 负责任务拆解与汇总，N 个子 Agent 并发执行子任务。

## 架构

```
用户任务 → MainAgent(拆解 → 并发分发 → 汇总) → 最终输出
                  ↓
            N 个 SubAgent 并发执行
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入真实的 DEEPSEEK_API_KEY
```

### 3. 运行

```bash
# 交互式输入
python main.py

# 或直接传任务
python main.py "用 Python 实现一个 LRU 缓存，并写一份使用说明"
```

## 目录结构

```
.
├── .env.example          # 配置模板
├── config.py             # 读取 .env，构建 LLM 客户端
├── schemas.py            # 结构化通信：SubTask / Plan / SubResult
├── main.py               # 入口
├── agents/
│   ├── base.py           # BaseAgent：统一 LLM 调用
│   ├── main_agent.py     # 主 Agent：plan → dispatch → aggregate
│   └── sub_agent.py      # 子 Agent：执行单个子任务
├── prompts/
│   ├── planner.py        # 任务拆解 prompt
│   ├── worker.py         # 子 Agent 角色 prompt
│   └── aggregator.py     # 结果汇总 prompt
└── utils/
    └── logger.py         # 日志
```

## 核心流程

1. **Plan**：主 Agent 调用 LLM，将用户任务拆解为 2~6 个子任务（JSON）。
2. **Dispatch**：用 `asyncio.gather` + `Semaphore` 并发执行子 Agent。
3. **Aggregate**：主 Agent 再次调用 LLM，汇总所有子结果。

## 配置项说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | - |
| `DEEPSEEK_BASE_URL` | 接口地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | 模型名 | `deepseek-chat` |
| `REQUEST_TIMEOUT` | 请求超时（秒） | `60` |
| `MAX_SUB_AGENTS` | 子 Agent 最大并发数 | `5` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

## 可扩展方向

- 给子 Agent 增加 tools（搜索、代码执行）—— DeepSeek 支持 function calling
- 主 Agent 根据子结果动态追加新子任务（迭代规划）
- 接入 streamlit/gradio 做可视化前端
- 增加 token 用量统计与成本控制
