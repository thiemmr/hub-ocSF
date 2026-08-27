# Ornith-1.5-35B-A3B 模型调研报告

> 调研日期：2026-08-20  
> 模型仓库：[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)  
> 本地资料范围：模型卡、网络配置、生成配置、聊天模板、tokenizer 配置、多模态处理配置、权重索引  
> 结论边界：本报告未下载 BF16 权重，也未实际执行推理或复跑基准；能力数据均标记为发布方自报值。

## 1. 执行摘要

Ornith-1.5-35B-A3B 是 Ornith 1.5 系列的中型模型，定位于代码生成、软件工程、工具调用和长程智能体任务。它不是一个纯文本模型：配置中包含完整视觉编码器及图片、视频输入协议，因此更准确的定义是**面向代码与智能体任务的多模态推理型 MoE 模型**。

该模型最值得关注的不是网络结构创新，而是发布方提出的训练方法：在强化学习阶段，不只优化最终解题轨迹，还让系统自动提出训练任务、生成任务专属脚手架（harness/scaffold），再产生解决方案 rollout，并分别为三个阶段设计奖励。这是 Ornith-1.0 “自生成脚手架”方法向“自生成训练课程”的扩展。

核心判断如下：

- **架构基础成熟**：采用 `Qwen3_5MoeForConditionalGeneration` 架构，40 层、256 专家、每 token 选 8 个专家，并结合 30 层线性注意力与 10 层全注意力。
- **参数效率较好，但存储并不小**：总参数约 35.952B，发布方称每 token 激活约 3B；完整 BF16 权重仍约 71.90 GB，必须装载所有专家权重。
- **代码/智能体能力是主线**：模型卡在 Terminal-Bench、SWE-bench、NL2Repo、MCP-Atlas、Toolathlon、ClawEval 等任务上给出了明显提升。
- **具有原生多模态结构**：包含 27 层视觉编码器，聊天模板和处理器同时支持图片与视频。
- **原生长上下文为 262,144 tokens**：发布方给出了利用 YaRN 扩展到约 1M tokens 的配置，但静态 RoPE 缩放可能损害普通长度请求质量。
- **“自我改进”发生在训练阶段**：没有证据表明已下载的静态检查点会在推理期间自主更新参数、产生训练数据或持续学习。
- **开放性不等于可复现**：仓库缺少训练代码、训练数据、任务生成器、reward/verifier 实现、训练超参数和完整评测脚本，无法从当前资料复现训练结论。
- **许可证存在资料缺口**：模型卡声明 MIT，但当前仓库文件树没有实际 `LICENSE` 文件，模型卡中的许可证链接返回 404。部署和再分发前应等待发布方补充正文或书面确认。

## 2. 资料来源与证据等级

### 2.1 本地文件

| 文件 | 主要信息 | 本报告用途 |
|---|---|---|
| [`README (1).md`](./README%20(1).md) | 模型定位、训练方法摘要、基准、部署建议 | 发布方自述与评测数据 |
| [`config.json`](./config.json) | 模型架构、MoE、注意力、视觉编码器、上下文 | 架构事实的主要证据 |
| [`generation_config.json`](./generation_config.json) | 默认采样和停止 token | 推理默认值 |
| [`chat_template.jinja`](./chat_template.jinja) | 消息、思考、图片/视频、工具调用格式 | 推理协议分析 |
| [`tokenizer_config.json`](./tokenizer_config.json) | 特殊 token、嵌入式聊天模板、最大长度 | tokenizer 和模板分析 |
| [`processor_config.json`](./processor_config.json) | 图片与视频预处理 | 多模态输入分析 |
| [`preprocessor_config.json`](./preprocessor_config.json) | 图片预处理兼容配置 | 图片输入分析 |
| [`model.safetensors.index.json`](./model.safetensors.index.json) | 1,811 个张量到 16 个分片的映射 | 权重结构与模块核验 |

### 2.2 外部一手资料

- [Ornith-1.5 官方博客：From Self-Scaffolding to Self-Improvement](https://ornith.ai/ornith_1_5.html)
- [Hugging Face 模型仓库](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/main)

### 2.3 证据标签

本报告使用以下措辞区分证据强度：

- **配置确认**：能由本地 JSON、Jinja 或权重索引直接验证。
- **发布方自述**：来自模型卡或官方博客，但尚未独立复测。
- **分析推断**：根据结构、张量名称或配置关系得出的解释，并明确写出推断性质。

## 3. 模型定位

### 3.1 目标任务

模型卡将 Ornith-1.5-35B-A3B 定位为：

- agentic coding；
- 仓库级软件工程；
- 长程问题求解；
- 工具调用；
- 通用推理；
- 图片和视频理解。

其重点不是低延迟闲聊，而是在带工具、测试环境和较长上下文的任务中完成多步工作。

### 3.2 模型家族

Ornith 1.5 同时发布三个主要规模：

| 版本 | 结构 | 典型定位 |
|---|---|---|
| Ornith-1.5-9B | 9B dense | 本地、边缘和移动设备 |
| Ornith-1.5-35B-A3B | 35B MoE，约 3B 激活 | 性能与部署成本折中 |
| Ornith-1.5-397B | 397B MoE | 集群级旗舰模型 |

当前报告只分析 35B-A3B 版本。

### 3.3 与 Qwen3.5 的关系

`config.json` 明确记录：

```json
{
  "architectures": ["Qwen3_5MoeForConditionalGeneration"],
  "model_type": "qwen3_5_moe"
}
```

因此可以确认其网络实现兼容 Qwen3.5-MoE 架构。需要注意：当前模型卡 front matter 没有 `base_model` 字段，也没有公开说明一个可精确核验的起始 checkpoint，因此只能确认**架构家族**，不能仅凭现有文件断言它具体从哪个 Qwen 权重版本继续训练。

## 4. 网络架构

### 4.1 总体数据流

```text
文本 ───────────────┐
                    │
图片/视频 → 视觉编码器 → 多模态 token → 40 层混合注意力 MoE 语言模型 → LM Head
                    │                                         └→ MTP 辅助预测分支
工具定义/历史消息 ──┘
```

### 4.2 主要结构参数

| 参数 | 值 | 证据 |
|---|---:|---|
| 精度 | BF16 | `dtype` |
| 总参数量 | 约 35,951,822,704 | 权重索引 `total_size / 2` 推算 |
| 发布方标称激活参数 | 约 3B/token | 模型卡自述 |
| 语言层数 | 40 | `num_hidden_layers` |
| 隐藏维度 | 2,048 | `hidden_size` |
| 词表大小 | 248,320 | `vocab_size` |
| 原生最大位置 | 262,144 | `max_position_embeddings` |
| 归一化 | RMSNorm，epsilon = 1e-6 | `rms_norm_eps` |
| 激活函数 | SiLU | `hidden_act` |
| Embedding/LM Head 共享 | 否 | `tie_word_embeddings=false` |
| MTP 层数 | 1 | `mtp_num_hidden_layers` |

### 4.3 混合注意力

40 层按固定节奏排列：

```text
Linear → Linear → Linear → Full Attention
```

该四层模式重复 10 次，因此共有：

- 30 层线性注意力；
- 10 层全注意力；
- `full_attention_interval = 4`。

这种混合结构的目标通常是在保留周期性全局交互能力的同时，降低长序列计算成本。这里的“降低成本”是结构目的分析，不代表本报告已测得实际吞吐提升。

#### 线性注意力参数

| 参数 | 值 |
|---|---:|
| Key heads | 16 |
| Key head dim | 128 |
| Value heads | 32 |
| Value head dim | 128 |
| 卷积核维度 | 4 |
| 状态计算 dtype | FP32 (`mamba_ssm_dtype`) |

权重索引中可见 `A_log`、`dt_bias`、`conv1d`、`in_proj_*` 和 `out_proj` 等线性注意力张量。虽然配置字段包含 `mamba_ssm_dtype`，但模型类型仍是 Qwen3.5 线性注意力；不能据此简单把它称为标准 Mamba 模型。

#### 全注意力参数

| 参数 | 值 |
|---|---:|
| Query heads | 16 |
| KV heads | 2 |
| Head dim | 256 |
| RoPE 比例 | 0.25 |
| RoPE theta | 10,000,000 |

`16 Q heads / 2 KV heads` 表明全注意力使用 GQA，以减少 KV cache 体积。

### 4.4 MoE 前馈层

| 参数 | 值 |
|---|---:|
| 专家总数 | 256 |
| 每 token 选中专家 | 8 |
| 路由稀疏率 | 8/256 = 3.125% |
| 单专家中间维度 | 512 |
| 共享专家中间维度 | 512 |
| Router auxiliary loss coefficient | 0.0（当前推理配置） |

每个语言层都包含：

- 路由器 `mlp.gate.weight`；
- 256 个路由专家；
- 一个共享专家；
- 共享专家 gate。

在主语言层中，256 个专家权重被打包为两个融合张量：

```text
mlp.experts.gate_up_proj
mlp.experts.down_proj
```

权重索引中正好出现 80 个此类融合专家张量，即 `40 层 × 2`。这验证了 40 层均配置 MoE，而不是只有部分层使用专家。

发布方的“约 3B 激活参数”是整体口径，包含共享路径、注意力、选中专家及可能的视觉/MTP 组件。当前配置不足以精确复算每种输入下的动态激活量，因此本报告保留发布方口径，不将其当作独立测量值。

### 4.5 MTP 辅助分支

配置包含一个 MTP（Multi-Token Prediction）隐藏层。权重索引中出现独立的：

- `mtp.fc`；
- `mtp.norm`；
- `mtp.layers.0.self_attn`；
- `mtp.layers.0.mlp`；
- 256 个 MTP 专家；
- embedding/hidden 预归一化层。

MTP 分支共对应约 785 个索引张量。它说明 checkpoint 保留了额外预测头，可供兼容运行时利用多 token 预测或推测解码能力；是否实际提速取决于 vLLM、SGLang、llama.cpp 等运行时是否启用并正确支持，不能仅凭权重存在推断默认生效。

## 5. 多模态能力

### 5.1 视觉编码器

`vision_config` 给出的结构如下：

| 参数 | 值 |
|---|---:|
| 层数 | 27 |
| 隐藏维度 | 1,152 |
| 输出维度 | 2,048 |
| 注意力头 | 16 |
| MLP 中间维度 | 4,304 |
| Patch size | 16 |
| Spatial merge | 2 |
| Temporal patch | 2 |
| 激活函数 | GELU tanh approximation |

权重索引中有 333 个 `model.visual.*` 张量，包括 patch embedding、position embedding、27 个视觉 block 和投影模块，因此视觉编码器并非空配置。

### 5.2 图片预处理

- RGB 转换；
- 缩放、重采样、归一化；
- 均值和标准差均为 `[0.5, 0.5, 0.5]`；
- rescale factor 为 `1/255`；
- patch size 16；
- temporal patch size 2；
- merge size 2；
- processor 为 `Qwen3VLProcessor`。

### 5.3 视频预处理

`processor_config.json` 还记录：

- 默认采样 2 FPS；
- 最少 4 帧；
- 最多 768 帧；
- 视频 processor 为 `Qwen3VLVideoProcessor`。

当前本地目录没有单独的 `video_preprocessor_config.json`，但其核心参数已包含在 `processor_config.json` 内，因此不影响配置层面的分析。

### 5.4 音频能力边界

`tokenizer_config.json` 中存在 `<|audio_start|>`、`<|audio_end|>`、`<|audio_pad|>` 和 TTS token。这些很可能来自共享 tokenizer 词表。模型配置没有音频编码器、音频 processor 或音频输入路径，因此**不能因为存在音频 token 就认定该 checkpoint 支持音频理解或语音生成**。

## 6. Tokenizer 与对话协议

### 6.1 Tokenizer

| 项目 | 值 |
|---|---|
| tokenizer class | `Qwen2Tokenizer` |
| 最大长度 | 262,144 |
| EOS | `<|im_end|>`，ID 248046 |
| PAD/BOS fallback | `<|endoftext|>`，ID 248044 |
| 已登记新增 token | 33 |

模型还包含以下任务相关 token：

- 图片/视频：`<|vision_start|>`、`<|image_pad|>`、`<|video_pad|>`；
- 工具：`<tool_call>`、`<tool_response>`；
- 推理：`<think>`、`</think>`；
- FIM 代码补全：`<|fim_prefix|>`、`<|fim_middle|>`、`<|fim_suffix|>`；
- 仓库级代码：`<|repo_name|>`、`<|file_sep|>`；
- 视觉定位：object、box、quad 起止 token。

### 6.2 消息模板

聊天模板执行以下协议：

1. system 消息必须位于最前面；
2. system 消息禁止包含图片或视频；
3. 图片转换为 `<|vision_start|><|image_pad|><|vision_end|>`；
4. 视频转换为 `<|vision_start|><|video_pad|><|vision_end|>`；
5. 工具定义被注入 `<tools>...</tools>`；
6. 工具调用要求使用 XML 风格的 `<tool_call><function=...>` 格式；
7. 工具结果包装为 `<tool_response>...</tool_response>`；
8. assistant 回合支持把思考内容与最终答案拆开；
9. `enable_thinking=false` 时，生成提示插入空的 `<think></think>`，从而要求直接作答。

### 6.3 两份聊天模板不一致

本地同时存在：

- 独立文件 `chat_template.jinja`；
- `tokenizer_config.json` 内嵌的 `chat_template`。

两者**并不完全相同**：

- 独立模板会在历史 assistant 消息中始终重新插入 `<think>...</think>`；
- 内嵌模板只有在 `preserve_thinking=true`，或消息位于最新用户查询之后时，才保留思考内容；否则只保留最终内容。

这会影响：

- 长对话 token 消耗；
- KV cache 命中；
- 历史推理内容是否继续暴露给后续轮次；
- 不同运行时之间的输出一致性。

部署时必须实际检查运行时加载的是哪一份模板。例如在 Transformers 中打印：

```python
print(tokenizer.get_chat_template())
```

并用同一段多轮消息比较最终序列。生产环境不应默认假设两份模板等价。

## 7. 参数量、权重与存储

### 7.1 权重索引统计

| 项目 | 值 |
|---|---:|
| BF16 参数字节数（索引 metadata） | 71,903,645,408 bytes |
| 推算参数量 | 35,951,822,704 |
| 权重分片 | 16 |
| 索引张量 | 1,811 |
| 视觉张量 | 333 |
| 语言层非融合专家张量 | 610 |
| 语言层融合专家张量 | 80 |
| MTP 分支张量 | 约 785 |

参数量推算依据是 BF16 每参数 2 bytes。由于 safetensors 分片还带有文件头，实际下载大小会略高于索引 metadata 的纯张量字节数。

### 7.2 MoE 的部署含义

“A3B”不能理解为只需装载一个 3B 模型：

- 推理每个 token 只调用部分专家，因此 FLOPs 接近较小 dense 模型；
- 但不同 token 会选择不同专家，全部 35B 权重仍需要存储并可访问；
- BF16 部署还需要额外预留 KV cache、激活、CUDA graph、运行时缓冲区和视觉输入显存；
- 上下文越长，KV cache 与工作区开销越明显。

发布方为原生 256K 上下文建议 2×80GB GPU，这一建议体现的是“权重 + 超长上下文余量”，而不只是权重能否勉强放入显存。

## 8. 自我改进训练方法

### 8.1 概念边界

Ornith-1.5 的“self-improvement”是训练系统设计：

```text
提出任务 → 生成/改进脚手架 → 产生解决方案 rollout → 验证与奖励 → GRPO 更新
   ↑                                                        │
   └───────────────── 根据能力边界继续生成更难任务 ─────────┘
```

已发布 checkpoint 是训练结束后的静态权重。没有本地证据表明它能在普通推理服务中自行执行上述闭环。

### 8.2 三阶段联合优化

官方博客将每轮训练描述为：

1. **Task generation**：根据环境/代码库、任务类型和既往解决历史，提出比已解决任务更困难的新任务；
2. **Scaffold generation**：为任务生成或改进专属指令、工具、分解策略和编排；
3. **Solution rollout**：在任务和脚手架条件下生成解决轨迹。

三个阶段的奖励均用于 GRPO 更新，因此系统尝试同时学习“练什么”“怎样组织求解”和“怎样真正解决”。

### 8.3 任务奖励

官方定义：

$$
R_{task}=V(q,s)\times D(q,s,\{\tau_i\})\times N(q)
$$

其中：

- $V$：任务与脚手架是否有效、可运行、可验证；
- $D$：任务是否处于当前能力边界；
- $N$：相对历史任务是否足够新颖。

有效性可作为硬门：若任务或验证环境无效，则奖励直接归零。

### 8.4 Frontier difficulty

系统从 $N$ 次 rollout 的经验成功率估计难度：

$$
p=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[s(q,\tau_i)=success]
$$

然后奖励接近目标成功率 $p^*$ 的任务：

$$
D=\exp\left(-\frac{(p-p^*)^2}{2\sigma^2}\right)
$$

官方博客给出 $p^*=0.2$。也就是说，系统倾向于选择当前模型大约只有 20% 成功率、但仍有成功轨迹可供学习的任务。随着模型改善，原有任务成功率上升、奖励下降，课程生成器会转向更难任务。

### 8.5 新颖性

$$
N(q)=1-\max_{q_j\in\mathcal{B}}sim(q,q_j)
$$

其中 $\mathcal{B}$ 是既往生成或训练任务的缓冲区。其目的是减少围绕同一题型反复生成轻微变体。

### 8.6 Harness 与 rollout 奖励

官方将脚手架奖励写为：

$$
R_{harness}=C(q,h)\times F(h,\{\tau_i\})\times H(h)
$$

- $C$：与任务规格是否一致；
- $F$：reward 是否忠实反映候选方案质量；
- $H$：是否能抵抗 shortcut、评测器漏洞和 reward hacking。

rollout 则由生成的 harness 直接打分：

$$
R_{rollout}(\tau_i)=h(q,\tau_i)
$$

这一方法的关键风险也很明显：如果任务生成器、harness 和解题模型共享盲区，系统可能产生“可被自己验证但不真正有用”的任务。官方引入有效性、reward fidelity 和 hack resistance 来缓解，但仓库没有公开具体 judge、测试生成器和反作弊实现，无法独立审计这些机制的充分性。

## 9. 推理行为与运行配置

### 9.1 默认生成参数

`generation_config.json`：

```json
{
  "do_sample": true,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95
}
```

模型卡另建议：

- 一般任务：`temperature=0.6, top_p=0.95, top_k=20`；
- 复现官方基准：`temperature=1.0`。

因此 JSON 默认值偏向评测复现，不一定是普通生产任务的最稳配置。

### 9.2 思考模式

默认 assistant 生成以 `<think>` 开始。兼容服务端可将：

- 思考内容映射到 `reasoning_content`；
- 最终回答映射到 `content`；
- `<tool_call>` 映射到 OpenAI 风格 `tool_calls`。

如果运行时没有正确启用 reasoning/tool parser，原始 XML 标签可能直接出现在文本输出中。

### 9.3 官方运行时最低版本

根据模型卡：

- Transformers ≥ 5.8.1；
- vLLM ≥ 0.19.1；
- SGLang ≥ 0.5.9。

这些版本要求来自发布方，实际部署时应使用与 checkpoint 发布时间匹配的版本并锁定依赖。

### 9.4 长上下文

原生窗口为 262,144 tokens。发布方给出的 YaRN 关系近似为：

```text
扩展窗口 ≈ factor × 262,144
```

例如：

- factor 2：约 524K；
- factor 4：约 1M。

官方同时警告开源运行时通常静态应用 YaRN，同一缩放因子会用于短请求，可能轻微损害普通长度质量。因此应按真实工作负载选择最小必要 factor，而不是默认开启 1M。

## 10. 官方评测结果

下表只列 Ornith-1.5-35B-A3B、上一代 Ornith-1.0 和同规模 Qwen3.6。数值均来自模型卡，未由本报告复跑。

| 类别 | Benchmark | Ornith 1.5 | Ornith 1.0 | Qwen3.6-35B-A3B |
|---|---|---:|---:|---:|
| Coding | Terminal-Bench 2.1 (Terminus-2) | 67.8 | 64.2 | 52.5 |
| Coding | Terminal-Bench 2.1 (Claude Code) | 68.5 | 62.8 | 49.2 |
| Coding | SWE-bench Verified | 79.0 | 75.6 | 73.4 |
| Coding | SWE-bench Pro | 59.6 | 50.4 | 49.5 |
| Coding | SWE-bench Multilingual | 71.4 | 69.3 | 67.2 |
| Coding | DeepSWE | 22.0 | 0.0 | 0.0 |
| Coding | Frontier-Bench v0.1 | 5.1 | 1.4 | 1.4 |
| Coding | NL2Repo | 46.2 | 34.6 | 29.4 |
| Coding | SWE Atlas QnA | 39.8 | 37.1 | 15.5 |
| Reasoning | HLE（无工具） | 25.6 | 20.8 | 21.4 |
| Reasoning | HLE（有工具） | 33.4 | 30.1 | 28.9 |
| Reasoning | GPQA Diamond | 89.2 | 86.2 | 86.0 |
| Agentic | MCP-Atlas | 70.2 | 64.4 | 62.8 |
| Agentic | Toolathlon-Verified | 48.7 | 42.4 | 41.7 |
| Agentic | WideSearch | 67.8 | 63.4 | 60.1 |
| Agentic | BrowseComp | 67.6 | 63.5 | 62.0 |
| Agentic | ClawEval | 72.5 | 69.8 | 68.7 |

### 10.1 结果解读

对上一代 Ornith-1.0，较显著的提升包括：

- NL2Repo：+11.6；
- SWE-bench Pro：+9.2；
- MCP-Atlas：+5.8；
- Terminal-Bench（Claude Code）：+5.7；
- HLE 无工具：+4.8。

这与发布方“自生成更难任务”的叙事方向一致：提升集中在需要较长轨迹、工具、仓库操作或复杂执行环境的任务上。但“方向一致”不等于证明因果；没有消融实验时，无法区分改进来自任务生成、额外训练算力、数据变化、脚手架变化还是评测适配。

### 10.2 评测可信度注意事项

- 发布方称 Ornith-1.5 结果取五次独立运行平均值，这是积极因素；
- 多个任务采用不同 harness、上下文、温度和最大输出长度，分数不能跨 benchmark 横向比较；
- Terminal-Bench 使用经过 Qwen 模板适配的 harness；
- HLE 使用 Claude 4.6 Opus 作为 judge；
- MCP-Atlas 使用 Claude 4.8 Opus 作为 judge；
- SWE-bench 声称移除 Git 历史并关闭网络以降低泄漏和 reward hacking；
- NL2Repo 屏蔽指定仓库和 pip 包；
- 仓库没有提供可直接复跑这些数字的统一评测脚本、镜像摘要、任务版本和原始轨迹。

因此这些数据适合用于确定“值得进一步测试”，不宜直接作为采购或生产替换决策的唯一依据。

## 11. 适用场景

### 11.1 较适合

- 本地或私有化代码智能体；
- 仓库级问题定位、修改和测试；
- 带函数调用的工作流；
- 较长的代码库问答；
- 多步浏览、检索和工具编排；
- 图片/视频与代码、文档联合理解；
- 研究 MoE、混合注意力和训练期 self-improvement。

### 11.2 谨慎使用

- 对许可证完备性有严格要求的商业分发；
- 只有单张中低显存 GPU、但又要求 BF16 和超长上下文的环境；
- 强依赖稳定短回答、低延迟和确定性输出的业务；
- 医疗、法律、金融等高风险决策；
- 将公开 benchmark 分数直接当成真实生产通过率；
- 需要在线持续学习或运行时自我更新的项目。

## 12. 已知问题与资料缺口

### 12.1 仓库层面

- 模型卡声明 MIT，但仓库没有 `LICENSE` 文件；
- `license_link` 指向的 URL 当前返回 404；
- 本地 README 引用 `assets/ornith_logo.png` 和 `assets/ornith_35b_eval.png`，但当前目录未下载 assets，离线预览会显示缺图；
- 本地未下载 `configuration.json`，但该文件上游只包含框架和任务标签，研究价值较低；
- 本地未下载独立 `video_preprocessor_config.json`，但视频核心配置已在 `processor_config.json` 中；
- 未下载 `tokenizer.json`、`vocab.json` 和 `merges.txt`，因此不能离线执行真实分词或分析 token fertility；
- 未下载 16 个 safetensors 权重分片，因此不能加载模型、验证输出或检查张量数值。

### 12.2 训练复现

当前未公开：

- 预训练/后训练数据清单；
- 训练任务和自动生成任务样本；
- task proposer、harness generator 的提示和实现；
- verifier、judge、hack detector；
- GRPO 详细超参数；
- 训练步数、batch、学习率和算力；
- checkpoint 演化轨迹；
- 完整消融实验；
- 原始评测 rollout 与失败样例。

所以该发布更接近“开放权重 + 方法说明”，而不是完整可复现研究发布。

### 12.3 安全风险

模型专门强化工具调用和自主代码执行，这会提高以下风险：

- prompt injection 经工具链放大；
- shell、文件系统或网络权限误用；
- 测试环境与真实生产环境不一致；
- 长程任务中累积错误；
- 模型利用 verifier/harness 漏洞获取虚假成功；
- 保留历史 `<think>` 内容导致敏感信息继续进入后续上下文。

部署时应将模型置于最小权限沙箱，隔离秘密、生产凭据和不可恢复的数据操作，并将测试通过视为必要而非充分条件。

## 13. 建议的独立验证计划

### 第一阶段：静态核验

1. 补齐 tokenizer 文件和许可证；
2. 固定仓库 revision 与本地文件 SHA256；
3. 确认运行时实际采用的聊天模板；
4. 核对量化版本与 BF16 原版的 base revision；
5. 检查 vLLM/SGLang 对 MTP、视觉和工具解析的支持矩阵。

### 第二阶段：最小推理测试

1. 纯文本问答；
2. thinking on/off；
3. 单工具、多工具、工具报错恢复；
4. 图片理解；
5. 视频输入；
6. FIM 代码补全；
7. 32K、128K、256K 上下文稳定性；
8. 多轮历史思考保留行为。

### 第三阶段：业务基准

建议至少建立：

- 真实仓库 issue 修复集；
- 单元测试不可见的防过拟合设置；
- 无网络和有网络两套环境；
- 工具调用正确率、参数正确率和恢复率；
- 完成率、成本、首 token 延迟、总时长、峰值显存；
- 至少三个随机种子；
- BF16 与目标量化版本的差异测试；
- 人工审查失败样例，而不只报告平均分。

## 14. 本地文件完整性记录

| 文件 | Bytes | SHA256 |
|---|---:|---|
| `README (1).md` | 29,747 | `072CCF0EFBF7C0F9071641D839362F4B0EE134389C2E7B692F44F59E9F8871FA` |
| `chat_template.jinja` | 7,536 | `182E77DD83BD8E9CA818B240B82E28F243762CD5DDA32E6EEF327DF7B1CD107E` |
| `config.json` | 3,295 | `AF198CCBE6C17DD973DA98BBF6C92CE1C3EC7261C2E2FBC4C4F5150018418EA8` |
| `generation_config.json` | 202 | `E70C136C1B78DDC1FB0905BAC8E733A4DC448D4F852A5DD75143FFFC70BE550E` |
| `model.safetensors.index.json` | 165,731 | `0AD9FBFBB1A7514773ED881D3C9403019BD7AE9FF4C6AF3CFC65CA7DA784D78F` |
| `preprocessor_config.json` | 390 | `27225450AC9C6529872EE1924FCB0962FF5634834F817040F444118116F4E516` |
| `processor_config.json` | 1,191 | `D89EF49CE9CD37FB510158E13C1EF063D9286411C1EC9049932DBE0487143B1` |
| `tokenizer_config.json` | 16,718 | `5186F0DEFCD7F232382C7F0AEBCD2252D073BB921AB240E407B7AE8745D2B29B` |

## 15. 最终评价

Ornith-1.5-35B-A3B 的实际价值在于：它把一个可部署的 35B MoE 多模态模型，针对代码智能体、工具调用和复杂长轨迹进行了强后训练。约 3B 激活参数使它在理论计算量上具有吸引力，模型卡中的软件工程和智能体成绩也足以支持进一步测试。

但应避免两个误读：

1. **它不是只有 3B 大小的模型**：完整 BF16 权重仍约 72 GB，超长上下文部署成本很高；
2. **它不是运行时递归自我改进系统**：公开的是经过 self-improvement 训练流程产生的静态 checkpoint，而不是完整训练闭环。

综合判断：

- 作为**本地/私有代码智能体候选模型**：值得进入独立评测；
- 作为**研究自生成任务与脚手架训练的案例**：方法有启发性，但材料不足以复现；
- 作为**直接生产替代方案**：需先解决许可证、模板一致性、量化质量、真实业务成功率和沙箱安全问题；
- 作为**通用多模态模型**：结构上具备能力，但发布材料的主要证据仍集中在代码、推理与智能体任务，视觉能力需要单独验证。

