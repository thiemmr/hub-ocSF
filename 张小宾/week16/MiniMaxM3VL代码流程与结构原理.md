# MiniMax M3 VL 代码流程与结构原理教学（进阶版）

> **前置知识**：建议先掌握 Transformer 基础（Self-Attention、FFN、LayerNorm）、MoE 基本概念、GQA、RoPE。
> **配套源码**：
> - `src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py`（1548 行）
> - `src/transformers/models/minimax_m3_vl/configuration_minimax_m3_vl.py`（226 行）
> - `src/transformers/models/minimax_m3_vl/processing_minimax_m3_vl.py`
> - `src/transformers/models/minimax_m3_vl/image_processing_minimax_m3_vl.py`
> - `src/transformers/models/minimax_m3_vl/video_processing_minimax_m3_vl.py`

---

## 第 0 章 本讲目标与阅读地图

本讲不是泛泛的架构科普，而是**跟着代码走**：每一处结论都能在源码里找到对应行。读完你将能回答：

1. MiniMax M3 VL 的三份配置各管什么？每个关键数字为什么是它？
2. 一张图片 / 一段视频从"像素"到"logits"完整经过哪些环节？
3. Lightning Indexer（块级稀疏注意力）到底在代码里怎么实现？
4. 128 专家的 MoE 为什么用 Sigmoid 路由而不是 Softmax？
5. 视频和图像为什么能共用同一个视觉塔？

```
┌──────────────────────────────────────────────────────────────────┐
│                    第 1 章  模型全景图（一图流）                    │
├──────────────────────────────────────────────────────────────────┤
│                    第 2 章  配置层：设计蓝图                        │
│    TextConfig / VisionConfig / 顶层 Config 逐字段精讲 + 动机       │
├──────────────────────────────────────────────────────────────────┤
│                    第 3 章  代码流程：端到端数据流                   │
│    预处理 → 视觉特征 → 多模态融合 → 语言模型 → 输出                │
├──────────────────────────────────────────────────────────────────┤
│                    第 4 章  结构拆解：逐组件走代码                   │
│    视觉塔 / 投影器 / 文本主干 / DecoderLayer / Attention           │
│    / Lightning Indexer / MoE / Dense MLP                          │
├──────────────────────────────────────────────────────────────────┤
│                    第 5 章  机制原理深挖                            │
│    Sigmoid 路由 / 块稀疏 / GQA+部分RoPE / 共享视觉塔 / 稀疏缓存    │
├──────────────────────────────────────────────────────────────────┤
│                    第 6~8 章  速查表 / 误区 / 延伸                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 第 1 章 模型全景图

MiniMax M3 VL 是 MiniMax 出品的**多模态稀疏大模型**：它把"视觉塔 + 投影器 + 一个 60 层、混合 MoE/稠密、混合稀疏/全量注意力的文本大模型"拼成一个三明治。一句话类比：

> **一家能看图说话的公司**：前台（视觉塔）把图片切成小块看懂 → 翻译员（投影器）把视觉特征翻译成文字世界的语言 → 编辑部（文本模型）负责全文理解与续写。这个编辑部很特殊：**雇了 128 位专家，每件事只请 4 位来干**，还练就了"速读"（块级稀疏注意力）处理超长文本。

```
MiniMaxM3SparseForConditionalGeneration   ← 多模态成品入口（图文/视频→文字）
  └─ MiniMaxM3VLModel                     ← 骨架：三明治拼装
       ├─ MiniMaxM3VLVisionModel          ← 视觉塔（32 层 CLIP 风格 + Conv3d + 3D RoPE）
       ├─ MiniMaxM3VLMultiModalProjector  ← 翻译员（两级 GELU MLP + 空间合并）
       └─ MiniMaxM3VLTextModel            ← 语言模型（60 层）
            ├─ embed_tokens               ← 词嵌入 [vocab=200064, 6144]
            ├─ MiniMaxM3VLDecoderLayer ×60
            │    ├─ self_attn             ← 每层按 layer_types 决定是否挂 Indexer
            │    └─ mlp                   ← 每层按 mlp_layer_types 决定 Dense 或 MoE
            └─ norm (RMSNorm)
  └─ lm_head                              ← [6144 → 200064] 输出 logits
```

**参数规模速览**（来自配置默认值）：

| 模块 | 关键数字 |
|---|---|
| 词表 | 200,064（含多模态特殊 token） |
| 文本层数 / 隐藏维 | 60 层 / 6144 |
| 注意力头 | 64 查询头 / 4 KV 头（GQA 16 组）/ head_dim 128 |
| 上下文长度 | 524,288 |
| MoE | 128 路由专家 + 1 共享专家，每 token 激活 top-4 |
| 稀疏注意力 | Indexer 4 头、块大小 128、每 query 选 top-16 块 + 前 1 块 |
| 视觉塔 | 32 层 / 1280 维 / patch 14×14 / 时间 patch 2 / 空间合并 2 |

---

## 第 2 章 配置层：设计蓝图

配置文件里定义了三个 `PreTrainedConfig` 子类，形成**两级嵌套**：

```
MiniMaxM3VLConfig（顶层，model_type="minimax_m3_vl"）
 ├── vision_config: MiniMaxM3VLVisionConfig   （model_type="minimax_m3_vl_vision"）
 └── text_config:   MiniMaxM3VLTextConfig     （model_type="minimax_m3_vl_text"）
```

顶层 `__post_init__` 里做了三件关键事（configuration 文件 202-222 行）：

```python
def __post_init__(self, **kwargs):
    if isinstance(self.vision_config, dict):
        self.vision_config = MiniMaxM3VLVisionConfig(**self.vision_config)   # dict → 配置对象
    ...
    # 投影器"空间合并后"的通道维 = 文本 hidden_size × merge_size²
    self.merged_hidden_size = self.text_config.hidden_size * (self.vision_config.spatial_merge_size**2)
```

`merged_hidden_size = 6144 × 2² = 24576`，这是投影器第二阶段 MLP 的输入宽度。

### 2.1 MiniMaxM3VLTextConfig —— 文本大模型的设计蓝图

| 字段 | 默认值 | 一句话含义 |
|---|---|---|
| `vocab_size` | 200064 | 词表大小（含图像/视频占位符、多语言） |
| `hidden_size` | 6144 | 隐藏维 |
| `intermediate_size` | 3072 | **单个 MoE 专家**的中间维 |
| `dense_intermediate_size` | 12288 | 稠密层的中间维（是专家维的 4 倍） |
| `shared_intermediate_size` | 3072 | 共享专家的中间维 |
| `num_hidden_layers` | 60 | 解码层数 |
| `num_attention_heads` | 64 | 查询头数 |
| `num_key_value_heads` | 4 | KV 头数 → GQA 压缩 64/4 = 16 倍 |
| `head_dim` | 128 | 每个头的维度 |
| `max_position_embeddings` | 524288 | 最大上下文（512K） |
| `num_experts_per_tok` | 4 | 每 token 激活专家数（top-4） |
| `num_local_experts` | 128 | 路由专家总数 |
| `routed_scaling_factor` | 2.0 | 路由专家输出放大系数 |
| `router_aux_loss_coef` | 0.001 | MoE 负载均衡损失权重 |
| `rotary_dim` | 64 | 部分 RoPE：head_dim=128 只旋转前 64 维 |
| `swiglu_alpha` | 1.702 | SwiGLU-OAI 的 sigmoid 增益 |
| `swiglu_limit` | 7.0 | 门控/升维投影的 clamp 边界 |
| `index_n_heads` | 4 | Lightning Indexer 打分分支头数（= KV 头数） |
| `index_head_dim` | 128 | Indexer 每头通道维 |
| `index_block_size` | 128 | 每块包含的 key 数 |
| `index_topk_blocks` | 16 | 每 query 保留的 top 块数 |
| `index_local_blocks` | 1 | 前向紧邻的"必定可见"块数 |
| `layer_types` | 由 sparse_attention_freq 推导 | 逐层注意力类型（"minimax_m3_sparse"/"full_attention"） |
| `mlp_layer_types` | 由 moe_layer_freq 推导 | 逐层 MLP 类型（"sparse"/"dense"） |
| `rope_parameters.rope_theta` | 5,000,000 | RoPE 基频（支持长上下文） |

**两个"逐层列表"是混合架构的关键**（`__post_init__` 140-154 行）：

```python
# 没有显式给出时：稀疏注意力频率为 True 的层用 sparse，否则 full_attention
self.layer_types = [
    "minimax_m3_sparse" if f else "full_attention" for f in sparse_cfg["sparse_attention_freq"]
]
# 同理：MoE 频率为 True 的层用 sparse(MoE)，否则 dense
self.mlp_layer_types = ["sparse" if f else "dense" for f in moe_layer_freq]
```

> **教学点**：M3 不是"每一层都一样"的均匀模型——`layer_types[layer_idx]` 决定第 i 层的注意力是否挂 Indexer，`mlp_layer_types[layer_idx]` 决定第 i 层用 MoE 还是稠密 MLP。读代码时永远要带"逐层"的视角。

### 2.2 MiniMaxM3VLVisionConfig —— 视觉塔的设计蓝图

| 字段 | 默认值 | 含义 |
|---|---|---|
| `hidden_size` | 1280 | patch 嵌入维 |
| `intermediate_size` | 5120 | Vision MLP 中间维 |
| `num_hidden_layers` | 32 | 视觉编码器层数 |
| `num_attention_heads` | 16 | 注意力头数（head_dim = 1280/16 = 80） |
| `num_channels` | 3 | RGB 通道 |
| `image_size` | 2016 | 设计分辨率（预处理可动态缩放） |
| `patch_size` | 14 | 空间 patch 边长 |
| `temporal_patch_size` | 2 | 时间 patch（2 帧合成一个时间块） |
| `spatial_merge_size` | 2 | 空间合并：2×2 相邻 patch 合并成 1 个 token |
| `rope_parameters.rope_theta` | 10000 | 3D RoPE 基频 |

> **教学点**：`spatial_merge_size=2` 意味着**每 4 个视觉 patch 在投影器里合并成 1 个文本 token**——这是控制视觉 token 数量的关键旋钮。

### 2.3 MiniMaxM3VLConfig —— 顶层接线

| 字段 | 默认值 | 含义 |
|---|---|---|
| `image_token_index` | 200025 | 文本里图像占位符的 token id |
| `video_token_index` | 200026 | 文本里视频占位符的 token id |
| `projector_hidden_size` | 6144 | 投影器内部隐藏维（= 文本 hidden_size） |
| `merged_hidden_size` | 24576 | 空间合并后通道维（自动计算） |
| `tie_word_embeddings` | False | 不绑定 embedding 与 lm_head |

---

## 第 3 章 代码流程：端到端数据流

### 3.1 全景流程图

```
                        ┌────────────── 输入 ──────────────┐
                        │ 文本 prompt + 图片 + 视频          │
                        └──────────────┬──────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ 阶段一：预处理（Processor，processing / image / video 三个文件）   │
 │                                                                 │
 │ 文本:  把 "]<]image[>[" 替换成 N 个占位符 + 起止标记              │
 │        replace_image_token / replace_video_token                │
 │ 图片:  smart_resize → rescale/normalize → patchify               │
 │        → pixel_values [seq, 3·2·14·14] + image_grid_thw [1,h,w]  │
 │ 视频:  smart_resize → rescale/normalize → 帧数补齐 → patchify    │
 │        → pixel_values_videos + video_grid_thw [t,h,w]            │
 └─────────────────────────────────────────────────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ 阶段二：视觉特征提取（MiniMaxM3VLModel.get_image_features /      │
 │                        get_video_features）                      │
 │                                                                 │
 │ 视觉塔（共享！）：Conv3d patch embed → LayerNorm → 3D RoPE       │
 │                 → 32 层 VisionEncoderLayer                       │
 │ 投影器：linear→GELU→linear  (1280→6144)                          │
 │        → reshape 合并 2×2 patch → linear→GELU→linear (24576→6144)│
 └─────────────────────────────────────────────────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ 阶段三：多模态融合（MiniMaxM3VLModel.forward）                    │
 │                                                                 │
 │ 1. inputs_embeds = embed_tokens(input_ids)   [B, L, 6144]       │
 │ 2. get_placeholder_mask：找 image/video 占位符位置                │
 │ 3. inputs_embeds.masked_scatter(占位符, 视觉特征) ← 关键替换      │
 └─────────────────────────────────────────────────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ 阶段四：语言模型（MiniMaxM3VLTextModel.forward）                  │
 │                                                                 │
 │ DynamicCache(config) → create_causal_mask → rotary_emb          │
 │ → 60 × DecoderLayer（稀疏注意力/MoE 按层路由）→ RMSNorm           │
 └─────────────────────────────────────────────────────────────────┘
                                       ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ 阶段五：输出（MiniMaxM3SparseForConditionalGeneration.forward）   │
 │                                                                 │
 │ hidden_states → lm_head → logits [B, L, 200064]                 │
 │ labels 存在时 → cross entropy loss                               │
 └─────────────────────────────────────────────────────────────────┘
```

### 3.2 逐阶段代码走读

**阶段三的核心——占位符替换**（`MiniMaxM3VLModel.forward`，1368-1422 行）：

```python
inputs_embeds = self.get_input_embeddings()(input_ids)      # ① 文本→embedding

if pixel_values is not None:
    image_features = self.get_image_features(...).pooler_output   # ② 图片→视觉特征
if pixel_values_videos is not None:
    video_features = self.get_video_features(...).pooler_output   # ③ 视频→视觉特征

image_mask, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, ...)
if image_features is not None:
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)  # ④ 替换
if video_features is not None:
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)  # ⑤ 替换

outputs = self.language_model(attention_mask=..., inputs_embeds=inputs_embeds, ...)  # ⑥ 进语言模型
```

**教学点**：多模态融合在这里就是一行 `masked_scatter`——把文本 embedding 序列里"占位符位置"的向量，原样替换成视觉塔输出的特征向量。替换后，文本模型完全不知道这些 token 来自图片还是文字，统一按序列处理。这是 LLaVA 系架构的通用做法。

**`get_placeholder_mask` 的校验**（1325-1364 行）：它还会检查"占位符数量 == 视觉特征条数"，不匹配直接报错（`torch_compilable_check`），防止预处理与模型之间 token 数不一致。

---

## 第 4 章 结构拆解：逐组件走代码

### 4.1 视觉塔 MiniMaxM3VLVisionModel（1173-1211 行）

**Patch Embedding 用 3D 卷积**（`MiniMaxM3VLVisionEmbeddings`，982-1005 行）：

```python
kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]  # [2, 14, 14]
self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size)

def forward(self, hidden_states):
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )   # 输入被看成 [N, 3, 2, 14, 14] 的"视频块"
    hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)   # → [N, 1280]
```

**结构原理**：为什么用 Conv3d 而不是 Conv2d？因为 M3 的视觉塔要同时吃**图像和视频**。图像预处理时被"复制 2 帧"，视频按真实时间分成 `temporal_patch_size=2` 帧一组——两者都能被 `[2,14,14]` 的 3D 卷积核统一切成 patch。**一个卷积同时服务两种模态**，这就是"塔是 grid-agnostic（不关心网格维度）"的体现。

**3D RoPE**（`MiniMaxM3VL3DRotaryEmbedding`，1008-1049 行）：
- 可旋转维度 `2*(head_dim//2)` 平均分成 T/H/W 三段，每段 `axis_dim` 维，各自用坐标 `(t, h, w)` 乘自己的频率带；
- 超出 `3*axis_dim` 的维度直通不旋转（head_dim=80 时，每轴 26 维，共 78 维旋转，剩 2 维直通）。

```
|<----------- 旋转区 (3 × axis_dim) ----------->|<-直通->|
+----------+----------+----------+-------------+
|  T 轴频段 |  H 轴频段 |  W 轴频段 |  不旋转     |
| axis_dim | axis_dim | axis_dim |  (剩余)      |
+----------+----------+----------+-------------+
```

**视觉塔 forward**（1198-1210 行）：patch embed → `pre_layrnorm`（LayerNorm）→ 3D RoPE 生成 cos/sin → 32 层 `MiniMaxM3VLVisionEncoderLayer`（每层 = VisionAttention + VisionMLP，GELU）→ 输出 `last_hidden_state`。

> 注意：视觉塔**没有** final LayerNorm / final norm，池化用 `hidden_states[:, 0]` 占位，真正的"池化输出"被投影器覆盖（见 4.2）。

### 4.2 多模态投影器 MiniMaxM3VLMultiModalProjector（1213-1232 行）

```python
def forward(self, image_features):
    hidden_states = self.linear_2(self.act(self.linear_1(image_features)))   # ① 1280→6144（GELU）
    hidden_states = hidden_states.reshape(hidden_states.shape[0] // 4, -1)   # ② 2×2 合并→通道翻 4 倍
    return self.merge_linear_2(self.merge_act(self.merge_linear_1(hidden_states)))  # ③ 24576→6144
```

**教学点**：
- ①把视觉 patch 特征从 1280 维投影到文本世界的 6144 维；
- ②`reshape(seq//4, -1)` 把每 2×2 相邻 patch 的 4 条 6144 维特征**拼进通道维**（6144×4 = 24576），视觉 token 数量缩到 1/4；
- ③再压回 6144 维。**空间合并是控制 token 数的关键**：`merge_size²=4` 个 patch → 1 个 token。

> 所以处理器里 `replace_image_token` 计算占位符数时用的就是 `num_patches // merge_size**2`（processing 文件 57-59 行）。

### 4.3 文本主干 MiniMaxM3VLTextModel（716-792 行）

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
self.layers = nn.ModuleList([MiniMaxM3VLDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
self.norm = MiniMaxM3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.rotary_emb = MiniMaxM3VLRotaryEmbedding(config=config)
```

forward 里两个值得注意的点：

1. **稀疏缓存由配置驱动**（749-750 行）：
```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
```
`DynamicCache(config=...)` 会读取 `config.layer_types`，凡是 `"minimax_m3_sparse"` 的层自动使用 `MiniMaxM3VLSparseCacheLayer`（见第 5 章 5.5）。

2. **position_ids 逐层下传**（774-785 行）：`position_ids` 被传给每一层，供稀疏层的 Indexer 锚定"当前 query 在第几个块"。

### 4.4 解码层 MiniMaxM3VLDecoderLayer（638-677 行）

经典 Pre-Norm 残差结构，但**注意力和 MLP 都按层路由**：

```python
self.self_attn = MiniMaxM3VLAttention(config, layer_idx)          # 内部按 layer_types 决定是否挂 Indexer
self.mlp = (
    MiniMaxM3VLSparseMoeBlock(config)                              # 该层用 MoE
    if config.mlp_layer_types[layer_idx] == "sparse"
    else MiniMaxM3VLDenseMLP(config, intermediate_size=config.dense_intermediate_size)  # 该层用稠密
)
self.input_layernorm = MiniMaxM3VLRMSNorm(...)
self.post_attention_layernorm = MiniMaxM3VLRMSNorm(...)

def forward(...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(...)
    hidden_states = residual + hidden_states          # ① 残差
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states          # ② 残差
    return hidden_states
```

**RMSNorm 是 Gemma 风格**（142-158 行）：fp32 归一化，权重从 `1 + weight` 起（初始化全 0 → 等价于普通归一化），输出再转回原 dtype。

### 4.5 注意力 MiniMaxM3VLAttention（408-489 行）

M3 注意力的四个特征：

**① per-head QK-norm（Gemma 风格）**：q/k 投影后各自过一个 head_dim 维的 RMSNorm（426-427 行），让注意力分数更稳，避免长上下文下 logits 爆炸。

**② GQA**：64 查询头 / 4 KV 头 → `num_key_value_groups = 64/4 = 16`，KV 缓存压缩 16 倍。

**③ 部分 RoPE**：`apply_rotary_pos_emb` 只旋转前 `rotary_dim=64` 维，后 64 维直通（rotary_dim < head_dim）。

**④ 可选 Lightning Indexer**（428-430 行）：`layer_types[layer_idx] == "minimax_m3_sparse"` 时挂上：

```python
self.indexer = (
    MiniMaxM3VLIndexer(config, layer_idx)
    if config.layer_types[layer_idx] == "minimax_m3_sparse" else None
)
```

forward 中 Indexer 的工作（456-475 行）：先算 `block_indices`（每 query 选哪些 key 块），再在 eager/SDPA 路径下调用 `build_block_mask` 把块选择展开成完整 4D additive mask，交给标准注意力接口。

### 4.6 Lightning Indexer —— 全模型最亮的点（492-635 行）

**核心思想**：不做"每个 query 对所有 key"的完整注意力，而是先用一个**廉价打分分支**挑出"值得看"的 key 块，主注意力只在被选中的块上算。

```
                    ┌── 廉价打分分支（4 头，index_head_dim=128）──┐
 hidden_states ──→  q_proj → q_norm → 与全量 key 打分
                    k_proj → k_norm →（共享 key，1 份）
                    │
                    ▼
      scores [B, 4, S_q, S_k]  （float32 计算）
      │ ① 未来 token 打 -inf（因果）
      │ ② 补零到块边界
      ▼
      block_scores [B, 4, S_q, num_key_blocks]   ← 每块内 amax 池化
      │ ③ q_block = position_ids // block_size
      │ ④ 前 local_blocks 个块直接置 +inf（必定可见）
      ▼
      topk(16) → block_indices [B, 4, S_q, 16]   ← 每 query 选 16 个 key 块
      （无效槽位填 -1）
```

**forward 关键代码**（552-597 行）：

```python
scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))   # fp32 打分
token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
scores = scores.masked_fill(token_future, float("-inf"))                # 因果
scores = scores.view(batch, self.num_heads, q_len, num_key_blocks, self.block_size)
block_scores = scores.amax(dim=-1)                                      # 块内 max 池化
q_block = position_ids // self.block_size                               # query 所在块
# 前 local_blocks 个块直接赢（scatter +inf）
block_scores.scatter_(-1, local_idx, float("inf"))
topk_scores, topk_indices = block_scores.topk(topk, dim=-1)             # top-16 块
return topk_indices.masked_fill(topk_scores == float("-inf"), -1)       # 无效块→-1
```

**计算量对比**（教学点）：假设序列 128K、块大小 128：
- 主注意力若全量：128K × 128K 矩阵
- 稀疏后：每个 query 只看 16 块 × 128 key = **2048 个 key**（+1 局部块）
- 稀疏比 ≈ 2048 / 131072 ≈ **1.6%**，这就是 512K 长上下文能跑起来的原因。

**build_block_mask**（599-635 行）把块索引展开回稠密 mask：
1. `scatter` 把选中的块标记为 0、其余 -inf；
2. `repeat_interleave(block_size)` 把"块级判定"广播回每个 key；
3. `repeat_interleave(num_attention_heads // n_idx_heads)` 把 4 个 Indexer 头的选择广播到全部 64 个注意力头（因为 1 个 KV 组共享 1 个块选择）；
4. 与 padding mask / 因果 mask 组合，输出 `[B, 64, S_q, S_k]` 的 additive mask。

### 4.7 MoE 混合专家（183-264 行）

**路由：Sigmoid 打分 + 修正偏置 + top-4**（`MiniMaxM3VLTopKRouter`，223-241 行）：

```python
router_logits = F.linear(hidden_states, self.weight)                    # [*, 128]
routing_weights = F.sigmoid(router_logits.float())                      # 独立 sigmoid，非 softmax
scores_for_choice = routing_weights + self.e_score_correction_bias      # 可学习修正偏置
_, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1)      # top-4
top_k_weights = routing_weights.gather(1, top_k_index)
top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)                # 权重归一化
```

**专家权重存成 3D 张量**（`MiniMaxM3VLExperts`，183-220 行）：

```python
self.gate_up_proj = nn.Parameter(torch.empty(128, 2 * 3072, 6144))      # 128 个专家拼一起
self.down_proj    = nn.Parameter(torch.empty(128, 6144, 3072))

# forward：只对"命中的专家"做矩阵乘
mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()                # 找出命中的专家
for expert_idx in hit:
    current = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
    current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
    final.index_add_(0, token_idx, current.to(final.dtype))             # 累加回最终输出
```

**MoE 块结构**（`MiniMaxM3VLSparseMoeBlock`，244-264 行）：128 路由专家 + **1 个共享专家**（DenseMLP，intermediate=3072），输出：

```python
hidden_states = experts_out * self.routed_scaling_factor   # ×2.0
hidden_states = hidden_states + shared_output              # + 共享专家输出
```

> 共享专家保证每个 token 都有稳定的"通用知识"通道，路由专家负责"专长"，这在 M2/M3 系列中是标配设计。

### 4.8 Dense MLP 与 SwiGLU-OAI 激活（164-179 行）

稠密层中间维是 12288（专家维 3072 的 4 倍）。激活函数是 M3 特有的 **SwiGLU-OAI**：

```python
gate, up = gate_up.chunk(2, dim=-1)
gate = gate.clamp(max=self.swiglu_limit)                  # 上限 7.0
up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
glu = gate * torch.sigmoid(gate * self.swiglu_alpha)      # alpha=1.702
return self.down_proj((up + 1.0) * glu)
```

**原理**：相比标准 SwiGLU（`gate * silu(up)`），这里 `gate` 同时作为 sigmoid 的输入（增益 `alpha=1.702`），并对 gate/up 做 clamp 限制幅值——数值更稳、防止激活值爆炸，且权重不交错存储（注释明确说 "same as GPT OSS, but the weights are not interleaved"）。

---

## 第 5 章 机制原理深挖

### 5.1 为什么用 Sigmoid 路由而不是 Softmax？

Softmax 路由的本质是**零和博弈**：一个专家分数高，其他全被压低。Sigmoid 让每个专家**独立打分**（互不竞争），配合 `e_score_correction_bias`（每个专家一个可学习标量，初始化 0）可以整体抬升/压低某些专家的入选率。效果：路由更"宽松"，冷门专家也有机会被选到，训练更稳定。这也是 M2 就采用的方案（代码注释：`# Sigmoid scoring (not softmax), as in M2.`）。

### 5.2 Lightning Indexer 为什么省这么多计算？

省的不只是 FLOPs，还有**内存带宽**——KV 缓存读取是生成瓶颈。只读被选中的 16 个块（+局部块）而不是全量 KV：
- 主注意力的 KV 读取量降到约 1.6%；
- Indexer 自身只投影 128 维、单头 KV（4 头共享 1 份 key），成本远低于主注意力；
- 块选择是**每层独立**的（每层一个 Indexer），不同层可以关注不同位置，比固定稀疏模式（如 Sliding Window + Global）更灵活。

### 5.3 GQA + 部分 RoPE + QK-norm 的组合拳

- **GQA**：64→4 个 KV 头，KV 缓存减少 16 倍，长上下文 + 大批次推理的关键；
- **部分 RoPE**（rotary_dim=64 < head_dim=128）：只旋转一半维度，降低 RoPE 计算开销，且让另一半维度保留"绝对位置无关"的信息通道；
- **QK-norm**：fp32 归一化 q/k 后再算注意力，防止长序列注意力分数方差过大（这也是 Gemma/DeepSeek 系共同的选择）。

三者叠加 = **长上下文下又省内存、又稳数值、又便宜**。

### 5.4 为什么视频和图像共用同一个视觉塔？

代码里 `get_video_features` 与 `get_image_features` 都调用 `self.vision_tower`（1429-1439 行），注释写得很明白：*"Video frames flow through the same vision pipeline as images (the tower is grid-agnostic); only the placeholder token they scatter into differs."*

**塔是 grid-agnostic 的**：它只看 `grid_thw`（T/H/W 三维网格），不关心输入是静态图（T=1，预处理时复制成 2 帧）还是真视频（T=N）。位置信息由 3D RoPE 按 `grid_thw` 生成——**同一套参数同时建模空间与时间**。训练时图像和视频可以混合进同一个 batch，这是"原生多模态"的关键设计。

### 5.5 稀疏缓存：DynamicCache 与 SparseCacheLayer

文本模型的 KV 缓存有两种层类型（53-140 行）：

- `MiniMaxM3VLSparseCacheLayer`（动态）：除了存主注意力的 KV，还额外存 Indexer 的 `idx_keys`（打分分支的 key），`update_index` 把新 token 的 idx_k 追加进历史；
- `MiniMaxM3VLSparseStaticCacheLayer`（静态，cudagraph 友好）：预分配 `[B, 1, max_cache_len, D]` 零缓冲，原地 `index_copy_` 写入，支持 torch.compile 静态地址。

因为 Indexer 打分时需要**全量 key 历史**（而主注意力只读选中块），所以稀疏层必须把 idx_k 也缓存——这是生成阶段稀疏注意力的隐形支撑。

### 5.6 负载均衡损失（load_balancing_loss_func，795 行起）

MoE 训练还需要一个辅助损失防止"所有 token 都挤到少数专家"：

```python
loss = self.router_aux_loss_coef * load_balancing_loss_func(router_logits, ...)
```

代码里 `router_aux_loss_coef = 0.001`，损失很小，只在训练时叠加（`output_router_logits` 为 False 时默认不输出路由 logits）。

---

## 第 6 章 配置速查表

### 6.1 文本配置（MiniMaxM3VLTextConfig 默认值）

| 类别 | 字段 | 值 |
|---|---|---|
| 词表 | vocab_size | 200064 |
| 结构 | hidden_size / layers | 6144 / 60 |
| 注意力 | heads / kv_heads / head_dim | 64 / 4 / 128 |
| 上下文 | max_position_embeddings | 524288 |
| RoPE | rope_theta / rotary_dim | 5e6 / 64 |
| MoE | experts / per_tok / scaling / aux | 128 / 4 / 2.0 / 0.001 |
| MLP | dense_inter / expert_inter / shared_inter | 12288 / 3072 / 3072 |
| SwiGLU-OAI | alpha / limit | 1.702 / 7.0 |
| Indexer | heads / head_dim / block / topk / local | 4 / 128 / 128 / 16 / 1 |

### 6.2 视觉配置（MiniMaxM3VLVisionConfig 默认值）

| 字段 | 值 |
|---|---|
| hidden_size / layers / heads | 1280 / 32 / 16 |
| patch_size / temporal_patch / merge | 14 / 2 / 2 |
| image_size / channels | 2016 / 3 |
| rope_theta（3D RoPE） | 10000 |

### 6.3 顶层配置（MiniMaxM3VLConfig）

| 字段 | 值 |
|---|---|
| image_token_index / video_token_index | 200025 / 200026 |
| projector_hidden_size | 6144 |
| merged_hidden_size（自动） | 6144 × 2² = 24576 |

---

## 第 7 章 常见误区与 FAQ

1. **`MiniMaxM3VLForCausalLM` 能看图吗？** 不能！它是纯文本模型（`MiniMaxM3VLTextModel` + lm_head）。图文/视频要用 `MiniMaxM3SparseForConditionalGeneration`。
2. **所有 60 层都一样吗？** 不是。`layer_types` 和 `mlp_layer_types` 是逐层列表——有的层稀疏注意力、有的层全量；有的层 MoE、有的层稠密。读代码要带"逐层"视角。
3. **128 专家 = 参数量爆炸？** 专家权重存成 3D 张量 `[128, 2×3072, 6144]`，总参数大，但**每 token 只激活 top-4 专家 + 1 共享专家**，推理计算量 ≈ 稠密模型的一小部分。
4. **Indexer 和主注意力维度一样吗？** 不一样。Indexer 是独立的 4 头 128 维打分分支，自己有一套 q/k 投影和缓存，只负责"选块"，不产生主输出。
5. **左填充没问题吗？** 有问题。Indexer 的块锚定在绝对 key 槽位上（`q_block = position_ids // block_size`），左填充会错位（代码注释自己承认，与 DeepSeek-V4 同样限制）——只有右填充等价。
6. **视觉塔有 final norm 吗？** 没有。视觉塔只有 `pre_layrnorm`，最后 32 层的输出直接进投影器（池化输出被投影结果覆盖）。
7. **图像为什么会有 temporal 维度？** 图像预处理时把 patch 用 `expand` 复制成 `temporal_patch_size=2` 份，目的是对齐 Conv3d 的输入格式——"静态图像当作 2 帧相同画面"。

---

## 第 8 章 延伸阅读

- `processing_minimax_m3_vl.py`：`replace_image_token` / `replace_video_token` 如何把占位符扩成 N 个 token（视频还会加 `] 1.5 seconds [` 时间戳标记）。
- `image_processing_minimax_m3_vl.py` / `video_processing_minimax_m3_vl.py`：详见《MiniMax M3 VL 图像处理器与视频处理器对比》。
- 稀疏注意力同类实现对比：DeepSeek-V4 的 indexer、Qwen3 的 hybrid attention、Jamba 的 Mamba-Transformer 混合。
- `modular_minimax_m3_vl.py`：本模型全部源码的"唯一真源"，所有生成文件都从它产出（文件头有 CI 强制的说明）。
