# MiniMax M3 VL 源码解析与模型结构对比

> 对比对象：MiniMax M3 VL（多模态 MoE 稀疏注意力大模型）vs 目录内同期开源模型
> —— DeepSeek-V3、DeepSeek-V4（Flash / Pro）、Qwen3.5（dense）/ Qwen3.5-MoE、Kimi-K2.5、GLM-MoE-DSA
>
> 所有数值均来自各模型 `configuration_*.py` 默认值与建模源码（已逐项核对），代码级机制说明来自 transformers 官方实现。

---

## 0. MiniMax M3 VL 源码来源

MiniMax M3 VL 的建模实现位于 transformers 库 `models/minimax_m3_vl/` 目录，本次读取/核对的文件：

| 文件 | 行数 | 说明 |
|---|---|---|
| `configuration_minimax_m3_vl.py` | 226 | 三份配置类：Text / Vision / 顶层 Config |
| `modeling_minimax_m3_vl.py` | 1548 | 完整建模：视觉塔 + 投影器 + 文本模型 + MoE + Indexer |
| `image_processing_minimax_m3_vl.py` | — | 图像预处理器（smart_resize + Conv3d patchify） |
| `video_processing_minimax_m3_vl.py` | — | 视频预处理器（帧采样 + smart_resize + 3D patchify） |
| `processing_minimax_m3_vl.py` | — | 多模态 Processor 编排（图文 token 拼接入口） |

> 代码文件头部标注 `Copyright 2026 the MiniMax AI Team and HuggingFace Team`，由 `modular_minimax_m3_vl.py` 自动生成。

---

## 1. MiniMax M3 VL 结构总览

### 1.1 一句话定位

**MiniMax M3 VL = "视觉塔 + 投影器 + 128 专家 MoE + Lightning Indexer 块级稀疏注意力"的多模态三明治结构，文本侧用 60 层混合异构 Transformer 撑起 512K 超长上下文。**

### 1.2 三层拼装

```
MiniMaxM3SparseForConditionalGeneration   ← 多模态成品入口（图文视频 → 文字）
 └─ MiniMaxM3VLModel                      ← 总装车间
      ├─ MiniMaxM3VLVisionModel           ← 视觉塔（32 层 CLIP 风格，Conv3d 支持视频）
      ├─ MiniMaxM3VLMultiModalProjector   ← 投影器（两级 GELU MLP + 2×2 空间合并）
      └─ MiniMaxM3VLTextModel             ← 语言模型（60 层 Decoder）
           └─ MiniMaxM3VLDecoderLayer ×60
                ├─ self_attn  (GQA + QK-norm + 部分 RoPE，可挂 Lightning Indexer)
                └─ mlp       (DenseMLP 或 SparseMoeBlock 二选一，逐层异构)
```

### 1.3 文本侧核心参数（来自 `MiniMaxM3VLTextConfig` 默认值）

| 维度 | 值 | 代码出处 |
|---|---|---|
| 层数 | **60** | `num_hidden_layers: int = 60` |
| hidden_size | **6144** | `hidden_size: int = 6144` |
| vocab_size | **200064** | `vocab_size: int = 200064` |
| 注意力头 | Q **64** / KV **4**（GQA 16:1） | `num_attention_heads=64`, `num_key_value_heads=4` |
| head_dim | **128** | `head_dim: int = 128` |
| RoPE | theta **5,000,000**，partial rotary **64/128**（旋转前半） | `default_theta=5000000.0`, `rotary_dim=64` |
| 最大上下文 | **524,288（512K）** | `max_position_embeddings: int = 524288` |
| MoE 专家 | **128 路由 + 1 共享**，每 token 激活 **top-4** | `num_local_experts=128`, `num_experts_per_tok=4` |
| 专家中间维 | **3072**（路由+共享相同） | `intermediate_size=3072`, `shared_intermediate_size=3072` |
| 稠密层中间维 | **12288** | `dense_intermediate_size=12288` |
| 路由方式 | **Sigmoid + correction bias + topk 归一化**（非 softmax） | modeling L236: `F.sigmoid(router_logits)` |
| 路由缩放 | **2.0** | `routed_scaling_factor: float = 2.0` |
| 激活函数 | **SwiGLU-OAI**（clamp + sigmoid gate + alpha 1.702） | `swiglu_alpha=1.702`, `swiglu_limit=7.0` |
| 归一化 | RMSNorm 1e-6，Gemma 风格（权重 `1+weight`） | `rms_norm_eps=1e-6` |
| Lightning Indexer | 4 头 × 128 dim，块大小 128，top-16 块 + 1 局部块 | `index_n_heads=4`, `index_block_size=128`, `index_topk_blocks=16`, `index_local_blocks=1` |
| 逐层异构 | `layer_types`（全/稀疏注意力）+ `mlp_layer_types`（dense/sparse MoE） | `__post_init__` L143-154 |
| MTP | **无**（代码中无 MTP 模块） | — |

### 1.4 视觉侧核心参数（`MiniMaxM3VLVisionConfig`）

| 维度 | 值 |
|---|---|
| 层数 | **32** |
| hidden_size | **1280** |
| 注意力头 | **16**（全注意力，无 GQA） |
| image_size | **2016×2016** |
| patch_size | **14×14** → 144×144 = 20736 patch/帧 |
| temporal_patch_size | **2**（Conv3d 时间核，支持视频） |
| spatial_merge_size | **2**（2×2 合并 → 4 patch → 1 token） |
| 激活 | **GELU**（与文本侧 silu 不同） |
| 归一化 | **LayerNorm 1e-5**（不是 RMSNorm） |
| RoPE | 3D RoPE，theta 10000（与文本的 500 万不同） |

### 1.5 投影器（`MiniMaxM3VLMultiModalProjector`）

两级 GELU MLP + 空间合并：
1. 第一级 MLP：1280 → 6144（对齐文本 hidden_size）
2. 空间合并：2×2=4 个相邻 patch 在通道维拼接 → 6144×4 = 24576
3. 第二级 MLP：24576 → 6144（压回文本维度）

> 合并后 4 个视觉 patch → 1 个文本 token，序列长度直接 ÷4。

---

## 2. 七款模型结构速览表

| 维度 | **MiniMax M3 VL** | DeepSeek-V3 | DeepSeek-V4 (Flash) | Qwen3.5 (dense) | Qwen3.5-MoE | Kimi-K2.5 | GLM-MoE-DSA |
|---|---|---|---|---|---|---|---|
| 架构类 | `MiniMaxM3SparseForConditionalGeneration` | `DeepseekV3ForCausalLM` | `DeepseekV4ForCausalLM` | `Qwen3_5ForConditionalGeneration` | `Qwen3_5MoeForCausalLM` | `Kimi_K25ForConditionalGeneration` | `GlmMoeDsaForCausalLM` |
| 模态 | **多模态（图+视频+文）** | 纯文本 | 纯文本 | 多模态（图+视频） | 多模态（图+视频） | 多模态（图+视频） | 纯文本 |
| 层数 | **60** | 61 | 43 | 32 | 40 | 93（text_config 用 DeepSeekV3） | 78 |
| hidden_size | **6144** | 7168 | 4096 | 4096 | 2048 | 7168 | 6144 |
| vocab_size | **200064** | 129280 | 129280 | 248320 | 248320 | 163840 | 154880 |
| 注意力类型 | **混合**：全注意力 + Lightning Indexer 块稀疏 | **纯全注意力**（MLA） | **混合稀疏**：滑窗 + CSA + HCA | **混合**：3 线性 + 1 全注意力 | **混合**：3 线性 + 1 全注意力 | **纯全注意力**（MLA，text=DeepSeekV3） | **全层 DSA 稀疏 top-k** |
| 稀疏/线性注意力 | **Lightning Indexer**：4 头打分 → 128 key/块 max-pool → top-16 块 + 1 局部块 | 无 | **CSA Compressor**（ratio 4）+ **HCA Compressor**（ratio 128）+ learned Indexer top-512 | **Gated DeltaNet**（卷积 4，QK-L2 norm） | **Gated DeltaNet**（卷积 4） | 无 | **DSA Indexer**（32 头，ReLU + top-2048 token，IndexShare 每 4 层） |
| 全注意力层 | Q 64 / KV 4（**GQA 16:1**），head_dim **128**，**per-head QK-norm**（Gemma 风格） | MLA：128 头，q_lora 1536，kv_lora 512，qk_nope 128 + rope 64 | MLA-512：KV 头=1，head_dim **512**，q_lora 1024，o_lora 1024，o_groups 8 | GQA：16Q/4KV，head_dim **256**，Q/K norm | GQA：16Q/2KV，head_dim **256**，Q/K norm | MLA：128 头（复用 DeepSeekV3） | MLA：64 头，q_lora 2048，kv_lora 512，qk_nope 192 + rope 64 |
| 位置编码 | RoPE **5M**，partial **64/128**，**非交错** | RoPE 10K + YaRN ×40（160K） | RoPE 10K (main) + 160K (compress) + YaRN ×16 | RoPE（partial 0.25），**交错 mRoPE** | RoPE（partial 0.25），交错 mRoPE | RoPE 10K + YaRN（复用 DeepSeekV3） | RoPE **8M**，interleaved，partial 64/256 |
| FFN / MoE | **128 专家 / top-4** + 1 shared（3072）；部分层 dense 12288 | 前 3 层 dense 18432 + MoE 256/8 + 1 shared（2048） | 全 MoE：256/6 + 1 shared（2048）；前 3 层 hash 路由 | dense 12288 | MoE 256/8 + 1 shared（512） | 1 dense + MoE 256/8 + 1 shared（复用 V3） | 前 3 层 dense 12288 + MoE 256/8 + 1 shared（2048） |
| 路由方式 | **Sigmoid** + correction bias + topk 归一化 + aux loss 0.001 | Sigmoid + 分组 topk（8 组×4）+ noaux_tc | **sqrtsoftplus** + noaux_tc；hash 前 3 层 | — | TopK softmax，无 aux | Sigmoid + correction bias（复用 V3） | Sigmoid + correction bias + noaux_tc |
| 路由缩放 | **2.0** | 2.5 | 1.5 | — | — | 2.5 | 2.5 |
| 激活函数 | **SwiGLU-OAI**（clamp 7.0 + sigmoid(α=1.702) gate） | SwiGLU (silu) | SwiGLU (silu)，专家 clamp 10.0 | SwiGLU (silu) | SwiGLU (silu) | SwiGLU (silu) | SwiGLU (silu) |
| 归一化 | RMSNorm **1e-6** | RMSNorm 1e-6 | RMSNorm 1e-6 + **mHC 超连接**（Sinkhorn ×4） | RMSNorm 1e-6 | RMSNorm 1e-6 | RMSNorm 1e-6 | RMSNorm **1e-5** |
| 上下文 | **512K** | 160K（YaRN） | **1M** | 32K | 32K | 160K（YaRN） | 202K |
| MTP | **无** | 1 层 | 1 层 | 无 | 无 | 无 | 无 |
| 视觉编码器 | 32 层 ViT，1280 维，patch 14×14，Conv3d 时间核 2 | 无 | 无 | 27 层 ViT，1152 维，patch 16×16 | 27 层 ViT，1152 维，patch 16×16 | 27 层 ViT，1152 维，patch 14×14 | 无 |
| 空间合并 | 2×2 → 4 patch 并 1 token | — | — | 2×2 | 2×2 | 2×2 merge kernel | — |
| tie_embeddings | False | False | False | False | False | **True** | False |

---

## 3. 逐模型差异分析

### 3.1 vs DeepSeek-V3（同为 Sigmoid 路由 MoE，但注意力路线完全不同）

| 对比点 | MiniMax M3 VL | DeepSeek-V3 |
|---|---|---|
| 注意力流派 | **块级稀疏注意力**（Lightning Indexer 选 top-16 块） | **纯全注意力**（MLA，全序列 softmax） |
| KV 压缩 | GQA 4 KV 头 + 块稀疏选择（只算选中的块） | **MLA 低秩潜变量**（kv_lora 512 压缩缓存） |
| 长上下文手段 | Lightning Indexer 把注意力复杂度从 O(n²) 降到块稀疏级 | YaRN 拉伸（160K） |
| MoE 规模 | 128 专家 / top-4 + 1 shared（3072） | 256 专家 / top-8 + 1 shared（2048），分组路由 8 组×4 |
| 路由 | Sigmoid + correction bias + topk 归一化 | Sigmoid + 分组 topk + noaux_tc |
| 稠密层 | 逐层异构（`mlp_layer_types` 控制 dense/sparse） | 前 3 层固定 dense，后续全 MoE |
| 激活 | **SwiGLU-OAI**（clamp + sigmoid gate，α=1.702） | 标准 SwiGLU (silu) |
| 视觉 | 有（32 层 Conv3d ViT） | 无 |
| MTP | 无 | 1 层 |
| 上下文 | **512K** | 160K |

核心差异：**两者路由方式同源（Sigmoid + correction bias），但注意力方案完全正交**——MiniMax 用 Lightning Indexer 做块级稀疏选择（类 DeepSeek-V4/GLM-DSA 路线），DeepSeek-V3 用 MLA 低秩压缩做全注意力。MiniMax 的 MoE 更"宽而浅"（128 专家仅激活 4 个 = 3.1%），DeepSeek-V3 更"窄而深"（256 专家激活 8 个 = 3.1%，但中间维更小）。

### 3.2 vs DeepSeek-V4（最接近的"同类"——都有 Lightning Indexer 稀疏注意力）

| 对比点 | MiniMax M3 VL | DeepSeek-V4 (Flash) |
|---|---|---|
| Indexer 头数 | **4 头**（= KV 头数，每个 GQA 组独立选块） | **64 头**（远多于 MiniMax） |
| Indexer 块定义 | 固定 128 key/块，**max-pool** 取块最高分 | 压缩率 4（CSA）或 128（HCA），**门控池化** Compressor |
| Indexer top-k | **top-16 块** + 1 局部块 | **top-512 压缩条目** |
| Indexer 评分 | QK 点积 + QK-norm，**无 ReLU**，无 weights_proj | `∑_h w_h · ReLU(q · k)`，有可学习 weights_proj |
| Indexer 旋转 | 复用主注意力的 cos/sin（同 theta） | **独立 RoPE**（compress_rope_theta 160K） |
| 全注意力 | GQA 64Q/4KV，head_dim 128，per-head QK-norm | MLA-512：KV 头=1，head_dim **512**，q_lora 1024 |
| 注意力层类型 | `full_attention` / `minimax_m3_sparse` 二选一 | `sliding_attention` / `compressed_sparse_attention` / `heavily_compressed_attention` 三选一 |
| 残差连接 | 标准 Pre-Norm | **mHC 流形超连接**（每层 4 份状态 + Sinkhorn 归一化） |
| MoE 路由 | Sigmoid + correction bias | **sqrtsoftplus** + noaux_tc |
| MoE 前 3 层 | 逐层可配（`mlp_layer_types`） | **hash_moe**（冻结 `tid2eid` 查表） |
| 激活 | SwiGLU-OAI（clamp 7.0 + α=1.702） | SwiGLU (silu)，专家 clamp 10.0 |
| 上下文 | 512K | **1M** |
| 视觉 | 有 | 无 |

核心差异：**两者都采用"Lightning Indexer + 稀疏注意力"路线，但实现深度差异显著**——DeepSeek-V4 的 Indexer 更重（64 头 + ReLU + 独立 RoPE + weights_proj + CSA/HCA 双压缩率），MiniMax 的更轻（4 头 + 简单点积 + 复用主 RoPE）。DeepSeek-V4 还多了 mHC 超连接和 hash MoE 前置层，工程复杂度更高。**MiniMax 的 Indexer 可视为 DeepSeek-V4 Indexer 的"轻量简化版"**——代码注释也自承认"和 DeepSeek-V4 同样的左填充限制"。

### 3.3 vs Qwen3.5 / Qwen3.5-MoE（线性注意力 vs 块稀疏注意力）

| 对比点 | MiniMax M3 VL | Qwen3.5 (dense) | Qwen3.5-MoE |
|---|---|---|---|
| 注意力流派 | **块级稀疏**（Lightning Indexer 选块） | **线性注意力**（Gated DeltaNet，无 KV cache） | **线性注意力**（Gated DeltaNet） |
| 层排布 | 逐层异构（`layer_types` 控制 sparse/full） | **3 线性 + 1 全注意力**（每 4 层 1 个全注意力） | 3 线性 + 1 全注意力 |
| 全注意力层 | GQA 64Q/4KV，head_dim 128 | GQA 16Q/4KV，head_dim **256** | GQA 16Q/2KV，head_dim 256 |
| 位置编码 | RoPE 5M，partial 64/128，**非交错** | RoPE（partial 0.25），**交错 mRoPE** | RoPE（partial 0.25），交错 mRoPE |
| MoE | 128/4 + 1 shared | — | 256/8 + 1 shared |
| 路由 | Sigmoid + correction bias | — | TopK softmax（无 correction bias） |
| 激活 | SwiGLU-OAI | SwiGLU (silu) | SwiGLU (silu) |
| 视觉 | 32 层 ViT，1280 维，patch 14 | 27 层 ViT，1152 维，patch 16 | 27 层 ViT，1152 维，patch 16 |
| 上下文 | 512K | 32K | 32K |

核心差异：**Qwen3.5 用"线性注意力（Gated DeltaNet）"解决长上下文复杂度**，MiniMax 用"块级稀疏注意力（Lightning Indexer）"解决——两者是正交路线。Qwen3.5 的 Gated DeltaNet 有可学习衰减 + 写入门控 + QK-L2 归一化，属于 DeltaNet 家族；MiniMax 的 Lightning Indexer 属于 DeepSeek Sparse Attention 家族。**MiniMax 的上下文窗口（512K）远大于 Qwen3.5 默认值（32K）**，但 Qwen3.5 的线性注意力理论上可以扩展到更长。

### 3.4 vs Kimi-K2.5（同为多模态 + Sigmoid 路由 MoE，但 Kimi 复用 DeepSeek-V3 架构）

| 对比点 | MiniMax M3 VL | Kimi-K2.5 |
|---|---|---|
| 文本架构 | 自研（MiniMax M3） | **复用 DeepSeek-V3**（`text_config` model_type = `deepseek_v3`） |
| 注意力 | 块级稀疏（Lightning Indexer） | **纯全注意力**（MLA，无稀疏） |
| MoE | 128/4 + 1 shared（3072） | 256/8 + 1 shared（2048），分组路由 |
| 路由 | Sigmoid + correction bias | Sigmoid + correction bias（同 DeepSeek-V3） |
| 视觉 | 32 层 ViT，1280 维，Conv3d | 27 层 ViT，1152 维，2D RoPE |
| 上下文 | 512K | 160K（YaRN） |
| tie_embeddings | False | **True** |
| 投影器 | 两级 GELU MLP + 空间合并 | 单级投影（projection_hidden_size=1152） |

核心差异：**Kimi-K2.5 的文本侧本质就是 DeepSeek-V3**，创新在视觉和多模态整合；MiniMax 的文本侧是自研架构（Lightning Indexer + SwiGLU-OAI + 逐层异构），两者共享的只是"Sigmoid + correction bias"路由范式。

### 3.5 vs GLM-MoE-DSA（同为"稀疏注意力 + MoE"，但稀疏粒度和共享策略不同）

| 对比点 | MiniMax M3 VL | GLM-MoE-DSA |
|---|---|---|
| 稀疏注意力 | **Lightning Indexer**（4 头，块级 max-pool，top-16 **块**） | **DSA Indexer**（32 头，token 级 top-2048 **token**） |
| 稀疏粒度 | **块级**（128 key/块，选 16 块 = 2048 key） | **token 级**（直接选 2048 个 token） |
| Indexer 共享 | 每层独立运行 | **IndexShare**：每 4 层只跑 1 次完整索引器（`"full"` / `"shared"` 模式） |
| 全注意力层 | 有（`layer_types` 可配 `full_attention`） | **无**（全部 78 层都是 `deepseek_sparse_attention`） |
| 注意力 | GQA 64Q/4KV，head_dim 128 | MLA 64 头，q_lora 2048，kv_lora 512，qk_nope 192 + rope 64 |
| MoE | 128/4 + 1 shared（3072） | 256/8 + 1 shared（2048），前 3 层 dense 12288 |
| 路由 | Sigmoid + correction bias + aux 0.001 | Sigmoid + correction bias + noaux_tc |
| 位置编码 | RoPE 5M，partial 64/128 | RoPE 8M，interleaved，partial 64/256 |
| 归一化 | RMSNorm 1e-6 | RMSNorm **1e-5** |
| 上下文 | 512K | 202K |
| 模态 | 多模态 | 纯文本 |

核心差异：**两者都是"稀疏注意力 + MoE"路线，但稀疏策略不同**——MiniMax 做块级稀疏（128 key/块的 max-pool → top-16 块），GLM 做 token 级稀疏（直接 top-2048 token）。GLM 的 IndexShare 跨层共享机制是 MiniMax 没有的（MiniMax 每层独立跑 Indexer），这使 GLM 在长上下文下计算更省（1M 下 FLOPs 降 2.9×）。**MiniMax 的 RoPE base（5M）介于 GLM（8M）和 DeepSeek-V3（10K）之间**。

---

## 4. 五大技术流派归纳

目录内 7 款模型可归为 5 条技术路线：

### 流派 1：块级稀疏注意力 + MoE 流（MiniMax M3 VL）

- **代表**：MiniMax M3 VL
- **特征**：Lightning Indexer 用 4 头小打分分支给每个 query×key 打分 → 按 128 key/块做 max-pool → 每 query 选 top-16 块 + 1 局部块 → 主注意力只算选中的块
- **优势**：块级稀疏比 token 级更高效（block-sparse kernel 友好），512K 上下文
- **代价**：块锚定在绝对槽位，左填充会破坏块边界（代码注释自承认）
- **独特**：SwiGLU-OAI 激活（clamp + sigmoid gate）、逐层异构（`layer_types` + `mlp_layer_types`）、Sigmoid 路由

### 流派 2：MLA 全注意力 + 低秩 KV 压缩流（DeepSeek-V3 / Kimi-K2.5）

- **代表**：DeepSeek-V3、Kimi-K2.5（文本侧复用 V3）
- **特征**：纯 softmax 全序列注意力 + MLA 低秩潜变量压缩 KV cache（kv_lora 512）
- **优势**：结构最"保守"，KV cache 压缩比高
- **代价**：全注意力复杂度 O(n²)，仅靠 YaRN 撑到 160K
- **独特**：分组路由（8 组×4）、noaux_tc 负载均衡

### 流派 3：CSA/HCA 压缩稀疏 + mHC 超连接流（DeepSeek-V4）

- **代表**：DeepSeek-V4（Flash / Pro）
- **特征**：滑窗 128 + CSA Compressor（ratio 4）+ HCA Compressor（ratio 128）+ learned Indexer top-512 + mHC 流形超连接
- **优势**：1M 上下文，KV cache 仅为 V3 的 10%
- **代价**：工程复杂度最高（3 种注意力层类型 + 2 种 MoE 层类型 + Sinkhorn 归一化）
- **独特**：sqrtsoftplus 路由、hash MoE 前置层、o_groups 分组输出投影

### 流派 4：Gated DeltaNet 线性注意力流（Qwen 系）

- **代表**：Qwen3.5（dense）、Qwen3.5-MoE
- **特征**：3:1 线性:全注意力层排布，Gated DeltaNet 负责线性复杂度（无 KV cache，recurrent state），每 4 层 1 个全注意力保强检索
- **优势**：线性复杂度 + 低 KV 开销
- **代价**：默认上下文仅 32K（需扩展配置）
- **独特**：交错 mRoPE（三维网格 [T,H,W]）、partial rotary 0.25

### 流派 5：DSA token 级稀疏 + IndexShare 跨层共享流（GLM-MoE-DSA）

- **代表**：GLM-MoE-DSA
- **特征**：全层 DSA 稀疏 top-2048 token，IndexShare 每 4 层只跑 1 次完整索引器
- **优势**：无全注意力层也能工作，1M 下 FLOPs 降 2.9×
- **代价**：全部稀疏可能损失全局信息
- **独特**：RoPE 8M（最大 base）、interleaved mRoPE、RMSNorm 1e-5

### 共性趋势

| 趋势 | 说明 |
|---|---|
| ① 全部 RMSNorm + Pre-Norm | 7 款模型无一例外（仅 eps 不同：1e-5 或 1e-6） |
| ② 全部走向"混合/稀疏"注意力 | 无一使用纯 dense 全注意力（DeepSeek-V3 最接近，但有 MLA 压缩） |
| ③ MoE 成为大模型标配 | 仅 Qwen3.5 dense 版无 MoE；MiniMax 的 128 专家最多但 top-4 激活比最低（3.1%） |
| ④ Sigmoid 路由成为主流 | MiniMax / DeepSeek-V3 / Kimi / GLM 均用 Sigmoid + correction bias；Qwen 用 softmax；DeepSeek-V4 用 sqrtsoftplus |
| ⑤ 部分 RoPE 普遍存在 | MiniMax 64/128、Qwen 0.25（=64/256）、GLM 64/256、DeepSeek-V4 64/512 |
| ⑥ 多模态成为差异化 | MiniMax / Qwen / Kimi 支持图文视频，DeepSeek / GLM 为纯文本 |

---

## 5. MiniMax M3 VL 的独特设计点（在其他模型中找不到的）

### 5.1 SwiGLU-OAI 激活函数（全模型唯一）

```python
# modeling L214-220
def _apply_gate(self, gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=self.swiglu_limit)       # clamp 上界 7.0
    up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)  # clamp 上下界
    glu = gate * torch.sigmoid(gate * self.swiglu_alpha)  # sigmoid 门控，α=1.702
    return (up + 1.0) * glu
```

其他模型全部用标准 SwiGLU（`silu(gate) * up`），MiniMax 独有的 `sigmoid(α·gate) · gate + (up+1)` 组合 + clamp 稳定训练，灵感来自 GPT-OSS。

### 5.2 逐层异构的双维度控制

MiniMax 同时控制两个逐层维度：
- `layer_types`：注意力类型（`full_attention` / `minimax_m3_sparse`）
- `mlp_layer_types`：MLP 类型（`dense` / `sparse`）

这意味着可以配置出"第 1 层全注意力+稠密 MLP"、"第 2 层稀疏注意力+MoE"等任意组合。其他模型的逐层控制通常只有一个维度（DeepSeek-V4 也有双维度但值域不同）。

### 5.3 per-head QK-norm + 部分 RoPE 的组合

MiniMax 的 QK-norm 是 **per-head** 的（每个头独立 RMSNorm），且权重初始化为 `1+weight`（Gemma 风格）。结合 partial rotary 64/128（只旋转前半），这在 7 款模型中是独一无二的组合——Qwen 也有 QK-norm 但 head_dim=256 且 partial=0.25（64/256），GLM 用 MLA 不需要 per-head norm。

### 5.4 视觉塔 Conv3d 统一图像和视频

MiniMax 的视觉塔用 `Conv3d(3, 1280, kernel=(2,14,14))`，时间核 2 统一处理图像和视频：
- **图像**：`expand` 复制成 2 帧，Conv3d 正常处理
- **视频**：真实帧按 `grid_t = T//2` 分块

其他多模态模型（Qwen、Kimi）的视觉塔虽然也支持视频，但实现路径不同（Qwen 用 temporal_patch_size=2 的 3D Conv，Kimi 用 2D RoPE + 位置编码扩展）。

### 5.5 Lightning Indexer 的"轻量"定位

与 DeepSeek-V4 的 Indexer 相比，MiniMax 的 Lightning Indexer 刻意做了简化：

| 特性 | MiniMax Indexer | DeepSeek-V4 Indexer |
|---|---|---|
| 头数 | **4**（= KV 头数） | **64** |
| 评分函数 | 纯 QK 点积 + QK-norm | `∑ w_h · ReLU(q·k)` + weights_proj |
| 独立 RoPE | 否（复用主注意力 cos/sin） | 是（compress_rope_theta 160K） |
| 压缩机制 | 固定块 max-pool | CSA/HCA 双压缩率门控池化 |
| top-k 粒度 | **块级**（16 块 × 128 key = 2048 key） | **压缩条目级**（512 条目） |

这种"轻量"设计的代价是：MiniMax 的 Indexer 粒度更粗（块级 vs token/压缩级），但优势是计算开销更小（4 头 vs 64 头），适合在更多层上部署。

---

## 6. 关键结论

1. **MiniMax M3 VL 属于"块级稀疏注意力 + MoE"路线**，与 DeepSeek-V4 / GLM-DSA 同属稀疏注意力阵营，但粒度更粗（块级 vs token 级/压缩级），Indexer 更轻（4 头 vs 64/32 头）。与 Qwen3.5 的线性注意力路线、DeepSeek-V3 的纯全注意力路线形成三足鼎立。

2. **MoE 方面，MiniMax 的"128 专家 / top-4 激活"是目录内最激进的稀疏比**（3.1%），但单个专家的中间维（3072）与共享专家相同，总参数量可能小于 DeepSeek-V3（256 专家 × 2048）。Sigmoid + correction bias 路由与 DeepSeek-V3 / Kimi / GLM 同源，但 `routed_scaling_factor=2.0` 是独有的缩放策略。

3. **SwiGLU-OAI 激活函数是 MiniMax 独有的设计**，其他 6 款模型全部使用标准 SwiGLU (silu)。clamp + sigmoid gate 的组合灵感来自 GPT-OSS，在目录内独一无二。

4. **长上下文方案分阵营**：MiniMax（512K，块稀疏）、DeepSeek-V4 / GLM（1M/202K，压缩稀疏 + mHC/IndexShare）、DeepSeek-V3 / Kimi（160K，YaRN + MLA 全注意力）、Qwen3.5（32K 默认，线性注意力可扩展）。MiniMax 的 512K 是块稀疏路线中的最高原生上下文。

5. **多模态整合方面，MiniMax 的"Conv3d 统一图像和视频 + 2×2 空间合并 + 两级投影 MLP"方案与 Qwen3.5 高度相似**（都用 temporal_patch_size=2 + spatial_merge_size=2），但视觉塔更深（32 层 vs 27 层）、维度更大（1280 vs 1152）、patch 更小（14 vs 16）。

6. **逐层异构是 MiniMax 的结构特色**：`layer_types` + `mlp_layer_types` 双维度控制允许任意混合"全/稀疏注意力"和"dense/MoE MLP"，这在目录内只有 DeepSeek-V4 可以类比（V4 用 `layer_types` + `mlp_layer_types` 但值域不同）。

7. **MiniMax 是目录内唯一不启用 MTP 的 MoE 大模型**——DeepSeek-V3 / V4 都有 1 层 MTP，而 MiniMax 代码中完全没有 MTP 模块。这意味着 MiniMax 的推理提速完全依赖 Lightning Indexer 的块稀疏和 MoE 的稀疏激活，不走投机解码路线。
