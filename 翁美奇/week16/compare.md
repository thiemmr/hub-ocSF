# DeepSeek-V4-Pro / MiniMax-M3 / GLM-5.2 架构对比

> 本文档基于 `model_code/` 目录下的配置与建模文件整理：`deepseek-ai_DeepSeek-V4-Pro_config.json` + `deepseek_v4_modeling.py`、`minimax_m3.json` + `minimax_m3_vl_modeling.py`、`glm52_config.json` + `modeling_glm_moe_dsa.py` + `glm52_readme.md`。三家的架构数据均来自实际 config 与 modeling 代码。

## 一、总体定位

| 维度 | DeepSeek-V4-Pro | MiniMax-M3 (VL) | GLM-5.2 |
|---|---|---|---|
| 模态 | 纯文本 | 多模态（文+图+视频） | 纯文本 |
| 上下文长度 | 1M（`1048576`） | 1M（`1048576`） | 1M（`1048576`，solid 1M） |
| 架构类 | `DeepseekV4ForCausalLM` | `MiniMaxM3SparseForConditionalGeneration` | `GlmMoeDsaForCausalLM` |
| 模型类型 | `deepseek_v4` | `minimax_m3_vl` | `glm_moe_dsa` |
| 许可证 | Apache 2.0 | Apache 2.0 | MIT（Pure Open） |
| 精度 | BF16 + FP8 权重（专家 fp4） | BF16 | BF16 |
| 长上下文核心创新 | 三态注意力（滑动/CSA/HCA）+ KV 压缩 + 压缩域 Lightning Indexer | 块稀疏索引（原始 key 分块 top-k） | MLA 潜在 KV + DSA Indexer + **IndexShare 跨层共享** |
| 残差创新 | mHC 多流残差 + Sinkhorn 双随机约束 | 标准残差 | 标准残差 |

## 二、核心规模参数

| 参数 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| hidden_size | 7168 | 6144 | 6144 |
| num_hidden_layers | 61 | 60 | 78 |
| num_attention_heads | 128 | 64 | 64 |
| head_dim | **512** | 128 | 192 |
| num_key_value_heads | **1（Shared-KV MQA）** | 4（GQA） | 64（MLA 潜在压缩，等同全头共享） |
| vocab_size | 129280 | 200064 | 154880 |
| rope_theta | 10000（main）/ 160000（compress） | 5,000,000 | 8,000,000 |
| rms_norm_eps | 1e-6 | 1e-6 | 1e-5 |
| torch_dtype | bfloat16 | bfloat16 | bfloat16 |

## 三、注意力机制

### 3.1 注意力头与 KV 组织（三家路线根本不同）

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| KV 路线 | **Shared-KV MQA**：`num_key_value_heads=1`，`kv_proj` 只投影一次，K 与 V 共享同一张量 | **GQA**：4 KV 头，独立 `q_proj/k_proj/v_proj` | **MLA（Multi-head Latent Attention）**：`kv_lora_rank=512`，将 KV 压到潜在向量再按头展开 |
| KV 压缩 | 无潜在压缩，但 HCA/CSA 有窗口级 KV 压缩 | 无压缩 | `kv_a_proj_with_mqa`→`kv_a_layernorm`→`kv_b_proj` 展开为每头 K/V |
| QK-Norm | 自有 `DeepseekV4RMSNorm`（带 weight）+ `DeepseekV4UnweightedRMSNorm`（q_b_norm） | Gemma 风格 `MiniMaxM3VLRMSNorm`（`weight+1`），`use_qk_norm=true`，`qk_norm_type=per_head` | indexer 用 `nn.LayerNorm`；主注意力无显式 QK-norm（靠 MLA 潜在结构） |
| Attention Sink | 每头可学习 sink（`self.sinks`），eager 路径拼到 logits softmax 后丢弃 | 无 | 无 |
| 输出投影 | **分组低秩**：`o_groups=16, o_lora_rank=1024`（`o_a_proj` 分组降维 → `o_b_proj` 混合），规避 `heads*head_dim→hidden` 巨大开销 | 标准 `o_proj` | 标准 `o_proj`（`num_heads*v_head_dim → hidden`） |
| Q 低秩 | `q_lora_rank=1536`（`q_a_proj→q_a_norm→q_b_proj`） | 无 | `q_lora_rank=2048`（`q_a_proj→q_a_layernorm→q_b_proj`） |
| head 拆分 | rope 切片在尾部（`rope_head_dim=64`），nope 在前 | 前 64 维旋转，后 64 维直通 | `qk_nope_head_dim=192` + `qk_rope_head_dim=64`（共 `qk_head_dim=256`），`v_head_dim=256` |

### 3.2 RoPE

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| RoPE 类型 | **双 rope 体系**：`main`（θ=10000，滑动层）+ `compress`（θ=160000，CSA/HCA 压缩层），yarn 缩放（factor=16, original_max=65536, beta_fast=32, beta_slow=1） | 单一 default RoPE，θ=5,000,000，`partial_rotary_factor=0.5` | 单一 default RoPE，θ=8,000,000，`rope_type=default`（无 yarn） |
| 旋转方式 | **交错式（interleaved）**：成对通道，`repeat_interleave(2)`，对输出 rope 切片施加共轭旋转（-sin）抵消 V 上的 rope | 标准（非交错）：`torch.cat([freqs,freqs])` 复制，前半旋转后半直通 | **交错式**（`rope_interleave=true`，`indexer_rope_interleave=true`）：even/odd 切片直接旋转，bit-identical 于 de-interleave 但省一次拷贝 |
| 部分旋转 | `rope_head_dim=64`（head_dim=512 中尾部 64 维） | `rotary_dim=64`（head_dim=128 中前 64 维） | `qk_rope_head_dim=64`（qk_head_dim=256 中尾部 64 维） |
| mscale | — | — | `yarn_apply_mscale`（仅 yarn 时生效，default rope 下为 1.0） |

### 3.3 稀疏注意力（长上下文核心，三家路线完全不同）

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| 层类型 | `sliding_attention`（滑动窗 128）/ `compressed_sparse_attention`（CSA）/ `heavily_compressed_attention`（HCA），按 `compress_ratios` 交错：`[128,128,4,128,4,...,0]` | 前 3 层 dense，其余 `minimax_m3_sparse`（`sparse_attention_freq` 控制） | 全层 DSA（DeepSeek Sparse Attention），`indexer_types` 标注 `full`/`shared` |
| KV 压缩 | CSA 每 4 token 压一个；HCA 每 128 token 压一个；双序列 Ca/Cb 相邻窗共享 | 不压缩 KV，直接在原始 key 上分块 | MLA 潜在压缩（`kv_lora_rank=512`），不额外做窗口压缩 |
| Indexer | **Lightning Indexer**：在压缩 KV 上用 `∑_h w·ReLU(q·K)` 打分，每 query 取 top-`index_topk=1024` 压缩块；`index_n_heads=64, index_head_dim=128` | **Lightning Indexer**：在原始 key 上分块（`index_block_size=128`），每块 max-pool，每 query 取 top-`index_topk_blocks=16` + 局部块；`index_n_heads=4` | **DSA Indexer**：在展开后的 KV 上用 `∑_h w·ReLU(q·K)` 打分，每 query 取 top-`index_topk=2048` token；`index_n_heads=32, index_head_dim=128` |
| 选择粒度 | 压缩条目级（每条 4 或 128 token） | key 块级（每块 128 key） | token 级（无分块） |
| **跨层共享** | 无（每 CSA/HCA 层独立 indexer，与压缩器耦合） | 无（每稀疏层独立 indexer） | **IndexShare**：`index_topk_freq=4`，`indexer_types` 为 `full` + 3×`shared` 循环，shared 层复用前一 full 层的 top-k（`prev_topk_indices` 透传） |
| CSA 重叠 | 双序列 Ca/Cb，相邻窗共享前一窗 Ca，窗宽 2×stride | 无 | 无（无窗口压缩） |
| 因果性处理 | indexer 对未来压缩块置 -inf，无效索引置 -1 | `token_future` 掩码 + topk 后 `-1` 右填充 | `causal` 掩码置 -inf，或叠加 `attention_mask` |
| 缓存层 | `DeepseekV4HCACache`/`DeepseekV4CSACache`（含 compressor buffer、overlap 状态，不可回滚，`_is_stateful=True`） | `MiniMaxM3VLSparseCacheLayer`/`SparseStaticCacheLayer`（额外维护 `idx_keys` 历史） | `DynamicIndexedLayer`/`StaticIndexedLayer`（indexer key 独立缓存，`past_key_values.update_indexer`） |
| FLOPs 收益 | 压缩域 top-k | 块级 top-k | IndexShare 使 3/4 层免算 indexer，1M ctx 下 per-token FLOPs 降 2.9×（README 声明） |

## 四、MoE / FFN

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| 路由专家数 | 384（`n_routed_experts`） | 128（`num_local_experts`） | 256（`n_routed_experts`） |
| 共享专家 | 1（`n_shared_experts`） | 1（`n_shared_experts`） | 1（`n_shared_experts`） |
| 每 token 激活专家 | 6（`num_experts_per_tok`） | 4（`num_experts_per_tok`） | 8（`num_experts_per_tok`） |
| 专家中间维 | 3072（`moe_intermediate_size`） | 3072（`intermediate_size`） | 2048（`moe_intermediate_size`） |
| dense 层中间维 | —（仅 MoE） | 12288（`dense_intermediate_size`，前 3 层 dense MLP） | 12288（`intermediate_size`，前 3 层 dense MLP，`first_k_dense_replace=3`） |
| 共享专家中间维 | 同 MoE（3072） | 3072（`shared_intermediate_size`） | `moe_intermediate_size * n_shared_experts` = 2048 |
| dense/MoE 排布 | 全 MoE（无 `first_k_dense_replace`） | 前 3 层 dense + 57 层 MoE（`moe_layer_freq`） | 前 3 层 dense + 75 层 MoE（`mlp_layer_types`） |
| 路由打分函数 | `sqrtsoftplus`（`scoring_func`） | `sigmoid`（`scoring_func`） | `sigmoid`（`scoring_func`） |
| 路由方法 | `noaux_tc`（带偏置 topk + 归一化）+ **HashRouter**（前若干层冻结 `tid2eid[input_ids]` 表选专家） | sigmoid + `use_routing_bias`，`e_score_correction_bias` | `noaux_tc`，sigmoid + `e_score_correction_bias`，`n_group=1, topk_group=1`（无分组路由），`norm_topk_prob=true` |
| 路由缩放 | `routed_scaling_factor=2.5` | `routed_scaling_factor=2.0` | `routed_scaling_factor=2.5` |
| 专家权重精度 | `expert_dtype=fp4` | bf16（无量化） | bf16（无量化配置） |
| 整体量化 | FP8（`e4m3`, 动态 activation, `ue8m0` scale, block 128×128） | 无 | 无 |
| 激活函数 | `silu`（`hidden_act`），`swiglu_limit=10.0` clamp | `swigluoai`（自定义 SwiGLU：`gate*sigmoid(gate*alpha)`，`(up+1)*glu`，`swiglu_alpha=1.702, swiglu_limit=7.0`） | `silu`（`hidden_act`），无 clamp |
| router dtype | — | — | `moe_router_dtype=float32`（路由计算强制 fp32） |

## 五、残差连接 / 模块组织

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| 残差结构 | **Manifold-Constrained Hyper-Connections (mHC)**：`hc_mult=4` 路并行残差流 `[B,S,4,D]`，每层 2 个 `DeepseekV4HyperConnection`（attn 位 + mlp 位），`comb` 矩阵经 Sinkhorn-Knopp（`hc_sinkhorn_iters=20`）投影到双随机流形，`pre` 塌缩流、`post` 放置子层输出 | 标准残差：`residual + sublayer(norm(x))` | 标准残差：`residual + sublayer(norm(x))` |
| 最终聚合 | `DeepseekV4HyperHead` 将 4 路流塌缩回单序列后过 norm | 直接 `norm(hidden_states)` | 直接 `norm(hidden_states)` |
| 归一化 | `DeepseekV4RMSNorm`（标准 RMSNorm，weight 乘） | `MiniMaxM3VLRMSNorm`（Gemma 风格，`x*(1+w)`，`use_gemma_norm=true`） | `GlmMoeDsaRMSNorm`（标准 T5 风格 RMSNorm，weight 乘） |
| MTP（多 token 预测） | `num_nextn_predict_layers=1`（`mtp.*` 权重加载时忽略） | `num_mtp_modules=7, num_nextn_predict_layers=1`（`mtp.*` 忽略） | `num_nextn_predict_layers=1`，`index_share_for_mtp_iteration=true`（MTP 迭代复用 indexer 选择） |
| 词嵌入绑定 | `tie_word_embeddings=false` | `tie_word_embeddings=false` | `tie_word_embeddings=false`（但 modeling 中 `_tied_weights_keys` 引用 `lm_head=embed_tokens`） |

## 六、多模态（仅 MiniMax-M3）

MiniMax-M3 为 VL 模型，DeepSeek-V4-Pro 与 GLM-5.2 为纯文本。

- 视觉编码器：CLIP 风格 `clip_vision_model`，32 层，`hidden_size=1280`，`num_attention_heads=16`，`image_size=2016`，`patch_size=14`，`intermediate_size=5120`，`projection_dim=6144`。
- 位置编码：视觉侧 **3D RoPE**（`rope_mode=3d`，`rope_theta=10000`）。
- 图像 token 压缩：`patch_merge`，`spatial_merge_size=2`，`temporal_patch_size=2`；`image_seq_length=576`，支持 `dynamic_res` 与多分辨率 `image_grid_pinpoints`。
- 视频：`vision_segment_max_frames=4`，`video_token_index` 与 `image_token_index` 分离。
- 投影器：`multimodal_projector_bias=true`，`projector_hidden_size=6144`，`projector_hidden_act=gelu`，`vision_feature_select_strategy=full`。

## 七、训练/推理工程细节

| 项 | DeepSeek-V4-Pro | MiniMax-M3 | GLM-5.2 |
|---|---|---|---|
| 注意力后端 | **仅 eager**（FA2/3 与 SDPA/Flex 均关）：head_dim=512 超 FA 上限 256；SDPA 不带 sink；Flex 的 BlockMask 无法匹配压缩器运行时拼接的 KV 长度 | 支持 SDPA（`_supports_sdpa=true`），Flash/Flex 关闭，兼容 `MiniMaxAI/msa` | 支持 SDPA（`_supports_sdpa=true`），Flash/Flex 关闭（flash-mla kernel 待集成） |
| fullgraph 编译 | 关闭（压缩器缓存不兼容 StaticCache） | 开启（`_can_compile_fullgraph=true`） | 开启（`_can_compile_fullgraph=true`） |
| FP32 保留模块 | `attn_hc/ffn_hc/hc_head/sinks/position_bias/q_a_norm/kv_norm` 等（strict）；压缩器/indexer 的 kv_proj/gate_proj 非严格保留 | 无显式 FP32 保留列表 | `e_score_correction_bias`（strict），`indexer.weights_proj`（非 strict） |
| 有状态性 | `_is_stateful=true`（压缩器 rolling-window 不可回滚，禁用 assisted/prompt-lookup/contrastive 生成） | — | — |
| 被忽略的加载键 | `mtp.*` | `mtp.*` | `model.layers.78.*`（第 79 层权重忽略，疑为 MTP 层） |

## 八、GLM-5.2 特有差异点

- **IndexShare**（arXiv:2603.12201）：`indexer_types` 按 `full` + 3×`shared` 循环，`index_topk_freq=4`，shared 层通过 `prev_topk_indices` 透传复用前一 full 层的 top-k 选择，`index_skip_topk_offset=3`。3/4 的稀疏层免算 indexer，1M ctx 下 per-token FLOPs 降 2.9×。
- **MLA 潜在 KV**：`kv_lora_rank=512` 将 KV 压到潜在向量，缓存的是展开后的 K/V（代码注释：`Sparse-attention models cache the expanded K/V, not the compressed latents`）。
- **交错式 RoPE**：`rope_interleave=true`，indexer 与主注意力均用交错式（与 DeepSeek-V3.2 的非交错 half-split 不同，代码注释明确）。
- **MTP 与 IndexShare 联动**：`index_share_for_mtp_iteration=true`，MTP 投机解码迭代时复用 indexer 选择。
- **改进的 MTP**：acceptance length 最多 +20%（README 声明）。
- **1M solid context**：强调稳定维持长程任务。
- **灵活 thinking effort**：编码场景性能/延迟多档可调（README 声明）。
- 78 层（三家中最深），3 dense + 75 sparse。

## 九、关键架构差异速览

1. **KV 组织三路线**：DeepSeek-V4-Pro 用 Shared-KV MQA（1 KV 头，K=V 同张量）+ 分组低秩输出投影；MiniMax-M3 用 GQA（4 KV 头）+ per-head QK-norm；GLM-5.2 用 MLA（潜在 KV 压缩，`kv_lora_rank=512`）+ Q 低秩。
2. **长上下文三路线**：DeepSeek-V4-Pro 走"滑动窗 + KV 压缩(CSA/HCA) + 压缩域 Lightning Indexer"三态；MiniMax-M3 走"原始 key 分块 + 块级 top-k 稀疏"；GLM-5.2 走"MLA 潜在压缩 + token 级 DSA Indexer + IndexShare 跨层共享"。
3. **Indexer 跨层共享**：仅 GLM-5.2 有（3/4 层复用，FLOPs -2.9×）；DeepSeek-V4-Pro 每层独立且与压缩器耦合；MiniMax-M3 每稀疏层独立。
4. **残差**：仅 DeepSeek-V4-Pro 有 mHC 多流残差 + Sinkhorn 双随机约束（`hc_mult=4`）；另两者标准残差。
5. **MoE 规模与精度**：DeepSeek-V4-Pro 384 专家/6 激活 + fp4 + FP8 + Hash/Topk 双路由；MiniMax-M3 128/4 + bf16 + sigmoid + bias；GLM-5.2 256/8 + bf16 + sigmoid + noaux_tc（n_group=1 无分组）。
6. **激活函数**：DeepSeek-V4-Pro 与 GLM-5.2 均为 SiLU（前者带 clamp=10.0）；MiniMax-M3 为自定义 swigluoai（sigmoid 门控 + alpha=1.702）。
7. **head_dim**：DeepSeek-V4-Pro 512（致 FA 不可用，仅 eager）；MiniMax-M3 128；GLM-5.2 192（qk_head_dim 256）。
8. **RoPE**：DeepSeek-V4-Pro 双 theta（main/compress）+ 交错式；MiniMax-M3 单一大 theta + 非交错部分旋转；GLM-5.2 单一超大 theta(8e6) + 交错式。
9. **模态**：仅 MiniMax-M3 含视觉/视频分支（3D RoPE + patch_merge）。
10. **归一化**：DeepSeek-V4-Pro 与 GLM-5.2 用标准 RMSNorm（weight 乘）；MiniMax-M3 用 Gemma 风格（`x*(1+w)`）。

## 十、benchmark 对照（取自 GLM-5.2 README）

| Benchmark | GLM-5.2 | GLM-5.1 | Qwen3.7-Max | MiniMax M3 | DeepSeek-V4-Pro |
|---|---:|---:|---:|---:|---:|
| HLE | 40.5 | 31 | 41.4 | 37 | 37.7 |
| AIME 2026 | 99.2 | 95.3 | 97 | - | 94.6 |
| GPQA-Diamond | 91.2 | 86.2 | 90 | 93 | 90.1 |
| SWE-bench Pro | 62.1 | 58.4 | 60.6 | 59 | 55.4 |
| Terminal Bench 2.1 (Terminus-2) | 81.0 | 63.5 | 75 | 65 | 64 |
| FrontierSWE (Dominance) | 74.4 | 30.5 | - | - | 29.0 |
| MCP-Atlas (Public) | 76.8 | 71.8 | 76.4 | 74.2 | 73.6 |

> 完整 benchmark 与评测脚注见 `glm52_readme.md`。
