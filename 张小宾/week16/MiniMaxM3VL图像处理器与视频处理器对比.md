# MiniMax M3 VL：图像处理器 vs 视频处理器 对比详解

> **配套源码**：
> - `src/transformers/models/minimax_m3_vl/image_processing_minimax_m3_vl.py`（287 行）→ `MiniMaxM3VLImageProcessor`
> - `src/transformers/models/minimax_m3_vl/video_processing_minimax_m3_vl.py`（138 行）→ `MiniMaxM3VLVideoProcessor`
>
> 两者的产出最终都汇入同一个 `MiniMaxM3VLProcessor`（processing 文件），再喂给同一个视觉塔（vision tower）。

---

## 第 0 章 为什么要把两者放在一起讲

MiniMax M3 VL 的视觉塔是 **grid-agnostic（不关心输入网格）** 的：图像和视频共用同一套 Conv3d patch 嵌入 + 3D RoPE + 32 层编码器（代码注释原文：*"Video frames flow through the same vision pipeline as images (the tower is grid-agnostic); only the placeholder token they scatter into differs."*）。

正因为下游完全共享，**上游预处理的分工就变得微妙**：图像处理器和视频处理器既要产出"同构"的输入（让视觉塔无感），又要处理各自模态的差异（静态 vs 时间维）。这份文档就回答：**它们哪些地方一模一样，哪些地方各干各的，为什么。**

---

## 第 1 章 快速定位：两张"证件照"

| 维度 | MiniMaxM3VLImageProcessor | MiniMaxM3VLVideoProcessor |
|---|---|---|
| 基类 | `TorchvisionBackend` | `BaseVideoProcessor` |
| 默认 size | `{"shortest_edge": 3136, "longest_edge": 451584}` | `{"height": 672, "width": 672}` |
| 输入格式 | `[B, C, H, W]`（2D 图像） | `[B, T, C, H, W]`（5D 视频） |
| 输出字段 | `pixel_values` + `image_grid_thw` | `pixel_values_videos` + `video_grid_thw` |
| 网格形状 | `[1, grid_h, grid_w]`（T 恒为 1） | `[grid_t, grid_h, grid_w]`（有真实时间） |
| 共享参数 | patch_size=14, temporal_patch_size=2, merge_size=2 | 同左 |
| 尺寸控制 | min_pixels / max_pixels（面积预算） | size + min_pixels / max_pixels / total_pixels / 帧数约束 |

---

## 第 2 章 相同点（Shared）

### 2.1 共享同一个 `smart_resize` 工具函数

视频处理器**直接 import** 了图像处理器里定义的函数（video 文件第 20 行）：

```python
from .image_processing_minimax_m3_vl import smart_resize
```

`smart_resize`（image 文件 56-82 行）的规则：

1. 高和宽都必须是 `factor = patch_size × merge_size = 28` 的倍数；
2. 总像素数落在 `[min_pixels, max_pixels]` 区间内；
3. 尽量保持宽高比（先四舍五入到 28 的倍数，超上限就按比例缩小，低于下限就放大）。

**这就是"动态分辨率"的来源**：不把图固定缩放成 672×672，而是按面积预算动态算尺寸——长图、宽图都能保留原始比例。

### 2.2 完全相同的视觉参数

| 参数 | 值 | 作用 |
|---|---|---|
| `patch_size` | 14 | 空间 patch 边长 |
| `temporal_patch_size` | 2 | 时间 patch（2 帧一组） |
| `merge_size` | 2 | 空间合并（2×2 合成 1 token） |
| `resample` | BICUBIC | 缩放插值 |
| `image_mean / image_std` | CLIP 统计值 | 归一化 |
| `do_rescale` | 1/255 | 像素归一 |
| `do_convert_rgb` | True | RGB 转换 |

归一化管线都走同一套 `rescale_and_normalize`（基类方法），均值/方差都是 OpenAI CLIP 的经典值 `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`。

### 2.3 相同的输出哲学：`pixel_values + grid_thw`

两个处理器都产出"**一维 patch 序列 + 三维网格描述**"：

- `pixel_values` / `pixel_values_videos`：形状 `[num_patches, channel × temporal_patch_size × patch_size × patch_size]`，即 `[N, 3×2×14×14 = 1176]`；
- `image_grid_thw` / `video_grid_thw`：形状 `[num_inputs, 3]`，分别是 `[T, H, W]` 网格。

视觉塔拿到这两样东西，就能用 3D RoPE 重建每个 patch 的位置（T/H/W 三维坐标），完全不需要知道来源是图还是视频。

### 2.4 相同的"分组-处理-重排"批处理策略

两者都先把同形状的输入**分组**（`group_images_by_shape` / `group_videos_by_shape`），组内堆叠成 tensor 批量处理（torchvision 算子要求同形状），处理完再按原顺序**重排**（`reorder_images` / `reorder_videos`）。动态分辨率下，不同尺寸的图像/视频能混在一个 batch 里高效处理。

### 2.5 相同的核心管线顺序

```
convert RGB → (smart_resize) → rescale/normalize → patchify → 拼 grid_thw
```

---

## 第 3 章 不同点（Differences）

### 3.1 基类不同 → 能力边界不同

| | ImageProcessor | VideoProcessor |
|---|---|---|
| 基类 | `TorchvisionBackend`（torchvision.transforms.v2 后端） | `BaseVideoProcessor`（视频通用基类） |
| 额外能力 | 无 | 帧采样（`do_sample_frames`）、fps 元数据、帧数约束 |

### 3.2 输入维度不同（最根本的分歧）

- **图像**：`[B, C, H, W]`，没有时间维；
- **视频**：`[B, T, C, H, W]`，`T` 是帧数。

这个差异直接导致后续 patchify 的实现分叉（见 3.5）。

### 3.3 输出字段与 token 数量不同

| | 图像 | 视频 |
|---|---|---|
| 字段名 | `pixel_values` / `image_grid_thw` | `pixel_values_videos` / `video_grid_thw` |
| 网格 | `[1, grid_h, grid_w]` | `[grid_t, grid_h, grid_w]` |
| patch 总数 | `grid_h × grid_w` | `grid_t × grid_h × grid_w` |
| 占位符 token 数（合并后） | `(grid_h×grid_w) // 4` | `grid_t × (grid_h×grid_w) // 4` |

> 视频的 token 数是图像的 `grid_t` 倍，而 `grid_t = T // temporal_patch_size`。例如 8 帧 672×672 视频 ≈ 4 张同尺寸图像的 token 量。

### 3.4 尺寸控制策略不同

**图像处理器**（面积预算制）：size 本身就是 `min_pixels`/`max_pixels` 的容器——

```python
size = {"shortest_edge": 4 * 28 * 28, "longest_edge": 451584}   # 默认
# __init__ 里兼容旧 checkpoint 的 [672,672]
if size == [672, 672]:
    size = self.size
```

`min_pixels = 3136`，`max_pixels = 451584`（≈ 672×672 = 451584，正好等于正方形上限）。

**视频处理器**（双重约束制）：既有固定 size，又有面积/帧数预算——

```python
size = {"height": 672, "width": 672}
min_pixels = 4 * 28 * 28            # 3136（单帧最小像素）
max_pixels = 768 * 28 * 28          # 602112（单帧最大像素，约 28×28=784 个 patch）
total_pixels = int(64000 * 28 * 28 * 0.9)   # ≈ 4516 万（全视频总 patch 预算）
fps = 1.0
min_frames, max_frames = 4, 768
```

注意：视频的 `smart_resize` 是**对单帧**做的（`smart_resize(h, w, ...)`），`max_pixels` 限的是**一帧**的像素；而 `total_pixels` 限的是**整段视频**的 patch 总量（64000 个 patch，打 9 折）。

### 3.5 patchify 的实现差异（最关键的分叉）

**图像 patchify**（image 文件 163-196 行）——**expand 伪造时间维**：

```python
def patchify(self, images, patch_size, merge_size, temporal_patch_size):
    batch_size, channel, resized_height, resized_width = images.shape
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    patches = images.reshape(batch_size, channel,
                             grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)   # 排成 [B, gh/m, gw/m, m, m, C, p, p]
    flatten_patches = (
        patches.unsqueeze(6)                                  # 插入时间轴
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)  # 复制成 2 份！
        .reshape(batch_size, grid_h * grid_w,
                 channel * temporal_patch_size * patch_size * patch_size)
    )
```

**关键**：图像没有真实时间，就用 `expand` 把每个 patch **复制 2 份**，特征维变成 `3×2×14×14 = 1176`——纯粹是为了对齐 Conv3d 的输入格式。等价于"2 帧相同画面的视频"。

**视频 patchify**（video 文件 96-125 行）——**真实时间分块**：

```python
bs, grid_t, c = patches.shape[:3]              # grid_t = T // temporal_patch_size
grid_t = grid_t // temporal_patch_size
patches = patches.view(
    bs, grid_t, temporal_patch_size, c,
    grid_h // merge_size, merge_size, patch_size,
    grid_w // merge_size, merge_size, patch_size,
)
patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)   # [B, gt, gh/m, gw/m, m, m, C, tp, p, p]
flat = patches.reshape(bs, grid_t * grid_h * grid_w,
                       c * temporal_patch_size * patch_size * patch_size)
```

**关键**：视频的 `grid_t = T // 2` 是真实的时间块数，每一块里含 `temporal_patch_size=2` 帧**不同的**画面；`grid_thw` 的第一个分量记录 `grid_t`。3D RoPE 会按真实时间坐标旋转。

| | 图像 | 视频 |
|---|---|---|
| 时间维来源 | `expand` 复制（假时间） | 真实帧分组（真时间） |
| grid_t | 恒为 1 | `T // temporal_patch_size` |
| 帧信息量 | 相同画面 ×2 | 连续画面 ×2 |

### 3.6 帧数补齐逻辑（视频独有）

视频处理器在 patchify 前会先处理"帧数不是 temporal_patch_size 的倍数"的情况（video 文件 104-106 行）：

```python
if pad := -patches.shape[1] % temporal_patch_size:
    repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)   # 复制最后一帧
    patches = torch.cat([patches, repeats], dim=1)
```

比如 7 帧视频：`-7 % 2 = 1`，复制最后一帧补成 8 帧，再按 2 帧一组切成 4 个时间块。图像没有这个环节（永远 1 帧）。

### 3.7 参数集合不同（VideosKwargs vs ImagesKwargs）

| 参数 | 图像 | 视频 |
|---|---|---|
| `min_pixels` / `max_pixels` | ✅ | ✅ |
| `patch_size` / `temporal_patch_size` / `merge_size` | ✅ | ✅ |
| `total_pixels` | ❌ | ✅（整段视频预算） |
| `min_frames` / `max_frames` | ❌ | ✅（帧数约束） |
| `fps` | ❌ | ✅（采样率/元数据） |
| `do_sample_frames` | ❌ | ✅（是否帧采样，默认 False） |
| `size` 语义 | 面积预算（shortest/longest_edge） | 固定分辨率（height/width） |

### 3.8 辅助方法不同

- 图像有 `get_number_of_image_patches(height, width)`（image 文件 255-283 行），**注释明确说明是给 vLLM 用的**——没有真实图片输入时，也能根据尺寸推算出占位符 token 数；
- 视频的同类方法 `get_number_of_video_patches` 来自 `BaseVideoProcessor` 基类，本文件未重写。

### 3.9 一处"历史遗留"的默认值差异

- 图像默认 `size["shortest_edge"] = 4*28*28 = 3136`，`longest_edge = 451584`；
- 视频默认单帧 `max_pixels = 768*28*28 = 602112`（比图像的 451584 大）。

含义：视频单帧允许更大的分辨率（因为还有时间维的压缩），而图像要在单帧预算里更保守。

---

## 第 4 章 代码级对比（并排）

```
                 ImageProcessor                          VideoProcessor
─────────────────────────────────────────────────────────────────────────────
输入            [B, C, H, W]                            [B, T, C, H, W]

分组            group_images_by_shape                  group_videos_by_shape
缩放            smart_resize(h, w, factor=28,          smart_resize(h, w, factor=28,
                 min=3136, max=451584)                   min=3136, max=602112)   ← 单帧
                                                        （view 成 2D 缩放再 view 回 5D）
帧补齐          （无）                                   pad 到最后 2 的倍数（复制末帧）
归一化          rescale_and_normalize                  rescale_and_normalize
patchify        expand 复制 temporal_patch_size 份      view 出真实 grid_t 时间块
网格            [1, grid_h, grid_w]                     [grid_t, grid_h, grid_w]
输出            pixel_values + image_grid_thw           pixel_values_videos + video_grid_thw
```

**视频 resize 的小技巧**（video 文件 89-92 行）：torchvision 的 resize 只吃 4D tensor，所以视频先 `view(bs*nf, c, h, w)` 拍平成帧序列 → resize → 再 `view(bs, nf, c, rh, rw)` 恢复 5D。

---

## 第 5 章 输出形状推导实例

**例 1：一张 672×672 的图像**

1. `smart_resize(672, 672, factor=28, min=3136, max=451584)` → 672×672 = 451584 = max，恰好不缩放；
2. `grid_h = grid_w = 672/14 = 48`；
3. patchify → `pixel_values: [2304, 1176]`（48×48=2304 个 patch）；
4. `image_grid_thw: [[1, 48, 48]]`；
5. 合并后占位符数 = `2304 // 4 = 576` 个 token。

**例 2：一段 8 帧 672×672 的视频**

1. 单帧 `smart_resize(672, 672, ..., max=602112)` → 不缩放；
2. 8 帧，`-8 % 2 = 0`，无需补齐；
3. `grid_t = 8/2 = 4`，`grid_h = grid_w = 48`；
4. patchify → `pixel_values_videos: [4×48×48 = 9216, 1176]`；
5. `video_grid_thw: [[4, 48, 48]]`；
6. 合并后占位符数 = `4 × 2304 // 4 = 2304` 个 token（= 4 张图）。

> 注意 token 量的线性关系：**视频 token ≈ 图像 token × grid_t**。这也是为什么 `total_pixels`（整段视频预算）必须存在——防止长视频把序列撑爆。

---

## 第 6 章 常见坑

1. **`image_grid_thw` 的第一维恒为 1，不代表没时间信息**：图像的时间信息由预处理 `expand` 伪造，视觉塔 3D RoPE 用 `[1, h, w]` 生成坐标，时间坐标恒为 0。
2. **视频单帧 max_pixels ≠ 整段视频预算**：`max_pixels` 限单帧（602112），`total_pixels` 才限整段（约 4516 万）。混用会得出错误的 token 数估算。
3. **帧数必须是 temporal_patch_size 的倍数**：不是的话处理器会静默复制最后一帧补齐——如果你在意"真实帧数"，需自行保证输入帧数。
4. **`size` 语义完全不同**：图像的 `size` 是面积预算（shortest/longest_edge），视频的 `size` 是固定分辨率（height/width）。不要拿图像处理器的最小/最大像素去套视频参数。
5. **别改 patch_size 忘了 merge_size**：`smart_resize` 的 factor 是 `patch_size × merge_size`，两者要联动，否则切出来的网格不是整数块。
6. **vLLM 无图推理**：占位符数量用 `get_number_of_image_patches` 计算，改过 `min_pixels/max_pixels` 记得同步（vLLM 会读取该方法的计算结果）。

---

## 第 7 章 总结表

| 对比项 | ImageProcessor | VideoProcessor | 说明 |
|---|---|---|---|
| 基类 | TorchvisionBackend | BaseVideoProcessor | 能力边界不同 |
| 输入维度 | 2D [B,C,H,W] | 5D [B,T,C,H,W] | 最根本分歧 |
| 时间维 | expand 复制（假） | 真实帧分块（真） | 见 3.5 |
| 帧补齐 | 无 | 复制末帧到 2 的倍数 | 视频独有 |
| size 语义 | 面积预算 | 固定分辨率 | 别混用 |
| 面积约束 | min/max_pixels | min/max_pixels + total_pixels | 视频多整段预算 |
| 额外参数 | 无 | fps / frames / sample_frames | 视频独有 |
| 输出字段 | pixel_values / image_grid_thw | pixel_values_videos / video_grid_thw | 下游分开 scatter |
| 网格 | [1, gh, gw] | [gt, gh, gw] | 共享 3D RoPE |
| 共享 | smart_resize / 视觉参数 / 归一化 / 分组重排 | 同左 | 同一套预处理哲学 |

**一句话总结**：两者共享"动态分辨率 + 一维 patch 序列 + 三维网格"的预处理哲学，只在**时间维的建模方式**上分道扬镳——图像用 `expand` 伪造 2 帧对齐 Conv3d，视频用真实帧分组产出 `grid_t` 时间块；其余差异（尺寸约束、帧数补齐、额外参数）都是时间维带来的衍生品。
