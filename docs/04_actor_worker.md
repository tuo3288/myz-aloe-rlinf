# Module 4: ALOE Actor Worker 优化 (`rlinf/workers/actor/fsdp_aloe_actor_worker.py`)

对应 patch：`../patches/fsdp_aloe_actor_worker.patch`

---

## 概述

本次对 `ALOEFSDPActorWorker` 的改动包含三个独立的功能点：

| 功能 | 作用 |
|------|------|
| Patch A：Replay 脏位 + REPLAY_REF.txt | 避免每步都重写完整 replay buffer |
| Patch B：Critic LR Scheduler 重启选项 | 支持 checkpoint 加载后重置调度器 |
| Patch C：1-ahead prefetch | GPU backward 时异步预取下一批数据 |
| Patch D：融合 VLM context 提取 | 两个 critic 前向 → 一个批量前向 |

---

## Patch A：Replay 脏位 + REPLAY_REF.txt 链式引用

### 背景

ALOE outer loop 中，纯 critic 训练步和纯 actor 训练步不会修改 replay buffer 内容，但原代码在每次 `save_checkpoint` 时都完整重写 replay buffer（可能 GB 级别），产生大量冗余 I/O。

### 实现

新增两个实例变量：

```python
self._replay_dirty: bool = False          # 自上次加载/保存以来 buffer 是否被修改
self._last_replay_load_path: Optional[str] = None  # 当前内存状态来源的磁盘路径
```

**触发 dirty=True 的操作：**
- `preload_replay_buffer`（加载后视为 dirty，因为需要首次持久化到新路径）
- `receive_trajectories`（新轨迹加入 buffer）

**`save_checkpoint` 逻辑：**
```python
if self._replay_dirty:
    self.replay_buffer.save_checkpoint(buffer_save_path)
    self._replay_dirty = False
else:
    # 只写一个 REPLAY_REF.txt 指向真实数据位置
    with open(ref_path, "w") as f:
        f.write(self._last_replay_load_path or "")
```

**`load_checkpoint` 逻辑：**
如果目录中没有 `.pt` 文件但有 `REPLAY_REF.txt`，则读取其中的路径并从那里加载（链式追溯）。

### 效果

纯训练步的 checkpoint 保存从 O(buffer大小) 降为写一个几十字节的文本文件。

---

## Patch B：Critic LR Scheduler 重启

### 背景

实验中有时希望 critic 在 resume 后以新的学习率 schedule 重新开始（cosine restart），而不是继续之前的 decay 曲线。

### 实现

在 config 中增加可选项：

```yaml
actor:
  critic_optim:
    reset_scheduler_on_load: true  # 默认 false
```

`load_checkpoint` 时检查此 flag：若为 true，跳过 `qf_lr_scheduler.load_state_dict`，调度器从 `last_epoch=0` 重新开始。

---

## Patch C：1-ahead Prefetch

### 背景

GPU 执行 backward pass 时，主线程空闲，可以利用这段时间提前从 replay buffer 采样下一批数据（涉及磁盘读取 + 数组切片）。

### 实现

新增 `ThreadPoolExecutor(max_workers=1)` 及四个辅助方法：

```python
def _start_critic_prefetch(self)   # 提交 sample_sequence_chunks 到后台
def _get_critic_batch(self)        # 取结果（已完成则直接返回，否则同步等待）
def _start_actor_prefetch(self)    # 同上，用于 sample_chunks
def _get_actor_batch(self)
def _drain_prefetch_future(self, fut)  # 强制等待并丢弃结果（边界清理用）
```

**调用时序：**
- critic update step 结束后（soft update 时）→ `_start_critic_prefetch()`
- 下一次 critic step 开始时 → `_get_critic_batch()`（通常已完成）
- critic block 结束、切换到 actor block 前 → `_drain_prefetch_future` 清理
- actor update step 结束后 → `_start_actor_prefetch()`
- 下一次 actor step 开始时 → `_get_actor_batch()`

**注意：** `torch.Generator`（random_generator）只在主线程操作，预取线程不访问它，避免竞态。`_drain_prefetch_future` 在 critic/actor block 边界确保后台线程完成。

---

## Patch D：融合 VLM Context 提取

### 背景

Critic update step 需要对 **当前状态** 和 **下一状态** 各提取一次 VLM context（即过一遍 pi0 prefix cache），原实现是两次串行的 `ALOE_CRITIC` forward。

### 实现

新方法 `_extract_critic_context_pair(forward_inputs_a, forward_inputs_b)`：

```python
# 形状匹配时，拼接 batch 做一次前向
merged = {k: torch.cat([a[k], b[k]], dim=0) for k in common_keys}
context = self.model(forward_type=ALOE_CRITIC, forward_inputs=merged)
ctx_a = {k: v[:B_a] for k, v in context.items()}
ctx_b = {k: v[B_a:B_a+B_b] for k, v in context.items()}
```

形状不匹配时（如 tokenized_prompt 长度不同），fallback 到两次串行调用。

### 效果

VLM prefix cache 计算从 2 次降为 1 次/critic step，减少约 50% 的 VLM forward overhead（适用于 H800，batch_size*2 在 VRAM 内）。

---

## 其他小改动

- `preload_replay_buffer` 中修复了 replay preload path 解析逻辑：原来只处理"路径不存在"的情况，现在也处理"路径是父目录（无 metadata.json）"的情况，自动拼接 `rank_N` 子目录。
- `preload_replay_buffer` 新增 `replay_success_ratio` 支持，加载后可调用 `filter_by_success_ratio`。
