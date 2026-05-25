# Module 3: Replay Buffer 并行加载与成功率过滤 (`rlinf/data/replay_buffer.py`)

对应 patch：`../patches/replay_buffer.patch`

---

## 改动 1：并行磁盘加载（cache miss 路径）

### 背景

`sample_sequence_chunks` 和 `sample_chunks` 都有一段"cache miss 路径"：当请求的轨迹不在内存缓存中时，需要从 BeeGFS 分布式文件系统逐个加载 `.pt` 文件。原实现串行加载，每个文件的 I/O 阻塞主线程。

### 改动

新增辅助方法：

```python
def _load_and_flatten_traj(self, trajectory_id: int) -> dict:
    mwid = self._trajectory_index[trajectory_id]["model_weights_id"]
    trajectory = self._load_trajectory(trajectory_id, mwid)
    return self._flatten_trajectory(trajectory)
```

在两处 cache miss 路径均改为：

```python
load_futures = {
    tid: self._checkpoint_executor.submit(self._load_and_flatten_traj, tid)
    for tid in miss_traj_ids
}
for tid in miss_traj_ids:
    flat_trajectory = load_futures[tid].result()
```

用已有的 `_checkpoint_executor`（ThreadPoolExecutor）并发提交所有 miss 的轨迹加载，保持结果收集顺序确定性（按 `miss_traj_ids` 顺序 `.result()`）。

### 效果

cache miss 时多个文件 I/O 并发进行，减少等待时间，在 BeeGFS 高延迟场景下收益明显。

---

## 改动 2：`filter_by_success_ratio`

### 背景

ALOE 的 BC-good-only 变体和成功率扫描实验需要控制 replay buffer 中成功/失败轨迹的比例。

### 接口

```python
def filter_by_success_ratio(self, target_success_ratio: float, rng=None) -> None
```

**判断成功的标准**：如果某条轨迹的任意 reward 值等于 0（RLinf 的稀疏奖励约定：成功=0，失败≠0），则视为成功轨迹。

> ⚠️ 注意：这个判断逻辑依赖具体的 reward 编码方式，在 libero 环境中是正确的，但不具备通用性。

### 实现逻辑

1. 遍历 `_trajectory_id_list`，每条轨迹加载 `.pt` 文件检查 rewards
2. 计算需要保留的失败轨迹数：`n_fail_keep = n_success * (1 - ratio) / ratio`
3. 随机采样保留 `n_fail_keep` 条失败轨迹，其余从索引中删除
4. 更新 `size`、`_total_samples`，清空 flat cache

**只操作内存索引，不修改磁盘文件。**

---

## 未改动部分

`add_trajectories`、`save_checkpoint`、`load_checkpoint` 等核心接口无变化。
