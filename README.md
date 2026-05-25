# RLinf × ALOE 实现改动记录

## 应用 patch

在 RLinf 仓库根目录执行：

```bash
git apply patches/aloe_losses.patch
git apply patches/openpi_action_model.patch
git apply patches/replay_buffer.patch
git apply patches/fsdp_aloe_actor_worker.patch
git apply patches/examples_aloe.patch
git apply patches/test_aloe_perf.patch
```

---

本项目记录在 RLinf 框架中实现 **ALOE（Actor-Critic Offline-to-Online RL with Language Embedding）** 算法时所做的代码改动，供代码审阅使用。

> 改动来源：RLinf 仓库分支 `clean-orchestration-from-zero`，相对 `main` 分支的 diff。

---

## 改动总览

| 文件 | 类型 | 核心内容 |
|------|------|----------|
| `rlinf/algorithms/aloe_losses.py` | 性能优化 | 多步 TD target 向量化（消除 H-step Python for-loop） |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | **Bug 修复** | Flow matching 时间步约定反转修复（关键） |
| `rlinf/data/replay_buffer.py` | 功能 + 性能 | 并行磁盘加载；`filter_by_success_ratio` |
| `rlinf/workers/actor/fsdp_aloe_actor_worker.py` | 功能 + 性能 | 4 个优化 patch（见下） |
| `examples/aloe/` 脚本 | 运行编排 | Outer loop 实现（对应论文 Algorithm 1） |
| `tests/unit_tests/test_aloe_perf.py` | **新增测试** | TD target 向量化 + 融合 forward 的数值正确性验证 |

---

## Bug 修复（最重要）

**文件**：`rlinf/models/embodiment/openpi/openpi_action_model.py`

Flow matching noisy action 的时间步混合系数写反了，导致 ALOE actor 的 timestep embedding 与 SFT 训练时相反，梯度方向完全错误，表现为 libero_10 成功率归零。

```python
# 修复前（错误）：eta=1 → clean action
noisy_action = eta * action + (1 - eta) * epsilon

# 修复后（正确）：t=0 → clean, t=1 → pure noise
noisy_action = (1 - t) * action + t * epsilon
```

详见 `docs/02_openpi_action_model.md`。

---

## 功能改动

### 1. 多步 TD target 向量化

`rlinf/algorithms/aloe_losses.py`：`compute_q_chunk_td_target` 中的 H-step 累积折扣奖励由 Python for-loop 改为 `torch.cumprod` 实现的 exclusive prefix product，语义不变，性能提升。

→ `docs/01_aloe_losses.md`

### 2. Replay Buffer 并行加载 + 成功率过滤

`rlinf/data/replay_buffer.py`：
- cache miss 路径改为用 `ThreadPoolExecutor` 并发加载多条轨迹
- 新增 `filter_by_success_ratio`：按目标成功率比例删除失败轨迹

→ `docs/03_replay_buffer.md`

### 3. Actor Worker 四项优化

`rlinf/workers/actor/fsdp_aloe_actor_worker.py`：

| Patch | 内容 |
|-------|------|
| A | Replay 脏位 + REPLAY_REF.txt：只在 buffer 真正变化时才重写磁盘 |
| B | Critic LR Scheduler 重启选项：`reset_scheduler_on_load` |
| C | 1-ahead prefetch：GPU backward 期间异步预取下一批数据 |
| D | 融合 VLM context 提取：两次 ALOE_CRITIC forward → 一次批量 forward |

→ `docs/04_actor_worker.md`

### 4. Outer Loop 脚本

`examples/aloe/outer_loop.sh` 及配套脚本实现了论文 Algorithm 1 的 collect → critic train → actor train 迭代循环，支持中途断点续跑。

→ `docs/05_examples_scripts.md`

### 5. 正确性测试（新增文件）

`tests/unit_tests/test_aloe_perf.py` 验证两个性能优化改动（TD target 向量化 + 融合 VLM context 提取）的数值正确性，覆盖边界情况（全零 done、首步 done、shape mismatch fallback）。

→ `docs/06_test_aloe_perf.md`

---

## 目录结构

```
aloe-rlinf/
├── README.md
├── docs/
│   ├── 01_aloe_losses.md           # TD target 向量化
│   ├── 02_openpi_action_model.md   # Flow matching bug 修复
│   ├── 03_replay_buffer.md         # 并行加载 + 成功率过滤
│   ├── 04_actor_worker.md          # Actor worker 4 项优化
│   ├── 05_examples_scripts.md      # Outer loop 脚本说明
│   └── 06_test_aloe_perf.md        # 正确性测试说明
└── patches/
    ├── aloe_losses.patch
    ├── openpi_action_model.patch
    ├── replay_buffer.patch
    ├── fsdp_aloe_actor_worker.patch
    ├── examples_aloe.patch          # 仅保留代表性脚本
    └── test_aloe_perf.patch
```
