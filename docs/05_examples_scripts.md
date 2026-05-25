# Module 5: ALOE 运行脚本 (`examples/aloe/`)

对应 patch：`../patches/examples_aloe.patch`（仅保留代表性脚本）

---

## 脚本结构

ALOE outer loop 的运行入口由以下 4 个脚本组成：

```
examples/aloe/
├── outer_loop.sh         # 主循环编排（对应论文 Algorithm 1）
├── collect.sh            # 数据采集（actor rollout → replay buffer）
├── train_critic.sh       # Critic 训练（Q 函数更新）
└── train_actor.sh        # Actor 训练（加权 flow-matching）
```

单次完整实验的入口是 `jobs/run_aloe_loop_l10.sh`，它设置超参数并调用 `outer_loop.sh`。

---

## `outer_loop.sh` —— 主循环

实现论文 Algorithm 1 的 outer loop：

```
for iteration in 1..N_ITER:
    1. collect.sh       → 采集新轨迹，追加到 replay buffer
    2. train_critic.sh  → 用 replay buffer 训练 critic（N_Q 步）
    3. train_actor.sh   → 用 replay buffer 训练 actor（N_PI 步）
    4. checkpoint       → 保存当前状态，供下次迭代 resume
```

关键设计：
- 每个阶段通过 `--resume_dir` 传入上一阶段的 checkpoint 路径，实现跨阶段状态续传
- `collect.sh` 的输出（replay buffer）作为 `train_critic.sh` / `train_actor.sh` 的 `replay_preload_path`
- 支持 `--start_iter` 参数从中途某轮恢复（per-step resume）

---

## `collect.sh` —— 数据采集

- 启动 RLinf embodied actor，在 LIBERO 环境中 rollout 当前 actor policy
- 输出轨迹写入 `replay_buffer/rank_N/` 目录
- 支持 `replay_preload_path`：加载上一轮的 replay buffer 后追加新轨迹（off-policy 数据复用）

---

## `train_critic.sh` —— Critic 训练

- 加载 replay buffer（通过 `replay_preload_path`）
- 调用 `ALOEFSDPActorWorker` 的 critic update step
- 配置项：`n_critic_updates`、`critic_batch_size`、`critic_sequence_len`、`gamma`

---

## `train_actor.sh` —— Actor 训练

- 加载 replay buffer，同时加载当前 critic 权重
- 计算 ALOE 优势权重：`w = exp(clip(A/beta, -eps, eps))`
- 加权 flow-matching loss backward
- 配置项：`n_actor_updates`、`actor_batch_size`、`beta`

---

## `jobs/run_aloe_loop_l10.sh` —— 完整实验入口（最简单）

设置 libero_10 实验的超参数并调用 `outer_loop.sh`，是最简洁的端到端入口：

```bash
bash outer_loop.sh \
  --task libero_10 \
  --n_iter 10 \
  --n_collect 50 \
  --n_critic_updates 200 \
  --n_actor_updates 50 \
  --beta 1.0 \
  ...
```
