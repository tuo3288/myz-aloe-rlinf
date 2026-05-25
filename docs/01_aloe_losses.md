# Module 1: ALOE 损失函数 (`rlinf/algorithms/aloe_losses.py`)

对应 patch：`../patches/aloe_losses.patch`

---

## 改动内容

文件已存在于 `main`，本次改动只涉及 **`compute_q_chunk_td_target` 函数中多步 TD target 的实现**。

### 改动前（Python for-loop）

```python
alive = torch.ones(B, 1, ...)
cum_rewards = torch.zeros(B, 1, ...)
for k in range(H):
    cum_rewards = cum_rewards + alive * (gamma**k) * rewards_h[:, k : k + 1]
    alive = alive * (1.0 - dones_h[:, k : k + 1])
```

逐步累积折扣奖励，done 之后的奖励被 `alive` mask 置零。

### 改动后（向量化 cumprod）

```python
survival = 1.0 - dones_h                                       # [B, H]
alive_mask = torch.cumprod(
    torch.cat([torch.ones(B, 1, ...), survival[:, :-1]], dim=1), dim=1
)                                                               # [B, H] exclusive cumprod
gammas = (gamma ** torch.arange(H, ...)).unsqueeze(0)          # [1, H]
cum_rewards = (alive_mask * gammas * rewards_h).sum(dim=1, keepdim=True)
alive = torch.cumprod(survival, dim=1)[:, -1:]                 # [B, 1]
```

用 exclusive cumprod 代替 for-loop，语义完全等价，但可以：
- 消除 Python 级别的 H 次串行迭代
- 全程在 GPU SRAM 上完成，避免 launch overhead
- 便于后续 `torch.compile` / JIT

### 语义验证

| 位置 k | `alive_mask[:, k]` | 含义 |
|--------|-------------------|------|
| 0      | 1（全1前缀）         | step 0 之前没有 done |
| 1      | `1 - done[0]`      | step 0 done → 屏蔽 step 1+ |
| k      | `prod_{j<k}(1-done[j])` | exclusive cumprod |

最终 `alive = prod_{j=0}^{H-1}(1-done[j])`，用于 bootstrap 项。

---

## 未改动的函数（算法核心）

这三个函数是 ALOE 算法的主体，**本次没有改动，但是理解整个算法必须知道它们的作用**：

### `compute_aloe_critic_loss(q_pred, td_target)`

Critic 的 MSE 损失。输入：online critic 的 Q 预测 `[B, K]`（K 个 ensemble head）和上面 `compute_q_chunk_td_target` 算出的 TD target `[B]`。对所有 head 求 MSE 均值。被 `_run_critic_update_step` 调用。

### `compute_aloe_advantage_and_weight(q_data, q_policy, beta, epsilon_clip)`

ALOE 的优势权重计算，对应论文公式：

```
A = Q_pess(s, a_data) - V_pi(s)
V_pi(s) = E_{a'~pi}[Q_pess(s, a')]
w = exp(clip(A / beta, -epsilon_clip, epsilon_clip))
```

`q_data` 是 replay buffer 中存的动作对应的 Q 值，`q_policy` 是当前 policy 采样的动作对应的 Q 值（用于估计 V_pi）。返回 `(weights, advantages)`，shape 均为 `[B]`。被 `_run_actor_update_step` 调用。

### `compute_aloe_actor_loss(flow_losses, weights)`

加权 flow-matching actor 损失。`flow_losses` 是模型前向返回的逐样本 flow MSE `[B, ...]`，乘以 `weights` 后取均值。flow-matching 的连续动作 log-prob 难以直接计算，ALOE 用这个加权 MSE 作为代理目标。被 `_run_actor_update_step` 调用。
