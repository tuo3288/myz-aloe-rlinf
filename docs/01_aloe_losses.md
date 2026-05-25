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

## 未改动的函数

| 函数 | 说明 |
|------|------|
| `compute_aloe_critic_loss` | MSE(Q_pred, td_target)，未改动 |
| `compute_aloe_advantage_and_weight` | A = Q_pess - V_pi，clip-exp 权重，未改动 |
| `compute_aloe_actor_loss` | 加权 flow-matching MSE，未改动 |
