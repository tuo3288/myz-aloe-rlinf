# Module 2: Flow Matching 时间步约定修复 (`rlinf/models/embodiment/openpi/openpi_action_model.py`)

对应 patch：`../patches/openpi_action_model.patch`

---

## 问题背景

ALOE actor 使用 flow-matching 前向计算加权 loss。改动前代码使用了与 SFT（`pi0_pytorch.forward`）**相反的时间步约定**，导致传给模型的 timestep embedding 被翻转，梯度方向完全错误。

### 约定定义（SFT 侧）

```
t=0 → 干净动作 (clean action)
t=1 → 纯噪声 (pure noise)
x_t = (1 - t) * action + t * noise
```

### 改动前（错误约定）

```python
eta = torch.rand((bsize,), ...)          # 采样名为 eta
noisy_action = eta * action + (1 - eta) * epsilon   # t≈1 时接近干净动作
```

即 `eta=1 → clean`，与 SFT 相反。

### 改动后（正确约定）

```python
t = torch.rand((bsize,), ...)            # 改名为 t，语义明确
noisy_action = (1 - t) * action + t * epsilon       # t=0 → clean，t=1 → noise
```

变量从 `eta` 改为 `t`，noisy_action 混合系数对调，传入 `get_velocity` 的参数名和返回字典键名同步修改。

---

## 影响范围

仅影响 `forward_aloe_actor` 方法（ALOE actor 训练前向）。SFT / 推理路径不受影响。

---

## 对应 bug 记录

见 `ACTOR_BUG_REPORT_20260516.md`：修复此 bug 后 libero_10 成功率从 0% 恢复正常。
