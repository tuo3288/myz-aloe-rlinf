# Module 6: 性能优化数值正确性测试 (`tests/unit_tests/test_aloe_perf.py`)

对应 patch：`../patches/test_aloe_perf.patch`

**新增文件**（main 上不存在）。

---

## 目的

本文件专门验证两个性能优化（TD target 向量化 + 融合 VLM context 提取）在改动后行为与改动前一致。测试分三组：

---

## T1：TD target 向量化数值等价性

验证 `compute_q_chunk_td_target`（vectorized cumprod）与原始 for-loop 实现输出差异 < 1e-5。

文件内保留了原始 for-loop 的完整副本作为 ground truth（`_td_target_loop_reference`），不依赖 main 分支代码。

测试用例覆盖：

| 测试 | 参数 / 场景 |
|------|------------|
| `test_td_target_vectorized_matches_loop` | parametrize: gamma∈{0.99,1.0,0.95}，(B,H)∈{(32,10),(1,10),(64,1)} |
| `test_td_target_all_done_zero` | 全程无 done，alive 始终为 1，完整 bootstrap |
| `test_td_target_done_at_step_zero` | 第 0 步立即 done，只有 r_0 贡献，bootstrap 被 mask 掉 |
| `test_td_target_single_step_branch_unchanged` | chunk_len=1 走单步分支，验证没有引入回归 |

---

## T2：`_extract_critic_context_pair` 单次 model 调用

验证融合前向（Patch D）确实只调用了 1 次 model，且两个 batch 的输出正确分割回各自的 B 维。

使用 `unittest.mock.MagicMock` 模拟 model，通过 `call_log` 记录每次调用时的 batch size，断言：
- 调用次数 == 1
- 调用时 batch size == 2B
- `ctx_a` 和 `ctx_b` 各自的 shape[0] == B

---

## T3：shape 不匹配时 fallback

当两个 batch 的 `tokenized_prompt` 长度不同（32 vs 48）时，`_extract_critic_context_pair` 应该退化为两次独立调用，而不是报错。

验证：
- 调用次数 == 2
- 每次调用 batch size == B

---

## 运行方式

```bash
pytest tests/unit_tests/test_aloe_perf.py -v
```
