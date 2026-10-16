# CausalBridge 调控伪时间模块 — 开发备忘录

> **状态**: 设计阶段，待实现
> **目标**: 基于 TF→peak→gene 调控传播一致性推断伪时间，替代纯细胞相似度方法
> **文件**: `CausalBridge/regulatory_pseudotime.py`（新建，不修改现有 `infer_pseudotime`）

---

## 1. 动机

现有伪时间方法（DPT/Palantir）完全基于**细胞相似度**：

```
细胞 i ≈ 细胞 j  →  i 和 j 在伪时间上相近
```

问题：
- 细胞相似度 ≠ 调控状态相似度。两个细胞可以 transcriptome-wide 相近，但在某个 TF→peak→gene 通路上处于完全不同的调控阶段
- 基于表达相似度的伪时间不能直接反映"调控事件的因果顺序"

**CausalBridge 的核心优势**是建模了 TF→peak→gene 三层因果调控网络。这个网络本身就编码了调控事件之间的先后关系——为什么不用它来推断伪时间？

**核心假设**：如果状态 i 的 TF 表达能解释状态 j 的 peak 开放模式（通过 motif），并且状态 j 的 peak 能解释其自身的 gene 表达，那么存在从 i 到 j 的调控过渡。

---

## 2. Peak-Centered Regulatory Transition Axis

### 2.1 关键洞察

在 TF→peak→gene 三层网络中，**peak 是承上启下的机械中介**：

```
TF (上游) ──motif──▶ Peak (中间状态) ──可及性──▶ Gene (下游)
```

- Peak 的开放状态记录了"哪些 TF 在此处结合过"
- Peak 的可及性变化是 TF 活性的**滞后读out**
- Gene 表达变化是 peak 可及性变化的**进一步滞后**

因此，**peak 状态是理想的调控时间锚点**：它同时编码了过去的 TF 活性和未来的 gene 表达潜力。

### 2.2 伪时间定义

不是"细胞 A 和细胞 B 在发育轴上多接近"，而是：

> **调控伪时间 = 细胞在 TF→peak→gene 调控传播链上的位置**

一个"早期"状态的特征：
- TF 已表达 → peak 尚未完全开放 → gene 尚未高表达
- TF→peak 一致性高，peak→gene 一致性低

一个"晚期"状态的特征：
- TF 表达 → peak 已开放 → gene 已高表达
- TF→peak 一致性高，peak→gene 一致性也高

---

## 3. 算法流程

```
输入:
  - rna_adata: AnnData (cells × genes, 含 TF 表达)
  - atac_adata: AnnData (cells × peaks)
  - motif_map: dict {peak_id: [tf_names]}   ← 来自 kinetics._scan_peaks_with_jaspar
  - clustering: Leiden 聚类结果 (来自 preprocess.cluster_cells)
  - n_bins: 分箱数上限 (默认 50)

输出:
  - pseudotime_df: DataFrame 列=[pseudotime, bin, context_id]
  - transition_matrices: dict {context_id: np.ndarray (n_bins × n_bins)}
  - regulatory_coordinate: 归一化的调控坐标 (0~1)

流程:
  Step 0: 构建 CausalBridge 上下文 (Leiden → PAGA merge)
  Step 1: 对每个 context，用 DPT 建立初始 bin 排序（弱先验）
  Step 2: 高斯软分箱聚合三层组学信号 (复用 _bin_matrix)
  Step 3: 构建候选转移图 (kNN + forward constraint)
  Step 4: 计算 transition score S(i→j) 对每条候选边
  Step 5: 构建转移矩阵 T = softmax_j(S(i→j))
  Step 6: 从 T 的平稳分布提取调控坐标
  Step 7: 映射回细胞级别
```

---

## 4. Step 0: 构建 CausalBridge 上下文

### 4.1 问题

Leiden 聚类（高 resolution）会把连续分化谱系切碎。直接在每个碎片内做伪时间推断会丢失跨簇的调控连续性。

### 4.2 PAGA 反碎片化

```
Leiden (resolution=0.8~1.5) → 碎片化的 cluster
    ↓
PAGA 连通性分析 → 识别哪些碎片属于同一谱系
    ↓
合并 PAGA-connected 的碎片 → CausalBridge context
```

PAGA 的作用**不是**引入拓扑约束防止跨谱系跳跃，而是**修复 Leiden 过度碎片化**——把本应连续的谱系重新拼接起来。

### 4.3 Context 定义

```python
# CausalBridge context = PAGA 合并后的细胞群
# 每个 context 内部独立运行调控伪时间推断
# 也支持用户提供的手动注释作为 context (cluster_key)
```

---

## 5. Step 1: DPT 初始 bin 排序

在每个 context 内部：

1. 用 DPT 计算基于细胞相似度的初始伪时间（弱先验）
2. 用 `pd.qcut` 对 DPT 值做分位数切割 → n_bins 个 bin
3. Bin 的**排序**作为弱先验（i < j 意味着 i 在 DPT 轴上早于 j）

注意：DPT 排序只用于定义 bin 的顺序约束（forward prior），transition score 的核心来自调控一致性。

---

## 6. Step 2: 高斯软分箱聚合信号

**不引入 metacell 概念**，直接复用现有的 `_bin_matrix` + `_bin_expression`：

```python
# 复用 kinetics.py 中的现有函数
from .kinetics import _bin_matrix, _adaptive_rebin

# (1) 自适应分箱: 决定 bin 数和细胞分配
pseudotime_df = _adaptive_rebin(pseudotime_df, n_cells, n_bins, min_cells_per_bin)

# (2) 高斯软分箱: 对每个 bin 聚合加权平均
#     - TF 表达矩阵: (n_bins × n_tfs)      → _bin_expression(rna_adata, smooth="gaussian")
#     - Peak 可及性矩阵: (n_bins × n_peaks) → _bin_expression(atac_adata, smooth="gaussian")
#     - Gene 表达矩阵: (n_bins × n_genes)   → _bin_expression(rna_adata, smooth="gaussian")
```

高斯软分箱的优势：
- 每个细胞可以部分属于多个相邻 bin → 平滑噪声
- 1D 排序（沿 DPT 轴）在单个 context 内是合理的（同一个 PAGA 连续谱系内）
- 无需构造额外的扁平化/聚类步骤
- 与 `kinetics.py` / `granger.py` / `perturbation.py` 下游模块完全一致

---

## 7. Step 3: 构建候选转移图

### 7.1 节点定义

每个 bin 是一个节点。节点 k 的状态向量：

```python
state[k] = {
    "tf_expr":    np.ndarray (n_tfs,),      # 高斯加权平均 TF 表达
    "peak_acc":   np.ndarray (n_peaks,),     # 高斯加权平均 peak 可及性
    "gene_expr":  np.ndarray (n_genes,),     # 高斯加权平均 gene 表达
    "dpt_median": float,                     # bin 内细胞 DPT 值中位数
}
```

### 7.2 候选边生成

不是全连接（O(n²)），而是 kNN + forward constraint：

```python
# (a) 在 joint latent space 中做 kNN
#     将 [tf_expr, peak_acc, gene_expr] concat → PCA → top 20 PCs → kNN
#     每个 bin 连接到 k 个最近邻 (k = min(10, n_bins // 5))

# (b) Forward constraint: 只保留 i → j where dpt_median[i] <= dpt_median[j]
#     DPT 排序作为弱先验——我们不禁止任何方向的调控推理，
#     但候选边的生成只考虑前向
```

---

## 8. Step 4: Transition Score S(i→j)

**核心**：S(i→j) 评估从 bin i 到 bin j 的调控过渡有多"合理"。

### 8.1 评分公式

```
S(i→j) = w1 × similarity(i,j)
        + w2 × forward_time_prior(i,j)
        + w3 × tf_peak_consistency(i,j)      ← 核心
        + w4 × peak_gene_consistency(i,j)
        + w5 × chain_completeness(i,j)
        - w6 × backward_penalty(i,j)
```

默认权重: `[w1, w2, w3, w4, w5, w6] = [0.10, 0.05, 0.35, 0.25, 0.15, 0.10]`

### 8.2 各分量定义

#### (a) similarity(i, j) — 状态相似度

```python
# Cosine similarity in joint latent space
# 确保相邻 bin 在多层组学空间中确实相近
similarity = cosine_similarity(state[i].concat_vector, state[j].concat_vector)
# 归一化到 [0, 1]
```

#### (b) forward_time_prior(i, j) — 前向时间先验

```python
# DPT 排序先验: i 早于 j 则加分
if dpt_median[i] < dpt_median[j]:
    forward_prior = min(1.0, (dpt_median[j] - dpt_median[i]) / typical_step)
else:
    forward_prior = 0.0
# typical_step = DPT range / n_bins
```

#### (c) tf_peak_consistency(i, j) — TF→Peak 调控一致性 ★ 核心

**逻辑（与现有 `fit_rna_to_atac` 一致）**：

> 给定状态 j 的 peak 开放模式 → 通过 motif 反推需要的 TF → 检查这些 TF 在状态 i 中是否表达

```python
def compute_tf_peak_consistency(i, j, state, motif_map):
    """
    逆向推断: peak(j) → required TFs → check expression in state(i)

    与现有模型的逻辑一致：
      - kinetics.fit_rna_to_atac: 用 Ridge 学习 TF(t-lag) → peak(t) 的映射
      - 这里做的是: 已知 peak(j) 需要哪些 TF, 检查 state(i) 是否有这些 TF
    """
    peaks_open_j = state[j].peak_acc > threshold  # 在 j 中开放的 peak
    tf_votes = defaultdict(float)

    for peak in peaks_open_j:
        required_tfs = motif_map.get(peak, [])
        for tf in required_tfs:
            # Vote: TF 在 state i 中的表达水平 × binding confidence
            tf_votes[tf] += state[i].tf_expr[tf] * motif_confidence[peak][tf]

    if len(tf_votes) == 0:
        return 0.0

    # 归一化: 被覆盖的 TF 比例 × 平均表达
    coverage = len(tf_votes) / len(set(tf for peak in peaks_open_j
                                       for tf in motif_map.get(peak, [])))
    mean_expr = np.mean(list(tf_votes.values()))
    return coverage * mean_expr
```

#### (d) peak_gene_consistency(i, j) — Peak→Gene 调控一致性

```python
# 检查在状态 j 中开放的 peak 是否与状态 j 中高表达的 gene 存在
# Granger 因果边 (复用 granger_test 的结果)

# 从 granger_edges (预计算) 获取 (peak, gene) pairs
# consistency = fraction of (open_peak_j, high_gene_j) pairs that have Granger edges
```

#### (e) chain_completeness(i, j) — 调控链完整度

```python
# 检查从 i 到 j 是否存在完整的 TF→peak→gene 链条:
#   ∃ TF in state(i) → motif → peak in state(j) → Granger → gene in state(j)
# 
# chain_completeness = (# 完整链条) / (# 可能的链条)
# 
# 一个"可能链条" = (TF ∈ expressed(i), peak ∈ open(j) with TF motif)
```

#### (f) backward_penalty(i, j) — 逆向惩罚

```python
# 如果 i 早于 j 但 tf_peak_consistency(i,j) < tf_peak_consistency(j,i)
# 说明调控方向可能是反的，施加惩罚
if dpt_median[i] < dpt_median[j]:
    forward_consistency = tf_peak_consistency(i, j)
    backward_consistency = tf_peak_consistency(j, i)
    if backward_consistency > forward_consistency:
        penalty = backward_consistency - forward_consistency
    else:
        penalty = 0.0
else:
    penalty = 0.0
```

---

## 9. Step 5: 构建转移矩阵

```python
# 对于每个 bin i:
#   候选目标 bins = {j: (i,j) 是候选边}
#   对所有候选 j 计算 S(i→j)
#   self_loop_score = S(i→i) = max(similarity baseline, 0.5)

# Softmax 归一化
T[i, j] = exp(S(i→j) / tau) / sum_k(exp(S(i→k) / tau))

# tau: temperature (默认 0.1) — 越小越 sharp (确定性转移)
# self_loop: 对角线 = self_loop_score, 允许状态停留
```

输出：`T` 是 (n_bins × n_bins) 的行随机矩阵（每行和为 1）。

---

## 10. Step 6: 提取调控坐标

### 10.1 平稳分布

```python
# 计算 T 的平稳分布 π
# π = π @ T, 用幂迭代
eigenvalues, eigenvectors = np.linalg.eig(T.T)
stationary = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
stationary = stationary / stationary.sum()
```

### 10.2 调控坐标

```python
# 对 bins 按以下规则排序 (不是 DPT 排序):

# (a) 每个 bin 的"出度质量" = mean(S(i→j) for j in forward_candidates)
# (b) 每个 bin 的"入度质量" = mean(S(j→i) for j in backward_candidates)
# (c) net_flow = out_quality - in_quality

# 高 net_flow → 调控源头 (TF 驱动状态变化)
# 低 net_flow → 调控终端 (gene 表达响应)

# 调控坐标 = 对 net_flow 做 min-max 归一化
regulatory_coordinate = (net_flow - net_flow.min()) / (net_flow.max() - net_flow.min())
```

### 10.3 与 DPT 的关系

```python
# 输出对比:
#   DPT 排序:      bin_0 → bin_1 → bin_2 → ... → bin_N  (表达相似度驱动)
#   调控坐标:      bin_3 → bin_0 → bin_7 → bin_1 → ...   (调控因果驱动)
#
# 相关性预期: 在分化谱系中两者应高度相关 (发育过程本身是按调控顺序展开的)
# 差异分析: 如果某个区段两者差异大，说明调控顺序与表达相似度脱钩
#          (可能是: 多个细胞类型共享 TF 但不同 peak/gene 响应)
```

---

## 11. Step 7: 映射到细胞级别

```python
# 高斯软分箱权重矩阵 W: (n_cells × n_bins)
#   W[c, b] = 细胞 c 属于 bin b 的权重 (高斯核)
#   sum_b W[c, :] = 1 (归一化)

# 细胞级别调控坐标
cell_regulatory_coordinate = W @ bin_regulatory_coordinate

# 输出到 pseudotime_df
pseudotime_df["regulatory_pseudotime"] = cell_regulatory_coordinate
pseudotime_df["bin"] = argmax(W, axis=1)  # 用调控坐标重排 bin 后分配
```

---

## 12. 函数签名

```python
# regulatory_pseudotime.py

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from anndata import AnnData

def infer_regulatory_pseudotime(
    rna_adata: AnnData,
    atac_adata: AnnData,
    motif_map: Dict[str, list],           # {peak_id: [tf_names]}
    granger_edges: pd.DataFrame,          # Granger 检验结果 (peak, gene, ...)
    clustering: Optional[pd.Series] = None,  # 细胞聚类标签, None=全局
    n_bins: int = 50,
    min_cells_per_bin: int = 10,
    n_neighbors_knn: int = 10,
    score_weights: Optional[Dict[str, float]] = None,
    temperature: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Peak-centered regulatory pseudotime inference.

    Returns:
        pseudotime_df: DataFrame with columns [pseudotime, bin, context_id, regulatory_coordinate]
        transition_matrices: dict {context_id: (n_bins, n_bins) transition matrix}
    """
    ...

# ---- 内部函数 ----

def _build_causalbridge_contexts(
    adata: AnnData,
    cluster_key: str,
    min_cells: int = 50,
) -> Dict[str, np.ndarray]:
    """Leiden → PAGA → 合并碎片 → CausalBridge contexts."""
    ...

def _compute_transition_score(
    i: int,
    j: int,
    states: Dict[int, dict],
    motif_map: Dict[str, list],
    granger_edges: pd.DataFrame,
    dpt_values: np.ndarray,
    weights: Dict[str, float],
) -> float:
    """Compute S(i→j) transition score."""
    ...

def _tf_peak_consistency(
    i: int, j: int,
    states: Dict[int, dict],
    motif_map: Dict[str, list],
) -> float:
    """
    Reverse inference: peak(j) → required TFs → check expression in state(i).
    Matches existing fit_rna_to_atac logic direction.
    """
    ...

def _build_transition_matrix(
    scores: np.ndarray,
    temperature: float = 0.1,
) -> np.ndarray:
    """Softmax row-normalize scores → transition matrix."""
    ...

def _extract_regulatory_coordinate(
    T: np.ndarray,
    states: Dict[int, dict],
) -> np.ndarray:
    """Extract regulatory coordinate from transition matrix."""
    ...
```

---

## 13. 与现有代码的关系

| 现有函数/模块 | 新模块中的使用 |
|-------------|-------------|
| `kinetics._adaptive_rebin()` | 定义 bin 数 + 分配细胞到 bin |
| `kinetics._bin_matrix(smooth="gaussian")` | 底层 Gaussian 加权聚合 |
| `kinetics._bin_expression(smooth="gaussian")` | 聚合每个 bin 的 TF/peak/gene 信号 |
| `kinetics._scan_peaks_with_jaspar()` | 提供 motif_map |
| `preprocess.cluster_cells()` | 提供 Leiden 聚类 |
| `granger.granger_test()` | 提供 Granger edges (peak-gene 因果边) |
| `granger.infer_pseudotime()` | **不修改** — 新模块独立，可选切换 |

### 配置扩增 (config.yaml)

```yaml
pseudotime:
  # ... 现有参数不变 ...
  
  # --- 调控伪时间 (新增) ---
  regulatory_pseudotime:
    enabled: false                        # true=使用调控伪时间替代 DPT
    score_weights:                        # Transition score 权重
      similarity: 0.10
      forward_prior: 0.05
      tf_peak_consistency: 0.35
      peak_gene_consistency: 0.25
      chain_completeness: 0.15
      backward_penalty: 0.10
    temperature: 0.1                      # Softmax temperature
    n_neighbors_knn: 10                   # 候选边 kNN
    paga_connectivity_threshold: 0.1      # PAGA 合并阈值
```

---

## 14. 实现优先级

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | `_tf_peak_consistency()` — 核心评分函数 | motif_map |
| P0 | `_compute_transition_score()` — 完整评分 | 上述 |
| P1 | `_build_causalbridge_contexts()` — PAGA 合并 | scanpy.tl.paga |
| P1 | `_build_transition_matrix()` — 转移矩阵 | scores |
| P1 | `_extract_regulatory_coordinate()` — 坐标提取 | T |
| P2 | `infer_regulatory_pseudotime()` — 主入口 | 上述全部 |
| P2 | 高斯软分箱集成（复用现有函数） | kinetics.py |
| P3 | config.yaml 集成 + run.py 切换逻辑 | 上述全部 |

---

## 15. 验证策略

### 15.1 单元验证

```python
# 在已知的分化谱系上测试 (如造血分化)
# 比较 regulatory_pseudotime vs DPT pseudotime:
#   预期: 两者高度相关 (Spearman r > 0.7)
#   但如果一致度 100%，说明调控信息没有提供增量

# 关键验证: 调控伪时间与已知 marker gene 表达的相关性
#   预期: 调控伪时间应与 stage-specific TFs 的表达更相关
```

### 15.2 消融验证

```python
# 消融 1: 去掉 tf_peak_consistency → 退化为 DPT 排序
#   预期: 与 DPT 几乎一致
#
# 消融 2: 去掉 forward_time_prior → 允许反向转移
#   预期: T 矩阵的双向性增加，平稳分布更均匀
#
# 消融 3: 高斯软分箱 vs 硬分箱
#   预期: 软分箱产生的 T 矩阵更平滑，噪音更小
```

### 15.3 下游影响验证

```python
# 将 regulatory_pseudotime 喂给现有的 Granger + kinetics + perturbation 流程
# 对比与 DPT 伪时间的结果差异:
#   - Granger 边数量和质量
#   - 传递函数 R²
#   - 扰动预测的方向一致率
```

---

## 16. 注意事项

1. **独立模块，不替换现有逻辑**：`infer_regulatory_pseudotime` 是 `infer_pseudotime` 的替代选项，通过 config 切换
2. **高斯软分箱一致性**：确保与 `kinetics.py` / `granger.py` 中的分箱逻辑完全一致（复用而非重写）
3. **PAGA 依赖**：如果 scanpy PAGA 不可用，回退为直接使用 Leiden 聚类（不做合并）
4. **Motif map 获取**：`_scan_peaks_with_jaspar` 已在现有流程中执行，新模块直接接收结果
5. **Computational cost**：transition score 计算是 O(n_bins² × n_peaks × n_tfs)，n_bins=50 时可接受；若性能瓶颈，可只对候选边计算完整 score
6. **p 值过滤教训**：任何统计检验必须实际用于过滤（参考 `feedback_pvalue_filtering.md`）
