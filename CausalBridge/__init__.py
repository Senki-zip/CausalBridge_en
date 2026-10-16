# =============================================================================
# CausalBridge: 基于 ATAC 桥接的因果扰动预测框架
# =============================================================================
# 核心范式：
#   将染色质可及性（ATAC）作为机械中介层，将基因扰动传播拆解为两步：
#     步骤1: RNA→ATAC — motif 扫描确定候选 TF，逐 TF/逐 lag 单特征 Pearson
#            检验确定 TF→peak 调控权重与滞后（首个显著 lag 胜出）
#     步骤2: ATAC→RNA — 独立 nn/spline/saturation 响应及支持组拟合
#   方向性由时间滞后设计与统计检验提供证据。
#
# 输入：仅需 WT 单细胞多组学数据（ATAC+RNA），无需任何扰动训练数据
# 输出：任意基因 KO 后的全转录组 + 全染色质状态偏移预测
# =============================================================================

from .io import load_config, load_data, save_results
from .preprocess import preprocess_rna, preprocess_atac, match_cells
from .granger import infer_pseudotime, granger_test, build_causal_grn
from .kinetics import fit_atac_to_rna, fit_rna_to_atac
from .perturbation import propagate_perturbation, perturb_peak

__version__ = "2.0.0"
__all__ = [
    "load_config", "load_data", "save_results",
    "preprocess_rna", "preprocess_atac", "match_cells",
    "infer_pseudotime", "granger_test", "build_causal_grn",
    "fit_atac_to_rna", "fit_rna_to_atac",
    "propagate_perturbation", "perturb_peak",
]
