# =============================================================================
# CausalBridge: a causal perturbation-prediction framework bridged by ATAC
# =============================================================================
# Core paradigm:
#   Chromatin accessibility (ATAC) is used as a mechanistic mediator layer, and
#   gene perturbation propagation is decomposed into two steps:
#     Step 1: RNA -> ATAC — motif scanning identifies candidate TFs; a
#             single-feature Pearson test per TF and lag determines TF -> peak
#             regulatory weights and lag (the first significant lag wins).
#     Step 2: ATAC -> RNA — a shared neural network (NN) fits the nonlinear
#             peak -> gene transfer function.
#   Directionality is supported by temporal-lag design and statistical testing;
#   nonlinear fitting is performed by the NN.
#
# Input: only WT single-cell multi-omics data (ATAC + RNA); no perturbation
# training data are required.
# Output: predicted transcriptome-wide and chromatin-wide state shifts after KO
# of any gene.
# =============================================================================

from .io import load_config, load_data, save_results
from .preprocess import preprocess_rna, preprocess_atac, match_cells
from .granger import infer_pseudotime, granger_test, build_causal_grn
from .kinetics import fit_atac_to_rna, fit_rna_to_atac
from .perturbation import propagate_perturbation, perturb_peak

__version__ = "1.39"
__all__ = [
    "load_config", "load_data", "save_results",
    "preprocess_rna", "preprocess_atac", "match_cells",
    "infer_pseudotime", "granger_test", "build_causal_grn",
    "fit_atac_to_rna", "fit_rna_to_atac",
    "propagate_perturbation", "perturb_peak",
]
