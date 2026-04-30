"""
Algorithms module for WeightLoRA implementation.

This module contains core optimization algorithms:
- StoIHT: Stochastic Iterative Hard Thresholding for ℓ0-norm sparsity
- WeightOptimizer: Manages trainable weight vector ω with sparsity constraints
- RankExpander: Implements two-phase rank expansion for WeightLoRA+
- SparsityConstraint: ℓ0-norm constraint enforcement utilities
"""

from .sto_iht import StoIHTOptimizer, StoIHTTrainer, compute_layer_contribution, validate_sparsity_constraint
from .weight_optimizer import WeightOptimizer, WeightOptimizerConfig, create_weight_optimizer
from .rank_expander import RankExpander, expand_rank_random, expand_rank_qr
from .sparsity_constraint import SparsityConstraint, enforce_sparsity, compute_sparsity_penalty

__all__ = [
    'StoIHTOptimizer',
    'StoIHTTrainer',
    'compute_layer_contribution',
    'validate_sparsity_constraint',
    'WeightOptimizer',
    'WeightOptimizerConfig',
    'create_weight_optimizer',
    'RankExpander',
    'expand_rank_random',
    'expand_rank_qr',
    'SparsityConstraint',
    'enforce_sparsity',
    'compute_sparsity_penalty',
]
