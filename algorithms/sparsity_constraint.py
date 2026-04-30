"""
Sparsity Constraint Module for WeightLoRA

This module implements ℓ0-norm constraint enforcement utilities for WeightLoRA,
providing functions to enforce sparsity constraints on the trainable weight vector ω.

Key Features:
- ℓ0-norm sparsity enforcement (||ω||_0 ≤ K)
- Hard thresholding for adapter selection
- Sparsity penalty computation
- Constraint validation utilities
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import numpy as np


class SparsityConstraint:
    """
    ℓ0-norm sparsity constraint enforcement for WeightLoRA.
    
    Implements the constraint ||ω||_0 ≤ K where K is the maximum number of
    active adapters allowed. This is the core mechanism for reducing trainable
    parameters by ~86% while maintaining performance.
    
    Attributes:
        K (int): Maximum number of non-zero weights allowed
        tolerance (float): Tolerance for floating point comparisons
    """
    
    def __init__(self, K: int = 10, tolerance: float = 1e-6):
        """
        Initialize sparsity constraint.
        
        Args:
            K: Maximum number of active adapters (sparsity level)
            tolerance: Tolerance for floating point comparisons
        """
        self.K = K
        self.tolerance = tolerance
    
    def enforce_sparsity(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Enforce sparsity constraint by keeping only top-K weights with largest absolute values.
        
        Implements hard thresholding: set all weights except top-K to zero.
        
        Args:
            weights: Weight vector of shape (n_layers,)
            
        Returns:
            weights: Sparsified weight vector with ||ω||_0 = K
        """
        if weights.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got {weights.dim()}D")
        
        # Handle edge case: all weights are zero
        if torch.allclose(weights, torch.zeros_like(weights), atol=self.tolerance):
            # Randomly select K indices to activate
            indices = torch.randperm(len(weights))[:self.K]
            weights = weights.clone()
            weights[indices] = 1.0
            return weights
        
        # Get absolute values and find top-K indices
        abs_weights = weights.abs()
        top_k_values, top_k_indices = torch.topk(abs_weights, min(self.K, len(weights)))
        
        # Create mask with 1s at top-K positions
        mask = torch.zeros_like(weights)
        mask[top_k_indices] = 1.0
        
        # Apply mask
        sparsified_weights = weights * mask
        
        return sparsified_weights
    
    def compute_sparsity_penalty(self, weights: torch.Tensor, 
                                  sparsity_lambda: float = 0.01) -> torch.Tensor:
        """
        Compute ℓ0-norm sparsity penalty for use in loss function.
        
        The penalty encourages sparsity by penalizing non-zero weights.
        Note: True ℓ0-norm is not differentiable, so we use a proxy.
        
        Args:
            weights: Weight vector
            sparsity_lambda: Penalty coefficient (default 0.01)
            
        Returns:
            sparsity_penalty: Scalar penalty value
        """
        # ℓ0-norm: count of non-zero elements
        # Use threshold for floating point comparison
        non_zero_count = (weights.abs() > self.tolerance).sum()
        
        # Penalty: λ * ||ω||_0
        penalty = sparsity_lambda * non_zero_count
        
        return penalty
    
    def validate_constraint(self, weights: torch.Tensor, 
                           K: Optional[int] = None) -> Tuple[bool, int]:
        """
        Validate that weight vector satisfies sparsity constraint.
        
        Args:
            weights: Weight vector to validate
            K: Maximum allowed sparsity (uses self.K if not provided)
            
        Returns:
            Tuple of (is_valid, actual_sparsity)
            - is_valid: True if ||ω||_0 ≤ K
            - actual_sparsity: Actual number of non-zero weights
        """
        if K is None:
            K = self.K
        
        # Count non-zero weights
        actual_sparsity = (weights.abs() > self.tolerance).sum().item()
        
        is_valid = actual_sparsity <= K
        
        return is_valid, int(actual_sparsity)
    
    def get_active_indices(self, weights: torch.Tensor) -> List[int]:
        """
        Get indices of active (non-zero) weights.
        
        Args:
            weights: Weight vector
            
        Returns:
            List of indices where weights are non-zero
        """
        active_mask = weights.abs() > self.tolerance
        return active_mask.nonzero(as_tuple=True)[0].tolist()
    
    def get_inactive_indices(self, weights: torch.Tensor) -> List[int]:
        """
        Get indices of inactive (zero) weights.
        
        Args:
            weights: Weight vector
            
        Returns:
            List of indices where weights are zero
        """
        inactive_mask = weights.abs() <= self.tolerance
        return inactive_mask.nonzero(as_tuple=True)[0].tolist()
    
    def compute_reduction_ratio(self, weights: torch.Tensor, 
                                total_adapters: int) -> float:
        """
        Compute parameter reduction ratio achieved by sparsity.
        
        Args:
            weights: Weight vector
            total_adapters: Total number of adapters (layers)
            
        Returns:
            Reduction ratio as fraction (0.0 = no reduction, 1.0 = all adapters removed)
        """
        active_count = self.get_active_indices(weights)
        active_ratio = len(active_count) / total_adapters
        
        reduction_ratio = 1.0 - active_ratio
        
        return reduction_ratio


def hard_thresholding(weights: torch.Tensor, K: int) -> torch.Tensor:
    """
    Apply hard thresholding to enforce ||ω||_0 = K.
    
    Keeps only the K weights with largest absolute values,
    sets all others to zero.
    
    Args:
        weights: Weight vector
        K: Number of weights to keep
        
    Returns:
        Sparsified weight vector
    """
    constraint = SparsityConstraint(K=K)
    return constraint.enforce_sparsity(weights)


def soft_thresholding(weights: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply soft thresholding (proximal operator for ℓ1-norm).
    
    Note: This is for comparison purposes. WeightLoRA uses hard thresholding
    for strict ℓ0-norm sparsity.
    
    Args:
        weights: Weight vector
        threshold: Threshold value
        
    Returns:
        Soft-thresholded weight vector
    """
    return torch.sign(weights) * torch.relu(weights.abs() - threshold)


def compute_sparsity_statistics(weights: torch.Tensor, 
                                K: int,
                                total_adapters: int) -> Dict[str, float]:
    """
    Compute comprehensive statistics about sparsity configuration.
    
    Args:
        weights: Weight vector
        K: Target sparsity level
        total_adapters: Total number of adapters
        
    Returns:
        Dictionary with sparsity statistics
    """
    constraint = SparsityConstraint(K=K)
    
    # Basic counts
    active_indices = constraint.get_active_indices(weights)
    inactive_indices = constraint.get_inactive_indices(weights)
    
    active_count = len(active_indices)
    inactive_count = len(inactive_indices)
    
    # Reduction metrics
    reduction_ratio = constraint.compute_reduction_ratio(weights, total_adapters)
    
    # Sparsity penalty (with λ=0.01)
    sparsity_penalty = constraint.compute_sparsity_penalty(weights, sparsity_lambda=0.01)
    
    # Validation
    is_valid, actual_sparsity = constraint.validate_constraint(weights, K)
    
    return {
        'active_count': active_count,
        'inactive_count': inactive_count,
        'active_indices': active_indices,
        'inactive_indices': inactive_indices,
        'reduction_ratio': float(reduction_ratio),
        'sparsity_penalty': float(sparsity_penalty),
        'is_valid': is_valid,
        'actual_sparsity': actual_sparsity,
        'target_sparsity': K,
        'total_adapters': total_adapters
    }


def adaptive_sparsity_schedule(weights: torch.Tensor, 
                                current_step: int,
                                total_steps: int,
                                initial_K: int = 100,
                                final_K: int = 10) -> torch.Tensor:
    """
    Apply adaptive sparsity schedule that gradually reduces K during training.
    
    This allows the model to first learn with more adapters and then
    progressively select the most important ones.
    
    Args:
        weights: Current weight vector
        current_step: Current training step
        total_steps: Total training steps
        initial_K: Initial sparsity level (more adapters)
        final_K: Final sparsity level (fewer adapters)
        
    Returns:
        Sparsified weight vector with adaptive K
    """
    if total_steps <= 0:
        total_steps = 1
    
    # Linear interpolation from initial_K to final_K
    progress = current_step / total_steps
    K = int(initial_K + (final_K - initial_K) * progress)
    K = max(1, K)  # Ensure at least 1 adapter
    
    constraint = SparsityConstraint(K=K)
    return constraint.enforce_sparsity(weights)


def sparsity_regularization_loss(weights: torch.Tensor, 
                                  sparsity_lambda: float = 0.01) -> torch.Tensor:
    """
    Compute sparsity regularization loss for use in training loop.
    
    Args:
        weights: Weight vector
        sparsity_lambda: Penalty coefficient
        
    Returns:
        Scalar loss value
    """
    penalty = SparsityConstraint().compute_sparsity_penalty(weights, sparsity_lambda)
    return penalty


def enforce_weightlora_sparsity(model: torch.nn.Module, 
                                K: int = 10,
                                tolerance: float = 1e-6) -> Dict[str, List[int]]:
    """
    Enforce sparsity on a WeightLoRA model by disconnecting inactive adapters.
    
    This function applies the sparsity constraint to the actual model,
    zeroing out adapter parameters for inactive layers.
    
    Args:
        model: WeightLoRA model with trainable weight vectors
        K: Maximum number of active adapters
        tolerance: Tolerance for floating point comparison
        
    Returns:
        Dictionary mapping layer names to active/inactive status
    """
    # Get weight vector from model
    weight_vector = model.get_weight_vector()
    
    # Get active indices
    constraint = SparsityConstraint(K=K, tolerance=tolerance)
    active_indices = constraint.get_active_indices(weight_vector)
    
    # Disconnect inactive adapters
    disconnected = []
    for i, layer in enumerate(model.layers_with_adapters):
        if i not in active_indices:
            layer.disconnect()
            disconnected.append(i)
    
    return {
        'active_indices': active_indices,
        'disconnected_indices': disconnected,
        'active_count': len(active_indices),
        'disconnected_count': len(disconnected)
    }


def enforce_sparsity(weights: "torch.Tensor", K: int = 10) -> "torch.Tensor":
    """Standalone wrapper: apply L0 sparsity keeping top-K weights."""
    return SparsityConstraint(K=K).enforce_sparsity(weights)


def compute_sparsity_penalty(weights: "torch.Tensor", K: int = 10, lambda_reg: float = 0.01) -> "torch.Tensor":
    """Standalone wrapper: compute L0 sparsity regularization penalty."""
    return SparsityConstraint(K=K).compute_sparsity_penalty(weights, lambda_reg)
