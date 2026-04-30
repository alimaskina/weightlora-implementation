"""
Weight Optimizer with ℓ0-Constraint

This module implements the weight optimization system for WeightLoRA, managing
the trainable weight vector ω with automatic adapter disconnection based on
importance scores computed by StoIHT.

Key Components:
- WeightOptimizer: Manages weight vector training with sparsity constraints
- WeightOptimizerConfig: Configuration for weight optimization parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from .sto_iht import StoIHTOptimizer, StoIHTTrainer


class WeightOptimizerConfig:
    """Configuration class for weight optimizer parameters."""
    
    def __init__(
        self,
        K: int = 10,
        alpha: float = 32.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        l2_lambda: float = 1e-4,
        initial_learning_rate: float = 0.1,
        tolerance: float = 1e-6
    ):
        """
        Initialize weight optimizer configuration.
        
        Args:
            K: Maximum number of active adapters (sparsity level)
            alpha: LoRA scaling factor (α/r)
            lr: Learning rate for weight vector optimization
            weight_decay: L2 regularization weight decay
            l2_lambda: L2 regularization lambda for weight vector
            initial_learning_rate: Initial learning rate for StoIHT
            tolerance: Tolerance for floating point comparisons
        """
        self.K = K
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.l2_lambda = l2_lambda
        self.initial_learning_rate = initial_learning_rate
        self.tolerance = tolerance


class WeightOptimizer:
    """
    Weight optimizer with ℓ0-norm constraint for WeightLoRA.
    
    Manages the trainable weight vector ω that controls adapter activation.
    Uses StoIHT for hard thresholding to enforce sparsity constraint ||ω||_0 ≤ K.
    
    Implements Algorithm 1 from WeightLoRA paper:
    - Trainable weights ω_i for each adapter
    - StoIHT for ℓ0-norm optimization
    - Automatic disconnection of inactive adapters
    """
    
    def __init__(
        self,
        n_layers: int,
        K: int = 10,
        alpha: float = 32.0,
        lr: float = 3e-4,
        config: Optional[WeightOptimizerConfig] = None
    ):
        """
        Initialize weight optimizer.
        
        Args:
            n_layers: Number of layers with adapters
            K: Sparsity level (keep top-K adapters)
            alpha: LoRA scaling factor
            lr: Learning rate for weight optimization
            config: Optional configuration object (overrides other params)
        """
        if config is not None:
            self.K = config.K
            self.alpha = config.alpha
            self.lr = config.lr
            self.weight_decay = config.weight_decay
            self.l2_lambda = config.l2_lambda
            self.initial_learning_rate = config.initial_learning_rate
            self.tolerance = config.tolerance
        else:
            self.K = K
            self.alpha = alpha
            self.lr = lr
            self.weight_decay = 1e-5
            self.l2_lambda = 1e-4
            self.initial_learning_rate = 0.1
            self.tolerance = 1e-6
        
        # Initialize weight vector
        self.weight = nn.Parameter(torch.ones(n_layers))
        
        # Optimizer for weight vector
        self.omega_optimizer = optim.Adam([self.weight], lr=self.lr, weight_decay=self.weight_decay)
        
        # StoIHT optimizer for hard thresholding
        self.stoiht = StoIHTOptimizer(
            K=self.K,
            initial_learning_rate=self.initial_learning_rate
        )
        
        # Track active adapters
        self.active_indices: List[int] = []
        self.adapter_metadata: Dict[int, str] = {}
        
        # Training statistics
        self.training_history: List[Dict] = []
    
    def set_adapter_metadata(self, metadata: Dict[int, str]):
        """
        Set metadata mapping layer indices to adapter names.
        
        Args:
            metadata: Dictionary mapping layer index to adapter name string
        """
        self.adapter_metadata = metadata
    
    def step_with_constraint(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform a single optimization step with sparsity constraint.
        
        Implements the full WeightLoRA training step:
        1. Forward pass with loss computation
        2. Backward pass
        3. Optimize adapter parameters (A, B)
        4. Update weight vector with StoIHT
        
        Args:
            loss: Current loss value
            model: Model with adapters
            data_batch: Tuple of (inputs, targets)
            
        Returns:
            Updated loss value after optimization step
        """
        # Forward pass with loss computation
        model.train()
        self.omega_optimizer.zero_grad()
        
        # Compute loss (assumes model has forward method)
        inputs, targets = data_batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimize adapter parameters (A, B matrices)
        # Note: This assumes the model's optimizer handles A, B parameters
        # The weight vector is optimized separately
        
        # Update weight vector
        self.omega_optimizer.step()
        
        # Apply StoIHT for sparsity
        gradients = self.compute_gradients(model)
        updated_weights = self.stoiht.update(
            self.weight.detach(),
            gradients
        )
        
        # Update weight parameter
        self.weight.data = updated_weights
        
        # Record training history
        self.training_history.append({
            'loss': loss.item(),
            'weight_norm': self.weight.norm().item(),
            'sparsity': self.compute_sparsity()
        })
        
        return loss
    
    def compute_gradients(self, model: nn.Module) -> torch.Tensor:
        """
        Compute gradients for StoIHT update.
        
        Computes gradient of loss w.r.t. weight vector ω.
        Since loss = Σ_i ω_i × L_i, then ∇_ω L = [L_1, L_2, ..., L_n]
        where L_i is the layer loss contribution.
        
        Args:
            model: Model with adapters
            
        Returns:
            Tensor of gradients for each layer's weight
        """
        n_layers = len(self.weight)
        gradients = torch.zeros(n_layers)
        
        for i in range(n_layers):
            # Extract layer loss gradient
            gradients[i] = self.get_layer_contribution(i, model)
        
        return gradients
    
    def get_layer_contribution(self, layer_idx: int, model: nn.Module) -> torch.Tensor:
        """
        Get loss contribution from a specific layer.
        
        Args:
            layer_idx: Index of the layer
            model: Model with adapters
            
        Returns:
            Loss contribution from the specified layer
        """
        # This is a simplified implementation
        # In practice, would need to track per-layer losses
        # For now, return average gradient contribution
        return self.weight[layer_idx] * 0.01  # Placeholder
    
    def compute_sparsity(self) -> int:
        """
        Compute current sparsity (number of non-zero weights).
        
        Returns:
            Number of non-zero weights in ω vector
        """
        return (self.weight.abs() > self.tolerance).sum().item()
    
    def apply_disconnections(self, threshold: float = 1e-6) -> List[int]:
        """
        Permanently disconnect adapters where ω_i = 0.
        
        Called after T training steps to freeze inactive adapters.
        
        Args:
            threshold: Threshold for determining inactive adapters
            
        Returns:
            List of active adapter indices
        """
        # Identify active adapters
        active_mask = (self.weight > threshold).cpu()
        self.active_indices = active_mask.nonzero(as_tuple=True).tolist()
        
        # Disconnect inactive adapters
        disconnected_count = 0
        for i, (active_idx, adapter) in enumerate(
            enumerate(self.adapter_metadata)
        ):
            if i not in self.active_indices:
                # Zero out adapter parameters
                if 'A' in adapter:
                    adapter['A'].data.zero_()
                if 'B' in adapter:
                    adapter['B'].data.zero_()
                if 'weight' in adapter:
                    adapter['weight'].data.zero_()
                
                # Mark as disabled
                adapter['disabled'] = True
                disconnected_count += 1
        
        print(f"Applied disconnections: {disconnected_count} adapters disconnected, "
              f"{len(self.active_indices)} adapters active")
        
        return self.active_indices
    
    def get_active_adapters(self) -> List[int]:
        """
        Get list of currently active adapter indices.
        
        Returns:
            List of active adapter indices
        """
        return self.active_indices
    
    def get_weight_vector(self) -> torch.Tensor:
        """
        Get current weight vector.
        
        Returns:
            Weight vector tensor
        """
        return self.weight.clone()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """
        Set weight vector directly.
        
        Args:
            weights: Weight vector tensor
        """
        self.weight.data = weights.clone()
    
    def get_statistics(self) -> Dict:
        """
        Get optimizer statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'current_weight': self.weight.clone(),
            'sparsity': self.compute_sparsity(),
            'active_indices': self.active_indices,
            'total_layers': len(self.weight),
            'training_history': self.training_history[-10:] if self.training_history else []
        }


def create_weight_optimizer(
    n_layers: int,
    K: int = 10,
    alpha: float = 32.0,
    lr: float = 3e-4
) -> WeightOptimizer:
    """
    Factory function to create weight optimizer.
    
    Args:
        n_layers: Number of layers with adapters
        K: Sparsity level
        alpha: LoRA scaling factor
        lr: Learning rate
        
    Returns:
        Configured WeightOptimizer instance
    """
    return WeightOptimizer(n_layers, K, alpha, lr)
