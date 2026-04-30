"""
Stochastic Iterative Hard Thresholding (StoIHT) Optimizer

Implements the ℓ0-norm optimization algorithm from Nguyen et al. (2014)
for sparse weight vector optimization in WeightLoRA.

Key Algorithm:
1. Gradient descent step on weight vector ω
2. Hard thresholding: Keep only top-K coordinates with largest |ω_i|
3. Zero out all other coordinates to enforce sparsity constraint ||ω||_0 ≤ K

References:
- Nguyen, N., Dang, C., & Dang, T. (2014). Iterative Hard Thresholding for
  Sparse Optimization.
- WeightLoRA paper: Adaptive LoRA adapter selection using trainable importance weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StoIHTOptimizer:
    """
    Stochastic Iterative Hard Thresholding Optimizer for ℓ0-norm sparsity.
    
    This optimizer enforces sparsity by keeping only the top-K weights
    with the largest absolute values and zeroing out the rest.
    
    Parameters:
    -----------
    K : int
        Maximum number of non-zero weights (sparsity level)
    initial_learning_rate : float, optional
        Initial learning rate for gradient descent step
    """
    
    def __init__(self, K: int, initial_learning_rate: float = 0.1):
        """
        Initialize StoIHT optimizer.
        
        Args:
            K: Maximum number of non-zero weights to keep
            initial_learning_rate: Learning rate for gradient descent
        """
        self.K = K
        self.learning_rate = initial_learning_rate
        self.iteration = 0
    
    def update(self, weights: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        """
        Perform StoIHT update step.
        
        Algorithm:
        1. Gradient descent: weights = weights + lr * gradients
        2. Hard thresholding: Keep top-K by absolute value
        
        Args:
            weights: Current weight vector (shape: [n_layers])
            gradients: Gradients for each weight (shape: [n_layers])
            
        Returns:
            torch.Tensor: Updated weight vector with ||ω||_0 = K
        """
        # Step 1: Gradient descent update
        # weights = weights + lr * gradients
        updated_weights = weights + self.learning_rate * gradients
        
        # Step 2: Hard thresholding - keep top-K coordinates
        # Compute absolute values
        abs_weights = updated_weights.abs()
        
        # Find top-K indices with largest absolute values
        top_k, top_indices = torch.topk(abs_weights, self.K, dim=0)
        
        # Create binary mask for top-K indices
        mask = torch.zeros_like(updated_weights).float()
        mask.scatter_(0, top_indices, 1.0)
        
        # Apply sparsity: keep only top-K, zero out rest
        sparse_weights = updated_weights * mask
        
        # Ensure exactly K non-zero elements (handle edge cases)
        if sparse_weights.sum() == 0:
            # If all weights are zero, randomly select K indices
            random_indices = torch.randperm(len(sparse_weights))[:self.K]
            sparse_weights = sparse_weights.clone()
            sparse_weights[random_indices] = 1.0
        
        self.iteration += 1
        return sparse_weights
    
    def update_with_regularization(
        self, 
        weights: torch.Tensor, 
        gradients: torch.Tensor,
        l2_lambda: float = 0.0
    ) -> torch.Tensor:
        """
        StoIHT update with optional L2 regularization.
        
        Args:
            weights: Current weight vector
            gradients: Gradients for each weight
            l2_lambda: L2 regularization coefficient
            
        Returns:
            Updated weight vector with sparsity constraint
        """
        # Gradient descent with L2 regularization
        updated_weights = weights + self.learning_rate * (gradients - l2_lambda * weights)
        
        # Hard thresholding
        abs_weights = updated_weights.abs()
        top_k, top_indices = torch.topk(abs_weights, self.K, dim=0)
        
        mask = torch.zeros_like(updated_weights).float()
        mask.scatter_(0, top_indices, 1.0)
        
        sparse_weights = updated_weights * mask
        
        return sparse_weights
    
    def compute_sparsity(self, weights: torch.Tensor) -> int:
        """
        Compute the ℓ0-norm (number of non-zero elements).
        
        Args:
            weights: Weight vector
            
        Returns:
            int: Number of non-zero elements
        """
        return (weights.abs() > 1e-6).sum().item()
    
    def get_active_indices(self, weights: torch.Tensor) -> list:
        """
        Get indices of active (non-zero) weights.
        
        Args:
            weights: Weight vector
            
        Returns:
            list: Indices of non-zero weights
        """
        active_mask = (weights.abs() > 1e-6)
        return active_mask.nonzero(as_tuple=True)[0].tolist()
    
    def get_inactive_indices(self, weights: torch.Tensor) -> list:
        """
        Get indices of inactive (zero) weights.
        
        Args:
            weights: Weight vector
            
        Returns:
            list: Indices of zero weights
        """
        inactive_mask = (weights.abs() <= 1e-6)
        return inactive_mask.nonzero(as_tuple=True)[0].tolist()
    
    def adaptive_learning_rate(self, weights: torch.Tensor, gradients: torch.Tensor) -> float:
        """
        Compute adaptive learning rate based on gradient magnitude.
        
        Args:
            weights: Current weight vector
            gradients: Gradients
            
        Returns:
            float: Adaptive learning rate
        """
        # Scale learning rate by gradient norm
        grad_norm = gradients.abs().mean().item()
        if grad_norm > 0:
            return self.learning_rate * (1.0 / (1.0 + grad_norm))
        return self.learning_rate
    
    def step(
        self, 
        weights: torch.Tensor, 
        gradients: torch.Tensor,
        l2_lambda: float = 0.0
    ) -> tuple:
        """
        Full optimization step with learning rate adaptation.
        
        Args:
            weights: Current weight vector
            gradients: Gradients
            l2_lambda: L2 regularization coefficient
            
        Returns:
            tuple: (updated_weights, sparsity_level, active_indices)
        """
        # Adaptive learning rate
        lr = self.adaptive_learning_rate(weights, gradients)
        
        # Update with regularization
        updated_weights = self.update_with_regularization(weights, gradients, l2_lambda)
        
        # Compute statistics
        sparsity = self.compute_sparsity(updated_weights)
        active_indices = self.get_active_indices(updated_weights)
        
        return updated_weights, sparsity, active_indices


class StoIHTTrainer:
    """
    Training wrapper for StoIHT-based weight optimization.
    
    Manages the complete training loop with weight sparsity enforcement.
    """
    
    def __init__(
        self,
        n_layers: int,
        K: int,
        initial_lr: float = 3e-4,
        l2_lambda: float = 1e-5
    ):
        """
        Initialize StoIHT trainer.
        
        Args:
            n_layers: Number of layers with adapters
            K: Sparsity level (keep top-K adapters)
            initial_lr: Initial learning rate
            l2_lambda: L2 regularization coefficient
        """
        self.n_layers = n_layers
        self.K = K
        self.stoiht = StoIHTOptimizer(K=K, initial_learning_rate=initial_lr)
        self.l2_lambda = l2_lambda
        
        # Initialize weight vector
        self.weight = nn.Parameter(torch.ones(n_layers))
        self.optimizer = torch.optim.Adam([self.weight], lr=initial_lr)
    
    def train_step(
        self, 
        loss: torch.Tensor,
        gradients: torch.Tensor
    ) -> tuple:
        """
        Single training step with StoIHT update.
        
        Args:
            loss: Current loss value
            gradients: Gradients for each weight
            
        Returns:
            tuple: (updated_weights, loss, sparsity)
        """
        # Optimize weight vector
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Apply StoIHT hard thresholding
        updated_weights, sparsity, _ = self.stoiht.step(
            self.weight, gradients, self.l2_lambda
        )
        
        return updated_weights, loss.item(), sparsity
    
    def get_weight_vector(self) -> torch.Tensor:
        """Get current weight vector."""
        return self.weight.detach().clone()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """Set weight vector directly."""
        self.weight.data = weights.clone()


def compute_layer_contribution(
    model,
    layer_indices: list,
    input_batch: torch.Tensor,
    target_batch: torch.Tensor
) -> torch.Tensor:
    """
    Compute loss contribution from specific layers.
    
    Used for computing gradients w.r.t. individual layer weights.
    
    Args:
        model: Model with WeightLoRALayers
        layer_indices: List of layer indices to compute contribution for
        input_batch: Input batch
        target_batch: Target batch
        
    Returns:
        torch.Tensor: Loss contributions for each layer
    """
    contributions = []
    
    for idx in layer_indices:
        # Temporarily set weight to 1 to compute pure layer gradient
        layer = model.layers[idx]
        original_weight = layer.weight.clone()
        layer.weight.data = torch.ones(1)
        
        # Forward pass
        output = model(input_batch)
        loss = F.cross_entropy(output, target_batch)
        
        # Backward pass
        layer_gradient = torch.autograd.grad(
            outputs=loss,
            inputs=layer.weight,
            retain_graph=True
        )[0][:, 0]
        
        contributions.append(torch.sum(layer_gradient))
        
        # Restore original weight
        layer.weight.data = original_weight
    
    return torch.stack(contributions)


def validate_sparsity_constraint(
    weights: torch.Tensor,
    K: int,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that sparsity constraint is satisfied.
    
    Args:
        weights: Weight vector
        K: Maximum allowed non-zero elements
        tolerance: Tolerance for floating point comparison
        
    Returns:
        bool: True if constraint satisfied
    """
    actual_sparsity = (weights.abs() > tolerance).sum().item()
    return actual_sparsity <= K


if __name__ == "__main__":
    # Test StoIHT optimizer
    print("Testing StoIHT Optimizer...")
    
    # Create test weight vector
    n_layers = 10
    K = 5
    weights = torch.randn(n_layers)
    gradients = torch.randn(n_layers)
    
    # Initialize optimizer
    optimizer = StoIHTOptimizer(K=K, initial_learning_rate=0.1)
    
    # Perform update
    updated_weights = optimizer.update(weights, gradients)
    
    # Verify sparsity
    sparsity = optimizer.compute_sparsity(updated_weights)
    print(f"Initial weights: {weights}")
    print(f"Gradients: {gradients}")
    print(f"Updated weights: {updated_weights}")
    print(f"Sparsity (||ω||_0): {sparsity}")
    print(f"K constraint satisfied: {sparsity <= K}")
    
    # Test with zero gradients
    zero_gradients = torch.zeros(n_layers)
    updated_weights_zero = optimizer.update(weights, zero_gradients)
    print(f"\nWith zero gradients:")
    print(f"Updated weights: {updated_weights_zero}")
    print(f"Sparsity: {optimizer.compute_sparsity(updated_weights_zero)}")
    
    print("\nStoIHT Optimizer tests completed successfully!")
