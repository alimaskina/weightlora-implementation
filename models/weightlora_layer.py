"""
WeightLoRA Layer Implementation

This module implements the core WeightLoRA layer with trainable weight ω_i for adaptive
adapter activation control. It extends the standard LoRA adapter with a learnable weight
vector that controls the contribution of each adapter.

Key Formulas:
- Forward: h_i = W_i × x + ω_i × A_i × B_i × x  [Eq. 3]
- Sparsity: ||ω||_0 = Σ I(ω_i ≠ 0), count of non-zero weights
- Constraint: ||ω||_0 ≤ K (keep only top-K adapters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding


class WeightLoRALayer(nn.Module):
    """
    WeightLoRA Layer with trainable weight ω_i for adaptive activation.
    
    This layer implements the core innovation of WeightLoRA: a trainable weight
    vector ω that controls which adapters are active during training and inference.
    
    Args:
        pretrained_layer: The original pretrained layer (nn.Linear or nn.Embedding)
        adapter: LoRA adapter module (LoRAAdapter, LoRALinear, or LoRAEmbedding)
        layer_name: Name of the layer for tracking purposes
        rank: Rank of the LoRA adapter (default: 8)
        dropout: Dropout probability for adapter (default: 0.05)
    """
    
    def __init__(
        self,
        pretrained_layer: nn.Module,
        adapter: nn.Module,
        layer_name: str = "layer_0",
        rank: int = 8,
        dropout: float = 0.05
    ):
        super().__init__()
        
        self.layer_name = layer_name
        self.pretrained = pretrained_layer
        self.adapter = adapter
        self.rank = rank
        
        # Eq 3: Trainable weight ω_i for this layer
        # Initialize to 1.0 (fully active)
        self.weight = nn.Parameter(torch.ones(1))
        
        # Register buffers for debugging and monitoring
        self.register_buffer('adam_a', adapter.A)
        self.register_buffer('adam_b', adapter.B)
        
        # Track if this layer has been disconnected
        self.register_buffer('is_disconnected', torch.tensor([False]))
        
        # Store original weight for reference
        self.original_pretrained_weight = pretrained_layer.weight.clone()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Eq 3: h_i = W_i × x + ω_i × A_i × B_i × x
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            
        Returns:
            Output tensor with shape matching input
        """
        # Compute pretrained layer output
        h_pretrained = self.pretrained(x)
        
        # Compute adapter output
        h_adapter = self.adapter(x)
        
        # Apply trainable weight ω_i
        # When weight = 1: full adapter contribution
        # When weight = 0: adapter is disconnected
        h_out = h_pretrained + (self.weight * h_adapter)
        
        return h_out
    
    def set_weight(self, value: float) -> None:
        """
        Set the weight to disconnect (0) or enable (1) the adapter.
        
        Args:
            value: Target weight value (0.0 for disconnect, 1.0 for full activation)
        """
        self.weight.data = torch.full_like(self.weight, value)
    
    def disconnect(self) -> None:
        """Permanently disconnect this adapter by setting weight to 0."""
        self.weight.data = torch.zeros(1)
        self.is_disconnected.data = torch.tensor([True])
    
    def enable(self) -> None:
        """Enable this adapter by setting weight to 1."""
        self.weight.data = torch.ones(1)
        self.is_disconnected.data = torch.tensor([False])
    
    def get_weight(self) -> torch.Tensor:
        """Get current weight value."""
        return self.weight.data.clone()
    
    def get_param_count(self) -> int:
        """
        Get the number of trainable parameters in this layer.
        
        Returns:
            Total parameter count including pretrained weights, A, B, and ω
        """
        pretrained_params = sum(p.numel() for p in self.pretrained.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        weight_params = self.weight.numel()
        
        return pretrained_params + adapter_params + weight_params
    
    def zero_adapter(self) -> None:
        """Zero out adapter parameters A and B (temporary disconnection)."""
        self.adapter.A.data.zero_()
        self.adapter.B.data.zero_()
    
    def reset_adapter(self) -> None:
        """Reset adapter parameters to initial values."""
        # Reset A to N(0, 0.1)
        self.adapter.A.data = torch.randn_like(self.adapter.A) * 0.1
        # Reset B to zeros
        self.adapter.B.data.zero_()
    
    def compute_layer_loss_contribution(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss contribution from this specific layer.
        
        Used for computing gradients w.r.t. ω_i.
        
        Args:
            loss: Total loss tensor
            
        Returns:
            Gradient contribution from this layer
        """
        # Temporarily set weight to 1 to compute pure layer gradient
        original_weight = self.weight.data.clone()
        self.weight.data = torch.ones(1)
        
        try:
            layer_gradient = torch.autograd.grad(
                output=loss,
                inputs=self.weight,
                retain_graph=True
            )[0][:, 0]
        except:
            layer_gradient = torch.tensor([0.0])
        
        self.weight.data = original_weight
        return layer_gradient
    
    def __repr__(self) -> str:
        return (
            f"WeightLoRALayer(layer_name='{self.layer_name}', "
            f"rank={self.rank}, weight={self.weight.item():.4f})"
        )


class WeightLoRAWrapper(nn.Module):
    """
    Wrapper class for applying WeightLoRA to a pretrained model.
    
    This wrapper automatically adds WeightLoRALayer instances to all target layers
    in a pretrained model, managing the adapter lifecycle.
    
    Args:
        model: Pretrained model to wrap
        target_modules: List of module names to apply LoRA to
        rank: LoRA rank (default: 8)
        alpha: LoRA scaling factor (default: 32.0)
        dropout: Dropout probability (default: 0.05)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_modules: list,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Track all WeightLoRALayer instances
        self.weightlora_layers = nn.ModuleDict()
        
        # Apply adapters to target modules
        self._apply_adapters()
    
    def _apply_adapters(self) -> None:
        """Apply LoRA adapters to all target modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if any(target in name for target in self.target_modules):
                    # Create LoRA adapter
                    adapter = LoRAAdapter(
                        module.in_features,
                        module.out_features,
                        r=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout
                    )
                    
                    # Create WeightLoRALayer
                    weightlora_layer = WeightLoRALayer(
                        pretrained_layer=module,
                        adapter=adapter,
                        layer_name=name,
                        rank=self.rank
                    )
                    
                    # Replace original module with WeightLoRALayer
                    setattr(self.model, name, weightlora_layer)
                    
                    # Store reference
                    self.weightlora_layers[name] = weightlora_layer
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)
    
    def get_active_layers(self) -> list:
        """Get list of currently active (weight > 0) layers."""
        active = []
        for name, layer in self.weightlora_layers.items():
            if layer.weight.data > 1e-6:
                active.append(name)
        return active
    
    def get_disconnected_layers(self) -> list:
        """Get list of currently disconnected (weight = 0) layers."""
        disconnected = []
        for name, layer in self.weightlora_layers.items():
            if layer.weight.data <= 1e-6:
                disconnected.append(name)
        return disconnected
    
    def disconnect_layer(self, layer_name: str) -> None:
        """Disconnect a specific layer."""
        if layer_name in self.weightlora_layers:
            self.weightlora_layers[layer_name].disconnect()
    
    def enable_layer(self, layer_name: str) -> None:
        """Enable a specific layer."""
        if layer_name in self.weightlora_layers:
            self.weightlora_layers[layer_name].enable()
    
    def get_weight_vector(self) -> torch.Tensor:
        """Get the full weight vector ω."""
        weights = []
        for layer in self.weightlora_layers.values():
            weights.append(layer.weight.data)
        return torch.cat(weights)
    
    def set_weight_vector(self, weights: torch.Tensor) -> None:
        """Set the full weight vector ω."""
        idx = 0
        for layer in self.weightlora_layers.values():
            layer.weight.data = weights[idx:idx + 1]
            idx += 1


def create_weightlora_layer(
    pretrained_layer: nn.Module,
    adapter: nn.Module,
    layer_name: str = "layer_0",
    rank: int = 8
) -> WeightLoRALayer:
    """
    Factory function to create a WeightLoRALayer.
    
    Args:
        pretrained_layer: Original pretrained layer
        adapter: LoRA adapter module
        layer_name: Name for tracking
        rank: LoRA rank
        
    Returns:
        WeightLoRALayer instance
    """
    return WeightLoRALayer(
        pretrained_layer=pretrained_layer,
        adapter=adapter,
        layer_name=layer_name,
        rank=rank
    )
