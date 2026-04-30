"""
Standard LoRA (Low-Rank Adaptation) Implementation

This module provides the baseline LoRA implementation as described in:
Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

LoRA decomposes weight updates into low-rank matrices A and B:
    ΔW = A × B, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k)
    
Forward pass: h = W×x + (α/r) × A×B×x

Key characteristics:
- A: Randomly initialized with N(0, 0.1)
- B: Initialized to zeros (prevents gradient explosion)
- Scaling: α/r where α is the LoRA alpha parameter
- Memory: O(r×d×k) parameters per adapter (vs O(d×k) for full FT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LoRAAdapter(nn.Module):
    """
    Standard LoRA adapter module implementing the Hu et al. (2021) approach.
    
    This adapter adds trainable low-rank matrices A and B to a pretrained layer,
    allowing fine-tuning with significantly fewer parameters than full fine-tuning.
    
    Args:
        in_features: Input dimension of the layer
        out_features: Output dimension of the layer
        r: Rank of the low-rank decomposition (typically 4, 8, 16, or 32)
        alpha: LoRA alpha parameter (typically 32, 64, or 128)
        dropout: Dropout probability for regularization (default: 0.05)
        scale: Manual scaling factor (overrides α/r if specified)
    
    Attributes:
        A: Low-rank matrix A with random initialization N(0, 0.1)
        B: Low-rank matrix B with zero initialization
        scaling: Computed scaling factor α/r
        dropout: Dropout layer for regularization
    
    Example:
        >>> adapter = LoRAAdapter(768, 768, r=8, alpha=32)
        >>> # Use with pretrained layer
        >>> layer.adapter = adapter
        >>> output = layer(x)  # Uses h = W×x + (α/r) × A×B×x
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        scale: Optional[float] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        
        # Compute scaling factor: α/r
        # This is critical for maintaining performance parity with full FT
        self.scaling = alpha / r if scale is None else scale
        
        # Initialize matrices as per LoRA paper:
        # A: Random Gaussian N(0, 0.1) - provides learnable capacity
        # B: Zeros - prevents gradient explosion during training
        self.A = nn.Parameter(torch.randn(in_features, r) * 0.1)
        self.B = nn.Parameter(torch.zeros(r, out_features))
        
        # Dropout for regularization (optional but recommended)
        self.dropout = nn.Dropout(dropout)
        
        # Register buffers for debugging and monitoring
        self.register_buffer('adam_a', self.A)
        self.register_buffer('adam_b', self.B)
        
        # Track parameter count for memory analysis
        self.register_buffer('param_count', torch.tensor(
            in_features * r + r * out_features, dtype=torch.long
        ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LoRA adapter.
        
        Args:
            x: Input tensor of shape (batch_size, in_features) or 
               (batch_size, seq_len, in_features)
        
        Returns:
            Output tensor: h = W×x + (α/r) × A×B×x
            Note: This returns only the delta (A×B×x), not the full output
        """
        # Compute delta W = A × B
        delta_w = torch.matmul(self.A, self.B)  # Shape: (in_features, out_features)
        
        # Apply delta to input: x × ΔW
        delta_output = torch.matmul(x, delta_w)
        
        # Apply dropout and scaling
        delta_output = self.dropout(delta_output) * self.scaling
        
        return delta_output
    
    def set_scale(self, scale: float):
        """
        Manually set the scaling factor.
        
        Args:
            scale: New scaling factor to apply
        """
        self.scaling = scale
    
    def get_scaling(self) -> float:
        """Get current scaling factor."""
        return self.scaling
    
    def get_param_count(self) -> int:
        """Get total number of trainable parameters in this adapter."""
        return self.A.numel() + self.B.numel()
    
    def zero_adapter(self):
        """
        Zero out adapter parameters (useful for disconnection).
        
        This is different from setting weight=0 in WeightLoRA -
        it completely zeros A and B matrices.
        """
        self.A.data.zero_()
        self.B.data.zero_()
    
    def reset_parameters(self):
        """Reset adapter parameters to initial state."""
        torch.nn.init.normal_(self.A, mean=0.0, std=0.1)
        torch.nn.init.zeros_(self.B)
    
    def extra_repr(self) -> str:
        """String representation for logging."""
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class LoRALinear(nn.Module):
    """
    LoRA wrapper for linear layers.
    
    This class wraps a standard nn.Linear layer and adds LoRA adapters
    to it, allowing for easy integration with pretrained models.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA adapter
        self.lora_adapter = LoRAAdapter(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h = W×x + (α/r) × A×B×x
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor with LoRA delta added
        """
        # Original linear transformation (frozen)
        h = self.linear(x)
        
        # LoRA delta
        delta = self.lora_adapter(x)
        
        return h + delta
    
    def get_lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the LoRA delta contribution."""
        return self.lora_adapter(x)
    
    def set_lora_scale(self, scale: float):
        """Set LoRA scaling factor."""
        self.lora_adapter.set_scale(scale)

    # Read-only proxy properties so external code can inspect like LoRAAdapter
    @property
    def A(self):
        return self.lora_adapter.A

    @property
    def B(self):
        return self.lora_adapter.B

    @property
    def r(self):
        return self.lora_adapter.r

    def update_lora_params(self, A: 'nn.Parameter', B: 'nn.Parameter', r: int):
        """Update A, B, r on the inner adapter (used by rank expanders)."""
        self.lora_adapter.A = A
        self.lora_adapter.B = B
        self.lora_adapter.r = r


class LoRAEmbedding(nn.Module):
    """
    LoRA wrapper for embedding layers.
    
    Args:
        num_embeddings: Number of embeddings
        embedding_dim: Embedding dimension
        r: LoRA rank
        alpha: LoRA alpha parameter
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        alpha: float = 32.0
    ):
        super().__init__()
        
        # Original embedding layer (frozen)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # LoRA adapter for embedding
        self.lora_adapter = LoRAAdapter(
            in_features=embedding_dim,
            out_features=embedding_dim,
            r=r,
            alpha=alpha
        )
        
        # Freeze original weights
        self.embedding.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: h = E×x + (α/r) × A×B×x
        
        Args:
            x: Input token indices tensor
        
        Returns:
            Output embeddings with LoRA delta added
        """
        # Original embedding lookup (frozen)
        h = self.embedding(x)
        
        # LoRA delta
        delta = self.lora_adapter(h)
        
        return h + delta
    
    def get_lora_delta(self, x: torch.Tensor) -> torch.Tensor:
        """Get only the LoRA delta contribution."""
        return self.lora_adapter(x)


def create_lora_adapter(
    layer_type: str,
    in_features: int,
    out_features: int,
    r: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05
) -> nn.Module:
    """
    Factory function to create appropriate LoRA adapter based on layer type.
    
    Args:
        layer_type: Type of layer ('linear', 'embedding', 'conv1d', etc.)
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        alpha: LoRA alpha parameter
        dropout: Dropout probability
    
    Returns:
        Appropriate LoRA adapter module
    """
    lt = layer_type.lower()
    if lt == 'linear':
        return LoRALinear(in_features, out_features, r, alpha, dropout)
    elif lt == 'embedding':
        return LoRAEmbedding(out_features, in_features, r, alpha)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")


if __name__ == "__main__":
    # Test LoRA adapter
    print("Testing LoRA Adapter...")
    
    # Create adapter
    adapter = LoRAAdapter(in_features=768, out_features=768, r=8, alpha=32)
    
    # Check parameter count
    print(f"Adapter parameters: {adapter.get_param_count()}")
    print(f"Scaling factor: {adapter.get_scaling()}")
    
    # Test forward pass
    x = torch.randn(2, 10, 768)  # batch=2, seq_len=10, features=768
    output = adapter(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test LoRALinear wrapper
    print("\nTesting LoRALinear...")
    linear = LoRALinear(in_features=768, out_features=768, r=8, alpha=32)
    output_linear = linear(x)
    print(f"LoRALinear output shape: {output_linear.shape}")
    
    print("\nLoRA adapter tests passed!")
