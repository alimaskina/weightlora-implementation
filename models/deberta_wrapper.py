"""
DeBERTaV3 WeightLoRA Wrapper

This module provides DeBERTaV3 model integration with WeightLoRA adapters for
classification tasks (GLUE benchmark), wrapping the pretrained DeBERTaV3 model
with trainable weight vectors for adaptive adapter selection.

Based on: microsoft/deberta-v3-base (184M parameters)
Target layers: attention query, key, value, output projections
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any

from .adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding, create_lora_adapter
from .weightlora_layer import WeightLoRALayer, WeightLoRAWrapper, create_weightlora_layer
from .adapter_manager import AdapterManager, is_target_layer, create_lora_adapter as create_adapter


class DebertaV3WeightLoRA(nn.Module):
    """
    DeBERTaV3 model with WeightLoRA adapters for GLUE classification tasks.
    
    Implements adaptive LoRA adapter selection using trainable weight vectors ω
    with ℓ0-norm sparsity constraints to reduce trainable parameters by ~86%.
    
    Target layers (following paper):
    - Attention query projection
    - Attention key projection
    - Attention value projection
    - Attention output projection
    
    Args:
        model_name: HuggingFace model name (default: microsoft/deberta-v3-base)
        rank: LoRA rank r (default: 8)
        alpha: LoRA scaling factor α (default: 32.0)
        dropout: Dropout probability (default: 0.05)
        K: Maximum number of active adapters (default: 10)
        target_modules: List of target module names for LoRA (default: attention projections)
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        K: int = 10,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha
        self.K = K
        self.dropout = dropout
        
        # Load base DeBERTaV3 model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Default for GLUE tasks (binary classification)
        )
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            model=self.base_model,
            K=K,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules
        )
        
        # Track weight vector for sparsity control
        self.weight_vector = nn.Parameter(torch.ones(len(self.adapter_manager)))
        
        # Store original model for reference
        self.original_model = self.base_model
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass with WeightLoRA adapter activation.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels for classification
            
        Returns:
            outputs: Model outputs (logits, loss if labels provided)
        """
        # Get embeddings from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply WeightLoRA to target layers
        if self.adapter_manager:
            outputs = self.adapter_manager.forward(
                outputs,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return outputs
    
    def get_active_adapters(self) -> List[str]:
        """
        Get list of currently active adapter names.
        
        Returns:
            List of adapter names with weight > 1e-6
        """
        return self.adapter_manager.get_active_adapters()
    
    def get_disconnected_adapters(self) -> List[str]:
        """
        Get list of currently disconnected adapter names.
        
        Returns:
            List of adapter names with weight ≈ 0
        """
        return self.adapter_manager.get_inactive_adapters()
    
    def disconnect_adapters(self, adapter_names: Optional[List[str]] = None):
        """
        Disconnect specified adapters by setting weight to 0.
        
        Args:
            adapter_names: List of adapter names to disconnect. If None, disconnect all inactive.
        """
        if adapter_names is None:
            adapter_names = self.get_disconnected_adapters()
        self.adapter_manager.deactivate_adapters(adapter_names)
    
    def enable_adapters(self, adapter_names: Optional[List[str]] = None):
        """
        Enable specified adapters by setting weight to 1.
        
        Args:
            adapter_names: List of adapter names to enable. If None, enable all active.
        """
        if adapter_names is None:
            adapter_names = self.get_active_adapters()
        self.adapter_manager.activate_adapters(adapter_names)
    
    def get_weight_vector(self) -> torch.Tensor:
        """
        Get current weight vector ω.
        
        Returns:
            Weight vector tensor of shape (n_layers,)
        """
        return self.weight_vector.clone()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """
        Set weight vector ω.
        
        Args:
            weights: Weight vector tensor of shape (n_layers,)
        """
        self.weight_vector.data = weights.clone()
    
    def apply_disconnections(self):
        """
        Permanently disconnect adapters where ω_i = 0.
        
        This is called after T training steps to freeze selected adapters.
        """
        self.adapter_manager.apply_disconnections()
    
    def get_param_count(self) -> Dict[str, int]:
        """
        Get parameter count statistics.
        
        Returns:
            Dictionary with total, trainable, and adapter parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = self.adapter_manager.get_total_param_count()
        active_params = self.adapter_manager.get_active_param_count()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'adapter_total': adapter_params,
            'adapter_active': active_params,
            'reduction': (1 - active_params / adapter_params) * 100 if adapter_params > 0 else 0
        }
    
    def zero_adapter(self, adapter_name: str):
        """
        Zero out a specific adapter's parameters.
        
        Args:
            adapter_name: Name of adapter to zero
        """
        self.adapter_manager.zero_adapter(adapter_name)
    
    def reset_adapter(self, adapter_name: str):
        """
        Reset a specific adapter to initial state.
        
        Args:
            adapter_name: Name of adapter to reset
        """
        self.adapter_manager.reset_adapter(adapter_name)
    
    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def __repr__(self):
        return (
            f"DebertaV3WeightLoRA("
            f"model={self.model_name}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"K={self.K}, "
            f"active_adapters={len(self.get_active_adapters())})"
        )


def create_deberta_weightlora(
    model_name: str = "microsoft/deberta-v3-base",
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05,
    K: int = 10,
    target_modules: Optional[List[str]] = None
) -> Tuple[DebertaV3WeightLoRA, Any]:
    """
    Factory function to create DeBERTaV3 model with WeightLoRA.
    
    Args:
        model_name: HuggingFace model name
        rank: LoRA rank r
        alpha: LoRA scaling factor α
        dropout: Dropout probability
        K: Maximum active adapters
        target_modules: Target module names for LoRA
        
    Returns:
        Tuple of (DebertaV3WeightLoRA model, tokenizer)
    """
    model = DebertaV3WeightLoRA(
        model_name=model_name,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        K=K,
        target_modules=target_modules
    )
    return model, model.tokenizer


# Target modules for DeBERTaV3 (following paper specification)
# These are the layers where LoRA adapters are applied
DEBERTA_TARGET_MODULES = [
    "attention.q_proj",
    "attention.k_proj",
    "attention.v_proj",
    "attention.out_proj",
]


if __name__ == "__main__":
    # Test DeBERTaV3 WeightLoRA wrapper
    print("Testing DeBERTaV3 WeightLoRA wrapper...")
    
    # Create model
    model, tokenizer = create_deberta_weightlora(
        model_name="microsoft/deberta-v3-base",
        rank=8,
        alpha=32.0,
        K=10
    )
    
    # Check parameter counts
    params = model.get_param_count()
    print(f"Total params: {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")
    print(f"Adapter params (total): {params['adapter_total']:,}")
    print(f"Adapter params (active): {params['adapter_active']:,}")
    print(f"Parameter reduction: {params['reduction']:.2f}%")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 512))
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Output shape: {outputs.logits.shape}")
    
    print("Test completed successfully!")
