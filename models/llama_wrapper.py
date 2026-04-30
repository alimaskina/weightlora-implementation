"""
Llama-3-7B WeightLoRA Wrapper

Provides Llama-3-7B model integration with WeightLoRA adapters for causal language modeling tasks.
Implements adaptive adapter selection with trainable weight vectors for sparsity control.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding, create_lora_adapter
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper, create_weightlora_layer
from models.adapter_manager import AdapterManager, is_target_layer, create_lora_adapter as create_adapter


class LlamaWeightLoRA(nn.Module):
    """
    Llama-3-7B model with WeightLoRA adapters for causal language modeling.
    
    Implements adaptive adapter selection with trainable weight vectors ω_i
    to achieve sparsity constraints ||ω||_0 ≤ K while maintaining performance.
    
    Target layers (following paper specification):
    - Attention: query, key, value, output projections
    - MLP: gate and up projections
    
    Args:
        model_name: Path to pretrained Llama-3-7B model
        rank: LoRA rank r (default: 8)
        alpha: LoRA scaling factor α (default: 32.0)
        dropout: Dropout probability (default: 0.05)
        K: Maximum number of active adapters (default: 10)
        target_modules: List of target module names for LoRA adaptation
    """
    
    # Target modules for Llama-3-7B following paper specification
    TARGET_MODULES = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-7b",
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        K: int = 10,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.K = K
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Load pretrained model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Wrap with WeightLoRA adapters
        self.adapter_manager = AdapterManager(
            self.base_model,
            K=K,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules or self.TARGET_MODULES
        )
        
        # Initialize weight vector
        self.weight_vector = nn.Parameter(torch.ones(len(self.adapter_manager)))
        
        # Optimizer for weight vector (separate from adapter parameters)
        self.weight_optimizer = torch.optim.Adam(
            [self.weight_vector],
            lr=3e-4
        )
        
        # Store original model for reference
        self.original_model = self.base_model
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """
        Forward pass with WeightLoRA adapter selection.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments passed to base model
            
        Returns:
            Model outputs with logits
        """
        # Forward pass through WeightLoRA-wrapped model
        outputs = self.adapter_manager.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        return outputs
    
    def get_active_adapters(self) -> List[str]:
        """Return list of currently active adapter names."""
        return self.adapter_manager.get_active_adapters()
    
    def get_disconnected_adapters(self) -> List[str]:
        """Return list of currently disconnected adapter names."""
        return self.adapter_manager.get_disconnected_adapters()
    
    def disconnect_adapters(self, adapter_names: Optional[List[str]] = None):
        """
        Disconnect specified adapters by setting weight to 0.
        
        Args:
            adapter_names: List of adapter names to disconnect. If None, disconnect all inactive adapters.
        """
        if adapter_names is None:
            adapter_names = self.get_disconnected_adapters()
        self.adapter_manager.deactivate_adapters(adapter_names)
    
    def enable_adapters(self, adapter_names: List[str]):
        """Enable specified adapters by setting weight to 1."""
        self.adapter_manager.activate_adapters(adapter_names)
    
    def get_weight_vector(self) -> torch.Tensor:
        """Get current weight vector."""
        return self.weight_vector.clone()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """Set weight vector."""
        self.weight_vector.data = weights.clone()
    
    def apply_disconnections(self):
        """
        Permanently disconnect adapters where ω_i = 0.
        
        Called after T training steps to freeze selected adapters.
        """
        self.adapter_manager.apply_disconnections()
    
    def get_param_count(self) -> Dict[str, int]:
        """
        Get parameter counts for memory reduction verification.
        
        Returns:
            Dictionary with total, trainable, and adapter parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = self.adapter_manager.get_total_param_count()
        active_adapter_params = self.adapter_manager.get_active_param_count()
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "adapter_total": adapter_params,
            "adapter_active": active_adapter_params,
            "reduction": (1 - active_adapter_params / adapter_params) * 100 if adapter_params > 0 else 0
        }
    
    def zero_adapter(self, adapter_name: str):
        """Zero out adapter parameters for specified adapter."""
        self.adapter_manager.zero_adapter(adapter_name)
    
    def reset_adapter(self, adapter_name: str):
        """Reset adapter parameters to initial state."""
        self.adapter_manager.reset_adapter(adapter_name)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.tokenizer.set_seed(seed)
    
    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_model, name)


def create_llama_weightlora(
    model_name: str = "meta-llama/Llama-3-7b",
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05,
    K: int = 10,
    target_modules: Optional[List[str]] = None
) -> Tuple[LlamaWeightLoRA, Any]:
    """
    Factory function to create Llama-3-7B model with WeightLoRA.
    
    Args:
        model_name: Path to pretrained Llama-3-7B model
        rank: LoRA rank r
        alpha: LoRA scaling factor α
        dropout: Dropout probability
        K: Maximum number of active adapters
        target_modules: List of target module names
        
    Returns:
        Tuple of (LlamaWeightLoRA model, tokenizer)
    """
    model = LlamaWeightLoRA(
        model_name=model_name,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        K=K,
        target_modules=target_modules
    )
    return model, model.tokenizer


if __name__ == "__main__":
    # Test instantiation
    print("Testing LlamaWeightLoRA instantiation...")
    try:
        model, tokenizer = create_llama_weightlora(
            model_name="meta-llama/Llama-3-7b",
            rank=8,
            K=10
        )
        print(f"Model created successfully")
        print(f"Total params: {model.get_param_count()['total']}")
        print(f"Active adapters: {len(model.get_active_adapters())}")
    except Exception as e:
        print(f"Note: Model download may require internet access: {e}")