"""
WeightLoRA Models Package

This package provides model wrappers and adapter management for WeightLoRA implementation.
Includes standard LoRA baseline and WeightLoRA with adaptive weight control.
"""

from .adapter_base import (
    LoRAAdapter,
    LoRALinear,
    LoRAEmbedding,
    create_lora_adapter
)

from .weightlora_layer import (
    WeightLoRALayer,
    WeightLoRAWrapper,
    create_weightlora_layer
)

from .adapter_manager import (
    AdapterManager,
    is_target_layer,
    create_lora_adapter as create_adapter
)

__all__ = [
    # Adapter base components
    'LoRAAdapter',
    'LoRALinear',
    'LoRAEmbedding',
    'create_lora_adapter',
    
    # WeightLoRA components
    'WeightLoRALayer',
    'WeightLoRAWrapper',
    'create_weightlora_layer',
    
    # Adapter management
    'AdapterManager',
    'is_target_layer',
    'create_adapter',
]
