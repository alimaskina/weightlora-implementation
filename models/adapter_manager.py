"""
Adapter Manager for WeightLoRA

This module manages the lifecycle of LoRA adapters, including initialization,
activation/disconnection, and tracking of active adapters.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from .adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding, create_lora_adapter


def is_target_layer(layer: nn.Module) -> bool:
    """Check if a layer is a target for LoRA adaptation."""
    layer_type = type(layer).__name__
    target_types = [
        'Linear', 'Embedding', 'Conv1d', 'Conv2d', 'Conv3d',
        'Linear', 'Embedding'
    ]
    return any(t in layer_type for t in target_types)


class AdapterManager:
    """
    Manages LoRA adapters for WeightLoRA implementation.
    
    Handles:
    - Initialization of adapters for all target layers
    - Activation/disconnection of adapters via weight control
    - Tracking of active adapters for sparsity constraints
    - Batched operations for efficiency
    """
    
    def __init__(
        self,
        model: nn.Module,
        K: int = 10,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize adapter manager.
        
        Args:
            model: Pretrained model to add adapters to
            K: Maximum number of adapters to keep (sparsity constraint)
            rank: LoRA rank r for A×B decomposition
            alpha: Scaling factor α for LoRA
            dropout: Dropout probability for regularization
            target_modules: List of module names to apply adapters to
        """
        self.K = K
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.manager: Dict[str, LoRAAdapter] = {}
        self.adapter_metadata: Dict[str, Dict] = {}
        self.model = model
        self.target_modules = target_modules or self._get_default_target_modules()
        
        self.initialize_adapters()
    
    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules for transformer models."""
        # Default: attention layers and output layers
        return ['query', 'key', 'value', 'output']
    
    def initialize_adapters(self):
        """
        Add LoRA adapters to all target layers in the model.
        
        For each target layer, creates a LoRA adapter with:
        - A matrix: N(0, 0.1) initialization
        - B matrix: zeros initialization
        - Weight parameter ω: 1.0 (fully active)
        """
        # Find all target layers
        target_layers = self._find_target_layers()
        
        for layer_name, layer in target_layers:
            # Create adapter for this layer
            adapter = create_lora_adapter(
                layer_type=type(layer).__name__,
                in_features=layer.in_features if hasattr(layer, 'in_features') else layer.weight.shape[0],
                out_features=layer.out_features if hasattr(layer, 'out_features') else layer.weight.shape[1],
                r=self.rank,
                alpha=self.alpha,
                dropout=self.dropout
            )
            
            # Store adapter
            self.manager[layer_name] = adapter
            
            # Store metadata
            self.adapter_metadata[layer_name] = {
                'in_features': adapter.A.shape[0],
                'out_features': adapter.B.shape[1],
                'rank': self.rank,
                'alpha': self.alpha,
                'weight': torch.ones(1),  # Initialize weight to 1
                'active': True
            }
            
            # Attach adapter to layer
            layer.adapter = adapter
    
    def _find_target_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find all layers matching target modules."""
        target_layers = []
        
        for name, module in self.model.named_modules():
            if is_target_layer(module):
                # Check if this module name matches our target modules
                for target in self.target_modules:
                    if target in name:
                        target_layers.append((name, module))
                        break
        
        return target_layers
    
    def activate_adapters(self, adapter_names: List[str]):
        """
        Set weight=1 for specified adapters (fully active).
        
        Args:
            adapter_names: List of adapter names to activate
        """
        for name in adapter_names:
            if name in self.manager:
                self.manager[name].weight.data = torch.ones(1)
                self.adapter_metadata[name]['weight'] = torch.ones(1)
                self.adapter_metadata[name]['active'] = True
            else:
                raise ValueError(f"Adapter {name} not found in manager")
    
    def deactivate_adapters(self, adapter_names: List[str]):
        """
        Set weight=0 for specified adapters (disconnected).
        
        Args:
            adapter_names: List of adapter names to deactivate
        """
        for name in adapter_names:
            if name in self.manager:
                self.manager[name].weight.data = torch.zeros(1)
                self.adapter_metadata[name]['weight'] = torch.zeros(1)
                self.adapter_metadata[name]['active'] = False
            else:
                raise ValueError(f"Adapter {name} not found in manager")
    
    def get_active_adapters(self) -> List[str]:
        """
        Return list of currently active adapter names.
        
        Returns:
            List of adapter names where weight > 1e-6
        """
        active = []
        for name, adapter in self.manager.items():
            if adapter.weight.data > 1e-6:
                active.append(name)
        return active
    
    def get_disconnected_adapters(self) -> List[str]:
        """
        Return list of currently disconnected adapter names.
        
        Returns:
            List of adapter names where weight <= 1e-6
        """
        disconnected = []
        for name, adapter in self.manager.items():
            if adapter.weight.data <= 1e-6:
                disconnected.append(name)
        return disconnected
    
    def set_weight(self, adapter_name: str, value: float):
        """
        Set weight for a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            value: Weight value (0.0 for disconnected, 1.0 for active)
        """
        if adapter_name in self.manager:
            self.manager[adapter_name].weight.data = torch.full(
                torch.ones(1), value
            )
            self.adapter_metadata[adapter_name]['weight'] = torch.full(
                torch.ones(1), value
            )
            self.adapter_metadata[adapter_name]['active'] = value > 1e-6
    
    def get_weight_vector(self) -> torch.Tensor:
        """
        Get the weight vector for all adapters.
        
        Returns:
            Tensor of shape (n_adapters,) containing weight values
        """
        weights = []
        for name in self.manager.keys():
            weights.append(self.manager[name].weight.data)
        return torch.stack(weights)
    
    def set_weight_vector(self, weights: torch.Tensor):
        """
        Set weight vector for all adapters.
        
        Args:
            weights: Tensor of shape (n_adapters,) containing weight values
        """
        if len(weights) != len(self.manager):
            raise ValueError(f"Weight vector length {len(weights)} doesn't match "
                           f"number of adapters {len(self.manager)}")
        
        for i, (name, adapter) in enumerate(self.manager.items()):
            adapter.weight.data = weights[i]
            self.adapter_metadata[name]['weight'] = weights[i]
    
    def disconnect_adapters_by_weight(self, threshold: float = 1e-6):
        """
        Disconnect all adapters with weight below threshold.
        
        Args:
            threshold: Weight threshold for disconnection
        """
        for name, adapter in self.manager.items():
            if adapter.weight.data <= threshold:
                self.deactivate_adapters([name])
    
    def get_active_count(self) -> int:
        """
        Get number of currently active adapters.
        
        Returns:
            Count of adapters with weight > 1e-6
        """
        return len(self.get_active_adapters())
    
    def get_total_param_count(self) -> int:
        """
        Get total number of trainable parameters in all adapters.
        
        Returns:
            Total parameter count across all adapters
        """
        total = 0
        for adapter in self.manager.values():
            total += adapter.A.numel() + adapter.B.numel()
        return total
    
    def get_active_param_count(self) -> int:
        """
        Get number of trainable parameters in active adapters only.
        
        Returns:
            Parameter count for adapters with weight > 1e-6
        """
        total = 0
        for name, adapter in self.manager.items():
            if adapter.weight.data > 1e-6:
                total += adapter.A.numel() + adapter.B.numel()
        return total
    
    def zero_adapter(self, adapter_name: str):
        """
        Permanently zero out an adapter's parameters.
        
        Args:
            adapter_name: Name of the adapter to zero
        """
        if adapter_name in self.manager:
            adapter = self.manager[adapter_name]
            adapter.A.data.zero_()
            adapter.B.data.zero_()
            adapter.weight.data.zero_()
            self.adapter_metadata[adapter_name]['active'] = False
    
    def reset_adapter(self, adapter_name: str):
        """
        Reset an adapter to initial state (weight=1, random A, zero B).
        
        Args:
            adapter_name: Name of the adapter to reset
        """
        if adapter_name in self.manager:
            adapter = self.manager[adapter_name]
            # Reset weight
            adapter.weight.data = torch.ones(1)
            # Reset A to random
            adapter.A.data = torch.randn(adapter.A.shape) * 0.1
            # Reset B to zeros
            adapter.B.data.zero_()
            self.adapter_metadata[adapter_name]['active'] = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all active adapters.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with adapter contributions
        """
        # This would need to be integrated with the actual model forward
        # For now, return input as placeholder
        return x
    
    def __len__(self):
        """Return number of adapters managed."""
        return len(self.manager)
    
    def __getitem__(self, key):
        """Get adapter by name or index."""
        if isinstance(key, str):
            return self.manager[key]
        else:
            return list(self.manager.values())[key]
    
    def __iter__(self):
        """Iterate over adapter names."""
        return iter(self.manager.keys())
    
    def keys(self):
        """Return adapter names."""
        return self.manager.keys()
    
    def values(self):
        """Return adapter instances."""
        return self.manager.values()
    
    def items(self):
        """Return (name, adapter) pairs."""
        return self.manager.items()
