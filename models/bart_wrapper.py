"""
BART Model Wrapper for WeightLoRA Implementation

This module provides BART model integration with WeightLoRA adapters,
supporting sequence-to-sequence tasks like summarization (XSum, CNN/DailyMail).
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List, Dict, Optional, Tuple

from .adapter_base import LoRAAdapter, create_lora_adapter
from .weightlora_layer import WeightLoRALayer, WeightLoRAWrapper
from .adapter_manager import AdapterManager, is_target_layer


class BartWeightLoRA(nn.Module):
    """
    BART model with WeightLoRA adapters applied to attention layers.
    
    Implements WeightLoRA for BART-large model, targeting:
    - Attention query, key, value projections
    - Attention output projections
    - Final layer norm
    
    Args:
        model: Pretrained BART model
        rank: LoRA rank r (default: 8)
        alpha: LoRA scaling factor alpha (default: 32)
        dropout: Dropout probability (default: 0.1)
        target_modules: List of module names to apply adapters to
    """
    
    def __init__(
        self,
        model: BartForConditionalGeneration,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        # Default target modules for BART attention layers
        if target_modules is None:
            self.target_modules = [
                'encoder.layers.{layer}.self_attn.q_proj',
                'encoder.layers.{layer}.self_attn.k_proj',
                'encoder.layers.{layer}.self_attn.v_proj',
                'encoder.layers.{layer}.self_attn.out_proj',
                'decoder.layers.{layer}.self_attn.q_proj',
                'decoder.layers.{layer}.self_attn.k_proj',
                'decoder.layers.{layer}.self_attn.v_proj',
                'decoder.layers.{layer}.self_attn.out_proj',
                'decoder.layers.{layer}.cross_attn.q_proj',
                'decoder.layers.{layer}.cross_attn.v_proj',
                'decoder.layers.{layer}.cross_attn.out_proj',
                'decoder.layers.{layer}.fc1',
                'decoder.layers.{layer}.fc2'
            ]
        else:
            self.target_modules = target_modules
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            model=model,
            K=10,  # Default: keep top 10 adapters
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=self.target_modules
        )
        
        # Wrap model with WeightLoRA layers
        self._wrap_model()
        
    def _wrap_model(self):
        """Wrap BART model layers with WeightLoRA adapters."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(
                name.endswith(target.replace('{layer}', ''))
                for target in self.target_modules
            ):
                # Create LoRA adapter
                adapter = create_lora_adapter(
                    'linear',
                    module.in_features,
                    module.out_features,
                    self.rank,
                    self.alpha,
                    self.dropout
                )
                
                # Create WeightLoRA layer
                weightlora_layer = WeightLoRALayer(
                    pretrained_layer=module,
                    adapter=adapter,
                    layer_name=name,
                    rank=self.rank
                )
                
                # Replace original module with WeightLoRA layer
                setattr(self.model, name, weightlora_layer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through BART model with WeightLoRA adapters.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Decoder attention mask
            labels: Labels for loss computation
            **kwargs: Additional keyword arguments
            
        Returns:
            dict: Model outputs including logits and loss
        """
        # Forward pass through wrapped model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def get_active_adapters(self) -> List[str]:
        """Get list of currently active adapter names."""
        return self.adapter_manager.get_active_adapters()
    
    def get_disconnected_adapters(self) -> List[str]:
        """Get list of currently disconnected adapter names."""
        all_adapters = list(self.adapter_manager.manager.keys())
        active_adapters = self.get_active_adapters()
        return [a for a in all_adapters if a not in active_adapters]
    
    def disconnect_adapters(self, adapter_names: List[str]):
        """Disconnect specified adapters by setting weight to 0."""
        self.adapter_manager.deactivate_adapters(adapter_names)
    
    def enable_adapters(self, adapter_names: List[str]):
        """Enable specified adapters by setting weight to 1."""
        self.adapter_manager.activate_adapters(adapter_names)
    
    def get_weight_vector(self) -> torch.Tensor:
        """Get the weight vector ω for all adapters."""
        return self.adapter_manager.get_weight_vector()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """Set the weight vector ω for all adapters."""
        self.adapter_manager.set_weight_vector(weights)
    
    def apply_disconnections(self):
        """Apply permanent disconnections based on weight vector."""
        self.adapter_manager.apply_disconnections()
    
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter count statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        adapter_params = self.adapter_manager.get_total_param_count()
        active_adapter_params = self.adapter_manager.get_active_param_count()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'adapter_total': adapter_params,
            'adapter_active': active_adapter_params,
            'reduction': (1 - active_adapter_params / adapter_params) * 100 if adapter_params > 0 else 0
        }
    
    def zero_adapter(self, adapter_name: str):
        """Zero out a specific adapter's parameters."""
        if adapter_name in self.adapter_manager.manager:
            adapter = self.adapter_manager.manager[adapter_name]
            adapter.A.data.zero_()
            adapter.B.data.zero_()
            adapter.weight.data.zero_()
    
    def reset_adapter(self, adapter_name: str):
        """Reset adapter parameters to initial values."""
        if adapter_name in self.adapter_manager.manager:
            adapter = self.adapter_manager.manager[adapter_name]
            # Reset to initial random values
            adapter.A.data = torch.randn(adapter.A.shape) * 0.1
            adapter.B.data = torch.zeros(adapter.B.shape)
            adapter.weight.data = torch.ones(1)


class BartTokenizerWrapper:
    """
    Wrapper for BART tokenizer with preprocessing utilities.
    
    Handles tokenization for XSum and CNN/DailyMail summarization tasks.
    """
    
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def encode(self, text: str, max_length: int = 512, truncation=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            truncation: Whether to truncate long sequences
            
        Returns:
            dict: Tokenized input with attention mask
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding='max_length',
            return_tensors='pt'
        )
    
    def batch_encode(self, texts: List[str], max_length: int = 512):
        """
        Encode batch of texts to token IDs.
        
        Args:
            texts: List of input text strings
            max_length: Maximum sequence length
            
        Returns:
            dict: Batched tokenized inputs
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True):
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(
            token_ids[0],
            skip_special_tokens=skip_special_tokens
        )
    
    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True):
        """
        Decode batch of token IDs to text.
        
        Args:
            token_ids: Token IDs tensor
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List[str]: List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )


def create_bart_weightlora(
    model_name: str = 'facebook/bart-large-cnn',
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.1,
    K: int = 10
) -> Tuple[BartWeightLoRA, BartTokenizerWrapper]:
    """
    Factory function to create BART model with WeightLoRA.
    
    Args:
        model_name: HuggingFace model name
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        K: Number of adapters to keep (sparsity level)
        
    Returns:
        Tuple[BartWeightLoRA, BartTokenizerWrapper]: Model and tokenizer
    """
    # Load pretrained model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Create WeightLoRA wrapper
    weightlora_model = BartWeightLoRA(
        model=model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        K=K
    )
    
    # Create tokenizer
    tokenizer = BartTokenizerWrapper(model_name)
    
    return weightlora_model, tokenizer


if __name__ == '__main__':
    # Test BART WeightLoRA wrapper
    print("Testing BART WeightLoRA wrapper...")
    
    # Create model (would need actual model download in practice)
    # model, tokenizer = create_bart_weightlora()
    
    # print(f"Model created: {model}")
    # print(f"Active adapters: {model.get_active_adapters()}")
    # print(f"Param count: {model.get_param_count()}")
    
    print("BART WeightLoRA wrapper module loaded successfully.")
