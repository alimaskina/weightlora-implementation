"""
Rank Expansion Module for WeightLoRA+ Two-Phase Training
Implements rank expansion for selected adapters after initial adapter selection.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from models.adapter_base import LoRAAdapter


def _set_adapter_params(adapter, A: nn.Parameter, B: nn.Parameter, r: int):
    """Set A, B, r on any adapter type (LoRAAdapter or LoRALinear wrapper)."""
    if hasattr(adapter, 'update_lora_params'):
        adapter.update_lora_params(A, B, r)
    else:
        adapter.A = A
        adapter.B = B
        adapter.r = r


class RankExpander:
    """
    Implements rank expansion for WeightLoRA+ two-phase training.
    
    Phase 1: Adapter selection with small rank (r=4)
    Phase 2: Rank expansion for selected adapters (r=8, 16, etc.)
    
    Reference: WeightLoRA paper, Section 4.2, Algorithm 2
    """
    
    def __init__(
        self,
        selected_adapters: List[str],
        initial_rank: int = 4,
        target_rank: int = 8,
        strategy: str = 'random'
    ):
        """
        Initialize rank expander.
        
        Args:
            selected_adapters: List of adapter names/indices to expand
            initial_rank: Rank used in Phase 1 (default: 4)
            target_rank: Target rank for Phase 2 (default: 8)
            strategy: Expansion strategy - 'random' or 'qr_factorization'
        """
        self.selected_adapters = selected_adapters
        self.initial_rank = initial_rank
        self.target_rank = target_rank
        self.strategy = strategy
        
        # Track expansion history
        self.expansion_history = []
        
    def expand_rank(self, adapter: LoRAAdapter) -> None:
        """
        Expand rank from r_old to r_new for a selected adapter.
        
        Implements Eq 5 from paper:
        - Strategy 1: Random extension with N(0, 0.1) initialization
        - Strategy 2: QR factorization to preserve current representation
        
        Args:
            adapter: LoRAAdapter instance to expand
        """
        r_old = adapter.r
        r_new = self.target_rank
        
        if r_new <= r_old:
            return  # No expansion needed
        
        # Store old parameters for reference
        old_A = adapter.A.detach().clone()
        old_B = adapter.B.detach().clone()
        
        if self.strategy == 'random':
            # Strategy 1: Random extension
            # A: (in_features, r_old) -> (in_features, r_new)
            # B: (r_old, out_features) -> (r_new, out_features)
            
            # Extend A along rank dim: (in_features, r_old) -> (in_features, r_new)
            A_extended = torch.cat([
                old_A,
                torch.randn(old_A.shape[0], r_new - r_old) * 0.1
            ], dim=1)

            # Extend B along rank dim: (r_old, out_features) -> (r_new, out_features)
            B_extended = torch.cat([
                old_B,
                torch.zeros(r_new - r_old, old_B.shape[1])
            ], dim=0)
            
            # Update adapter parameters
            _set_adapter_params(adapter, nn.Parameter(A_extended), nn.Parameter(B_extended), r_new)
            
        elif self.strategy == 'qr_factorization':
            # Strategy 2: QR factorization preserves representation
            # Q, R = QR(A_old)
            # A_new = Q @ [I | 0]  (extend with identity and zeros)
            
            Q, R = torch.linalg.qr(old_A)
            
            # Create extended matrix: [I | 0] where I is r_old x r_old
            # and 0 is (r_new - r_old) x r_old
            I = torch.eye(r_old)
            zero_block = torch.zeros(r_new - r_old, r_old)
            extended_block = torch.cat([I, zero_block], dim=1)
            
            # A_new = Q @ extended_block
            A_extended = Q @ extended_block
            
            # Extend B with zeros: (r_old, out_features) -> (r_new, out_features)
            B_extended = torch.cat([
                old_B,
                torch.zeros(r_new - r_old, old_B.shape[1])
            ], dim=0)
            
            # Update adapter parameters
            _set_adapter_params(adapter, nn.Parameter(A_extended), nn.Parameter(B_extended), r_new)
            
        else:
            raise ValueError(f"Unknown expansion strategy: {self.strategy}")
        
        # Record expansion
        self.expansion_history.append({
            'adapter': adapter,
            'old_rank': r_old,
            'new_rank': r_new,
            'strategy': self.strategy
        })
        
    def expand_rq_factorization(self, adapter: LoRAAdapter) -> None:
        """
        Alternative rank expansion using RQ factorization.
        
        RQ factorization can be more numerically stable for certain matrices.
        
        Args:
            adapter: LoRAAdapter instance to expand
        """
        r_old = adapter.r
        r_new = self.target_rank
        
        if r_new <= r_old:
            return
        
        old_A = adapter.A.detach().clone()
        
        # RQ factorization: A = R @ Q (vs QR: A = Q @ R)
        R, Q = torch.linalg.rq(old_A)
        
        # Extend Q with zeros
        I = torch.eye(r_old)
        zero_block = torch.zeros(r_new - r_old, r_old)
        extended_block = torch.cat([I, zero_block], dim=1)
        
        # A_new = R @ extended_block
        A_extended = R @ extended_block
        
        adapter.A = nn.Parameter(A_extended)
        adapter.r = r_new
        
    def expand_all_selected(self, adapter_dict: Dict[str, LoRAAdapter]) -> None:
        """
        Expand rank for all selected adapters.
        
        Args:
            adapter_dict: Dictionary mapping adapter names to LoRAAdapter instances
        """
        for adapter_name in self.selected_adapters:
            if adapter_name in adapter_dict:
                adapter = adapter_dict[adapter_name]
                self.expand_rank(adapter)
                print(f"Expanded adapter {adapter_name}: r={self.initial_rank} -> r={self.target_rank}")
            else:
                print(f"Warning: Adapter {adapter_name} not found")
                
    def get_expansion_stats(self) -> Dict:
        """
        Get statistics about rank expansions performed.
        
        Returns:
            Dictionary with expansion statistics
        """
        if not self.expansion_history:
            return {
                'total_expansions': 0,
                'total_rank_increase': 0,
                'strategies_used': []
            }
        
        total_increase = sum(
            exp['new_rank'] - exp['old_rank']
            for exp in self.expansion_history
        )
        
        strategies = list(set(exp['strategy'] for exp in self.expansion_history))
        
        return {
            'total_expansions': len(self.expansion_history),
            'total_rank_increase': total_increase,
            'strategies_used': strategies,
            'expansion_details': self.expansion_history
        }


class RankExpansionConfig:
    """
    Configuration class for rank expansion parameters.
    """
    
    def __init__(
        self,
        phase_1_rank: int = 4,
        phase_2_rank: int = 8,
        expansion_strategy: str = 'random',
        expansion_threshold: float = 0.5
    ):
        """
        Initialize rank expansion configuration.
        
        Args:
            phase_1_rank: Rank used in Phase 1 (default: 4)
            phase_2_rank: Rank used in Phase 2 (default: 8)
            expansion_strategy: 'random' or 'qr_factorization'
            expansion_threshold: Threshold for deciding when to expand
        """
        self.phase_1_rank = phase_1_rank
        self.phase_2_rank = phase_2_rank
        self.expansion_strategy = expansion_strategy
        self.expansion_threshold = expansion_threshold


def expand_rank_random(adapter: LoRAAdapter, target_rank: int) -> None:
    """Expand adapter rank using random initialization strategy."""
    expander = RankExpander(
        selected_adapters=[],
        initial_rank=adapter.r,
        target_rank=target_rank,
        strategy='random',
    )
    expander.expand_rank(adapter)


def expand_rank_qr(adapter: LoRAAdapter, target_rank: int) -> None:
    """Expand adapter rank using QR factorization strategy."""
    expander = RankExpander(
        selected_adapters=[],
        initial_rank=adapter.r,
        target_rank=target_rank,
        strategy='qr_factorization',
    )
    expander.expand_rank(adapter)


def create_rank_expander(
    selected_adapters: List[str],
    initial_rank: int = 4,
    target_rank: int = 8,
    strategy: str = 'random'
) -> RankExpander:
    """
    Factory function to create a RankExpander instance.
    
    Args:
        selected_adapters: List of adapter names to expand
        initial_rank: Initial rank (Phase 1)
        target_rank: Target rank (Phase 2)
        strategy: Expansion strategy
        
    Returns:
        Configured RankExpander instance
    """
    return RankExpander(
        selected_adapters=selected_adapters,
        initial_rank=initial_rank,
        target_rank=target_rank,
        strategy=strategy
    )
