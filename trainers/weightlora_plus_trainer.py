"""
WeightLoRA+ Two-Phase Training Implementation

This module implements the WeightLoRA+ training pipeline with rank expansion,
following Algorithm 2 from the WeightLoRA paper. The two-phase approach:
- Phase 1: Adapter selection with initial rank (r=4)
- Phase 2: Rank expansion for selected adapters (r=8, 16, etc.)

Key Features:
- Two-phase training with phase transition at T steps
- Rank expansion using random or QR factorization strategies
- Performance improvement over standard WeightLoRA (Table 2, Page 13)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from models.adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding
from models.adapter_manager import AdapterManager, is_target_layer
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper
from algorithms.sto_iht import StoIHTOptimizer, StoIHTTrainer
from algorithms.weight_optimizer import WeightOptimizer, WeightOptimizerConfig
from algorithms.rank_expander import RankExpander, create_rank_expander
from trainers.trainer_utils import (
    load_model, load_dataset, create_dataloaders, 
    save_checkpoint, load_checkpoint, setup_device, set_seed,
    compute_metrics, load_config
)
from utils.evaluation import compute_glue_metrics, compute_squad_metrics, compute_rouge_metrics


class WeightLoRAPlusTrainer:
    """
    Two-phase training pipeline for WeightLoRA+ with rank expansion.
    
    Implements Algorithm 2 from the paper:
    Phase 1: Select top-K adapters with initial rank r_init
    Phase 2: Expand rank of selected adapters to r_target
    
    Performance Target: WeightLoRA+ > WeightLoRA > LoRA (Table 2)
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        K: int = 10,
        phase_1_steps: int = 200,
        rank_expansion: int = 8,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        validation_freq: int = 50,
        seed: int = 42,
        config_path: Optional[str] = None
    ):
        """
        Initialize WeightLoRA+ trainer.
        
        Args:
            model_name: Name of pretrained model (deberta-v3-base, bart-large, llama-3-7b)
            dataset_name: Name of dataset (glue, squad, xsum, cnn_dailymail)
            K: Number of adapters to keep (sparsity level)
            phase_1_steps: Number of steps in Phase 1 (adapter selection)
            rank_expansion: Target rank for Phase 2
            rank: Initial rank for Phase 1
            alpha: LoRA scaling factor (α/r)
            dropout: Dropout probability for adapters
            lr: Learning rate for adapter parameters
            device: GPU/CPU device
            batch_size: Training batch size
            num_epochs: Total training epochs
            validation_freq: Frequency of validation checks
            seed: Random seed for reproducibility
            config_path: Path to YAML configuration file
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.K = K
        self.phase_1_steps = phase_1_steps
        self.rank_expansion = rank_expansion
        self.rank = rank  # Initial rank for Phase 1
        self.alpha = alpha
        self.dropout = dropout
        self.lr = lr
        self.device = device if device else setup_device()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.seed = seed
        
        # Set random seed
        set_seed(seed)
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            self.config = load_config(config_path)
        
        # Load model and dataset
        self.model = load_model(model_name, num_labels=1, device=self.device)
        self.dataset = load_dataset(dataset_name, max_length=128)
        
        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.dataset, batch_size=self.batch_size, device=self.device
        )
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            self.model, K=K, rank=rank, alpha=alpha, dropout=dropout,
            target_modules=self.config.get('target_modules', ['attention.q_proj', 'attention.k_proj', 'attention.v_proj', 'attention.out_proj'])
        )
        
        # Initialize weight optimizer
        n_layers = len(self.adapter_manager.manager)
        self.weight_optimizer = WeightOptimizer(
            n_layers=n_layers, K=K, alpha=alpha, lr=lr
        )
        
        # Initialize StoIHT optimizer
        self.stoiht = StoIHTOptimizer(K=K, initial_learning_rate=0.1)
        
        # Initialize rank expander for Phase 2
        self.rank_expander = create_rank_expander(
            selected_adapters=[],
            initial_rank=rank,
            target_rank=rank_expansion,
            strategy='random'
        )
        
        # Training state
        self.optimizer = optim.Adam(
            list(self.adapter_manager.model.parameters()),
            lr=lr
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.active_indices = []
        self.phase = 1  # 1 = Phase 1, 2 = Phase 2
        
    def train(self, epochs: int = None, validation_freq: int = None) -> Dict[str, Any]:
        """
        Execute two-phase WeightLoRA+ training.
        
        Args:
            epochs: Number of epochs (uses self.num_epochs if None)
            validation_freq: Validation frequency (uses self.validation_freq if None)
            
        Returns:
            Dictionary with training statistics and final model
        """
        if epochs is None:
            epochs = self.num_epochs
        if validation_freq is None:
            validation_freq = self.validation_freq
        
        print("=" * 80)
        print("WeightLoRA+ Two-Phase Training")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"K (sparsity): {self.K}")
        print(f"Phase 1 steps: {self.phase_1_steps}")
        print(f"Phase 2 rank expansion: {self.rank_expansion}")
        print(f"Initial rank: {self.rank}")
        print("=" * 80)
        
        # Phase 1: Adapter Selection
        print("\n" + "=" * 80)
        print("PHASE 1: ADAPTER SELECTION")
        print("=" * 80)
        phase_1_results = self._train_phase_1(epochs, validation_freq)
        
        # Get selected adapters after Phase 1
        selected_adapters = self.active_indices
        print(f"\nPhase 1 Complete. Selected {len(selected_adapters)} adapters.")
        
        # Phase 2: Rank Expansion
        print("\n" + "=" * 80)
        print("PHASE 2: RANK EXPANSION")
        print("=" * 80)
        phase_2_results = self._train_phase_2(epochs, validation_freq)
        
        # Apply final disconnections
        self.apply_disconnections()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Final active adapters: {len(self.active_indices)}")
        print(f"Active indices: {self.active_indices}")
        
        return {
            'phase_1_results': phase_1_results,
            'phase_2_results': phase_2_results,
            'final_active_adapters': self.active_indices,
            'model': self.adapter_manager.model
        }
    
    def _train_phase_1(self, epochs: int, validation_freq: int) -> Dict[str, Any]:
        """
        Phase 1: Train with initial rank to select top-K adapters.
        
        This phase focuses on adapter selection using StoIHT-based weight optimization.
        After T steps, adapters with ω_i ≈ 0 are permanently disconnected.
        
        Args:
            epochs: Number of epochs for Phase 1
            validation_freq: Frequency of validation checks
            
        Returns:
            Dictionary with Phase 1 training statistics
        """
        phase_stats = {
            'train_losses': [],
            'val_scores': [],
            'active_adapters': []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch()
            phase_stats['train_losses'].append(train_loss)
            
            # Validation
            if (epoch + 1) % validation_freq == 0:
                val_score = self._validate()
                phase_stats['val_scores'].append(val_score)
                phase_stats['active_adapters'].append(len(self.active_indices))
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Score: {val_score:.4f}, Active: {len(self.active_indices)}")
            
            # Apply disconnections after T steps (end of Phase 1)
            if epoch == self.phase_1_steps:
                print(f"\nApplying adapter disconnections after {self.phase_1_steps} steps...")
                self.apply_disconnections()
                print(f"Active adapters: {len(self.active_indices)}")
        
        return phase_stats
    
    def _train_phase_2(self, epochs: int, validation_freq: int) -> Dict[str, Any]:
        """
        Phase 2: Train with expanded rank for selected adapters.
        
        Expands rank of selected adapters from Phase 1 to target rank.
        This improves performance while maintaining sparsity.
        
        Args:
            epochs: Number of epochs for Phase 2
            validation_freq: Frequency of validation checks
            
        Returns:
            Dictionary with Phase 2 training statistics
        """
        phase_stats = {
            'train_losses': [],
            'val_scores': [],
            'active_adapters': []
        }
        
        # Expand ranks of selected adapters
        print(f"\nExpanding rank of {len(self.active_indices)} selected adapters...")
        self._expand_selected_ranks()
        
        for epoch in range(self.phase_1_steps, epochs):
            # Training
            train_loss = self._train_epoch()
            phase_stats['train_losses'].append(train_loss)
            
            # Validation
            if (epoch + 1) % validation_freq == 0:
                val_score = self._validate()
                phase_stats['val_scores'].append(val_score)
                phase_stats['active_adapters'].append(len(self.active_indices))
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Score: {val_score:.4f}, Active: {len(self.active_indices)}")
            
            # Update weight vector with StoIHT
            gradients = self.weight_optimizer.compute_gradients()
            updated_weights = self.stoiht.update(
                self.weight_optimizer.weight, gradients
            )
        
        return phase_stats
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.adapter_manager.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.adapter_manager.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Step optimizer
            self.optimizer.step()
            
            # Update weight vector with StoIHT
            gradients = self.weight_optimizer.compute_gradients()
            updated_weights = self.stoiht.update(
                self.weight_optimizer.weight, gradients
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self) -> float:
        """
        Validate model on validation set.
        
        Returns:
            Validation loss or score
        """
        self.adapter_manager.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.adapter_manager.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _expand_selected_ranks(self):
        """
        Expand rank of selected adapters from Phase 1 to Phase 2.
        
        Uses the rank expander to increase adapter rank for selected adapters.
        """
        for adapter_name in self.active_indices:
            if adapter_name in self.adapter_manager.manager:
                adapter_instance = self.adapter_manager.manager[adapter_name]
                self.rank_expander.expand_rank(adapter_instance)
                print(f"Expanded rank for adapter {adapter_name}: {self.rank} -> {self.rank_expansion}")
    
    def apply_disconnections(self):
        """
        Permanently disconnect adapters where ω_i = 0.
        
        Called after Phase 1 to freeze selected adapters.
        """
        active_mask = (self.weight_optimizer.weight > 1e-6).cpu()
        self.active_indices = active_mask.nonzero().tolist()
        
        print(f"Active adapters: {len(self.active_indices)}")
        print(f"Active indices: {self.active_indices}")
        
        for i, (active_idx, adapter) in enumerate(
            enumerate(self.adapter_manager.manager)
        ):
            if i not in self.active_indices:
                # Zero out adapter parameters
                adapter.A.data.zero_()
                adapter.B.data.zero_()
                adapter.weight.data.zero_()
                adapter.register_buffer('disabled', torch.tensor([True]))
    
    def get_active_adapters(self) -> List[int]:
        """
        Get list of active adapter indices.
        
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
        return self.weight_optimizer.weight
    
    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        save_checkpoint(
            self.adapter_manager.model,
            self.optimizer,
            self.phase,
            path,
            self.weight_optimizer.weight
        )
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        model, optimizer, epoch, weight_vector = load_checkpoint(path)
        self.adapter_manager.model = model
        self.optimizer = optimizer
        self.phase = epoch
        self.weight_optimizer.weight = weight_vector


def create_weightlora_plus_trainer(
    model_name: str,
    dataset_name: str,
    K: int = 10,
    phase_1_steps: int = 200,
    rank_expansion: int = 8,
    **kwargs
) -> WeightLoRAPlusTrainer:
    """
    Factory function to create WeightLoRA+ trainer instance.
    
    Args:
        model_name: Name of pretrained model
        dataset_name: Name of dataset
        K: Number of adapters to keep
        phase_1_steps: Number of steps in Phase 1
        rank_expansion: Target rank for Phase 2
        **kwargs: Additional trainer parameters
        
    Returns:
        Configured WeightLoRAPlusTrainer instance
    """
    return WeightLoRAPlusTrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        K=K,
        phase_1_steps=phase_1_steps,
        rank_expansion=rank_expansion,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    trainer = WeightLoRAPlusTrainer(
        model_name="deberta-v3-base",
        dataset_name="glue",
        K=10,
        phase_1_steps=200,
        rank_expansion=8,
        lr=3e-4,
        batch_size=16
    )
    
    results = trainer.train(epochs=10, validation_freq=50)
    print(f"\nTraining complete. Final active adapters: {len(results['final_active_adapters'])}")
