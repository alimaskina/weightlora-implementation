"""
WeightLoRA Training Pipeline - Algorithm 1 Implementation

This module implements the complete WeightLoRA training pipeline including:
- Adapter initialization and management
- StoIHT-based weight optimization
- Phase-based adapter selection and disconnection
- Training loop with validation

References:
- Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models
- Nguyen et al. (2014) - Stochastic Iterative Hard Thresholding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional, Callable
import yaml
import os

from models.adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding
from models.adapter_manager import AdapterManager, is_target_layer
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper
from algorithms.sto_iht import StoIHTOptimizer, StoIHTTrainer
from algorithms.weight_optimizer import WeightOptimizer, WeightOptimizerConfig
from utils.metrics import compute_glue_metrics, compute_squad_metrics, compute_rouge_metrics


class WeightLoRATrainer:
    """
    Main WeightLoRA training pipeline implementing Algorithm 1.
    
    This trainer manages:
    1. Model initialization with LoRA adapters
    2. Weight vector optimization with StoIHT
    3. Two-phase training (selection + disconnection)
    4. Validation and metric computation
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        K: int = 10,
        T: int = 200,
        rank: int = 8,
        lr: float = 3e-4,
        alpha: float = 32.0,
        dropout: float = 0.05,
        device: str = 'auto',
        batch_size: int = 32,
        num_epochs: int = 10,
        validation_freq: int = 50,
        seed: int = 42,
        config_path: Optional[str] = None
    ):
        """
        Initialize WeightLoRA trainer.
        
        Args:
            model_name: Name of base model (deberta-v3-base, bart-large, llama-3-7b)
            dataset_name: Name of dataset for training
            K: Number of adapters to keep (sparsity level)
            T: Number of training steps before adapter disconnection
            rank: LoRA rank r
            lr: Learning rate for adapter parameters
            alpha: LoRA scaling factor (α)
            dropout: Dropout probability
            device: Device to train on ('auto', 'cuda', 'cpu')
            batch_size: Training batch size
            num_epochs: Total training epochs
            validation_freq: Frequency of validation (steps)
            seed: Random seed for reproducibility
            config_path: Optional path to YAML config file
        """
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.K = K
        self.T = T
        self.rank = rank
        self.lr = lr
        self.alpha = alpha
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        
        # Load configuration if provided
        self.config = self._load_config(config_path)
        
        # Initialize model and dataset
        self.model = self._load_model(model_name)
        self.dataset = self._load_dataset(dataset_name)
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            model=self.model,
            K=K,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=self.config.get('target_modules', ['query', 'value', 'output'])
        )
        
        # Initialize optimizers
        self.adapter_optimizer = optim.Adam(
            list(self.adapter_manager.adapter_params),
            lr=lr
        )
        
        # Initialize weight optimizer with StoIHT
        self.weight_optimizer = WeightOptimizer(
            n_layers=len(self.adapter_manager),
            K=K,
            alpha=alpha,
            lr=lr
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Training state
        self.current_step = 0
        self.best_val_score = -float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'active_adapters': [],
            'sparsity': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file if provided."""
        if config_path is None:
            return {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_path} not found")
            return {}
    
    def _load_model(self, model_name: str) -> nn.Module:
        """Load pretrained model and wrap with WeightLoRA."""
        # Model loading logic (placeholder - actual implementation in wrappers)
        if model_name == 'deberta-v3-base':
            from models.deberta_wrapper import DeBERTaV3Wrapper
            model = DeBERTaV3Wrapper.from_pretrained('microsoft/deberta-v3-base')
        elif model_name == 'bart-large':
            from models.bart_wrapper import BARTWrapper
            model = BARTWrapper.from_pretrained('facebook/bart-large')
        elif model_name == 'llama-3-7b':
            from models.llama_wrapper import LlamaWrapper
            model = LlamaWrapper.from_pretrained('meta-llama/Llama-3-7b')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with WeightLoRA
        self.adapter_manager.wrap_model(model)
        
        return model
    
    def _load_dataset(self, dataset_name: str) -> Tuple[DataLoader, DataLoader]:
        """Load training and validation datasets."""
        from datasets import load_dataset
        
        if dataset_name == 'glue':
            # Load GLUE dataset
            dataset = load_dataset('glue', 'mnli')
            train_loader = DataLoader(
                dataset['train'],
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                dataset['validation_matched'],
                batch_size=self.batch_size,
                shuffle=False
            )
        elif dataset_name == 'squad':
            # Load SQuAD dataset
            dataset = load_dataset('squad', 'v2.0')
            train_loader = DataLoader(
                dataset['train'],
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                dataset['validation'],
                batch_size=self.batch_size,
                shuffle=False
            )
        elif dataset_name == 'xsum':
            # Load XSum dataset
            dataset = load_dataset('xsum')
            train_loader = DataLoader(
                dataset['train'],
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                dataset['validation'],
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return train_loader, val_loader
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Execute Algorithm 1: WeightLoRA training pipeline.
        
        Returns:
            Dictionary containing training history and final model state
        """
        if epochs is None:
            epochs = self.num_epochs
        
        print(f"Starting WeightLoRA training on {self.dataset_name}")
        print(f"Model: {self.model_name}, K={self.K}, T={self.T}, rank={self.rank}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_train_loss = self._train_epoch()
            
            # Validate periodically
            if (epoch + 1) % self.validation_freq == 0:
                val_score = self._validate()
                
                # Log progress
                self.training_history['train_loss'].append(epoch_train_loss)
                self.training_history['val_loss'].append(val_score)
                self.training_history['active_adapters'].append(
                    self.weight_optimizer.get_active_count()
                )
                self.training_history['sparsity'].append(
                    self._compute_sparsity()
                )
                
                print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, "
                      f"Val Score: {val_score:.4f}, Active: {self.weight_optimizer.get_active_count()}")
                
                # Save best model
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self._save_checkpoint(epoch, val_score)
            
            # Phase transition: Apply adapter disconnections after T steps
            if self.current_step >= self.T:
                print(f"Phase transition at step {self.current_step}")
                self.adapter_manager.apply_disconnections()
                print(f"Active adapters after disconnection: {self.weight_optimizer.get_active_count()}")
            
            self.current_step += 1
        
        # Final validation
        final_val_score = self._validate()
        print(f"Final validation score: {final_val_score:.4f}")
        
        return {
            'training_history': self.training_history,
            'best_val_score': self.best_val_score,
            'final_val_score': final_val_score,
            'active_adapters': self.weight_optimizer.get_active_indices()
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.dataset.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.adapter_manager.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.adapter_optimizer.zero_grad()
            loss.backward()
            
            # Update adapter parameters
            self.adapter_optimizer.step()
            
            # Update weight vector with StoIHT
            gradients = self.weight_optimizer.compute_gradients()
            updated_weights = self.weight_optimizer.stoiht.update(
                self.weight_optimizer.weight, gradients
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate(self) -> float:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.dataset.val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.adapter_manager.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_sparsity(self) -> float:
        """Compute sparsity ratio (fraction of zero weights)."""
        weight_vector = self.weight_optimizer.weight.cpu().numpy()
        zero_count = sum(w == 0 for w in weight_vector)
        return zero_count / len(weight_vector)
    
    def _save_checkpoint(self, epoch: int, val_score: float):
        """Save training checkpoint."""
        checkpoint_path = f"checkpoints/weightlora_epoch_{epoch}_score_{val_score:.4f}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'val_score': val_score,
            'model_state_dict': self.model.state_dict(),
            'adapter_manager_state': self.adapter_manager.state_dict(),
            'weight_optimizer_state': self.weight_optimizer.state_dict(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def get_active_adapters(self) -> List[int]:
        """Get list of active adapter indices."""
        return self.weight_optimizer.get_active_indices()
    
    def get_weight_vector(self) -> torch.Tensor:
        """Get current weight vector."""
        return self.weight_optimizer.weight.clone()
    
    def set_weight_vector(self, weights: torch.Tensor):
        """Set weight vector."""
        self.weight_optimizer.weight.data = weights


class WeightLoRAPlusTrainer(WeightLoRATrainer):
    """
    WeightLoRA+ Training Pipeline - Algorithm 2 Implementation.
    
    Implements two-phase training with rank expansion for selected adapters.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        K: int = 10,
        phase_1_steps: int = 200,
        rank_expansion: int = 8,
        **kwargs
    ):
        """
        Initialize WeightLoRA+ trainer.
        
        Args:
            model_name: Name of base model
            dataset_name: Name of dataset
            K: Number of adapters to keep
            phase_1_steps: Number of steps in Phase 1 (adapter selection)
            rank_expansion: Target rank for expansion in Phase 2
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(model_name, dataset_name, K, **kwargs)
        self.phase_1_steps = phase_1_steps
        self.rank_expansion = rank_expansion
        self.rank_expander = None
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """
        Execute Algorithm 2: WeightLoRA+ two-phase training.
        
        Phase 1: Adapter selection with small rank
        Phase 2: Rank expansion for selected adapters
        """
        if epochs is None:
            epochs = self.num_epochs
        
        print(f"Starting WeightLoRA+ training on {self.dataset_name}")
        print(f"Phase 1: {self.phase_1_steps} steps, Phase 2: rank expansion to {self.rank_expansion}")
        
        # Phase 1: Initial adapter selection
        print("\n=== Phase 1: Adapter Selection ===")
        selected_adapters = self._train_phase_1(self.phase_1_steps)
        
        # Rank expansion for selected adapters
        print("\n=== Expanding ranks for selected adapters ===")
        self._expand_selected_ranks(selected_adapters)
        
        # Phase 2: Further training with expanded ranks
        print("\n=== Phase 2: Rank Expansion Training ===")
        for epoch in range(self.phase_1_steps, epochs):
            self._train_epoch()
            
            if (epoch + 1) % self.validation_freq == 0:
                val_score = self._validate()
                print(f"Phase 2 Epoch {epoch+1}: Val Score: {val_score:.4f}")
        
        # Final validation
        final_val_score = self._validate()
        print(f"\nFinal validation score: {final_val_score:.4f}")
        
        return {
            'phase_1_adapters': selected_adapters,
            'final_val_score': final_val_score,
            'active_adapters': self.weight_optimizer.get_active_indices()
        }
    
    def _train_phase_1(self, steps: int) -> List[int]:
        """Train Phase 1: adapter selection only."""
        for step in range(steps):
            self._train_epoch()
            
            # Apply disconnections after selection
            if step == steps - 1:
                self.adapter_manager.apply_disconnections()
                return self.weight_optimizer.get_active_indices()
        
        return []
    
    def _expand_selected_ranks(self, selected_adapters: List[int]):
        """Expand ranks of selected adapters."""
        from algorithms.rank_expander import RankExpander
        
        self.rank_expander = RankExpander(
            selected_adapters,
            initial_rank=self.rank,
            target_rank=self.rank_expansion
        )
        
        for adapter_idx in selected_adapters:
            if adapter_idx in self.adapter_manager.manager:
                adapter_instance = self.adapter_manager.manager[adapter_idx]
                self.rank_expander.expand_rank(adapter_instance)
                print(f"Expanded rank for adapter {adapter_idx}: {self.rank} -> {self.rank_expansion}")


def create_weightlora_trainer(
    model_name: str,
    dataset_name: str,
    K: int = 10,
    T: int = 200,
    rank: int = 8,
    lr: float = 3e-4,
    alpha: float = 32.0,
    config_path: Optional[str] = None
) -> WeightLoRATrainer:
    """
    Factory function to create WeightLoRA trainer.
    
    Args:
        model_name: Model name
        dataset_name: Dataset name
        K: Sparsity level
        T: Steps before disconnection
        rank: LoRA rank
        lr: Learning rate
        alpha: Scaling factor
        config_path: Optional config file
        
    Returns:
        Configured WeightLoRATrainer instance
    """
    return WeightLoRATrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        K=K,
        T=T,
        rank=rank,
        lr=lr,
        alpha=alpha,
        config_path=config_path
    )


def create_weightlora_plus_trainer(
    model_name: str,
    dataset_name: str,
    K: int = 10,
    phase_1_steps: int = 200,
    rank_expansion: int = 8,
    **kwargs
) -> WeightLoRAPlusTrainer:
    """
    Factory function to create WeightLoRA+ trainer.
    
    Args:
        model_name: Model name
        dataset_name: Dataset name
        K: Sparsity level
        phase_1_steps: Phase 1 steps
        rank_expansion: Target rank
        **kwargs: Additional arguments
        
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
