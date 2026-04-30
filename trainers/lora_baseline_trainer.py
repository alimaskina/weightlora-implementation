"""
LoRA Baseline Trainer
=====================

This module implements the standard LoRA (Low-Rank Adaptation) baseline trainer
for comparison with WeightLoRA. It provides a reference implementation following
Hu et al. (2021) without the adaptive weight selection mechanism.

Key Features:
- Standard LoRA training with A×B decomposition
- Fixed adapter activation (no sparsity control)
- Baseline for performance comparison
- Supports all model types (DeBERTaV3, BART, Llama)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader

from models.adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding
from models.adapter_manager import AdapterManager, is_target_layer
from trainers.trainer_utils import (
    load_model, load_dataset, create_dataloaders, 
    save_checkpoint, load_checkpoint, setup_device, set_seed,
    compute_metrics, load_config
)
from utils.evaluation import compute_glue_metrics, compute_squad_metrics, compute_rouge_metrics


class LoraBaselineTrainer:
    """
    Standard LoRA baseline trainer for comparison with WeightLoRA.
    
    Implements the original LoRA approach without adaptive weight selection.
    All adapters remain active throughout training.
    
    Parameters:
        model_name (str): Name of pretrained model to load
        dataset_name (str): Name of dataset to use
        rank (int): LoRA rank r
        alpha (float): LoRA scaling factor α
        dropout (float): Dropout probability
        lr (float): Learning rate
        device (torch.device): Device for training
        batch_size (int): Batch size
        num_epochs (int): Number of training epochs
        validation_freq (int): Frequency of validation
        seed (int): Random seed for reproducibility
        config_path (str): Path to configuration file
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
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
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lr = lr
        self.device = device if device else setup_device()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.seed = seed
        
        # Set random seed for reproducibility
        set_seed(seed)
        
        # Load model and wrap with LoRA adapters
        self.model = self._create_model()
        
        # Initialize adapter manager
        self.adapter_manager = AdapterManager(
            self.model,
            K=len(self.adapter_manager.manager),  # All adapters active
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Load dataset and create dataloaders
        self.dataset = load_dataset(dataset_name, max_length=128)
        self.train_loader, self.val_loader = create_dataloaders(
            self.dataset, batch_size=self.batch_size, device=self.device
        )
        
        # Optimizer for LoRA parameters only
        lora_params = [
            p for name, p in self.model.named_parameters()
            if 'lora' in name.lower() or 'adapter' in name.lower()
        ]
        self.optimizer = torch.optim.Adam(lora_params, lr=lr)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _create_model(self) -> nn.Module:
        """
        Create and wrap model with LoRA adapters.
        
        Returns:
            nn.Module: Model with LoRA adapters applied
        """
        # Load base model
        model = load_model(self.model_name, num_labels=3, device=self.device)
        
        # Wrap with LoRA adapters
        self.adapter_manager.initialize_adapters(model)
        
        return model
    
    def train(self) -> Dict[str, Any]:
        """
        Execute LoRA baseline training.
        
        Returns:
            Dict[str, Any]: Training results including metrics and history
        """
        print(f"Starting LoRA baseline training on {self.dataset_name}")
        print(f"Model: {self.model_name}, Rank: {self.rank}, Alpha: {self.alpha}")
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss = self._train_epoch()
            
            # Validation phase
            if (epoch + 1) % self.validation_freq == 0 or epoch == self.num_epochs - 1:
                val_loss, val_acc = self._validate()
                
                # Log progress
                print(f"Epoch {epoch + 1}/{self.num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
                
                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
        
        # Final evaluation
        final_results = self._final_evaluation()
        
        return final_results
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss
    
    def _validate(self) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            Tuple[float, float]: (validation_loss, validation_accuracy)
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _final_evaluation(self) -> Dict[str, Any]:
        """
        Perform final evaluation and return results.
        
        Returns:
            Dict[str, Any]: Complete evaluation results
        """
        self.model.eval()
        
        # Collect predictions for detailed metrics
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        metrics = compute_glue_metrics(
            all_predictions, all_targets, task='mnli'
        )
        
        # Get parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get active adapter count
        active_adapters = self.adapter_manager.get_active_adapters()
        
        results = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'rank': self.rank,
            'alpha': self.alpha,
            'num_epochs': self.num_epochs,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_acc': self.history['val_acc'][-1],
            'metrics': metrics,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'active_adapters': len(active_adapters),
            'total_adapters': len(self.adapter_manager.manager),
            'history': self.history
        }
        
        return results
    
    def save_checkpoint(self, path: str, epoch: int = -1):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save checkpoint
            epoch (int): Current epoch (use -1 for latest)
        """
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            path,
            weight_vector=None  # Not applicable for LoRA baseline
        )
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to checkpoint
        """
        model, optimizer, epoch = load_checkpoint(path)
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch


def create_lora_baseline_trainer(
    model_name: str,
    dataset_name: str,
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05,
    lr: float = 3e-4,
    **kwargs
) -> LoraBaselineTrainer:
    """
    Factory function to create LoRA baseline trainer.
    
    Args:
        model_name (str): Name of pretrained model
        dataset_name (str): Name of dataset
        rank (int): LoRA rank
        alpha (float): LoRA scaling factor
        dropout (float): Dropout probability
        lr (float): Learning rate
        **kwargs: Additional arguments passed to LoraBaselineTrainer
    
    Returns:
        LoraBaselineTrainer: Configured trainer instance
    """
    return LoraBaselineTrainer(
        model_name=model_name,
        dataset_name=dataset_name,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        lr=lr,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='LoRA Baseline Trainer')
    parser.add_argument('--model', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--dataset', type=str, default='glue/mnli')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=32.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    
    args = parser.parse_args()
    
    trainer = create_lora_baseline_trainer(
        model_name=args.model,
        dataset_name=args.dataset,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    results = trainer.train()
    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {results['final_val_acc']:.4f}")
    print(f"Active Adapters: {results['active_adapters']}/{results['total_adapters']}")
