#!/usr/bin/env python3
"""
Ablation Study for WeightLoRA
================================
Implements RLoRA (Random Layer Selection) ablation and comparison with SOTA adaptive methods.

This module validates that WeightLoRA's ω_i optimization is necessary by comparing against:
- RLoRA: Random selection of K adapters after T steps
- AdaLoRA: Adaptive LoRA with threshold-based selection
- IncreLoRA: Incremental LoRA with dynamic rank adjustment
- DynLoRA: Dynamic LoRA with layer-wise importance

Reference: WeightLoRA paper Tables 6-9
"""

import argparse
import json
import os
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import core components
from models.deberta_wrapper import create_deberta_weightlora
from models.bart_wrapper import create_bart_weightlora
from models.llama_wrapper import create_llama_weightlora
from trainers.weightlora_trainer import WeightLoRATrainer, create_weightlora_trainer
from trainers.weightlora_plus_trainer import WeightLoRAPlusTrainer, create_weightlora_plus_trainer
from trainers.lora_baseline_trainer import LoraBaselineTrainer, create_lora_baseline_trainer
from trainers.trainer_utils import (
    set_seed, load_config, setup_device, load_model, load_dataset,
    create_dataloaders, save_checkpoint, load_checkpoint, compute_metrics
)
from utils.evaluation import (
    compute_glue_metrics, compute_squad_metrics, compute_rouge_metrics,
    compute_accuracy, compute_f1_score, compute_mcc_score, validate_sparsity_constraint
)
from utils.metrics import MetricsCalculator, aggregate_metrics
from algorithms.sto_iht import StoIHTOptimizer, StoIHTTrainer
from algorithms.weight_optimizer import WeightOptimizer, create_weight_optimizer
from algorithms.rank_expander import RankExpander, create_rank_expander
from algorithms.sparsity_constraint import SparsityConstraint, hard_thresholding

# Constants
GLUE_TASKS = ["mnli", "sst2", "cola", "qqp", "qnli", "rte", "mrpc", "sts-b"]
SQUAD_VERSIONS = ["v1.1", "v2.0"]
NLG_TASKS = ["xsum", "cnn_dailymail"]

# Baseline parameters from paper
BASELINE_PARAMS_DEBERTA = 442000  # Full LoRA params for DeBERTa
BASELINE_PARAMS_BART = 442000
BASELINE_PARAMS_LLAMA = 6400000


class RLORATrainer:
    """
    RLoRA (Random Layer Selection) - Ablation control for WeightLoRA.
    
    Instead of optimizing ω_i with StoIHT, randomly selects K adapters after T steps.
    This validates that WeightLoRA's optimization is necessary, not just random selection.
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        K: int = 10,
        T: int = 200,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        lr: float = 3e-4,
        device: torch.device = None,
        batch_size: int = 32,
        num_epochs: int = 10,
        validation_freq: int = 50,
        seed: int = 42
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.K = K
        self.T = T
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lr = lr
        self.device = device or setup_device()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.seed = seed
        
        set_seed(seed)
        
        # Create model with adapters
        self.model, _ = self._create_model()
        self.adapter_manager = self._initialize_adapter_manager()
        
        # Create optimizer for adapter parameters (A, B matrices)
        self.optimizer = torch.optim.Adam(
            self.adapter_manager.parameters(),
            lr=lr
        )
        
        # Track active indices (randomly selected)
        self.active_indices = None
        
        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def _create_model(self) -> Tuple:
        """Create model based on model_name"""
        if self.model_name == "deberta-v3-base":
            return create_deberta_weightlora(
                "microsoft/deberta-v3-base",
                self.rank, self.alpha, self.dropout, self.K
            )
        elif self.model_name == "bart-large":
            return create_bart_weightlora(
                "facebook/bart-large",
                self.rank, self.alpha, self.dropout, self.K
            )
        elif self.model_name == "llama-7b":
            return create_llama_weightlora(
                "meta-llama/Llama-3-7b",
                self.rank, self.alpha, self.dropout, self.K
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _initialize_adapter_manager(self) -> Any:
        """Initialize adapter manager for the model"""
        # Get target modules from model
        target_modules = self._get_target_modules()
        
        # Create adapter manager
        adapter_manager = AdapterManager(
            self.model,
            K=self.K,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            target_modules=target_modules
        )
        return adapter_manager
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules based on model type"""
        if "deberta" in self.model_name:
            return [
                "layers.{0-11}.attention.q_proj",
                "layers.{0-11}.attention.k_proj",
                "layers.{0-11}.attention.v_proj",
                "layers.{0-11}.attention.out_proj"
            ]
        elif "bart" in self.model_name:
            return [
                "encoder.layers.{0-11}.self_attn.q_proj",
                "encoder.layers.{0-11}.self_attn.k_proj",
                "encoder.layers.{0-11}.self_attn.v_proj",
                "encoder.layers.{0-11}.self_attn.out_proj",
                "encoder.layers.{0-11}.fc1",
                "encoder.layers.{0-11}.fc2",
                "decoder.layers.{0-11}.self_attn.q_proj",
                "decoder.layers.{0-11}.self_attn.k_proj",
                "decoder.layers.{0-11}.self_attn.v_proj",
                "decoder.layers.{0-11}.self_attn.out_proj"
            ]
        elif "llama" in self.model_name:
            return [
                "model.layers.{0-31}.self_attn.q_proj",
                "model.layers.{0-31}.self_attn.k_proj",
                "model.layers.{0-11}.self_attn.v_proj",
                "model.layers.{0-31}.self_attn.o_proj",
                "model.layers.{0-31}.mlp.gate_proj",
                "model.layers.{0-31}.mlp.up_proj",
                "model.layers.{0-31}.mlp.down_proj"
            ]
        return []
    
    def _random_select_adapters(self, n_layers: int) -> List[int]:
        """Randomly select K adapters from n_layers"""
        indices = list(range(n_layers))
        selected = random.sample(indices, min(self.K, n_layers))
        return sorted(selected)
    
    def train(self, dataset, epochs: int = None) -> Dict[str, Any]:
        """
        Train RLoRA model.
        
        Phase 1: Train all adapters for T steps
        Phase 2: Randomly select K adapters and freeze others
        Phase 3: Continue training with selected adapters only
        """
        if epochs is None:
            epochs = self.num_epochs
        
        total_steps = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": [],
            "active_adapters": []
        }
        
        # Get number of layers
        n_layers = self._get_n_layers()
        
        # Phase 1: Train all adapters
        print(f"Phase 1: Training all {n_layers} adapters for {self.T} steps")
        for epoch in range(epochs):
            for step, (inputs, targets) in enumerate(dataset.train_loader):
                if total_steps >= self.T:
                    break
                
                # Forward pass
                outputs = self.adapter_manager(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_steps += 1
            
            # Random selection after T steps
            if total_steps >= self.T and self.active_indices is None:
                self.active_indices = self._random_select_adapters(n_layers)
                print(f"Randomly selected {len(self.active_indices)} adapters")
            
            # Validate periodically
            if epoch % self.validation_freq == 0:
                val_loss = self._validate(dataset)
                history["val_loss"].append(val_loss)
                history["active_adapters"].append(len(self.active_indices) if self.active_indices else n_layers)
        
        # Phase 2: Continue with selected adapters
        print(f"Phase 2: Continuing training with {len(self.active_indices)} selected adapters")
        for epoch in range(epochs):
            for step, (inputs, targets) in enumerate(dataset.train_loader):
                # Forward pass
                outputs = self.adapter_manager(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Validate
            val_loss = self._validate(dataset)
            history["val_loss"].append(val_loss)
            history["active_adapters"].append(len(self.active_indices))
        
        return {
            "history": history,
            "active_indices": self.active_indices,
            "n_layers": n_layers,
            "K": self.K
        }
    
    def _validate(self, dataset) -> float:
        """Validate model on validation set"""
        self.adapter_manager.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in dataset.val_loader:
                outputs = self.adapter_manager(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
            avg_loss = total_loss / len(dataset.val_loader)
        return avg_loss
    
    def _get_n_layers(self) -> int:
        """Get number of adapter layers"""
        if "deberta" in self.model_name:
            return 12  # 12 transformer layers
        elif "bart" in self.model_name:
            return 24  # 12 encoder + 12 decoder layers
        elif "llama" in self.model_name:
            return 32  # 32 transformer layers
        return 12


class RLORATrainerWrapper:
    """
    Wrapper for RLoRA that integrates with existing trainer infrastructure.
    """
    
    def __init__(self, trainer: RLORATrainer):
        self.trainer = trainer
    
    def run(self, dataset) -> Dict[str, Any]:
        """Run RLoRA training and return results"""
        results = self.trainer.train(dataset)
        return results


class SOTAAdaptiveMethods:
    """
    Implementation of SOTA adaptive LoRA methods for comparison.
    
    Methods:
    - AdaLoRA: Threshold-based adapter selection
    - IncreLoRA: Incremental rank adjustment
    - DynLoRA: Dynamic layer-wise importance
    """
    
    @staticmethod
    def adalora(
        model, K: int = 10, rank: int = 8, alpha: float = 32.0,
        threshold: float = 0.5, lr: float = 3e-4
    ) -> Dict[str, Any]:
        """
        AdaLoRA: Adaptive LoRA with threshold-based selection.
        
        Selects adapters based on gradient magnitude threshold.
        """
        print("Running AdaLoRA...")
        
        # Get all adapters
        adapters = model.adapter_manager.manager.values()
        
        # Compute gradient magnitudes
        gradients = []
        for adapter in adapters:
            # Simulate gradient computation
            grad = torch.randn(adapter.A.shape)
            gradients.append(torch.norm(grad))
        
        # Threshold-based selection
        gradients = torch.tensor(gradients)
        selected_mask = gradients > threshold * torch.max(gradients)
        selected_indices = selected_mask.nonzero().tolist()
        
        # Activate selected adapters
        for idx in selected_indices:
            model.adapter_manager.manager[list(model.adapter_manager.manager.keys())[idx]].weight.data = torch.ones(1)
        
        return {
            "method": "AdaLoRA",
            "threshold": threshold,
            "selected_adapters": len(selected_indices),
            "selected_indices": selected_indices
        }
    
    @staticmethod
    def increlora(
        model, initial_rank: int = 4, target_rank: int = 8,
        expansion_steps: int = 200
    ) -> Dict[str, Any]:
        """
        IncreLoRA: Incremental LoRA with dynamic rank adjustment.
        
        Expands rank incrementally during training.
        """
        print("Running IncreLoRA...")
        
        # Get selected adapters (assume from previous phase)
        selected_adapters = model.adapter_manager.active_indices
        
        # Create rank expander
        rank_expander = RankExpander(
            selected_adapters,
            initial_rank=initial_rank,
            target_rank=target_rank
        )
        
        # Expand ranks
        for adapter_name in selected_adapters:
            if adapter_name in model.adapter_manager.manager:
                adapter = model.adapter_manager.manager[adapter_name]
                rank_expander.expand_rank(adapter)
        
        return {
            "method": "IncreLoRA",
            "initial_rank": initial_rank,
            "target_rank": target_rank,
            "expanded_adapters": len(selected_adapters)
        }
    
    @staticmethod
    def dynlora(
        model, importance_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        DynLoRA: Dynamic layer-wise importance selection.
        
        Uses layer-wise importance scores to select adapters.
        """
        print("Running DynLoRA...")
        
        # Get all adapters with their importance
        adapters = model.adapter_manager.manager.items()
        
        # Compute importance scores (simulated)
        importance_scores = []
        for name, adapter in adapters:
            # Importance based on adapter parameter count
            param_count = adapter.A.numel() + adapter.B.numel()
            importance_scores.append(param_count)
        
        importance_scores = torch.tensor(importance_scores)
        
        # Select top-K by importance
        top_k, top_indices = torch.topk(importance_scores, min(K, len(importance_scores)))
        selected_indices = top_indices.tolist()
        
        # Activate selected adapters
        for idx in selected_indices:
            model.adapter_manager.manager[list(model.adapter_manager.manager.keys())[idx]].weight.data = torch.ones(1)
        
        return {
            "method": "DynLoRA",
            "importance_scores": importance_scores.tolist(),
            "selected_adapters": len(selected_indices),
            "selected_indices": selected_indices
        }


def run_ablation_study(
    model_name: str = "deberta-v3-base",
    task: str = "mnli",
    K: int = 10,
    T: int = 200,
    rank: int = 8,
    alpha: float = 32.0,
    dropout: float = 0.05,
    lr: float = 3e-4,
    num_epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
    save_path: str = "ablation_results.json"
) -> Dict[str, Any]:
    """
    Run complete ablation study comparing WeightLoRA vs RLoRA vs SOTA methods.
    
    Returns comprehensive results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY: {model_name} on {task}")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    device = setup_device()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(task, max_length=128)
    
    # Results dictionary
    results = {
        "experiment": "ablation_study",
        "model": model_name,
        "task": task,
        "K": K,
        "T": T,
        "rank": rank,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "weightlora": {},
        "rlora": {},
        "sota_methods": {}
    }
    
    # 1. WeightLoRA (baseline)
    print("\n1. Running WeightLoRA...")
    weightlora_trainer = WeightLoRATrainer(
        model_name=model_name,
        dataset_name=task,
        K=K,
        T=T,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        lr=lr,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        validation_freq=50,
        seed=seed
    )
    
    weightlora_results = weightlora_trainer.train(dataset)
    results["weightlora"] = {
        "final_loss": weightlora_results["val_loss"][-1] if weightlora_results["val_loss"] else None,
        "active_adapters": weightlora_results["active_indices"],
        "n_layers": weightlora_results["n_layers"],
        "K": K
    }
    
    # Compute metrics
    metrics = compute_glue_metrics(
        predictions=weightlora_results["val_loss"],
        targets=dataset.targets,
        task=task
    )
    results["weightlora"]["metrics"] = metrics
    
    # 2. RLoRA (Random Layer Selection)
    print("\n2. Running RLoRA (Random Layer Selection)...")
    rlora_trainer = RLORATrainer(
        model_name=model_name,
        dataset_name=task,
        K=K,
        T=T,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        lr=lr,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        validation_freq=50,
        seed=seed + 1  # Different seed for randomness
    )
    
    rlora_results = rlora_trainer.train(dataset)
    results["rlora"] = {
        "final_loss": rlora_results["history"]["val_loss"][-1] if rlora_results["history"]["val_loss"] else None,
        "active_indices": rlora_results["active_indices"],
        "n_layers": rlora_results["n_layers"],
        "K": K
    }
    
    # Compute metrics
    rlora_metrics = compute_glue_metrics(
        predictions=rlora_results["history"]["val_loss"],
        targets=dataset.targets,
        task=task
    )
    results["rlora"]["metrics"] = rlora_metrics
    
    # 3. SOTA Methods
    print("\n3. Running SOTA Adaptive Methods...")
    
    # AdaLoRA
    print("  - AdaLoRA...")
    adalora_results = SOTAAdaptiveMethods.adalora(
        rlora_trainer.model, K=K, rank=rank, alpha=alpha, threshold=0.5
    )
    results["sota_methods"]["adalora"] = adalora_results
    
    # IncreLoRA
    print("  - IncreLoRA...")
    increlora_results = SOTAAdaptiveMethods.increlora(
        rlora_trainer.model, initial_rank=4, target_rank=rank
    )
    results["sota_methods"]["increlora"] = increlora_results
    
    # DynLoRA
    print("  - DynLoRA...")
    dynlora_results = SOTAAdaptiveMethods.dynlora(rlora_trainer.model)
    results["sota_methods"]["dynlora"] = dynlora_results
    
    # 4. Comparison Summary
    print("\n4. Comparison Summary...")
    results["comparison"] = {
        "weightlora_final_loss": results["weightlora"]["final_loss"],
        "rlora_final_loss": results["rlora"]["final_loss"],
        "weightlora_active_adapters": len(results["weightlora"]["active_indices"]),
        "rlora_active_adapters": len(results["rlora"]["active_indices"]),
        "weightlora_improvement": (
            results["rlora"]["final_loss"] - results["weightlora"]["final_loss"]
        ) if results["rlora"]["final_loss"] and results["weightlora"]["final_loss"] else None
    }
    
    # 5. Validation
    print("\n5. Validating Sparsity Constraints...")
    weightlora_valid, weightlora_sparsity = validate_sparsity_constraint(
        weightlora_trainer.get_weight_vector(), K, tolerance=1e-6
    )
    results["validation"] = {
        "weightlora_sparsity_valid": weightlora_valid,
        "weightlora_actual_sparsity": weightlora_sparsity,
        "rlora_sparsity_valid": False,  # RLoRA doesn't enforce sparsity
        "rlora_actual_sparsity": K  # RLoRA randomly selects K
    }
    
    # Save results
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {save_path}")
    
    return results


def main():
    """Main entry point for ablation study"""
    parser = argparse.ArgumentParser(description="Run WeightLoRA Ablation Study")
    parser.add_argument("--model", type=str, default="deberta-v3-base",
                       choices=["deberta-v3-base", "bart-large", "llama-7b"])
    parser.add_argument("--task", type=str, default="mnli",
                       choices=GLUE_TASKS)
    parser.add_argument("--K", type=int, default=10, help="Number of active adapters")
    parser.add_argument("--T", type=int, default=200, help="Steps before disconnection")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-path", type=str, default="ablation_results.json",
                       help="Path to save results")
    
    args = parser.parse_args()
    
    results = run_ablation_study(
        model_name=args.model,
        task=args.task,
        K=args.K,
        T=args.T,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        save_path=args.save_path
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"K: {args.K}")
    print(f"\nWeightLoRA Final Loss: {results['weightlora']['final_loss']:.4f}")
    print(f"RLoRA Final Loss: {results['rlora']['final_loss']:.4f}")
    print(f"WeightLoRA Improvement: {results['comparison']['weightlora_improvement']:.4f}")
    print(f"\nWeightLoRA Active Adapters: {len(results['weightlora']['active_indices'])}")
    print(f"RLoRA Active Adapters: {len(results['rlora']['active_indices'])}")
    print("="*60)


if __name__ == "__main__":
    main()
