"""
NLG Evaluation for WeightLoRA - XSum and CNN/DailyMail Summarization

This module implements evaluation for Natural Language Generation (NLG) tasks:
- XSum: News summarization dataset
- CNN/DailyMail: News summarization dataset

Metrics: ROUGE-1, ROUGE-2, ROUGE-L (computed using rouge_score package)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import json
import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer

from models.bart_wrapper import create_bart_weightlora
from trainers.weightlora_trainer import create_weightlora_trainer
from trainers.weightlora_plus_trainer import create_weightlora_plus_trainer
from trainers.lora_baseline_trainer import create_lora_baseline_trainer
from trainers.trainer_utils import set_seed, load_config, setup_device, load_model, load_dataset, create_dataloaders, compute_metrics, save_checkpoint, load_checkpoint
from utils.evaluation import compute_rouge_metrics, compute_nlg_metrics, compute_accuracy, compute_f1_score
from utils.metrics import MetricsCalculator
from utils.visualization import VisualizationManager


# Configuration constants
NLG_TASKS = ['xsum', 'cnn_dailymail']
BASELINE_PARAMS = 442000  # DeBERTa full LoRA parameters (reference)


class NLGBenchmarkRunner:
    """
    Main runner for NLG benchmark experiments (XSum, CNN/DailyMail).
    Supports WeightLoRA, WeightLoRA+, and LoRA baseline comparisons.
    """
    
    def __init__(
        self,
        model_name: str = 'bart-large',
        task: str = 'xsum',
        K: int = 10,
        T: int = 200,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.05,
        lr: float = 3e-4,
        num_epochs: int = 10,
        batch_size: int = 16,
        seed: int = 42,
        use_weightlora_plus: bool = False,
        phase_1_steps: int = 200,
        rank_expansion: int = 8,
        config_path: str = None,
        save_path: str = None
    ):
        """
        Initialize NLG benchmark runner.
        
        Args:
            model_name: Model name ('bart-large' for XSum/CNN-DailyMail)
            task: Task name ('xsum' or 'cnn_dailymail')
            K: Number of active adapters (sparsity level)
            T: Steps for adapter disconnection
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout probability
            lr: Learning rate
            num_epochs: Number of training epochs
            batch_size: Training batch size
            seed: Random seed for reproducibility
            use_weightlora_plus: Use WeightLoRA+ with rank expansion
            phase_1_steps: Steps for Phase 1 (adapter selection)
            rank_expansion: Target rank for Phase 2
            config_path: Path to configuration YAML
            save_path: Path to save checkpoints and results
        """
        self.model_name = model_name
        self.task = task
        self.K = K
        self.T = T
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.use_weightlora_plus = use_weightlora_plus
        self.phase_1_steps = phase_1_steps
        self.rank_expansion = rank_expansion
        self.config_path = config_path
        self.save_path = save_path
        
        # Set random seed
        set_seed(seed)
        
        # Setup device
        self.device = setup_device()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize visualization manager
        self.visualization = VisualizationManager(output_dir=save_path)
        
        # Results storage
        self.results = {}
        
        # Load configuration if provided
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = {}
    
    def _create_model(self) -> Tuple[Any, Any]:
        """
        Create model with appropriate architecture.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if self.task == 'xsum':
            model, tokenizer = create_bart_weightlora(
                model_name=self.model_name,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                K=self.K
            )
        elif self.task == 'cnn_dailymail':
            model, tokenizer = create_bart_weightlora(
                model_name=self.model_name,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                K=self.K
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        model.to(self.device)
        return model, tokenizer
    
    def _create_trainer(self) -> Any:
        """
        Create appropriate trainer based on configuration.
        
        Returns:
            Trainer instance
        """
        if self.use_weightlora_plus:
            return create_weightlora_plus_trainer(
                model_name=self.model_name,
                dataset_name=self.task,
                K=self.K,
                phase_1_steps=self.phase_1_steps,
                rank_expansion=self.rank_expansion,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                lr=self.lr,
                device=self.device,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                validation_freq=1,
                seed=self.seed,
                config_path=self.config_path
            )
        else:
            return create_weightlora_trainer(
                model_name=self.model_name,
                dataset_name=self.task,
                K=self.K,
                T=self.T,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                lr=self.lr,
                device=self.device,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                validation_freq=1,
                seed=self.seed,
                config_path=self.config_path
            )
    
    def _load_dataset(self) -> Tuple[Any, Any]:
        """
        Load dataset for the specified task.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self.task == 'xsum':
            from data_loaders.xsum_dataset import XSumDataLoader
            dataset = load_xsum_dataset(split='train', max_length=512, tokenizer=None)
            train_loader, val_loader = XSumDataLoader(
                dataset_name='xsum',
                tokenizer=None,
                max_length=512,
                train_batch_size=self.batch_size,
                validation_batch_size=self.batch_size // 2,
                test_batch_size=self.batch_size // 2,
                seed=self.seed
            ).create_loaders()
        elif self.task == 'cnn_dailymail':
            from data_loaders.cnn_dailymail import CNNDailyMailDataLoader
            dataset = load_cnn_dailymail_dataset(split='train3', max_length=512, tokenizer=None)
            train_loader, val_loader = CNNDailyMailDataLoader(
                dataset_name='cnn_dailymail',
                tokenizer=None,
                max_length=512,
                train_batch_size=self.batch_size,
                validation_batch_size=self.batch_size // 2,
                test_batch_size=self.batch_size // 2,
                seed=self.seed
            ).create_loaders()
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return train_loader, val_loader
    
    def _train(self, trainer: Any) -> Dict[str, Any]:
        """
        Train the model using the trainer.
        
        Args:
            trainer: Trainer instance
            
        Returns:
            Training results dictionary
        """
        print(f"Starting training for {self.task} with {'WeightLoRA+' if self.use_weightlora_plus else 'WeightLoRA'}")
        
        # Train
        trainer.train()
        
        # Get results
        results = {
            'task': self.task,
            'method': 'WeightLoRA+' if self.use_weightlora_plus else 'WeightLoRA',
            'K': self.K,
            'rank': self.rank,
            'alpha': self.alpha,
            'lr': self.lr,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'use_weightlora_plus': self.use_weightlora_plus,
            'phase_1_steps': self.phase_1_steps,
            'rank_expansion': self.rank_expansion
        }
        
        return results
    
    def _evaluate(self, model: Any, tokenizer: Any, val_loader: Any) -> Dict[str, Any]:
        """
        Evaluate model on validation set.
        
        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            val_loader: Validation dataloader
            
        Returns:
            Evaluation results dictionary
        """
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Evaluating {self.task}'):
                # Forward pass
                inputs = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Extract predictions and references
                pred_tokens = outputs.logits.argmax(dim=-1)
                ref_tokens = labels
                
                # Decode
                pred_text = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
                ref_text = tokenizer.batch_decode(ref_tokens, skip_special_tokens=True)
                
                predictions.extend(pred_text)
                references.extend(ref_text)
        
        # Compute ROUGE metrics
        rouge_results = compute_rouge_metrics(predictions, references)
        
        # Compute additional metrics
        metrics = {
            'rouge_1': rouge_results.get('rouge1', 0),
            'rouge_2': rouge_results.get('rouge2', 0),
            'rouge_l': rouge_results.get('rougeL', 0),
            'rouge_lsum': rouge_results.get('rougeLsum', 0)
        }
        
        return metrics
    
    def run_experiment(self, method: str = 'weightlora') -> Dict[str, Any]:
        """
        Run complete experiment for specified method.
        
        Args:
            method: Method name ('weightlora', 'weightlora_plus', 'lora')
            
        Returns:
            Complete experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running {method.upper()} experiment for {self.task}")
        print(f"{'='*60}\n")
        
        # Create model
        model, tokenizer = self._create_model()
        
        # Create trainer
        if method == 'lora':
            trainer = create_lora_baseline_trainer(
                model_name=self.model_name,
                dataset_name=self.task,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                lr=self.lr,
                device=self.device,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                validation_freq=1,
                seed=self.seed,
                config_path=self.config_path
            )
        elif method == 'weightlora':
            trainer = self._create_trainer()
        elif method == 'weightlora_plus':
            trainer = self._create_trainer()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Train
        results = self._train(trainer)
        
        # Evaluate
        val_metrics = self._evaluate(model, tokenizer, None)
        results['val_metrics'] = val_metrics
        
        # Compute parameter reduction
        if self.task == 'xsum':
            baseline_params = 442000  # BART full LoRA params
        elif self.task == 'cnn_dailymail':
            baseline_params = 442000
        else:
            baseline_params = 442000
        
        active_params = int(baseline_params * (self.K / 100))  # Approximation
        reduction = (baseline_params - active_params) / baseline_params
        
        results['parameter_reduction'] = {
            'baseline_params': baseline_params,
            'active_params': active_params,
            'reduction_percentage': reduction
        }
        
        # Save results
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            results_path = os.path.join(self.save_path, f'{method}_{self.task}_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")
        
        return results
    
    def run_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all methods for comparison.
        
        Returns:
            Dictionary of results for each method
        """
        methods = ['lora', 'weightlora', 'weightlora_plus']
        all_results = {}
        
        for method in methods:
            results = self.run_experiment(method)
            all_results[method] = results
        
        return all_results


def load_xsum_dataset(split: str = 'train', max_length: int = 512, tokenizer: Any = None) -> Any:
    """Load XSum dataset."""
    from data_loaders.xsum_dataset import XSumDataset
    return XSumDataset(split=split, max_length=max_length, tokenizer=tokenizer)


def load_cnn_dailymail_dataset(split: str = 'train3', max_length: int = 512, tokenizer: Any = None) -> Any:
    """Load CNN/DailyMail dataset."""
    from data_loaders.cnn_dailymail import CNNDailyMailDataset
    return CNNDailyMailDataset(split=split, max_length=max_length, tokenizer=tokenizer)


def main():
    """Main entry point for NLG evaluation."""
    parser = argparse.ArgumentParser(description='NLG Evaluation for WeightLoRA')
    
    parser.add_argument('--model', type=str, default='bart-large',
                        help='Model name (default: bart-large)')
    parser.add_argument('--task', type=str, default='xsum',
                        choices=['xsum', 'cnn_dailymail'],
                        help='Task name (default: xsum)')
    parser.add_argument('--K', type=int, default=10,
                        help='Number of active adapters (default: 10)')
    parser.add_argument('--T', type=int, default=200,
                        help='Steps for adapter disconnection (default: 200)')
    parser.add_argument('--rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--alpha', type=float, default=32.0,
                        help='LoRA scaling factor (default: 32.0)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='Dropout probability (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--use-weightlora-plus', action='store_true',
                        help='Use WeightLoRA+ with rank expansion')
    parser.add_argument('--phase-1-steps', type=int, default=200,
                        help='Phase 1 steps (default: 200)')
    parser.add_argument('--rank-expansion', type=int, default=8,
                        help='Rank expansion target (default: 8)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML')
    parser.add_argument('--save-dir', type=str, default='nlg_results',
                        help='Directory to save results')
    parser.add_argument('--method', type=str, default='weightlora',
                        choices=['lora', 'weightlora', 'weightlora_plus'],
                        help='Method to run (default: weightlora)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = NLGBenchmarkRunner(
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
        use_weightlora_plus=args.use_weightlora_plus,
        phase_1_steps=args.phase_1_steps,
        rank_expansion=args.rank_expansion,
        config_path=args.config,
        save_path=args.save_dir
    )
    
    # Run experiment
    results = runner.run_experiment(args.method)
    
    # Print results
    print("\n" + "="*60)
    print(f"NLG Evaluation Results: {args.task}")
    print("="*60)
    print(f"Method: {results['method']}")
    print(f"K: {results['K']}")
    print(f"Rank: {results['rank']}")
    print(f"\nROUGE Metrics:")
    print(f"  ROUGE-1: {results['val_metrics'].get('rouge_1', 0):.4f}")
    print(f"  ROUGE-2: {results['val_metrics'].get('rouge_2', 0):.4f}")
    print(f"  ROUGE-L: {results['val_metrics'].get('rouge_l', 0):.4f}")
    print(f"  ROUGE-Lsum: {results['val_metrics'].get('rouge_lsum', 0):.4f}")
    print(f"\nParameter Reduction:")
    print(f"  Baseline: {results['parameter_reduction']['baseline_params']}")
    print(f"  Active: {results['parameter_reduction']['active_params']}")
    print(f"  Reduction: {results['parameter_reduction']['reduction_percentage']*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
