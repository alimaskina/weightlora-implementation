"""
Comprehensive metrics computation utilities for WeightLoRA evaluation.
Provides standard metrics for GLUE, SQuAD, and NLG benchmarks.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_recall_fscore_support


class MetricsCalculator:
    """Centralized metrics calculator for all benchmark types."""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_glue_metrics(self, predictions: List[int], targets: List[int], task: str) -> Dict[str, float]:
        """
        Compute GLUE benchmark metrics based on task type.
        
        Args:
            predictions: List of predicted labels
            targets: List of ground truth labels
            task: GLUE task name (mnli, sst2, cola, etc.)
        
        Returns:
            Dictionary with task-specific metrics
        """
        metrics = {}
        
        if task in ['mnli', 'mnli_mismatched']:
            # Multi-class classification
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['f1_macro'] = f1_score(targets, predictions, average='macro')
            metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted')
        elif task in ['sst2', 'cola', 'qnli', 'rte']:
            # Binary classification
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['f1'] = f1_score(targets, predictions, average='binary')
            metrics['precision'] = precision_recall_fscore_support(targets, predictions, average='binary')[0]
            metrics['recall'] = precision_recall_fscore_support(targets, predictions, average='binary')[1]
            metrics['mcc'] = matthews_corrcoef(targets, predictions)
        elif task == 'qqp':
            # Binary classification with many negatives
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['f1'] = f1_score(targets, predictions, average='binary')
            metrics['precision'] = precision_recall_fscore_support(targets, predictions, average='binary')[0]
            metrics['recall'] = precision_recall_fscore_support(targets, predictions, average='binary')[1]
        elif task == 'mrpc':
            # Binary classification
            metrics['accuracy'] = accuracy_score(targets, predictions)
            metrics['f1'] = f1_score(targets, predictions, average='binary')
            metrics['precision'] = precision_recall_fscore_support(targets, predictions, average='binary')[0]
            metrics['recall'] = precision_recall_fscore_support(targets, predictions, average='binary')[1]
        elif task == 'sts-b':
            # Regression task - compute Pearson correlation
            predictions = np.array(predictions)
            targets = np.array(targets)
            metrics['pearson'] = np.corrcoef(predictions, targets)[0, 1]
            metrics['spearman'] = self._compute_spearman(targets, predictions)
            metrics['mae'] = np.mean(np.abs(predictions - targets))
            metrics['mse'] = np.mean((predictions - targets) ** 2)
        
        return metrics
    
    def compute_squad_metrics(self, predictions: List[Tuple[int, int]], 
                             references: List[Tuple[int, int]]) -> Dict[str, float]:
        """
        Compute SQuAD evaluation metrics (Exact Match and F1).
        
        Args:
            predictions: List of (start, end) token positions
            references: List of (start, end) token positions
        
        Returns:
            Dictionary with EM and F1 scores
        """
        exact_matches = 0
        total_matches = 0
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_start, pred_end = pred
            ref_start, ref_end = ref
            
            # Exact match
            if pred_start == ref_start and pred_end == ref_end:
                exact_matches += 1
            
            # Compute F1 score
            pred_tokens = set(range(pred_start, pred_end + 1))
            ref_tokens = set(range(ref_start, ref_end + 1))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
            else:
                intersection = len(pred_tokens & ref_tokens)
                union = len(pred_tokens | ref_tokens)
                f1 = 2 * intersection / union if union > 0 else 0.0
                f1_scores.append(f1)
            
            total_matches += 1
        
        em_score = exact_matches / total_matches if total_matches > 0 else 0.0
        f1_score_val = np.mean(f1_scores) if f1_scores else 0.0
        
        return {
            'exact_match': em_score,
            'f1': f1_score_val
        }
    
    def compute_rouge_metrics(self, predictions: List[str], 
                             references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE metrics for summarization tasks.
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if len(predictions) == 0 or len(references) == 0:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Compute ROUGE scores
        scores = self.scorer.score_many(predictions, references)
        
        rouge1_scores = [s['rouge1'].fmeasure for s in scores]
        rouge2_scores = [s['rouge2'].fmeasure for s in scores]
        rougel_scores = [s['rougeL'].fmeasure for s in scores]
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougel_scores)
        }
    
    def compute_nlg_metrics(self, predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """
        Compute NLG benchmark metrics (ROUGE-1 and ROUGE-L).
        
        Args:
            predictions: List of generated summaries
            references: List of reference summaries
        
        Returns:
            Dictionary with ROUGE-1 and ROUGE-L scores
        """
        rouge_scores = self.compute_rouge_metrics(predictions, references)
        return {
            'rouge1': rouge_scores['rouge1'],
            'rougeL': rouge_scores['rougeL']
        }
    
    def _compute_spearman(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman correlation coefficient."""
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        correlation = np.corrcoef(x_rank, y_rank)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def compute_sparsity_metrics(self, weights: torch.Tensor, K: int, 
                                 total_adapters: int) -> Dict[str, float]:
        """
        Compute sparsity metrics for WeightLoRA weight vector.
        
        Args:
            weights: Weight vector ω
            K: Target sparsity level
            total_adapters: Total number of adapters
        
        Returns:
            Dictionary with sparsity statistics
        """
        non_zero = (weights.abs() > 1e-6).sum().item()
        sparsity_ratio = non_zero / total_adapters
        reduction_ratio = 1.0 - sparsity_ratio
        
        return {
            'non_zero_weights': int(non_zero),
            'sparsity_ratio': sparsity_ratio,
            'reduction_ratio': reduction_ratio,
            'target_K': K,
            'actual_sparsity': int(non_zero)
        }
    
    def compute_memory_reduction(self, baseline_params: int, 
                                 active_params: int) -> Dict[str, float]:
        """
        Compute parameter reduction achieved by WeightLoRA.
        
        Args:
            baseline_params: Total parameters in full LoRA
            active_params: Parameters in active adapters
        
        Returns:
            Dictionary with reduction statistics
        """
        reduction = 1.0 - (active_params / baseline_params)
        return {
            'baseline_params': baseline_params,
            'active_params': active_params,
            'reduction_percentage': reduction * 100,
            'reduction_ratio': reduction
        }
    
    def aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple runs.
        
        Args:
            results: List of result dictionaries
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        aggregated = {}
        
        for metric in results[0].keys():
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return aggregated


def compute_glue_metrics(predictions: List[int], targets: List[int], task: str) -> Dict[str, float]:
    """Compute GLUE benchmark metrics."""
    calculator = MetricsCalculator()
    return calculator.compute_glue_metrics(predictions, targets, task)


def compute_squad_metrics(predictions: List[Tuple[int, int]], 
                         references: List[Tuple[int, int]]) -> Dict[str, float]:
    """Compute SQuAD evaluation metrics."""
    calculator = MetricsCalculator()
    return calculator.compute_squad_metrics(predictions, references)


def compute_rouge_metrics(predictions: List[str], 
                         references: List[str]) -> Dict[str, float]:
    """Compute ROUGE metrics for summarization."""
    calculator = MetricsCalculator()
    return calculator.compute_rouge_metrics(predictions, references)


def compute_nlg_metrics(predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
    """Compute NLG benchmark metrics."""
    calculator = MetricsCalculator()
    return calculator.compute_nlg_metrics(predictions, references)


def compute_sparsity_metrics(weights: torch.Tensor, K: int, 
                            total_adapters: int) -> Dict[str, float]:
    """Compute sparsity metrics for weight vector."""
    calculator = MetricsCalculator()
    return calculator.compute_sparsity_metrics(weights, K, total_adapters)


def compute_memory_reduction(baseline_params: int, 
                            active_params: int) -> Dict[str, float]:
    """Compute parameter reduction achieved by WeightLoRA."""
    calculator = MetricsCalculator()
    return calculator.compute_memory_reduction(baseline_params, active_params)


def validate_sparsity_constraint(weights: torch.Tensor, K: int, 
                                tolerance: float = 1e-6) -> Tuple[bool, int]:
    """
    Validate that weight vector satisfies sparsity constraint ||ω||_0 ≤ K.
    
    Args:
        weights: Weight vector ω
        K: Target sparsity level
        tolerance: Tolerance for floating point comparison
    
    Returns:
        Tuple of (is_valid, actual_sparsity)
    """
    non_zero = (weights.abs() > tolerance).sum().item()
    is_valid = non_zero <= K
    return is_valid, int(non_zero)


def compute_weightlora_reduction(model, K: int, baseline_params: int) -> Dict[str, float]:
    """
    Compute parameter reduction achieved by WeightLoRA.
    
    Args:
        model: WeightLoRA model instance
        K: Target sparsity level
        baseline_params: Total parameters in full LoRA
    
    Returns:
        Dictionary with reduction statistics
    """
    # Count active parameters
    active_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 1:  # Weight vector
            active_params += (param.abs() > 1e-6).sum().item()
    
    # Estimate adapter parameters (simplified)
    # In practice, would need to track adapter parameter counts
    reduction = compute_memory_reduction(baseline_params, active_params)
    reduction['K'] = K
    reduction['baseline_params'] = baseline_params
    
    return reduction


def compute_accuracy(predictions: List[int], targets: List[int]) -> float:
    """Compute accuracy for classification tasks."""
    return accuracy_score(targets, predictions)


def compute_f1_score(predictions: List[int], targets: List[int]) -> float:
    """Compute F1 score for classification tasks."""
    return f1_score(targets, predictions, average='binary')


def compute_mcc_score(predictions: List[int], targets: List[int]) -> float:
    """Compute Matthews Correlation Coefficient."""
    return matthews_corrcoef(targets, predictions)


def compute_loss_contribution(model, layer_indices: List[int], 
                             input_batch: torch.Tensor, 
                             target_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute loss contribution from specific layers for StoIHT gradient computation.
    
    Args:
        model: Model with WeightLoRA layers
        layer_indices: List of layer indices to compute contribution for
        input_batch: Input tensor
        target_batch: Target tensor
    
    Returns:
        Tensor of loss contributions for each layer
    """
    contributions = torch.zeros(len(layer_indices))
    
    for i, layer_idx in enumerate(layer_indices):
        # Temporarily set weight to 1 to compute pure layer gradient
        if hasattr(model, 'layers'):
            layer = model.layers[layer_idx]
            original_weight = layer.weight.clone()
            layer.weight.data = torch.ones_like(layer.weight)
            
            # Forward pass
            output = model(input_batch)
            loss = torch.nn.functional.cross_entropy(output, target_batch)
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Extract gradient
            if hasattr(layer, 'weight_grad'):
                contributions[i] = torch.sum(layer.weight_grad).item()
            
            # Restore original weight
            layer.weight.data = original_weight
    
    return contributions


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics from multiple runs."""
    calculator = MetricsCalculator()
    return calculator.aggregate_metrics(results)
