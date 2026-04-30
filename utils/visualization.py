"""
Visualization utilities for WeightLoRA implementation.
Provides functions for plotting training curves, performance comparisons,
memory usage analysis, and sparsity evolution.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import seaborn as sns
import pandas as pd


class VisualizationManager:
    """Manages all visualization outputs for WeightLoRA experiments."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        self.figures = {}
    
    def save_figure(self, fig, filename: str, dpi: int = 300):
        """Save a matplotlib figure to file."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure: {filepath}")
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """Load experiment results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_training_curves(self, results: Dict[str, Any], 
                            output_filename: str = "training_curves.png"):
        """
        Plot training loss curves for different methods.
        
        Args:
            results: Dictionary containing training history from experiments
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('WeightLoRA Training Curves', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Plot 1: Training Loss
        ax = axes[0, 0]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            train_loss = history.get('train_loss', [])
            if train_loss:
                ax.plot(range(len(train_loss)), train_loss, 
                       label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            val_loss = history.get('val_loss', [])
            if val_loss:
                ax.plot(range(len(val_loss)), val_loss,
                       label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Over Epochs')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy/F1 Score
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            val_score = history.get('val_score', [])
            if val_score:
                ax.plot(range(len(val_score)), val_score,
                       label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Score')
        ax.set_title('Validation Performance Over Epochs')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Sparsity Evolution
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            active_adapters = history.get('active_adapters', [])
            if active_adapters:
                ax.plot(range(len(active_adapters)), active_adapters,
                       label=method, color=colors[i], linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Active Adapters')
        ax.set_title('Adapter Selection Over Training')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_parameter_reduction(self, results: Dict[str, Any],
                                output_filename: str = "parameter_reduction.png"):
        """
        Plot parameter reduction achieved by WeightLoRA at different K values.
        
        Args:
            results: Dictionary containing parameter counts
            output_filename: Output filename for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('WeightLoRA Parameter Reduction', fontsize=16, fontweight='bold')
        
        # Extract K values and parameter counts
        k_values = []
        param_counts = []
        reduction_percentages = []
        
        for method, data in results.items():
            params = data.get('param_count', {})
            k_values.append(params.get('K', 10))
            param_counts.append(params.get('active_params', 0))
            baseline = params.get('baseline_params', 442000)
            reduction = (baseline - param_counts[-1]) / baseline * 100
            reduction_percentages.append(reduction)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
        ax.bar(range(len(k_values)), param_counts, color=colors, alpha=0.8,
               edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('K Value (Number of Active Adapters)', fontsize=12)
        ax.set_ylabel('Number of Trainable Parameters', fontsize=12)
        ax.set_title('Parameter Count vs K Value', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=10)
        
        # Add reduction percentage annotation
        for i, (k, count, reduction) in enumerate(zip(k_values, param_counts, reduction_percentages)):
            ax.annotate(f'{reduction:.1f}% reduction',
                       (i, count),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=9,
                       color='red')
        
        # Add target line
        ax.axhline(y=61500, color='green', linestyle='--', 
                  label='Target (K=5): 61.5K params', alpha=0.7)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_memory_comparison(self, results: Dict[str, Any],
                              output_filename: str = "memory_comparison.png"):
        """
        Compare memory usage across different methods.
        
        Args:
            results: Dictionary containing memory usage data
            output_filename: Output filename for the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Memory Usage Comparison', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        memory_values = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            memory = results.get(method, {}).get('memory_usage', {})
            trainable_params = memory.get('trainable_params', 0)
            memory_values.append(trainable_params)
            ax.bar(method, trainable_params, color=colors[i], alpha=0.8,
                   edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Trainable Parameters (K)', fontsize=12)
        ax.set_title('Memory Usage by Method', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (method, value) in enumerate(zip(methods, memory_values)):
            ax.text(i, value + 5, f'{value/1000:.1f}K',
                   ha='center', va='bottom', fontsize=10)
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_sparsity_evolution(self, results: Dict[str, Any],
                               output_filename: str = "sparsity_evolution.png"):
        """
        Plot the evolution of sparsity (weight vector ω) over training.
        
        Args:
            results: Dictionary containing weight vector history
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Sparsity Evolution Over Training', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Plot 1: Weight vector values
        ax = axes[0]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('weight_vector_history', [])
            if history:
                weights = [h.get('weight', []) for h in history]
                if weights:
                    ax.plot(range(len(weights[0])), weights[0],
                           label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value (ω_i)')
        ax.set_title('Weight Vector Evolution')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Active (ω=1)')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Inactive (ω=0)')
        
        # Plot 2: Sparsity ratio
        ax = axes[1]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('sparsity_history', [])
            if history:
                sparsity = [h.get('sparsity_ratio', 0) for h in history]
                if sparsity:
                    ax.plot(range(len(sparsity)), sparsity,
                           label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Sparsity Ratio (1 - active/K)')
        ax.set_title('Sparsity Ratio Over Training')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.9, color='green', linestyle='--', 
                  label='Target (90% sparsity)', alpha=0.7)
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_performance_comparison(self, results: Dict[str, Any],
                                   output_filename: str = "performance_comparison.png"):
        """
        Compare final performance across different methods and benchmarks.
        
        Args:
            results: Dictionary containing final performance metrics
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance Comparison Across Methods', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Plot 1: GLUE Average Score
        ax = axes[0, 0]
        glue_scores = []
        for method in methods:
            metrics = results.get(method, {}).get('metrics', {})
            glue_avg = metrics.get('glue_average', 0)
            if glue_avg > 0:
                glue_scores.append(glue_avg)
                ax.bar(method, glue_avg, color=colors[methods.index(method)],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('GLUE Average Score', fontsize=12)
        ax.set_title('GLUE Benchmark Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.8562, color='red', linestyle='--', 
                  label='LoRA Baseline: 0.8562', alpha=0.7)
        ax.legend(loc='upper left')
        
        # Plot 2: SQuAD F1 Score
        ax = axes[0, 1]
        squad_scores = []
        for method in methods:
            metrics = results.get(method, {}).get('metrics', {})
            squad_f1 = metrics.get('squad_f1', 0)
            if squad_f1 > 0:
                squad_scores.append(squad_f1)
                ax.bar(method, squad_f1, color=colors[methods.index(method)],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('SQuAD v2.0 F1 Score', fontsize=12)
        ax.set_title('SQuAD Question Answering', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.8234, color='red', linestyle='--', 
                  label='Target: 0.8234', alpha=0.7)
        ax.legend(loc='upper left')
        
        # Plot 3: ROUGE-1 Score
        ax = axes[1, 0]
        rouge_scores = []
        for method in methods:
            metrics = results.get(method, {}).get('metrics', {})
            rouge1 = metrics.get('rouge1', 0)
            if rouge1 > 0:
                rouge_scores.append(rouge1)
                ax.bar(method, rouge1, color=colors[methods.index(method)],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('ROUGE-1 Score', fontsize=12)
        ax.set_title('Summarization Performance (XSum)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.3674, color='red', linestyle='--', 
                  label='Target: 0.3674', alpha=0.7)
        ax.legend(loc='upper left')
        
        # Plot 4: Parameter Efficiency
        ax = axes[1, 1]
        efficiency_scores = []
        for method in methods:
            metrics = results.get(method, {}).get('metrics', {})
            reduction = metrics.get('param_reduction', 0)
            if reduction > 0:
                efficiency_scores.append(reduction)
                ax.bar(method, reduction, color=colors[methods.index(method)],
                      alpha=0.8, edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Parameter Reduction (%)', fontsize=12)
        ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=86, color='green', linestyle='--', 
                  label='Target: 86%', alpha=0.7)
        ax.legend(loc='upper left')
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_weightlora_plus_comparison(self, results: Dict[str, Any],
                                       output_filename: str = "weightlora_plus_comparison.png"):
        """
        Compare WeightLoRA vs WeightLoRA+ performance.
        
        Args:
            results: Dictionary containing WeightLoRA and WeightLoRA+ results
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('WeightLoRA vs WeightLoRA+ Comparison', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Plot 1: GLUE Performance
        ax = axes[0, 0]
        for i, method in enumerate(methods):
            metrics = results.get(method, {}).get('metrics', {})
            glue_avg = metrics.get('glue_average', 0)
            if glue_avg > 0:
                ax.bar(method, glue_avg, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('GLUE Average', fontsize=12)
        ax.set_title('GLUE Benchmark', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Parameter Reduction
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            metrics = results.get(method, {}).get('metrics', {})
            reduction = metrics.get('param_reduction', 0)
            if reduction > 0:
                ax.bar(method, reduction, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Parameter Reduction (%)', fontsize=12)
        ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Active Adapters
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            final_active = history.get('final_active_adapters', 0)
            if final_active > 0:
                ax.bar(method, final_active, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Active Adapters', fontsize=12)
        ax.set_title('Final Active Adapters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training Time
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            epochs = history.get('epochs', 0)
            if epochs > 0:
                ax.bar(method, epochs, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Training Epochs', fontsize=12)
        ax.set_title('Training Duration', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_rank_expansion(self, results: Dict[str, Any],
                           output_filename: str = "rank_expansion.png"):
        """
        Visualize rank expansion in WeightLoRA+ training.
        
        Args:
            results: Dictionary containing rank expansion history
            output_filename: Output filename for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('Rank Expansion in WeightLoRA+', fontsize=16, fontweight='bold')
        
        # Extract rank history
        phases = list(results.keys())
        rank_values = []
        
        for phase in phases:
            history = results.get(phase, {}).get('rank_history', [])
            if history:
                ranks = [h.get('rank', 4) for h in history]
                rank_values.extend(ranks)
        
        if rank_values:
            ax.plot(range(len(rank_values)), rank_values,
                   color='blue', linewidth=2, marker='o', markersize=4)
        
        # Mark phase transition
        ax.axvline(x=len(results.get('phase_1', {}).get('rank_history', [])),
                  color='red', linestyle='--', alpha=0.7,
                  label='Phase Transition')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Adapter Rank', fontsize=12)
        ax.set_title('Rank Evolution Over Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_ablation_study(self, results: Dict[str, Any],
                           output_filename: str = "ablation_study.png"):
        """
        Plot ablation study results comparing WeightLoRA vs RLoRA.
        
        Args:
            results: Dictionary containing ablation study results
            output_filename: Output filename for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Ablation Study: WeightLoRA vs RLoRA', fontsize=16, fontweight='bold')
        
        methods = list(results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # Plot 1: GLUE Performance
        ax = axes[0, 0]
        for i, method in enumerate(methods):
            metrics = results.get(method, {}).get('metrics', {})
            glue_avg = metrics.get('glue_average', 0)
            if glue_avg > 0:
                ax.bar(method, glue_avg, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('GLUE Average', fontsize=12)
        ax.set_title('GLUE Benchmark Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Parameter Reduction
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            metrics = results.get(method, {}).get('metrics', {})
            reduction = metrics.get('param_reduction', 0)
            if reduction > 0:
                ax.bar(method, reduction, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Parameter Reduction (%)', fontsize=12)
        ax.set_title('Parameter Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Active Adapters
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            final_active = history.get('final_active_adapters', 0)
            if final_active > 0:
                ax.bar(method, final_active, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1.2)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Active Adapters', fontsize=12)
        ax.set_title('Final Active Adapters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Training Loss
        ax = axes[1, 1]
        for i, method in enumerate(methods):
            history = results.get(method, {}).get('history', {})
            train_loss = history.get('train_loss', [])
            if train_loss:
                ax.plot(range(len(train_loss)), train_loss,
                       label=method, color=colors[i], linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, output_filename)
        return fig
    
    def plot_all_results(self, results: Dict[str, Any], 
                        base_filename: str = "weightlora_results"):
        """
        Generate all visualization plots from results.
        
        Args:
            results: Dictionary containing all experiment results
            base_filename: Base filename for all plots
        """
        plots = [
            ("training_curves", self.plot_training_curves),
            ("parameter_reduction", self.plot_parameter_reduction),
            ("memory_comparison", self.plot_memory_comparison),
            ("sparsity_evolution", self.plot_sparsity_evolution),
            ("performance_comparison", self.plot_performance_comparison),
            ("weightlora_plus_comparison", self.plot_weightlora_plus_comparison),
            ("ablation_study", self.plot_ablation_study),
        ]
        
        for plot_name, plot_func in plots:
            try:
                plot_func(results, f"{base_filename}_{plot_name}.png")
            except Exception as e:
                print(f"Warning: Could not generate {plot_name}: {e}")
        
        print(f"Generated all plots in {self.output_dir}/")
        return self.output_dir


def generate_comparison_table(results: Dict[str, Any], 
                            output_path: str = "comparison_table.csv"):
    """
    Generate a CSV comparison table from experiment results.
    
    Args:
        results: Dictionary containing experiment results
        output_path: Output CSV file path
    """
    df_data = []
    
    for method, data in results.items():
        metrics = data.get('metrics', {})
        history = data.get('history', {})
        
        row = {
            'Method': method,
            'GLUE Average': metrics.get('glue_average', 0),
            'SQuAD F1': metrics.get('squad_f1', 0),
            'ROUGE-1': metrics.get('rouge1', 0),
            'ROUGE-L': metrics.get('rouge_l', 0),
            'Param Reduction (%)': metrics.get('param_reduction', 0),
            'Active Adapters': history.get('final_active_adapters', 0),
            'Training Epochs': history.get('epochs', 0),
            'Trainable Params (K)': data.get('param_count', {}).get('active_params', 0) / 1000
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    print(f"Generated comparison table: {output_path}")
    return df


def create_results_summary(results: Dict[str, Any], 
                         output_path: str = "results_summary.txt"):
    """
    Create a text summary of all results.
    
    Args:
        results: Dictionary containing experiment results
        output_path: Output text file path
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WEIGHTLoRA EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for method, data in results.items():
            f.write(f"Method: {method}\n")
            f.write("-" * 40 + "\n")
            
            metrics = data.get('metrics', {})
            f.write(f"  GLUE Average: {metrics.get('glue_average', 0):.4f}\n")
            f.write(f"  SQuAD F1: {metrics.get('squad_f1', 0):.4f}\n")
            f.write(f"  ROUGE-1: {metrics.get('rouge1', 0):.4f}\n")
            f.write(f"  ROUGE-L: {metrics.get('rouge_l', 0):.4f}\n")
            f.write(f"  Parameter Reduction: {metrics.get('param_reduction', 0):.2f}%\n")
            f.write(f"  Active Adapters: {data.get('param_count', {}).get('active_params', 0)}\n")
            f.write(f"  Baseline Params: {data.get('param_count', {}).get('baseline_params', 0)}\n")
            f.write(f"  Training Epochs: {data.get('history', {}).get('epochs', 0)}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"Generated results summary: {output_path}")


def main():
    """Main function to demonstrate visualization capabilities."""
    # Example usage
    print("Visualization utilities loaded successfully.")
    print("Available methods:")
    print("  - plot_training_curves()")
    print("  - plot_parameter_reduction()")
    print("  - plot_memory_comparison()")
    print("  - plot_sparsity_evolution()")
    print("  - plot_performance_comparison()")
    print("  - plot_weightlora_plus_comparison()")
    print("  - plot_ablation_study()")
    print("  - plot_all_results()")
    print("  - generate_comparison_table()")
    print("  - create_results_summary()")


if __name__ == "__main__":
    main()
