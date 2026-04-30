"""
Checkpointing utilities for WeightLoRA training pipeline.
Provides functions for saving and loading model checkpoints, including weight vectors,
optimizer states, and training progress for reproducibility.
"""

import os
import json
import torch
from typing import Dict, Tuple, Optional, Any
from datetime import datetime


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    path: str,
    weight_vector: Optional[torch.Tensor] = None,
    active_indices: Optional[list] = None,
    config: Optional[Dict] = None
) -> None:
    """
    Save complete model checkpoint including model state, optimizer, and weight vector.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current training epoch
        step: Current training step
        path: Directory path to save checkpoint
        weight_vector: WeightLoRA weight vector ω (optional)
        active_indices: List of active adapter indices (optional)
        config: Training configuration dictionary (optional)
    """
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if weight_vector is not None:
        checkpoint['weight_vector'] = weight_vector.detach().cpu()
    
    if active_indices is not None:
        checkpoint['active_indices'] = active_indices
    
    if config is not None:
        checkpoint['config'] = config
    
    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata
    metadata = {
        'epoch': epoch,
        'step': step,
        'timestamp': checkpoint['timestamp']
    }
    metadata_path = os.path.join(path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """
    Load model checkpoint from saved path.
    
    Args:
        path: Directory path containing checkpoint
        
    Returns:
        Tuple of (model, optimizer, epoch)
    """
    checkpoint_path = os.path.join(path, 'checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model and optimizer (assumed to be passed separately)
    # For now, return checkpoint data
    return checkpoint['model_state_dict'], checkpoint['optimizer_state_dict'], checkpoint['epoch']


def save_training_history(
    history: Dict[str, list],
    path: str,
    filename: str = 'training_history.json'
) -> None:
    """
    Save training history (losses, metrics) to JSON file.
    
    Args:
        history: Dictionary with keys 'train_loss', 'val_loss', 'accuracy', etc.
        path: Directory to save file
        filename: Output filename
    """
    os.makedirs(path, exist_ok=True)
    
    history_path = os.path.join(path, filename)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to {history_path}")


def load_training_history(path: str, filename: str = 'training_history.json') -> Dict:
    """
    Load training history from JSON file.
    
    Args:
        path: Directory containing history file
        filename: History filename
        
    Returns:
        Training history dictionary
    """
    history_path = os.path.join(path, filename)
    
    if not os.path.exists(history_path):
        return {}
    
    with open(history_path, 'r') as f:
        return json.load(f)


def save_experiment_results(
    results: Dict[str, Any],
    path: str,
    experiment_name: str = 'experiment'
) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Experiment results dictionary
        path: Directory to save file
        experiment_name: Name of experiment
    """
    os.makedirs(path, exist_ok=True)
    
    results_path = os.path.join(path, f'{experiment_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Experiment results saved to {results_path}")


def load_experiment_results(path: str, experiment_name: str = 'experiment') -> Dict:
    """
    Load experiment results from JSON file.
    
    Args:
        path: Directory containing results
        experiment_name: Experiment name
        
    Returns:
        Results dictionary
    """
    results_path = os.path.join(path, f'{experiment_name}_results.json')
    
    if not os.path.exists(results_path):
        return {}
    
    with open(results_path, 'r') as f:
        return json.load(f)


def save_model_only(
    model: torch.nn.Module,
    path: str,
    filename: str = 'model.pth'
) -> None:
    """
    Save only model weights (no optimizer state).
    
    Args:
        model: PyTorch model
        path: Directory to save file
        filename: Output filename
    """
    os.makedirs(path, exist_ok=True)
    
    model_path = os.path.join(path, filename)
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")


def load_model_only(
    model: torch.nn.Module,
    path: str,
    filename: str = 'model.pth'
) -> None:
    """
    Load model weights from saved file.
    
    Args:
        model: PyTorch model (state_dict will be loaded)
        path: Directory containing model file
        filename: Model filename
    """
    model_path = os.path.join(path, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Model loaded from {model_path}")


def save_optimizer_only(
    optimizer: torch.optim.Optimizer,
    path: str,
    filename: str = 'optimizer.pth'
) -> None:
    """
    Save only optimizer state (for resuming training).
    
    Args:
        optimizer: PyTorch optimizer
        path: Directory to save file
        filename: Output filename
    """
    os.makedirs(path, exist_ok=True)
    
    optimizer_path = os.path.join(path, filename)
    torch.save(optimizer.state_dict(), optimizer_path)
    
    print(f"Optimizer saved to {optimizer_path}")


def load_optimizer_only(
    optimizer: torch.optim.Optimizer,
    path: str,
    filename: str = 'optimizer.pth'
) -> None:
    """
    Load optimizer state from saved file.
    
    Args:
        optimizer: PyTorch optimizer (state_dict will be loaded)
        path: Directory containing optimizer file
        filename: Optimizer filename
    """
    optimizer_path = os.path.join(path, filename)
    
    if not os.path.exists(optimizer_path):
        raise FileNotFoundError(f"Optimizer file not found at {optimizer_path}")
    
    optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
    print(f"Optimizer loaded from {optimizer_path}")


def save_weight_vector(
    weight_vector: torch.Tensor,
    path: str,
    filename: str = 'weight_vector.pth'
) -> None:
    """
    Save WeightLoRA weight vector ω separately.
    
    Args:
        weight_vector: Weight vector tensor
        path: Directory to save file
        filename: Output filename
    """
    os.makedirs(path, exist_ok=True)
    
    weight_path = os.path.join(path, filename)
    torch.save(weight_vector.detach().cpu(), weight_path)
    
    print(f"Weight vector saved to {weight_path}")


def load_weight_vector(
    path: str,
    filename: str = 'weight_vector.pth'
) -> torch.Tensor:
    """
    Load WeightLoRA weight vector from saved file.
    
    Args:
        path: Directory containing weight vector file
        filename: Weight vector filename
        
    Returns:
        Loaded weight vector tensor
    """
    weight_path = os.path.join(path, filename)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight vector file not found at {weight_path}")
    
    weight_vector = torch.load(weight_path, map_location='cpu')
    print(f"Weight vector loaded from {weight_path}")
    return weight_vector


def save_full_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    weight_vector: torch.Tensor,
    active_indices: list,
    path: str,
    epoch: int,
    step: int,
    config: Optional[Dict] = None
) -> str:
    """
    Save complete checkpoint with all WeightLoRA components.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        weight_vector: WeightLoRA weight vector
        active_indices: Active adapter indices
        path: Directory to save checkpoint
        epoch: Current epoch
        step: Current step
        config: Training config (optional)
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'weight_vector': weight_vector.detach().cpu(),
        'active_indices': active_indices,
        'timestamp': datetime.now().isoformat()
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    checkpoint_path = os.path.join(path, 'full_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Full checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_full_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> Tuple[torch.Tensor, list]:
    """
    Load complete checkpoint with all WeightLoRA components.
    
    Args:
        path: Directory containing checkpoint
        model: PyTorch model (state_dict will be loaded)
        optimizer: Optimizer (state_dict will be loaded)
        
    Returns:
        Tuple of (weight_vector, active_indices)
    """
    checkpoint_path = os.path.join(path, 'full_checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Full checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    weight_vector = checkpoint['weight_vector']
    active_indices = checkpoint['active_indices']
    
    print(f"Full checkpoint loaded from {checkpoint_path}")
    return weight_vector, active_indices


def create_checkpoint_dir(
    base_path: str,
    experiment_name: str,
    model_name: str = 'deberta',
    k: int = 10,
    seed: int = 42
) -> str:
    """
    Create checkpoint directory with standardized naming.
    
    Args:
        base_path: Base directory
        experiment_name: Experiment identifier
        model_name: Model name (deberta, bart, llama)
        k: Sparsity level K
        seed: Random seed
        
    Returns:
        Path to created checkpoint directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(
        base_path,
        f'{experiment_name}_{model_name}_k{k}_seed{seed}_{timestamp}'
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory created: {checkpoint_dir}")
    return checkpoint_dir


def list_checkpoints(path: str) -> list:
    """
    List all checkpoint directories in a path.
    
    Args:
        path: Directory to search
        
    Returns:
        List of checkpoint directory names
    """
    if not os.path.exists(path):
        return []
    
    checkpoints = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            # Check if it contains checkpoint file
            checkpoint_file = os.path.join(item_path, 'checkpoint.pth')
            if os.path.exists(checkpoint_file):
                checkpoints.append(item)
    
    return checkpoints


def delete_checkpoint(path: str) -> None:
    """
    Delete checkpoint directory and all contents.
    
    Args:
        path: Directory to delete
    """
    import shutil
    
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Checkpoint directory deleted: {path}")
    else:
        print(f"Checkpoint directory not found: {path}")


def merge_checkpoints(
    checkpoint_paths: list,
    output_path: str,
    keep_largest: bool = True
) -> None:
    """
    Merge multiple checkpoints, keeping the largest one by step/epoch.
    
    Args:
        checkpoint_paths: List of checkpoint directory paths
        output_path: Output directory for merged checkpoint
        keep_largest: Keep only the largest checkpoint
        
    Returns:
        Path to kept checkpoint
    """
    checkpoints = []
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_file = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(checkpoint_file):
                checkpoints.append((path, checkpoint_file))
    
    if not checkpoints:
        raise ValueError("No checkpoints found")
    
    # Sort by step (if available)
    def get_step(path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            return checkpoint.get('step', 0)
        except:
            return 0
    
    checkpoints.sort(key=lambda x: get_step(x[1]), reverse=True)
    
    if keep_largest:
        best_path, best_file = checkpoints[0]
        output_path = os.path.join(output_path, 'merged_checkpoint.pth')
        torch.save(torch.load(best_file, map_location='cpu'), output_path)
        print(f"Kept checkpoint with largest step: {best_path}")
    else:
        # Keep all
        for path, file in checkpoints:
            torch.save(torch.load(file, map_location='cpu'), output_path)
    
    print(f"Merge complete: {output_path}")
    return output_path
