"""
Utility functions for WeightLoRA training pipeline.

This module provides essential utilities for:
- Model loading and initialization
- Dataset loading and preprocessing
- Checkpoint saving and loading
- Device configuration
- Random seed setting for reproducibility
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer
import yaml


def setup_device() -> torch.device:
    """
    Configure GPU/CPU device for training.
    
    Returns:
        torch.device: The device to use for computation (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA not available)")
    
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def load_model(model_name: str, num_labels: int = 2, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Load a pretrained model from HuggingFace.
    
    Args:
        model_name: Model name from HuggingFace (e.g., 'microsoft/deberta-v3-base')
        num_labels: Number of labels for classification tasks
        device: Device to load model on (default: auto-detect)
    
    Returns:
        torch.nn.Module: Loaded model instance
    """
    if device is None:
        device = setup_device()
    
    set_seed(42)  # Ensure reproducibility during model loading
    
    if "deberta" in model_name.lower():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    elif "bart" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif "llama" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif "squad" in model_name.lower():
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    model.to(device)
    model.eval()
    print(f"Loaded model: {model_name}")
    return model


def load_dataset(dataset_name: str, max_length: int = 128) -> Tuple[Dataset, Dataset]:
    """
    Load dataset from HuggingFace.
    
    Args:
        dataset_name: Dataset name (e.g., 'glue/mnli', 'squad_v2', 'xsum')
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Tuple[Dataset, Dataset]: (train_dataset, validation_dataset)
    """
    from datasets import load_dataset
    
    if "glue" in dataset_name.lower():
        # GLUE tasks
        dataset = load_dataset("glue", dataset_name.replace("glue/", ""))
        train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        val_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
    elif "squad" in dataset_name.lower():
        # SQuAD datasets
        if "v2" in dataset_name.lower():
            dataset = load_dataset("squad_v2")
        else:
            dataset = load_dataset("squad_v1")
        train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        val_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
    elif "xsum" in dataset_name.lower():
        # XSum summarization
        dataset = load_dataset("xsum")
        train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        val_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
    elif "cnn" in dataset_name.lower() or "dailymail" in dataset_name.lower():
        # CNN/DailyMail summarization
        dataset = load_dataset("rotten/daily_mail")
        train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        val_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
    else:
        # Generic dataset loading
        dataset = load_dataset(dataset_name)
        train_dataset = dataset["train"].select(range(min(5000, len(dataset["train"]))))
        val_dataset = dataset["validation"].select(range(min(1000, len(dataset["validation"]))))
    
    return train_dataset, val_dataset


def create_dataloaders(dataset: Dataset, batch_size: int = 32, device: Optional[torch.device] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size for data loading
        device: Device to use for data loading
    
    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, val_dataloader)
    """
    if device is None:
        device = setup_device()
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


def tokenize_glue_inputs(dataset, tokenizer, max_length: int = 128):
    """
    Tokenize GLUE dataset inputs.
    
    Args:
        dataset: GLUE dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tuple[Dataset, Dataset]: Tokenized train and validation datasets
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"] if "premise" in examples else examples["sentence1"],
            examples["hypothesis"] if "hypothesis" in examples else examples["sentence2"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_datasets


def tokenize_squad_inputs(dataset, tokenizer, max_length: int = 384):
    """
    Tokenize SQuAD dataset inputs.
    
    Args:
        dataset: SQuAD dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tuple[Dataset, Dataset]: Tokenized train and validation datasets
    """
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        tokenized_inputs["answers"] = examples["answers"]
        return tokenized_inputs
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_datasets


def tokenize_nlg_inputs(dataset, tokenizer, max_length: int = 512):
    """
    Tokenize NLG (summarization) dataset inputs.
    
    Args:
        dataset: NLG dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Tuple[Dataset, Dataset]: Tokenized train and validation datasets
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["article"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_datasets


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, path: str, weight_vector: Optional[torch.Tensor] = None) -> None:
    """
    Save model checkpoint including weights, optimizer state, and weight vector.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current training epoch
        path: Path to save checkpoint
        weight_vector: Optional weight vector to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "weight_vector": weight_vector.cpu() if weight_vector is not None else None
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def load_checkpoint(path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint file
    
    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int]: (model, optimizer, epoch)
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    
    model = torch.nn.Module()
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    
    print(f"Checkpoint loaded from: {path}, epoch: {epoch}")
    return model, optimizer, epoch


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor, task: str = "classification") -> Dict[str, float]:
    """
    Compute evaluation metrics based on task type.
    
    Args:
        outputs: Model outputs (logits or probabilities)
        targets: Ground truth labels
        task: Task type ('classification', 'qa', 'summarization')
    
    Returns:
        Dict[str, float]: Dictionary of computed metrics
    """
    metrics = {}
    
    if task == "classification":
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean().item()
        metrics["accuracy"] = accuracy
        metrics["loss"] = torch.nn.functional.cross_entropy(outputs, targets).item()
    
    elif task == "qa":
        # SQuAD metrics would require special handling
        metrics["accuracy"] = (outputs == targets).float().mean().item()
        metrics["loss"] = torch.nn.functional.cross_entropy(outputs, targets).item()
    
    elif task == "summarization":
        # ROUGE metrics would be computed separately
        metrics["accuracy"] = 0.0  # Placeholder
        metrics["loss"] = torch.nn.functional.cross_entropy(outputs, targets).item()
    
    return metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_model_param_count(model: torch.nn.Module) -> int:
    """
    Get total number of parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def get_trainable_param_count(model: torch.nn.Module) -> int:
    """
    Get number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, max_depth: int = 3) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to print
    """
    total_params = get_model_param_count(model)
    trainable_params = get_trainable_param_count(model)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Parameter ratio: {trainable_params / total_params * 100:.2f}%")
    print()
