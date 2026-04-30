"""
SQuAD v1.1 and v2.0 dataset loading utilities for WeightLoRA.
Provides standardized interfaces for question answering tasks with PyTorch DataLoader integration.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
import numpy as np


class SQuADDataset(Dataset):
    """
    SQuAD v1.1/v2.0 dataset wrapper for WeightLoRA training.
    
    Attributes:
        version: "v1.1" or "v2.0"
        split: "train", "validation", or "test"
        tokenizer: HuggingFace tokenizer for preprocessing
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        version: str = "v2.0",
        split: str = "train",
        tokenizer: Optional[Any] = None,
        max_length: int = 384,
        is_training: bool = True
    ):
        """
        Initialize SQuAD dataset.
        
        Args:
            version: Dataset version ("v1.1" or "v2.0")
            split: Dataset split ("train", "validation", "test")
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length
            is_training: Whether this is training data
        """
        self.version = version
        self.split = split
        self.max_length = max_length
        self.is_training = is_training
        
        # Load dataset from HuggingFace
        if version == "v1.1":
            self.dataset = load_dataset("squad", "v1.1", split=split)
        else:
            self.dataset = load_dataset("squad", "v2.0", split=split)
        
        # Tokenize if tokenizer provided
        if tokenizer is not None:
            self.dataset = self.dataset.map(
                lambda x: tokenizer(
                    x["question"],
                    x["context"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length"
                ),
                batched=True
            )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index with preprocessing.
        
        Returns:
            Dictionary with question, context, attention_mask, and labels
        """
        item = self.dataset[idx]
        
        # Prepare labels for v2.0
        if self.version == "v2.0":
            # v2.0 has is_impossible flag
            is_impossible = item.get("is_impossible", False)
            if is_impossible:
                # For impossible questions, use max_length as answer
                start_positions = torch.full(
                    (self.max_length,), 
                    self.max_length, 
                    dtype=torch.long
                )
                end_positions = torch.full(
                    (self.max_length,), 
                    self.max_length, 
                    dtype=torch.long
                )
            else:
                start_positions = torch.tensor(
                    item["start_positions"], 
                    dtype=torch.long
                )
                end_positions = torch.tensor(
                    item["end_positions"], 
                    dtype=torch.long
                )
        else:
            # v1.1 always has answers
            start_positions = torch.tensor(
                item["start_positions"], 
                dtype=torch.long
            )
            end_positions = torch.tensor(
                item["end_positions"], 
                dtype=torch.long
            )
        
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "start_positions": start_positions,
            "end_positions": end_positions,
            "is_impossible": is_impossible if self.version == "v2.0" else None,
            "question": item["question"],
            "context": item["context"]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "num_samples": len(self),
            "version": self.version,
            "split": self.split,
            "max_length": self.max_length
        }


class SQuADDataLoader:
    """
    Utility for creating SQuAD train/val/test DataLoaders.
    """
    
    @staticmethod
    def create_loaders(
        version: str = "v2.0",
        tokenizer: Optional[Any] = None,
        train_batch_size: int = 16,
        validation_batch_size: int = 16,
        max_length: int = 384,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders for SQuAD.
        
        Args:
            version: Dataset version
            tokenizer: HuggingFace tokenizer
            train_batch_size: Training batch size
            validation_batch_size: Validation batch size
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_loader, validation_loader)
        """
        np.random.seed(seed)
        
        # Create datasets
        train_dataset = SQuADDataset(
            version=version,
            split="train",
            tokenizer=tokenizer,
            max_length=max_length,
            is_training=True
        )
        
        val_dataset = SQuADDataset(
            version=version,
            split="validation",
            tokenizer=tokenizer,
            max_length=max_length,
            is_training=False
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    @staticmethod
    def get_task_info(version: str = "v2.0") -> Dict[str, Any]:
        """
        Get information about SQuAD task configuration.
        
        Returns:
            Dictionary with task configuration details
        """
        return {
            "version": version,
            "num_labels": 1,  # Single label for answer span
            "metric": "f1",
            "secondary_metric": "exact_match",
            "description": "SQuAD Question Answering Benchmark"
        }


def load_squad_dataset(
    version: str = "v2.0",
    split: str = "train",
    tokenizer: Optional[Any] = None,
    max_length: int = 384
) -> SQuADDataset:
    """
    Load a single SQuAD dataset split.
    
    Args:
        version: Dataset version ("v1.1" or "v2.0")
        split: Dataset split ("train", "validation", "test")
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        SQuADDataset instance
    """
    return SQuADDataset(
        version=version,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length
    )


def get_squad_versions() -> List[str]:
    """
    Get available SQuAD versions.
    
    Returns:
        List of available versions
    """
    return ["v1.1", "v2.0"]


if __name__ == "__main__":
    # Test SQuAD dataset loading
    print("Testing SQuAD dataset loading...")
    
    # Create a simple tokenizer for testing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load dataset
    dataset = load_squad_dataset(
        version="v2.0",
        split="validation",
        tokenizer=tokenizer,
        max_length=384
    )
    
    print(f"Loaded {len(dataset)} samples")
    
    # Get first sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Question length: {len(sample['question'])}")
    print(f"Context length: {len(sample['context'])}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    
    # Test DataLoader
    train_loader, val_loader = SQuADDataLoader.create_loaders(
        version="v2.0",
        tokenizer=tokenizer,
        train_batch_size=4,
        validation_batch_size=4,
        max_length=384
    )
    
    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Test iteration
    for batch in val_loader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        break
    
    print("SQuAD dataset loading test completed successfully!")
