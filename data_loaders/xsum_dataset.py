"""
XSum Dataset Implementation for WeightLoRA

This module provides dataset loading utilities for the XSum news summarization benchmark.
XSum is a single-document summarization dataset with short summaries.

Public Interface:
- Class XSumDataset: XSum dataset wrapper for training/evaluation
- Class XSumDataLoader: Utility for creating train/val/test DataLoaders
- Function load_xsum_dataset: Load XSum dataset with preprocessing
"""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from transformers import AutoTokenizer


class XSumDataset(Dataset):
    """
    XSum dataset wrapper for summarization tasks.
    
    XSum contains news articles with short summaries.
    Each sample has: article (input), summary (target)
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        max_length: Maximum sequence length for tokenization
        tokenizer: HuggingFace tokenizer instance
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 512,
        tokenizer: Optional[Any] = None
    ):
        """Initialize XSum dataset."""
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('xsum')
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset('xsum', split=split)
        
        # Preprocess data
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess dataset for training."""
        # XSum format: 'article' (input) and 'summary' (target)
        self.encoded_data = []
        
        for item in self.dataset:
            article = item['article']
            summary = item['summary']
            
            # Tokenize input (article)
            article_tokens = self.tokenizer.encode(
                article,
                max_length=self.max_length,
                truncation=True
            )
            
            # Tokenize target (summary)
            summary_tokens = self.tokenizer.encode(
                summary,
                max_length=self.max_length,
                truncation=True
            )
            
            self.encoded_data.append({
                'input_ids': article_tokens,
                'attention_mask': [1] * len(article_tokens),
                'labels': summary_tokens
            })
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.encoded_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index."""
        item = self.encoded_data[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }


class XSumDataLoader:
    """
    Utility for creating train/val/test DataLoaders for XSum.
    
    Args:
        dataset_name: Dataset name ('train', 'validation', 'test')
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length
        train_batch_size: Batch size for training
        validation_batch_size: Batch size for validation
        test_batch_size: Batch size for testing
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset_name: str = 'train',
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        train_batch_size: int = 32,
        validation_batch_size: int = 16,
        test_batch_size: int = 16,
        seed: int = 42
    ):
        """Initialize XSumDataLoader."""
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('xsum')
        self.max_length = max_length
        self.seed = seed
        
        # Create datasets
        self.train_dataset = XSumDataset(
            split='train',
            max_length=max_length,
            tokenizer=self.tokenizer
        )
        self.validation_dataset = XSumDataset(
            split='validation',
            max_length=max_length,
            tokenizer=self.tokenizer
        )
        self.test_dataset = XSumDataset(
            split='test',
            max_length=max_length,
            tokenizer=self.tokenizer
        )
    
    def create_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Set seed for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.validation_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_task_info(self) -> Dict[str, Any]:
        """Return task information."""
        return {
            'name': 'XSum',
            'type': 'summarization',
            'num_train_samples': len(self.train_dataset),
            'num_val_samples': len(self.validation_dataset),
            'num_test_samples': len(self.test_dataset),
            'max_length': self.max_length
        }


def load_xsum_dataset(
    split: str = 'train',
    max_length: int = 512,
    tokenizer: Optional[Any] = None
) -> XSumDataset:
    """
    Load XSum dataset with preprocessing.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        max_length: Maximum sequence length
        tokenizer: Optional tokenizer instance
    
    Returns:
        XSumDataset instance
    """
    return XSumDataset(split=split, max_length=max_length, tokenizer=tokenizer)


# Example usage
if __name__ == '__main__':
    # Create dataset
    dataset = load_xsum_dataset(split='train', max_length=512)
    print(f"XSum Train Dataset: {len(dataset)} samples")
    
    # Create dataloader
    loader = XSumDataLoader()
    train_loader, val_loader, test_loader = loader.create_loaders()
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Sample batch
    batch = next(iter(train_loader))
    print(f"Sample batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
