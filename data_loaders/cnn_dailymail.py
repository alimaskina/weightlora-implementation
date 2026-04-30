"""
CNN/DailyMail Dataset Loading Utilities for WeightLoRA

This module provides dataset loading utilities for the CNN/DailyMail news summarization benchmark,
including dataset wrapper, data loader, and preprocessing functions for training and evaluation.

The CNN/DailyMail dataset is a large-scale corpus of news articles with summaries, used for
sequence-to-sequence summarization tasks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


class CNNDailyMailDataset(Dataset):
    """
    CNN/DailyMail dataset wrapper for summarization tasks.
    
    This class wraps the CNN/DailyMail dataset from HuggingFace, providing
    standardized preprocessing and tokenization for WeightLoRA training.
    
    Attributes:
        split: Dataset split ('train', 'validation', 'test')
        max_length: Maximum sequence length for tokenization
        tokenizer: Tokenizer instance for preprocessing
        data: Raw dataset from HuggingFace
    """
    
    def __init__(
        self,
        split: str = 'train',
        max_length: int = 512,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        """
        Initialize CNN/DailyMail dataset.
        
        Args:
            split: Dataset split to load ('train', 'validation', 'test')
            max_length: Maximum sequence length for tokenization
            tokenizer: Optional tokenizer instance (uses default if None)
        """
        self.split = split
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('facebook/bart-large')
        
        # Load dataset from HuggingFace
        self.data = load_dataset('cnn_dailymail', '3.0.0', split=split)
        
        # Preprocess data
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Preprocess dataset by separating articles and summaries."""
        self.articles = []
        self.summaries = []
        
        for item in self.data:
            article = item['article']
            high_level_summary = item['highlights']
            
            # Join highlights into single summary
            summary = ' '.join(high_level_summary)
            
            self.articles.append(article)
            self.summaries.append(summary)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.articles)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index with preprocessing.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'input_ids', 'attention_mask', 'labels' tensors
        """
        article = self.articles[idx]
        summary = self.summaries[idx]
        
        # Tokenize article (input)
        article_tokens = self.tokenizer(
            article,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize summary (target)
        summary_tokens = self.tokenizer(
            summary,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (shifted right for teacher forcing)
        input_ids = article_tokens['input_ids'].squeeze(0)
        attention_mask = article_tokens['attention_mask'].squeeze(0)
        
        # Prepare labels (shift right by 1, pad with -100)
        labels = summary_tokens['input_ids'].squeeze(0)
        labels = torch.cat([torch.tensor([-100] * len(labels)), labels[:-1]], dim=0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CNNDailyMailDataLoader:
    """
    Utility class for creating train/val/test DataLoaders for CNN/DailyMail.
    
    This class provides centralized DataLoader creation with consistent
    configuration across experiments.
    """
    
    def __init__(
        self,
        dataset_name: str = 'cnn_dailymail',
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        train_batch_size: int = 16,
        validation_batch_size: int = 16,
        test_batch_size: int = 16,
        seed: int = 42
    ):
        """
        Initialize CNN/DailyMail DataLoader.
        
        Args:
            dataset_name: Dataset identifier
            tokenizer: Optional tokenizer instance
            max_length: Maximum sequence length
            train_batch_size: Training batch size
            validation_batch_size: Validation batch size
            test_batch_size: Test batch size
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('facebook/bart-large')
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.seed = seed
        
        # Create datasets
        self.train_dataset = CNNDailyMailDataset(
            split='train3',
            max_length=max_length,
            tokenizer=tokenizer
        )
        self.val_dataset = CNNDailyMailDataset(
            split='validation',
            max_length=max_length,
            tokenizer=tokenizer
        )
        self.test_dataset = CNNDailyMailDataset(
            split='test',
            max_length=max_length,
            tokenizer=tokenizer
        )
    
    def create_loaders(
        self,
        train: bool = True,
        validation: bool = True,
        test: bool = False
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        Create DataLoaders for specified splits.
        
        Args:
            train: Include training DataLoader
            validation: Include validation DataLoader
            test: Include test DataLoader
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = None
        val_loader = None
        test_loader = None
        
        if train:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
        
        if validation:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.validation_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        if test:
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        return train_loader, val_loader, test_loader
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-specific information.
        
        Returns:
            Dictionary with task metadata
        """
        return {
            'name': 'CNN/DailyMail',
            'type': 'summarization',
            'num_train_samples': len(self.train_dataset),
            'num_val_samples': len(self.val_dataset),
            'num_test_samples': len(self.test_dataset),
            'max_length': self.max_length,
            'batch_sizes': {
                'train': self.train_batch_size,
                'validation': self.validation_batch_size,
                'test': self.test_batch_size
            }
        }


def load_cnn_dailymail_dataset(
    split: str = 'train3',
    max_length: int = 512,
    tokenizer: Optional[AutoTokenizer] = None
) -> CNNDailyMailDataset:
    """
    Load CNN/DailyMail dataset with preprocessing.
    
    Args:
        split: Dataset split ('train3', 'validation', 'test')
        max_length: Maximum sequence length
        tokenizer: Optional tokenizer instance
        
    Returns:
        CNNDailyMailDataset instance
    """
    return CNNDailyMailDataset(
        split=split,
        max_length=max_length,
        tokenizer=tokenizer
    )


def get_cnn_dailymail_splits() -> List[str]:
    """
    Get available CNN/DailyMail splits.
    
    Returns:
        List of available split names
    """
    return ['train3', 'validation', 'test']


if __name__ == '__main__':
    # Test CNN/DailyMail dataset loading
    print("Testing CNN/DailyMail dataset loading...")
    
    # Create dataset
    dataset = CNNDailyMailDataset(split='train3', max_length=256)
    print(f"Loaded {len(dataset)} training samples")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Test iteration
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}")
        print(f"Batch {i}: attention_mask shape = {batch['attention_mask'].shape}")
        print(f"Batch {i}: labels shape = {batch['labels'].shape}")
    
    print("CNN/DailyMail dataset test completed successfully!")
