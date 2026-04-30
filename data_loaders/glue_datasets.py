"""
GLUE Benchmark Dataset Loading and Preprocessing

This module provides dataset loading utilities for all 8 GLUE benchmark tasks:
- MNLI: Multi-Genre Natural Language Inference
- SST-2: Stanford Sentiment Treebank (Binary Classification)
- CoLA: Corpus of Linguistic Acceptability
- QQP: Quora Question Pairs (Binary Classification)
- QNLI: Question Natural Language Inference
- RTE: Recognizing Textual Entailment
- MRPC: Microsoft Research Paraphrase Corpus
- STS-B: Semantic Textual Similarity Benchmark

All datasets are loaded from HuggingFace Datasets library with standardized preprocessing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


# Task-specific configurations
GLUE_TASKS = {
    'mnli': {
        'name': 'Multi-Genre Natural Language Inference',
        'num_labels': 3,  # entailment, contradiction, neutral
        'label_map': {'contradiction': 0, 'entailment': 1, 'neutral': 2},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation_matched',
        'test_split': 'test_matched',
    },
    'sst2': {
        'name': 'Stanford Sentiment Treebank',
        'num_labels': 2,  # negative, positive
        'label_map': {'negative': 0, 'positive': 1},
        'max_length': 64,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'cola': {
        'name': 'Corpus of Linguistic Acceptability',
        'num_labels': 2,  # unacceptable, acceptable
        'label_map': {'unacceptable': 0, 'acceptable': 1},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'qqp': {
        'name': 'Quora Question Pairs',
        'num_labels': 2,  # not paraphrase, paraphrase
        'label_map': {'not paraphrase': 0, 'paraphrase': 1},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'qnli': {
        'name': 'Question Natural Language Inference',
        'num_labels': 2,  # entailment, not entailment
        'label_map': {'entailment': 0, 'not entailment': 1},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'rte': {
        'name': 'Recognizing Textual Entailment',
        'num_labels': 2,  # not entailment, entailment
        'label_map': {'not entailment': 0, 'entailment': 1},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'mrpc': {
        'name': 'Microsoft Research Paraphrase Corpus',
        'num_labels': 2,  # not paraphrase, paraphrase
        'label_map': {'not paraphrase': 0, 'paraphrase': 1},
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
    'sts-b': {
        'name': 'Semantic Textual Similarity Benchmark',
        'num_labels': 1,  # regression task (0-5 similarity score)
        'label_map': None,  # regression task, no label mapping
        'max_length': 128,
        'train_split': 'train',
        'validation_split': 'validation',
        'test_split': 'test',
    },
}


class GLUEDataset(Dataset):
    """
    Base class for GLUE benchmark datasets with standardized preprocessing.
    
    Provides tokenization, padding, and label formatting for all GLUE tasks.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer: AutoTokenizer,
        split: str = 'train',
        max_length: int = 128,
        label_map: Optional[Dict[str, int]] = None,
        is_regression: bool = False
    ):
        """
        Initialize GLUE dataset.
        
        Args:
            dataset_name: Name of GLUE task (mnli, sst2, cola, etc.)
            tokenizer: HuggingFace tokenizer instance
            split: Dataset split (train, validation, test)
            max_length: Maximum sequence length
            label_map: Mapping from labels to integers (None for regression)
            is_regression: Whether task is regression (STS-B)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.label_map = label_map
        self.is_regression = is_regression
        
        # Load dataset from HuggingFace
        self.raw_data = load_dataset('glue', dataset_name, split=split)
        
        # Tokenize dataset
        self.tokenized_data = self._tokenize()
        
        print(f"Loaded {len(self.tokenized_data)} samples from {dataset_name} ({split})")
    
    def _tokenize(self) -> List[Dict[str, Any]]:
        """Tokenize dataset samples."""
        tokenized = []
        
        for item in self.raw_data:
            # Handle different input formats per task
            if self.dataset_name == 'mnli':
                # MNLI: premise, hypothesis, label
                encoding = self.tokenizer(
                    item['premise'],
                    item['hypothesis'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'sst2':
                # SST-2: sentence, label
                encoding = self.tokenizer(
                    item['sentence'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'cola':
                # CoLA: sentence, label
                encoding = self.tokenizer(
                    item['sentence'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'qqp':
                # QQP: question1, question2, label
                encoding = self.tokenizer(
                    item['question1'],
                    item['question2'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'qnli':
                # QNLI: question, premise, label
                encoding = self.tokenizer(
                    item['premise'],
                    item['question'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'rte':
                # RTE: sentence1, sentence2, label
                encoding = self.tokenizer(
                    item['sentence1'],
                    item['sentence2'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'mrpc':
                # MRPC: sentence1, sentence2, label
                encoding = self.tokenizer(
                    item['sentence1'],
                    item['sentence2'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = self.label_map[item['label']]
            elif self.dataset_name == 'sts-b':
                # STS-B: sentence1, sentence2, label (regression)
                encoding = self.tokenizer(
                    item['sentence1'],
                    item['sentence2'],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                )
                label = item['label']  # Keep as float for regression
            else:
                raise ValueError(f"Unknown GLUE task: {self.dataset_name}")
            
            tokenized.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding.get('token_type_ids', []),
                'labels': label
            })
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized sample as PyTorch tensors."""
        item = self.tokenized_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long) if item['token_type_ids'] else torch.zeros_like(item['input_ids']),
            'labels': torch.tensor(item['labels'], dtype=torch.float32) if self.is_regression else torch.tensor(item['labels'], dtype=torch.long)
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Return dataset configuration."""
        return {
            'name': self.dataset_name,
            'split': self.split,
            'num_labels': GLUE_TASKS[self.dataset_name]['num_labels'],
            'is_regression': self.is_regression,
            'max_length': self.max_length
        }


class GLUEDataLoader:
    """
    Utility class for creating and managing GLUE dataset loaders.
    
    Handles train/val/test splits and creates DataLoaders with appropriate
    batch sizes and shuffle settings.
    """
    
    @staticmethod
    def create_loaders(
        dataset_name: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        train_batch_size: int = 32,
        validation_batch_size: int = 16,
        test_batch_size: int = 16,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders for a GLUE task.
        
        Args:
            dataset_name: Name of GLUE task
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            train_batch_size: Batch size for training
            validation_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_loader, validation_loader, test_loader)
        """
        # Set random seed
        np.random.seed(seed)
        
        # Create datasets
        train_dataset = GLUEDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split='train',
            max_length=max_length,
            label_map=GLUE_TASKS[dataset_name]['label_map'],
            is_regression=dataset_name == 'sts-b'
        )
        
        val_dataset = GLUEDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=GLUE_TASKS[dataset_name]['validation_split'],
            max_length=max_length,
            label_map=GLUE_TASKS[dataset_name]['label_map'],
            is_regression=dataset_name == 'sts-b'
        )
        
        test_dataset = GLUEDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            split=GLUE_TASKS[dataset_name]['test_split'],
            max_length=max_length,
            label_map=GLUE_TASKS[dataset_name]['label_map'],
            is_regression=dataset_name == 'sts-b'
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def get_task_info(dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration information for a GLUE task.
        
        Args:
            dataset_name: Name of GLUE task
        
        Returns:
            Dictionary with task configuration
        """
        if dataset_name not in GLUE_TASKS:
            raise ValueError(f"Unknown GLUE task: {dataset_name}")
        
        return {
            'name': GLUE_TASKS[dataset_name]['name'],
            'num_labels': GLUE_TASKS[dataset_name]['num_labels'],
            'label_map': GLUE_TASKS[dataset_name]['label_map'],
            'max_length': GLUE_TASKS[dataset_name]['max_length'],
            'train_split': GLUE_TASKS[dataset_name]['train_split'],
            'validation_split': GLUE_TASKS[dataset_name]['validation_split'],
            'test_split': GLUE_TASKS[dataset_name]['test_split'],
            'is_regression': dataset_name == 'sts-b'
        }


def load_glue_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    split: str = 'train',
    max_length: int = 128
) -> GLUEDataset:
    """
    Convenience function to load a single GLUE dataset split.
    
    Args:
        dataset_name: Name of GLUE task
        tokenizer: HuggingFace tokenizer
        split: Dataset split (train, validation, test)
        max_length: Maximum sequence length
    
    Returns:
        GLUEDataset instance
    """
    return GLUEDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        label_map=GLUE_TASKS[dataset_name]['label_map'],
        is_regression=dataset_name == 'sts-b'
    )


def get_all_glue_tasks() -> List[str]:
    """
    Return list of all available GLUE tasks.
    
    Returns:
        List of task names
    """
    return list(GLUE_TASKS.keys())


if __name__ == '__main__':
    # Test GLUE dataset loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    # Test loading different tasks
    for task in ['sst2', 'mnli', 'cola']:
        print(f"\nLoading {task}...")
        loaders = GLUEDataLoader.create_loaders(
            dataset_name=task,
            tokenizer=tokenizer,
            max_length=128,
            train_batch_size=8,
            validation_batch_size=8,
            test_batch_size=8
        )
        
        print(f"  Train samples: {len(loaders[0].dataset)}")
        print(f"  Val samples: {len(loaders[1].dataset)}")
        print(f"  Test samples: {len(loaders[2].dataset)}")
        
        # Test data loading
        for batch in loaders[0]:
            print(f"  Batch keys: {batch.keys()}")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            break
