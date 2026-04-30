"""
Datasets module initialization for WeightLoRA implementation.

This module provides dataset loading utilities for all benchmark tasks:
- GLUE: MNLI, SST-2, CoLA, QQP, QNLI, RTE, MRPC, STS-B
- SQuAD: v1.1 and v2.0 question answering
- NLG: XSum and CNN/DailyMail summarization

All datasets are loaded via HuggingFace Datasets library with standardized
preprocessing and PyTorch DataLoader integration.
"""

from .glue_datasets import (
    GLUEDataset,
    GLUEDataLoader,
    load_glue_dataset,
    get_all_glue_tasks,
    GLUE_TASKS
)

from .squad_datasets import (
    SQuADDataset,
    SQuADDataLoader,
    load_squad_dataset,
    get_squad_versions
)

from .xsum_dataset import (
    XSumDataset,
    XSumDataLoader,
    load_xsum_dataset
)

from .cnn_dailymail import (
    CNNDailyMailDataset,
    CNNDailyMailDataLoader,
    load_cnn_dailymail_dataset
)

__all__ = [
    # GLUE datasets
    'GLUEDataset',
    'GLUEDataLoader',
    'load_glue_dataset',
    'get_all_glue_tasks',
    'GLUE_TASKS',
    
    # SQuAD datasets
    'SQuADDataset',
    'SQuADDataLoader',
    'load_squad_dataset',
    'get_squad_versions',
    
    # XSum dataset
    'XSumDataset',
    'XSumDataLoader',
    'load_xsum_dataset',
    
    # CNN/DailyMail dataset
    'CNNDailyMailDataset',
    'CNNDailyMailDataLoader',
    'load_cnn_dailymail_dataset'
]
