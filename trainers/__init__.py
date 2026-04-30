"""
Trainer package initialization file.

Exports all training pipeline components including:
- WeightLoRATrainer: Main training pipeline with adapter management
- WeightLoRAPlusTrainer: Two-phase training with rank expansion
- LoraBaselineTrainer: Standard LoRA baseline for comparison
- Trainer utilities and helper functions
"""

from .weightlora_trainer import (
    WeightLoRATrainer,
    WeightLoRAPlusTrainer,
    create_weightlora_trainer,
    create_weightlora_plus_trainer,
)

from .lora_baseline_trainer import (
    LoraBaselineTrainer,
    create_lora_baseline_trainer,
)

from .trainer_utils import (
    load_model,
    load_dataset,
    create_dataloaders,
    compute_metrics,
    save_checkpoint,
    load_checkpoint,
    setup_device,
    set_seed,
)

__all__ = [
    # Main trainers
    "WeightLoRATrainer",
    "WeightLoRAPlusTrainer",
    "LoraBaselineTrainer",
    
    # Factory functions
    "create_weightlora_trainer",
    "create_weightlora_plus_trainer",
    "create_lora_baseline_trainer",
    
    # Utilities
    "load_model",
    "load_dataset",
    "create_dataloaders",
    "compute_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "setup_device",
    "set_seed",
]
