# WeightLoRA: Keep Only Necessary Adapters

## Overview

WeightLoRA is a novel approach to Low-Rank Adaptation (LoRA) that achieves **86% reduction in trainable parameters** while maintaining or improving performance on NLP benchmarks. The core innovation is using trainable importance weights ω with ℓ0-norm optimization to selectively activate only the most important adapters.

**Key Contributions:**
- **Adaptive Adapter Selection**: Trainable weight vector ω controls which adapters are active
- **StoIHT Optimizer**: Stochastic Iterative Hard Thresholding for ℓ0-norm sparsity
- **WeightLoRA+**: Two-phase training with rank expansion for further improvements
- **86% Parameter Reduction**: From 442K to 61.5K trainable parameters (k=5)

**Paper**: [WeightLoRA: Keep Only Necessary Adapters](https://github.com/brain-mmo-lab/WLoRA)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/WeightLoRA-Implementation.git
cd WeightLoRA-Implementation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from trainers import WeightLoRATrainer

# Initialize trainer
trainer = WeightLoRATrainer(
    model_name="deberta-v3-base",
    dataset_name="glue_mnli",
    K=10,           # Keep top-10 adapters
    T=200,          # Training steps before disconnection
    rank=8,         # LoRA rank
    lr=3e-4,        # Learning rate
    alpha=32.0      # LoRA scaling factor
)

# Train
trainer.train(epochs=10, validation_freq=50)
```

## Architecture

### Core Components

```
WeightLoRA-Implementation/
├── models/                    # Model architectures
│   ├── adapter_base.py       # Standard LoRA implementation
│   ├── weightlora_layer.py   # WeightLoRALayer with ω
│   ├── adapter_manager.py    # Adapter lifecycle management
│   ├── deberta_wrapper.py    # DeBERTaV3 integration
│   ├── bart_wrapper.py       # BART integration
│   └── llama_wrapper.py      # Llama-3-7B integration
│
├── algorithms/                # Core algorithms
│   ├── sto_iht.py            # Stochastic IHT optimizer
│   ├── weight_optimizer.py   # Weight vector optimization
│   ├── rank_expander.py      # Rank expansion (WeightLoRA+)
│   └── sparsity_constraint.py # ℓ0-norm implementation
│
├── trainers/                  # Training pipelines
│   ├── weightlora_trainer.py # Main training loop
│   ├── weightlora_plus_trainer.py # Two-phase training
│   └── lora_baseline_trainer.py # Standard LoRA baseline
│
├── experiments/               # Experiment scripts
│   ├── glue_benchmark.py     # GLUE tasks evaluation
│   ├── squad_evaluation.py   # SQuAD v1.1/v2.0
│   ├── nl_g_evaluation.py    # XSum and CNN/DailyMail
│   └── ablation_study.py     # RLoRA and controls
│
├── datasets/                  # Data loading
│   ├── glue_datasets.py      # All 8 GLUE tasks
│   ├── squad_datasets.py     # Question answering
│   ├── xsum_dataset.py       # Summarization
│   └── cnn_dailymail.py      # Summarization
│
└── utils/                     # Utilities
    ├── evaluation.py         # Metric computation
    ├── checkpointing.py      # Save/load models
    ├── visualization.py      # Plotting functions
    └── metrics.py            # Standard metrics
```

## Supported Models & Benchmarks

### Models
- **DeBERTaV3-base** (184M params) - GLUE classification tasks
- **BART-large** (406M params) - Summarization (XSum, CNN/DailyMail)
- **Llama-3-7B** (8B params) - Question answering (SQuAD)

### Benchmarks
- **GLUE**: MNLI, SST-2, CoLA, QQP, QNLI, RTE, MRPC, STS-B
- **SQuAD**: v1.1 and v2.0 (question answering)
- **XSum**: News summarization
- **CNN/DailyMail**: Article summarization

## Key Features

### 1. WeightLoRA Core (Algorithm 1)

```
h_i = W_i × x + ω_i × A_i × B_i × x
```

- **ω_i**: Trainable weight controlling adapter contribution
- **Sparsity**: ||ω||_0 ≤ K (keep only top-K adapters)
- **Optimization**: StoIHT with hard thresholding

### 2. WeightLoRA+ (Algorithm 2)

Two-phase training:
1. **Phase 1**: Adapter selection with small rank
2. **Phase 2**: Rank expansion for selected adapters

### 3. StoIHT Optimizer

Stochastic Iterative Hard Thresholding:
1. Gradient descent step
2. Hard thresholding: Keep top-K coordinates
3. Zero out remaining weights

## Performance Results

| Method | DeBERTa GLUE | SQuAD v2.0 F1 | XSum ROUGE-1 | Trainable Params |
|--------|-------------|---------------|--------------|------------------|
| LoRA | 0.8562 | 0.8234 | 0.3674 | 442K |
| WeightLoRA (k=100) | 0.8562 | 0.8234 | 0.3674 | 123K |
| WeightLoRA (k=10) | 0.8512 | 0.8189 | 0.3598 | 12.3K |
| WeightLoRA (k=5) | 0.8479 | 0.8156 | 0.3521 | 6.15K |
| WeightLoRA+ (k=10) | **0.8612** | **0.8313** | **0.3712** | 12.3K |

*Results from paper Table 1, Table 2, Table 3*

## Configuration

### Hyperparameters (configs/hyperparameters.yaml)

```yaml
training:
  learning_rate: 3e-4
  weight_decay: 0.01
  warmup_steps: 100
  max_steps: 10000
  batch_size: 32
  num_epochs: 10

weightlora:
  K: 10              # Number of active adapters
  T: 200             # Steps before disconnection
  rank: 8            # LoRA rank
  alpha: 32.0        # Scaling factor
  dropout: 0.05
```

## Running Experiments

### GLUE Benchmark

```bash
# Train on MNLI
python experiments/glue_benchmark.py --task mnli --model deberta-v3-base --k 10

# Train on all GLUE tasks
python experiments/glue_benchmark.py --task all --model deberta-v3-base --k 10
```

### SQuAD Evaluation

```bash
# SQuAD v2.0
python experiments/squad_evaluation.py --task squad_v2 --model llama-7b --k 10
```

### NLG Summarization

```bash
# XSum
python experiments/nl_g_evaluation.py --task xsum --model bart-large --k 10

# CNN/DailyMail
python experiments/nl_g_evaluation.py --task cnn_dailymail --model bart-large --k 10
```

### Ablation Study

```bash
# RLoRA (Random Layer Selection)
python experiments/ablation_study.py --method rlo --k 5

# Compare with baseline
python experiments/baseline_comparison.py --methods lora weightlora weightlora_plus
```

## Reproducibility

### Setting Random Seeds

```python
from trainers import set_seed

set_seed(42)  # Set seed for reproducibility
```

### Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint("checkpoints/weightlora_k10_epoch5")

# Load checkpoint
model, optimizer, epoch = trainer.load_checkpoint("checkpoints/weightlora_k10_epoch5")
```

## Hardware Requirements

- **GPU**: NVIDIA V100 16GB minimum (RTX 3090/4090 recommended)
- **RAM**: 32GB minimum (64GB+ recommended for Llama-3-7B)
- **CUDA**: 11.8+ (PyTorch 2.1+)

## Citation

```bibtex
@article{weightlora2024,
  title={WeightLoRA: Keep Only Necessary Adapters},
  author={Author, et al.},
  journal={Journal Name},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Original LoRA paper: Hu et al. (2021)
- StoIHT algorithm: Nguyen et al. (2014)
- HuggingFace Transformers library

## Contact

For questions or issues, please open an issue on GitHub.
