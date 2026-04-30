"""
SQuAD Evaluation Script for WeightLoRA

This script evaluates WeightLoRA models on SQuAD v1.1 and v2.0 datasets.
Implements Exact Match (EM) and F1 score computation as per standard SQuAD metrics.

Usage:
    python experiments/squad_evaluation.py --model deberta-v3-base --task squad_v2 --k 10
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from models.deberta_wrapper import create_deberta_weightlora
from models.bart_wrapper import create_bart_weightlora
from models.llama_wrapper import create_llama_weightlora
from trainers.trainer_utils import set_seed, load_config
from utils.evaluation import compute_squad_metrics


def load_squad_dataset(
    version: str = "v2.0",
    split: str = "validation",
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 384
) -> Tuple[Dict, Dict]:
    """
    Load SQuAD dataset from HuggingFace.
    
    Args:
        version: "v1.1" or "v2.0"
        split: "train", "validation", or "test"
        tokenizer: Optional tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Tuple of (train_data, validation_data) dictionaries
    """
    dataset_name = "squad_v2" if version == "v2.0" else "squad_v1"
    
    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_name, split=split)
    
    # Convert to dictionary format
    data = {
        "question": [],
        "context": [],
        "answers": []
    }
    
    for item in dataset:
        data["question"].append(item["question"])
        data["context"].append(item["context"])
        # Handle v2.0 is_impossible flag
        if version == "v2.0" and item.get("is_impossible", False):
            data["answers"].append({"text": [], "answer_start": []})
        else:
            data["answers"].append({
                "text": item["answers"]["text"],
                "answer_start": item["answers"]["answer_start"]
            })
    
    return data


def preprocess_squad_batch(
    batch: Dict,
    tokenizer: AutoTokenizer,
    max_length: int = 384
) -> Dict:
    """
    Preprocess SQuAD batch for model input.
    
    Args:
        batch: Dictionary with question, context, answers
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Preprocessed batch with tokenized inputs
    """
    # Tokenize question and context
    tokenized = tokenizer(
        batch["question"],
        batch["context"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Prepare answers for evaluation
    answers = []
    for q, ctx, ans in zip(batch["question"], batch["context"], batch["answers"]):
        # Handle impossible questions
        if len(ans["text"]) == 0:
            answers.append({"text": [], "answer_start": []})
        else:
            # Convert answer_start to token indices
            answer_starts = []
            for start in ans["answer_start"]:
                # Approximate character to token index
                token_start = ctx.find(ctx[start:start+10]) // 10
                answer_starts.append(max(0, min(len(tokenized["input_ids"][0]) - 1, token_start)))
            answers.append({
                "text": ans["text"],
                "answer_start": answer_starts
            })
    
    return {
        **tokenized,
        "answers": answers
    }


def evaluate_squad(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset_path: str = "validation",
    version: str = "v2.0",
    batch_size: int = 16,
    max_length: int = 384,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Evaluate model on SQuAD dataset.
    
    Args:
        model: Trained model instance
        tokenizer: Tokenizer instance
        dataset_path: Path to dataset file or "validation"/"test"
        version: "v1.1" or "v2.0"
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        device: Device to use for computation
    
    Returns:
        Dictionary with EM and F1 scores
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Load dataset
    print(f"Loading {version} dataset from {dataset_path}...")
    data = load_squad_dataset(version, dataset_path, tokenizer, max_length)
    
    # Prepare dataloader
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Evaluation metrics
    all_predictions = []
    all_answers = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Preprocess batch
            batch = preprocess_squad_batch(batch, tokenizer, max_length)
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract predictions
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Get predicted start and end positions
            start_positions = start_logits.argmax(dim=-1)
            end_positions = end_logits.argmax(dim=-1)
            
            # Decode predictions
            for i in range(len(input_ids)):
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()
                
                # Decode answer
                answer_tokens = tokenizer.decode(
                    input_ids[i][start_idx:end_idx + 1],
                    skip_special_tokens=True
                )
                
                all_predictions.append(answer_tokens)
                all_answers.append(batch["answers"][i])
    
    # Compute metrics
    metrics = compute_squad_metrics(all_predictions, all_answers)
    
    print(f"\nSQuAD {version} Evaluation Results:")
    print(f"  Exact Match (EM): {metrics['em']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    
    return metrics


def run_squad_experiment(
    model_name: str,
    task: str,
    k: int = 10,
    rank: int = 8,
    seed: int = 42
) -> Dict:
    """
    Run complete SQuAD experiment with WeightLoRA.
    
    Args:
        model_name: Model name (deberta-v3-base, bart-large, llama-3-7b)
        task: "squad_v1" or "squad_v2"
        k: Number of active adapters
        rank: LoRA rank
        seed: Random seed
    
    Returns:
        Dictionary with evaluation metrics
    """
    set_seed(seed)
    
    # Load config
    config = load_config(f"configs/{model_name.replace('-', '_')}_config.yaml")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create model with WeightLoRA
    if "deberta" in model_name.lower():
        model, _ = create_deberta_weightlora(
            model_name,
            rank=rank,
            alpha=config.get("weightlora.alpha", 32.0),
            dropout=0.05,
            K=k,
            target_modules=config.get("target_modules", [])
        )
    elif "bart" in model_name.lower():
        model, _ = create_bart_weightlora(
            model_name,
            rank=rank,
            alpha=config.get("weightlora.alpha", 32.0),
            dropout=0.05,
            K=k,
            target_modules=config.get("target_modules", [])
        )
    elif "llama" in model_name.lower():
        model, _ = create_llama_weightlora(
            model_name,
            rank=rank,
            alpha=config.get("weightlora.alpha", 32.0),
            dropout=0.05,
            K=k,
            target_modules=config.get("target_modules", [])
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Evaluate on validation set
    version = task.replace("_v", "_")
    metrics = evaluate_squad(
        model=model,
        tokenizer=tokenizer,
        dataset_path="validation",
        version=version,
        batch_size=config.get("training.batch_size", 16),
        max_length=config.get("max_length", 384),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Save results
    results_dir = "experiments/results"
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = os.path.join(
        results_dir,
        f"squad_{model_name.replace('-', '_')}_{task}_k{k}.json"
    )
    
    with open(result_file, "w") as f:
        json.dump({
            "model": model_name,
            "task": task,
            "k": k,
            "rank": rank,
            "metrics": metrics,
            "config": config
        }, f, indent=2)
    
    print(f"\nResults saved to {result_file}")
    
    return metrics


def main():
    """Main entry point for SQuAD evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate WeightLoRA on SQuAD datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deberta-v3-base",
        help="Model name (deberta-v3-base, bart-large, llama-3-7b)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="squad_v2",
        choices=["squad_v1", "squad_v2"],
        help="SQuAD version"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of active adapters"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    print(f"Running SQuAD evaluation with:")
    print(f"  Model: {args.model}")
    print(f"  Task: {args.task}")
    print(f"  K: {args.k}")
    print(f"  Rank: {args.rank}")
    print(f"  Seed: {args.seed}")
    print()
    
    metrics = run_squad_experiment(
        model_name=args.model,
        task=args.task,
        k=args.k,
        rank=args.rank,
        seed=args.seed
    )
    
    return metrics


if __name__ == "__main__":
    main()
