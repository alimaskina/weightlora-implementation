"""
Test suite for WeightLoRA training loop components.
Validates the complete training pipeline including WeightLoRATrainer, WeightLoRAPlusTrainer,
and integration with StoIHT optimizer, adapter management, and model wrappers.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter_base import LoRAAdapter
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper
from models.adapter_manager import AdapterManager
from algorithms.sto_iht import StoIHTOptimizer, StoIHTTrainer
from algorithms.weight_optimizer import WeightOptimizer, WeightOptimizerConfig
from algorithms.rank_expander import RankExpander
from algorithms.sparsity_constraint import SparsityConstraint, hard_thresholding
from trainers.weightlora_trainer import WeightLoRATrainer, create_weightlora_trainer
from trainers.weightlora_plus_trainer import WeightLoRAPlusTrainer, create_weightlora_plus_trainer
from trainers.trainer_utils import set_seed, setup_device, load_config
from utils.evaluation import compute_sparsity_metrics, validate_sparsity_constraint, compute_memory_reduction


class TestWeightLoRATrainer(unittest.TestCase):
    """Test suite for WeightLoRATrainer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.device = setup_device()
        self.K = 5
        self.T = 10
        self.rank = 4
        self.alpha = 32.0
        self.dropout = 0.1
        self.lr = 3e-4
        
    def test_trainer_initialization(self):
        """Test trainer initialization with valid parameters"""
        # Create a simple linear model for testing
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        # Should not raise exception
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.K, self.K)
        self.assertEqual(trainer.T, self.T)
        self.assertEqual(trainer.rank, self.rank)
    
    def test_trainer_training_step(self):
        """Test single training step execution"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run one training step
        loss = trainer._train_epoch()
        
        # Loss should be finite
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss, 0)
    
    def test_trainer_validation(self):
        """Test validation step"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run validation
        val_score = trainer._validate()
        
        # Score should be finite
        self.assertTrue(torch.isfinite(val_score))
    
    def test_trainer_apply_disconnections(self):
        """Test adapter disconnection after T steps"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Before disconnection, all weights should be active
        weights_before = trainer.get_weight_vector()
        self.assertEqual(weights_before.sum().item(), float(self.K))
        
        # Apply disconnections
        trainer.apply_disconnections()
        
        # After disconnection, weights should be binary (0 or 1)
        weights_after = trainer.get_weight_vector()
        active_count = (weights_after > 1e-6).sum().item()
        
        # Should have exactly K active adapters
        self.assertEqual(active_count, self.K)
    
    def test_trainer_weight_vector_sparsity(self):
        """Test that weight vector satisfies sparsity constraint"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Validate sparsity constraint
        is_valid, actual_sparsity = validate_sparsity_constraint(
            trainer.get_weight_vector(), self.K, tolerance=1e-6
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(actual_sparsity, self.K)


class TestWeightLoRAPlusTrainer(unittest.TestCase):
    """Test suite for WeightLoRAPlusTrainer class (two-phase training)"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.device = setup_device()
        self.K = 5
        self.phase_1_steps = 10
        self.rank_expansion = 8
        self.rank = 4
        self.alpha = 32.0
        self.dropout = 0.1
        self.lr = 3e-4
        
    def test_plus_trainer_initialization(self):
        """Test WeightLoRA+ trainer initialization"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRAPlusTrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            phase_1_steps=self.phase_1_steps,
            rank_expansion=self.rank_expansion,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.K, self.K)
        self.assertEqual(trainer.phase_1_steps, self.phase_1_steps)
        self.assertEqual(trainer.rank_expansion, self.rank_expansion)
    
    def test_plus_trainer_phase_1(self):
        """Test Phase 1: adapter selection"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRAPlusTrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            phase_1_steps=self.phase_1_steps,
            rank_expansion=self.rank_expansion,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run Phase 1
        selected_adapters = trainer._train_phase_1(self.phase_1_steps)
        
        # Should select exactly K adapters
        self.assertEqual(len(selected_adapters), self.K)
    
    def test_plus_trainer_phase_2(self):
        """Test Phase 2: rank expansion"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRAPlusTrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            phase_1_steps=self.phase_1_steps,
            rank_expansion=self.rank_expansion,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run Phase 1 to select adapters
        trainer._train_phase_1(self.phase_1_steps)
        
        # Expand ranks
        trainer._expand_selected_ranks(trainer.active_indices)
        
        # Verify rank expansion
        for idx in trainer.active_indices:
            adapter = trainer.adapter_manager.manager.get(f"layer_{idx}")
            if adapter is not None:
                self.assertEqual(adapter.A.shape[1], self.rank_expansion)
    
    def test_plus_trainer_full_training(self):
        """Test complete two-phase training"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRAPlusTrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            phase_1_steps=self.phase_1_steps,
            rank_expansion=self.rank_expansion,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run full training
        trainer.train(num_epochs=2, validation_freq=1)
        
        # Should complete without errors
        self.assertTrue(True)


class TestStoIHTOptimizer(unittest.TestCase):
    """Test suite for StoIHTOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.K = 5
        self.initial_lr = 0.1
        
    def test_hard_thresholding(self):
        """Test hard thresholding keeps exactly K non-zero weights"""
        weights = torch.randn(10)
        gradients = torch.randn(10)
        
        optimizer = StoIHTOptimizer(K=self.K, initial_learning_rate=self.initial_lr)
        
        # Update weights
        updated_weights = optimizer.update(weights, gradients, lambda x: x)
        
        # Should have exactly K non-zero weights
        non_zero_count = (updated_weights.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
    
    def test_gradient_update(self):
        """Test gradient descent step before thresholding"""
        weights = torch.ones(10) * 0.5
        gradients = torch.ones(10) * 0.1
        
        optimizer = StoIHTOptimizer(K=self.K, initial_learning_rate=self.initial_lr)
        
        # Update should apply gradient descent
        updated_weights = optimizer.update(weights, gradients, lambda x: x)
        
        # Weights should have changed due to gradient
        self.assertFalse(torch.allclose(weights, updated_weights))
    
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate scaling"""
        weights = torch.ones(10)
        gradients = torch.ones(10) * 100  # Large gradient
        
        optimizer = StoIHTOptimizer(K=self.K, initial_learning_rate=self.initial_lr)
        
        # Adaptive LR should scale down for large gradients
        updated_weights = optimizer.update(weights, gradients, lambda x: x)
        
        # Should not explode
        self.assertTrue(torch.isfinite(updated_weights).all())
    
    def test_get_active_indices(self):
        """Test active/inactive index retrieval"""
        weights = torch.randn(10)
        gradients = torch.randn(10)
        
        optimizer = StoIHTOptimizer(K=self.K, initial_learning_rate=self.initial_lr)
        updated_weights = optimizer.update(weights, gradients, lambda x: x)
        
        active_indices = optimizer.get_active_indices()
        inactive_indices = optimizer.get_inactive_indices()
        
        # Active + inactive should equal total
        self.assertEqual(len(active_indices) + len(inactive_indices), 10)
        # Active should equal K
        self.assertEqual(len(active_indices), self.K)


class TestWeightOptimizer(unittest.TestCase):
    """Test suite for WeightOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.n_layers = 10
        self.K = 5
        self.alpha = 32.0
        self.lr = 3e-4
        
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = WeightOptimizer(
            n_layers=self.n_layers,
            K=self.K,
            alpha=self.alpha,
            lr=self.lr
        )
        
        self.assertIsNotNone(optimizer)
        self.assertEqual(optimizer.K, self.K)
        self.assertEqual(optimizer.n_layers, self.n_layers)
    
    def test_step_with_constraint(self):
        """Test step with sparsity constraint"""
        optimizer = WeightOptimizer(
            n_layers=self.n_layers,
            K=self.K,
            alpha=self.alpha,
            lr=self.lr
        )
        
        # Create dummy model and loss
        model = nn.Linear(10, 5)
        loss = torch.tensor(1.0)
        
        # Should not raise exception
        optimizer.step_with_constraint(loss, model, None)
        
        # Weight vector should satisfy sparsity constraint
        weights = optimizer.get_weight_vector()
        is_valid, actual_sparsity = validate_sparsity_constraint(
            weights, self.K, tolerance=1e-6
        )
        self.assertTrue(is_valid)
    
    def test_apply_disconnections(self):
        """Test adapter disconnection"""
        optimizer = WeightOptimizer(
            n_layers=self.n_layers,
            K=self.K,
            alpha=self.alpha,
            lr=self.lr
        )
        
        # Apply disconnections
        active_indices = optimizer.apply_disconnections()
        
        # Should return exactly K active indices
        self.assertEqual(len(active_indices), self.K)
    
    def test_compute_gradients(self):
        """Test gradient computation"""
        optimizer = WeightOptimizer(
            n_layers=self.n_layers,
            K=self.K,
            alpha=self.alpha,
            lr=self.lr
        )
        
        # Compute gradients
        gradients = optimizer.compute_gradients()
        
        # Should return tensor of correct shape
        self.assertEqual(gradients.shape, (self.n_layers,))


class TestRankExpander(unittest.TestCase):
    """Test suite for RankExpander"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.initial_rank = 4
        self.target_rank = 8
        
    def test_rank_expansion_random(self):
        """Test random rank expansion strategy"""
        adapter = LoRAAdapter(in_features=10, out_features=5, r=self.initial_rank)
        
        expander = RankExpander(
            selected_adapters=[],
            initial_rank=self.initial_rank,
            target_rank=self.target_rank,
            strategy='random'
        )
        
        # Expand rank
        expander.expand_rank(adapter)
        
        # Rank should be expanded
        self.assertEqual(adapter.A.shape[1], self.target_rank)
    
    def test_rank_expansion_qr(self):
        """Test QR factorization rank expansion strategy"""
        adapter = LoRAAdapter(in_features=10, out_features=5, r=self.initial_rank)
        
        expander = RankExpander(
            selected_adapters=[],
            initial_rank=self.initial_rank,
            target_rank=self.target_rank,
            strategy='qr_factorization'
        )
        
        # Expand rank
        expander.expand_rq_factorization(adapter)
        
        # Rank should be expanded
        self.assertEqual(adapter.A.shape[1], self.target_rank)


class TestSparsityConstraint(unittest.TestCase):
    """Test suite for SparsityConstraint"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.K = 5
        self.tolerance = 1e-6
        
    def test_hard_thresholding(self):
        """Test hard thresholding function"""
        weights = torch.randn(10)
        
        result = hard_thresholding(weights, self.K)
        
        # Should have exactly K non-zero weights
        non_zero_count = (result.abs() > self.tolerance).sum().item()
        self.assertEqual(non_zero_count, self.K)
    
    def test_sparsity_constraint_validation(self):
        """Test sparsity constraint validation"""
        weights = torch.ones(10)
        weights[:self.K] = 1.0  # Set K weights to 1
        
        is_valid, actual_sparsity = validate_sparsity_constraint(
            weights, self.K, tolerance=self.tolerance
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(actual_sparsity, self.K)
    
    def test_sparsity_constraint_violation(self):
        """Test sparsity constraint violation detection"""
        weights = torch.ones(10)  # All weights non-zero
        
        is_valid, actual_sparsity = validate_sparsity_constraint(
            weights, self.K, tolerance=self.tolerance
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(actual_sparsity, 10)
    
    def test_compute_sparsity_metrics(self):
        """Test sparsity metrics computation"""
        weights = torch.ones(10)
        weights[:self.K] = 1.0
        
        metrics = compute_sparsity_metrics(weights, self.K, 10)
        
        # Should have sparsity_ratio close to 0.5 (K/10)
        self.assertAlmostEqual(metrics['sparsity_ratio'], 0.5, places=1)
        self.assertEqual(metrics['active_count'], self.K)
        self.assertEqual(metrics['inactive_count'], 10 - self.K)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete training pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        set_seed(42)
        self.device = setup_device()
        self.K = 5
        self.T = 10
        self.rank = 4
        self.alpha = 32.0
        self.dropout = 0.1
        self.lr = 3e-4
        
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from initialization to validation"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        # Create trainer
        trainer = WeightLoRATrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            T=self.T,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run training
        trainer.train(num_epochs=2, validation_freq=1)
        
        # Verify trainer state
        self.assertIsNotNone(trainer.get_weight_vector())
        self.assertIsNotNone(trainer.get_active_adapters())
    
    def test_weightlora_plus_complete_pipeline(self):
        """Test complete WeightLoRA+ pipeline with rank expansion"""
        model = nn.Linear(10, 5)
        dataset = SimpleDataset()
        
        trainer = WeightLoRAPlusTrainer(
            model_name="test",
            dataset_name="test",
            K=self.K,
            phase_1_steps=self.T,
            rank_expansion=self.rank,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            lr=self.lr,
            device=self.device
        )
        
        # Run training
        trainer.train(num_epochs=2, validation_freq=1)
        
        # Verify rank expansion occurred
        self.assertIsNotNone(trainer.active_indices)
        self.assertEqual(len(trainer.active_indices), self.K)


class SimpleDataset(Dataset):
    """Simple dataset for testing"""
    
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 5, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == '__main__':
    unittest.main()
