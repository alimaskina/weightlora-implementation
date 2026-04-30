"""
Test suite for StoIHT (Stochastic Iterative Hard Thresholding) optimizer.
Validates ℓ0-norm sparsity constraint enforcement and weight vector optimization.
"""

import torch
import torch.nn as nn
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.sto_iht import StoIHTOptimizer, StoIHTTrainer, compute_layer_contribution, validate_sparsity_constraint
from algorithms.weight_optimizer import WeightOptimizer, WeightOptimizerConfig
from models.adapter_base import LoRAAdapter
from models.weightlora_layer import WeightLoRALayer
from utils.evaluation import compute_sparsity_metrics


class TestStoIHTOptimizer(unittest.TestCase):
    """Test StoIHTOptimizer class for ℓ0-norm sparsity enforcement."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.K = 5  # Keep top 5 weights
        self.initial_lr = 0.1
        self.n_layers = 10
        
        # Create weight vector
        self.weights = torch.ones(self.n_layers)
        
        # Create optimizer
        self.stoiht = StoIHTOptimizer(K=self.K, initial_learning_rate=self.initial_lr)
        
    def test_initialization(self):
        """Test StoIHTOptimizer initialization."""
        optimizer = StoIHTOptimizer(K=10, initial_learning_rate=0.01)
        self.assertEqual(optimizer.K, 10)
        self.assertEqual(optimizer.learning_rate, 0.01)
        
    def test_hard_thresholding(self):
        """Test hard thresholding keeps exactly K non-zero weights."""
        weights = torch.ones(10)
        weights[0] = 0.1
        weights[1] = 0.2
        weights[2] = 0.3
        weights[3] = 0.4
        weights[4] = 0.5
        weights[5] = 0.6
        weights[6] = 0.7
        weights[7] = 0.8
        weights[8] = 0.9
        weights[9] = 1.0
        
        # K=5 should keep indices 5,6,7,8,9 (largest values)
        result = self.stoiht.update(weights, torch.zeros(10))
        
        # Check that exactly K weights are non-zero
        non_zero_count = (result.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
        
        # Check that top K values are preserved
        expected_top_k = torch.topk(weights.abs(), self.K)[0]
        self.assertTrue(torch.allclose(result[result.abs() > 1e-6], expected_top_k, atol=1e-5))
        
    def test_gradient_update(self):
        """Test gradient descent step before thresholding."""
        weights = torch.ones(10) * 0.5
        gradients = torch.ones(10) * 0.1
        
        # After gradient update: weights = weights + lr * gradients
        # Then hard thresholding keeps top K
        result = self.stoiht.update(weights, gradients)
        
        # All weights should be non-zero after update (all equal)
        # But hard thresholding should keep exactly K
        non_zero_count = (result.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
        
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate scaling."""
        weights = torch.ones(10)
        gradients = torch.ones(10) * 10.0  # Large gradient norm
        
        # With large gradient norm, learning rate should be scaled down
        result = self.stoiht.update(weights, gradients)
        
        # Should still have K non-zero weights
        non_zero_count = (result.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
        
    def test_edge_case_all_zero(self):
        """Test behavior when all weights are zero."""
        weights = torch.zeros(10)
        gradients = torch.ones(10)
        
        result = self.stoiht.update(weights, gradients)
        
        # Should have K non-zero weights after update
        non_zero_count = (result.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
        
    def test_get_active_indices(self):
        """Test get_active_indices returns correct indices."""
        weights = torch.arange(10).float()
        gradients = torch.zeros(10)
        
        result = self.stoiht.update(weights, gradients)
        active_indices = self.stoiht.get_active_indices()
        
        # Should return indices of top K weights
        self.assertEqual(len(active_indices), self.K)
        self.assertIn(9, active_indices)  # Largest value
        self.assertIn(8, active_indices)  # Second largest
        
    def test_get_inactive_indices(self):
        """Test get_inactive_indices returns correct indices."""
        weights = torch.arange(10).float()
        gradients = torch.zeros(10)
        
        result = self.stoiht.update(weights, gradients)
        inactive_indices = self.stoiht.get_inactive_indices()
        
        # Should return indices of bottom (n-K) weights
        self.assertEqual(len(inactive_indices), self.n_layers - self.K)
        self.assertIn(0, inactive_indices)  # Smallest value
        self.assertIn(1, inactive_indices)  # Second smallest


class TestStoIHTTrainer(unittest.TestCase):
    """Test StoIHTTrainer class for weight vector optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_layers = 10
        self.K = 5
        self.initial_lr = 0.1
        self.l2_lambda = 0.01
        
        self.trainer = StoIHTTrainer(
            n_layers=self.n_layers,
            K=self.K,
            initial_lr=self.initial_lr,
            l2_lambda=self.l2_lambda
        )
        
    def test_initialization(self):
        """Test StoIHTTrainer initialization."""
        trainer = StoIHTTrainer(n_layers=10, K=5, initial_lr=0.1, l2_lambda=0.01)
        self.assertEqual(trainer.K, 5)
        self.assertEqual(trainer.initial_lr, 0.1)
        self.assertEqual(trainer.l2_lambda, 0.01)
        
    def test_step(self):
        """Test single training step."""
        weights = torch.ones(self.n_layers)
        gradients = torch.ones(self.n_layers) * 0.1
        
        # Create a simple model for loss computation
        model = nn.Linear(10, 10)
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        # Perform training step
        new_weights = self.trainer.step(weights, gradients, model, input_batch, target_batch)
        
        # Should have K non-zero weights
        non_zero_count = (new_weights.abs() > 1e-6).sum().item()
        self.assertEqual(non_zero_count, self.K)
        
    def test_get_weight_vector(self):
        """Test weight vector retrieval."""
        weights = torch.arange(10).float()
        gradients = torch.zeros(10)
        
        result = self.trainer.step(weights, gradients, None, None, None)
        retrieved = self.trainer.get_weight_vector()
        
        self.assertTrue(torch.allclose(retrieved, result, atol=1e-5))
        
    def test_set_weight_vector(self):
        """Test weight vector setting."""
        new_weights = torch.arange(10).float()
        
        self.trainer.set_weight_vector(new_weights)
        retrieved = self.trainer.get_weight_vector()
        
        self.assertTrue(torch.allclose(retrieved, new_weights, atol=1e-5))


class TestComputeLayerContribution(unittest.TestCase):
    """Test compute_layer_contribution function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_layers = 5
        self.K = 3
        
        # Create a simple model with adapters
        self.model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        # Create input and target batches
        self.input_batch = torch.randn(4, 10)
        self.target_batch = torch.randn(4, 10)
        
    def test_compute_layer_contribution(self):
        """Test loss contribution computation for specific layers."""
        layer_indices = [0, 2]  # Compute contribution from layers 0 and 2
        
        contributions = compute_layer_contribution(
            self.model,
            layer_indices,
            self.input_batch,
            self.target_batch
        )
        
        # Should return tensor with correct shape
        self.assertEqual(contributions.shape, (len(layer_indices),))
        self.assertEqual(len(layer_indices), 2)
        
    def test_single_layer_contribution(self):
        """Test loss contribution for single layer."""
        layer_indices = [2]
        
        contributions = compute_layer_contribution(
            self.model,
            layer_indices,
            self.input_batch,
            self.target_batch
        )
        
        self.assertEqual(contributions.shape, (1,))
        self.assertTrue(torch.isfinite(contributions[0]))


class TestValidateSparsityConstraint(unittest.TestCase):
    """Test validate_sparsity_constraint function."""
    
    def test_valid_sparsity(self):
        """Test validation when sparsity constraint is satisfied."""
        weights = torch.arange(10).float()
        K = 5
        
        is_valid, actual_sparsity = validate_sparsity_constraint(weights, K, tolerance=1e-6)
        
        self.assertTrue(is_valid)
        self.assertEqual(actual_sparsity, K)
        
    def test_invalid_sparsity(self):
        """Test validation when sparsity constraint is violated."""
        weights = torch.ones(10)  # All weights non-zero
        K = 5
        
        is_valid, actual_sparsity = validate_sparsity_constraint(weights, K, tolerance=1e-6)
        
        self.assertFalse(is_valid)
        self.assertEqual(actual_sparsity, 10)
        
    def test_tolerance_handling(self):
        """Test tolerance parameter for floating point comparison."""
        weights = torch.arange(10).float()
        K = 5
        
        # With tight tolerance
        is_valid, _ = validate_sparsity_constraint(weights, K, tolerance=1e-10)
        self.assertTrue(is_valid)
        
        # With loose tolerance
        is_valid, _ = validate_sparsity_constraint(weights, K, tolerance=1e-3)
        self.assertTrue(is_valid)


class TestWeightOptimizerIntegration(unittest.TestCase):
    """Test WeightOptimizer integration with StoIHT."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_layers = 10
        self.K = 5
        self.alpha = 32.0
        self.lr = 3e-4
        
        self.optimizer = WeightOptimizer(
            n_layers=self.n_layers,
            K=self.K,
            alpha=self.alpha,
            lr=self.lr
        )
        
    def test_initialization(self):
        """Test WeightOptimizer initialization."""
        opt = WeightOptimizer(n_layers=10, K=5, alpha=32.0, lr=3e-4)
        self.assertEqual(opt.K, 5)
        self.assertEqual(opt.n_layers, 10)
        self.assertTrue(isinstance(opt.weight, nn.Parameter))
        
    def test_step_with_constraint(self):
        """Test step with sparsity constraint."""
        # Create a simple model
        model = nn.Linear(10, 10)
        
        # Create dummy data
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        # Create loss
        output = model(input_batch)
        loss = nn.functional.mse_loss(output, target_batch)
        
        # Perform step
        new_loss = self.optimizer.step_with_constraint(loss, model, (input_batch, target_batch))
        
        # Loss should be finite
        self.assertTrue(torch.isfinite(new_loss))
        
    def test_apply_disconnections(self):
        """Test adapter disconnection after training."""
        # Create a simple model with adapters
        model = nn.Linear(10, 10)
        adapter = LoRAAdapter(10, 10, r=8, alpha=32.0)
        layer = WeightLoRALayer(model, adapter, "test_layer", rank=8)
        
        # Create optimizer
        opt = WeightOptimizer(n_layers=1, K=1, alpha=32.0, lr=3e-4)
        
        # Simulate training
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        output = layer(input_batch)
        loss = nn.functional.mse_loss(output, target_batch)
        loss.backward()
        
        # Apply disconnections
        active_indices = opt.apply_disconnections()
        
        # Should have exactly K active adapters
        self.assertEqual(len(active_indices), self.K)
        
    def test_compute_gradients(self):
        """Test gradient computation for StoIHT update."""
        model = nn.Linear(10, 10)
        
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        output = model(input_batch)
        loss = nn.functional.mse_loss(output, target_batch)
        
        gradients = self.optimizer.compute_gradients()
        
        # Should return tensor with correct shape
        self.assertEqual(gradients.shape, (self.n_layers,))
        self.assertTrue(torch.isfinite(gradients).all())


class TestSparsityMetrics(unittest.TestCase):
    """Test sparsity metrics computation."""
    
    def test_compute_sparsity_metrics(self):
        """Test sparsity metrics computation."""
        weights = torch.arange(10).float()
        K = 5
        total_adapters = 10
        
        metrics = compute_sparsity_metrics(weights, K, total_adapters)
        
        # Should have expected keys
        self.assertIn('sparsity_ratio', metrics)
        self.assertIn('active_count', metrics)
        self.assertIn('inactive_count', metrics)
        
        # Check values
        self.assertEqual(metrics['active_count'], K)
        self.assertEqual(metrics['inactive_count'], total_adapters - K)
        self.assertEqual(metrics['sparsity_ratio'], K / total_adapters)
        
    def test_compute_memory_reduction(self):
        """Test memory reduction computation."""
        baseline_params = 442000  # Full LoRA params for DeBERTa
        active_params = 61500  # WeightLoRA k=5 params
        
        reduction = compute_memory_reduction(baseline_params, active_params)
        
        # Should have expected keys
        self.assertIn('reduction_percentage', reduction)
        self.assertIn('active_params', reduction)
        self.assertIn('baseline_params', reduction)
        
        # Check reduction percentage
        expected_reduction = (baseline_params - active_params) / baseline_params
        self.assertAlmostEqual(reduction['reduction_percentage'], expected_reduction, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete StoIHT workflow."""
    
    def test_complete_training_loop(self):
        """Test complete training loop with StoIHT."""
        # Create model
        model = nn.Linear(10, 10)
        
        # Create optimizer
        optimizer = WeightOptimizer(n_layers=1, K=1, alpha=32.0, lr=3e-4)
        
        # Create data
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        # Training loop
        for step in range(10):
            output = model(input_batch)
            loss = nn.functional.mse_loss(output, target_batch)
            loss.backward()
            
            # Update weights with constraint
            optimizer.step_with_constraint(loss, model, (input_batch, target_batch))
        
        # Verify sparsity constraint
        is_valid, actual_sparsity = validate_sparsity_constraint(
            optimizer.weight, 1, tolerance=1e-6
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(actual_sparsity, 1)
        
    def test_weight_vector_convergence(self):
        """Test that weight vector converges to binary values."""
        n_layers = 10
        K = 5
        
        # Create optimizer
        optimizer = WeightOptimizer(n_layers=n_layers, K=K, alpha=32.0, lr=3e-4)
        
        # Create model and data
        model = nn.Linear(10, 10)
        input_batch = torch.randn(4, 10)
        target_batch = torch.randn(4, 10)
        
        # Multiple training steps
        for step in range(50):
            output = model(input_batch)
            loss = nn.functional.mse_loss(output, target_batch)
            loss.backward()
            
            optimizer.step_with_constraint(loss, model, (input_batch, target_batch))
        
        # Check that weights are close to 0 or 1 (binary)
        binary_ratio = (
            (optimizer.weight.abs() > 0.9) | (optimizer.weight.abs() < 0.1)
        ).float().mean().item()
        
        # At least 80% of weights should be close to 0 or 1
        self.assertGreater(binary_ratio, 0.8)


if __name__ == '__main__':
    unittest.main()
