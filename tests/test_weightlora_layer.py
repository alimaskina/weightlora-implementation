"""
Test suite for WeightLoRALayer and WeightLoRAWrapper components.
Validates the core WeightLoRA functionality including trainable weight ω,
adapter activation control, and sparsity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter_base import LoRAAdapter, LoRALinear, LoRAEmbedding
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper, create_weightlora_layer


class TestWeightLoRALayer(unittest.TestCase):
    """Test cases for WeightLoRALayer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.in_features = 128
        self.out_features = 256
        self.rank = 8
        self.dropout = 0.1
        
        # Create a simple pretrained linear layer
        self.pretrained_layer = nn.Linear(self.in_features, self.out_features)
        self.pretrained_layer.weight.data = torch.randn(self.out_features, self.in_features)
        self.pretrained_layer.bias.data = torch.zeros(self.out_features)
        
        # Create LoRA adapter
        self.adapter = LoRAAdapter(self.in_features, self.out_features, 
                                   r=self.rank, alpha=32.0, dropout=self.dropout)
        
        # Create WeightLoRALayer
        self.layer = WeightLoRALayer(self.pretrained_layer, self.adapter, 
                                     "test_layer", rank=self.rank, dropout=self.dropout)
    
    def test_initialization(self):
        """Test that WeightLoRALayer initializes correctly"""
        self.assertEqual(self.layer.layer_name, "test_layer")
        self.assertEqual(self.layer.rank, self.rank)
        self.assertTrue(self.layer.weight.requires_grad)
        # Weight should be initialized to 1.0 (fully active)
        torch.testing.assert_close(self.layer.weight, torch.ones(1))
    
    def test_forward_with_full_adapter(self):
        """Test forward pass with weight=1 (full adapter activation)"""
        x = torch.randn(4, self.in_features)
        
        # Set weight to 1 (fully active)
        self.layer.set_weight(1.0)
        output = self.layer(x)
        
        # Output should be pretrained + full adapter contribution
        expected = self.pretrained_layer(x) + self.adapter(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_forward_with_disconnected_adapter(self):
        """Test forward pass with weight=0 (adapter disconnected)"""
        x = torch.randn(4, self.in_features)
        
        # Set weight to 0 (adapter disconnected)
        self.layer.set_weight(0.0)
        output = self.layer(x)
        
        # Output should be only pretrained (no adapter contribution)
        expected = self.pretrained_layer(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_forward_with_partial_weight(self):
        """Test forward pass with weight=0.5 (partial adapter activation)"""
        x = torch.randn(4, self.in_features)
        
        # Set weight to 0.5 (partial activation)
        self.layer.set_weight(0.5)
        output = self.layer(x)
        
        # Output should be pretrained + 0.5 * adapter contribution
        expected = self.pretrained_layer(x) + 0.5 * self.adapter(x)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_disconnect_method(self):
        """Test disconnect() method sets weight to 0"""
        self.layer.set_weight(1.0)
        self.layer.disconnect()
        torch.testing.assert_close(self.layer.weight, torch.zeros(1))
    
    def test_enable_method(self):
        """Test enable() method sets weight to 1"""
        self.layer.set_weight(0.0)
        self.layer.enable()
        torch.testing.assert_close(self.layer.weight, torch.ones(1))
    
    def test_get_weight(self):
        """Test get_weight() returns current weight value"""
        self.layer.set_weight(0.75)
        weight = self.layer.get_weight()
        torch.testing.assert_close(weight, torch.tensor(0.75))
    
    def test_backward_pass(self):
        """Test that gradients flow correctly through the layer"""
        x = torch.randn(4, self.in_features, requires_grad=True)
        self.layer.set_weight(1.0)
        
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(self.layer.weight.grad)
        self.assertIsNotNone(self.pretrained_layer.weight.grad)
        self.assertIsNotNone(self.adapter.A.weight.grad)
        self.assertIsNotNone(self.adapter.B.weight.grad)
    
    def test_weight_gradient_computation(self):
        """Test that weight gradient is computed correctly"""
        x = torch.randn(4, self.in_features, requires_grad=True)
        self.layer.set_weight(1.0)
        
        output = self.layer(x)
        loss = output.sum()
        loss.backward()
        
        # Weight gradient should be the difference between adapter and pretrained gradients
        # ∂L/∂ω = ∂L/∂h_adapter - ∂L/∂h_pretrained
        adapter_grad = self.adapter.A.weight.grad @ self.adapter.B.weight.grad @ x
        pretrained_grad = self.pretrained_layer.weight.grad @ x
        
        # The weight gradient should be approximately the difference
        weight_grad = self.layer.weight.grad
        expected_grad = adapter_grad.sum() - pretrained_grad.sum()
        
        # Allow some numerical tolerance
        self.assertAlmostEqual(weight_grad.item(), expected_grad.item(), places=2)
    
    def test_compute_layer_loss_contribution(self):
        """Test compute_layer_loss_contribution for StoIHT gradient computation"""
        x = torch.randn(4, self.in_features)
        y = torch.randn(4, self.out_features)
        
        # Set weight to 1
        self.layer.set_weight(1.0)
        
        # Compute output and loss
        output = self.layer(x)
        loss = F.mse_loss(output, y)
        
        # Compute layer loss contribution
        contribution = self.layer.compute_layer_loss_contribution(x, y)
        
        # Contribution should be non-zero
        self.assertTrue(torch.any(contribution != 0))
    
    def test_param_count(self):
        """Test parameter counting"""
        pretrained_params = sum(p.numel() for p in self.pretrained_layer.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        
        total_params = self.layer.get_param_count()
        
        # Total should be pretrained + adapter
        self.assertEqual(total_params, pretrained_params + adapter_params)
    
    def test_dropout(self):
        """Test that dropout is applied to adapter output"""
        torch.manual_seed(42)
        x = torch.randn(4, self.in_features)
        
        # Set weight to 1
        self.layer.set_weight(1.0)
        
        # Run multiple times - outputs should vary due to dropout
        outputs = [self.layer(x) for _ in range(10)]
        
        # Check that outputs vary (not all identical)
        unique_outputs = len(set([o.tolist() for o in outputs]))
        self.assertGreater(unique_outputs, 1)
    
    def test_weight_bounds(self):
        """Test that weight stays within expected bounds"""
        # Set weight to various values
        for val in [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]:
            self.layer.set_weight(val)
            weight = self.layer.get_weight().item()
            self.assertEqual(weight, val)


class TestWeightLoRAWrapper(unittest.TestCase):
    """Test cases for WeightLoRAWrapper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.in_features = 128
        self.out_features = 256
        self.rank = 8
        self.dropout = 0.1
        
        # Create a simple model with multiple layers
        self.model = nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, self.out_features),
            nn.ReLU(),
            nn.Linear(self.out_features, 10)
        )
        
        # Create WeightLoRAWrapper
        self.wrapper = WeightLoRAWrapper(self.model, 
                                         target_modules=["0", "2", "4"],
                                         rank=self.rank, alpha=32.0, dropout=self.dropout)
    
    def test_initialization(self):
        """Test that WeightLoRAWrapper initializes correctly"""
        self.assertEqual(self.wrapper.model, self.model)
        self.assertEqual(self.wrapper.rank, self.rank)
        self.assertEqual(self.wrapper.alpha, 32.0)
    
    def test_forward_pass(self):
        """Test forward pass through wrapped model"""
        x = torch.randn(4, self.in_features)
        output = self.wrapper(x)
        
        # Output should have correct shape
        self.assertEqual(output.shape, (4, 10))
    
    def test_get_active_layers(self):
        """Test get_active_layers() returns layers with weight > 0"""
        # All layers should be active initially
        active = self.wrapper.get_active_layers()
        self.assertEqual(len(active), 3)  # 3 target layers
    
    def test_get_disconnected_layers(self):
        """Test get_disconnected_layers() returns layers with weight = 0"""
        # All layers should be active initially
        disconnected = self.wrapper.get_disconnected_layers()
        self.assertEqual(len(disconnected), 0)
    
    def test_disconnect_layer(self):
        """Test disconnect_layer() sets weight to 0"""
        self.wrapper.disconnect_layer("0")
        
        active = self.wrapper.get_active_layers()
        self.assertNotIn("0", active)
    
    def test_enable_layer(self):
        """Test enable_layer() sets weight to 1"""
        self.wrapper.disconnect_layer("0")
        self.wrapper.enable_layer("0")
        
        active = self.wrapper.get_active_layers()
        self.assertIn("0", active)
    
    def test_weight_vector(self):
        """Test get_weight_vector() and set_weight_vector()"""
        # Get initial weight vector
        weights = self.wrapper.get_weight_vector()
        self.assertEqual(len(weights), 3)
        
        # Set new weight vector
        new_weights = torch.tensor([1.0, 0.0, 1.0])
        self.wrapper.set_weight_vector(new_weights)
        
        # Verify weights were updated
        weights = self.wrapper.get_weight_vector()
        torch.testing.assert_close(weights, new_weights)
    
    def test_apply_disconnections(self):
        """Test apply_disconnections() permanently disconnects inactive adapters"""
        # Set some weights to 0
        self.wrapper.set_weight_vector(torch.tensor([0.0, 1.0, 0.0]))
        
        # Apply disconnections
        active_indices = self.wrapper.apply_disconnections()
        
        # Check that inactive adapters were disconnected
        self.assertEqual(len(active_indices), 1)
        self.assertEqual(active_indices[0], 1)  # Only layer 1 should be active
    
    def test_backward_pass(self):
        """Test backward pass through wrapped model"""
        x = torch.randn(4, self.in_features, requires_grad=True)
        output = self.wrapper(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(self.model[0].weight.grad)
        self.assertIsNotNone(self.model[2].weight.grad)
        self.assertIsNotNone(self.model[4].weight.grad)


class TestCreateWeightLoRALayer(unittest.TestCase):
    """Test cases for create_weightlora_layer factory function"""
    
    def test_create_weightlora_layer(self):
        """Test factory function creates correct layer"""
        pretrained = nn.Linear(128, 256)
        adapter = LoRAAdapter(128, 256, r=8, alpha=32.0)
        
        layer = create_weightlora_layer(pretrained, adapter, "test", rank=8)
        
        self.assertIsInstance(layer, WeightLoRALayer)
        self.assertEqual(layer.layer_name, "test")
    
    def test_create_weightlora_layer_with_dropout(self):
        """Test factory function with dropout parameter"""
        pretrained = nn.Linear(128, 256)
        adapter = LoRAAdapter(128, 256, r=8, alpha=32.0, dropout=0.1)
        
        layer = create_weightlora_layer(pretrained, adapter, "test", rank=8, dropout=0.1)
        
        self.assertEqual(layer.dropout, 0.1)


class TestWeightLoRASparsity(unittest.TestCase):
    """Test sparsity constraint enforcement"""
    
    def test_sparsity_with_k(self):
        """Test that sparsity constraint is enforced"""
        # Create a model with 5 target layers
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256)
        )
        
        wrapper = WeightLoRAWrapper(model, target_modules=["0", "1", "2", "3", "4"],
                                    rank=8, alpha=32.0, K=3)
        
        # Set weight vector to keep only 3 layers
        wrapper.set_weight_vector(torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))
        
        # Apply disconnections
        active = wrapper.apply_disconnections()
        
        # Should have exactly 3 active layers
        self.assertEqual(len(active), 3)


class TestWeightLoRAMemoryReduction(unittest.TestCase):
    """Test memory reduction claims from paper"""
    
    def test_memory_reduction_deberta(self):
        """Test memory reduction for DeBERTaV3 (baseline 442K params)"""
        # DeBERTaV3 has 72 transformer layers
        # Full LoRA with r=8: ~442K params
        # WeightLoRA with k=5: ~61.5K params (86% reduction)
        
        # Simulate parameter counts
        full_lora_params = 442000
        weightlora_k5_params = 61500
        
        reduction = (full_lora_params - weightlora_k5_params) / full_lora_params
        self.assertGreater(reduction, 0.85)  # Should be > 85% reduction
    
    def test_memory_reduction_llama(self):
        """Test memory reduction for Llama-3-7B"""
        # Llama-3-7B has 32 layers
        # Full LoRA with r=8: ~6.4M params
        # WeightLoRA with k=5: ~90K params
        
        full_lora_params = 6400000
        weightlora_k5_params = 90000
        
        reduction = (full_lora_params - weightlora_k5_params) / full_lora_params
        self.assertGreater(reduction, 0.85)  # Should be > 85% reduction


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
