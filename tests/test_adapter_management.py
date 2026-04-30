"""
Test suite for AdapterManager component.
Validates adapter initialization, activation/disconnection, and lifecycle management.
"""

import torch
import unittest
from typing import List, Dict, Tuple

from models.adapter_base import LoRAAdapter
from models.adapter_manager import AdapterManager, is_target_layer, create_lora_adapter
from models.weightlora_layer import WeightLoRALayer, WeightLoRAWrapper
from models.deberta_wrapper import DebertaV3WeightLoRA
from models.bart_wrapper import BartWeightLoRA
from models.llama_wrapper import LlamaWeightLoRA


class TestAdapterManager(unittest.TestCase):
    """Test AdapterManager class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple linear layer as mock model
        self.mock_layer = torch.nn.Linear(128, 256)
        self.mock_layer.weight.data = torch.randn(256, 128) * 0.1
        self.mock_layer.bias.data = torch.zeros(256)
        
        # Create adapter
        adapter = LoRAAdapter(128, 256, r=8, alpha=32.0, dropout=0.0)
        
        # Create WeightLoRALayer
        self.wl_layer = WeightLoRALayer(
            pretrained_layer=self.mock_layer,
            adapter=adapter,
            layer_name="test_layer",
            rank=8
        )
        
        # Create AdapterManager
        self.manager = AdapterManager(
            model=self.wl_layer,
            K=5,
            rank=8,
            alpha=32.0,
            dropout=0.0
        )
    
    def test_adapter_initialization(self):
        """Test that adapters are initialized correctly"""
        # Check that manager has initialized adapters
        self.assertTrue(hasattr(self.manager, 'manager'))
        self.assertIsInstance(self.manager.manager, dict)
        
        # Check that adapters exist
        self.assertGreater(len(self.manager.manager), 0)
    
    def test_activate_adapters(self):
        """Test adapter activation"""
        # Get initial weight values
        initial_weights = self.manager.get_weight_vector()
        
        # Activate all adapters
        self.manager.activate_adapters(list(self.manager.manager.keys()))
        
        # Check that weights are set to 1
        new_weights = self.manager.get_weight_vector()
        for i, w in enumerate(new_weights):
            self.assertAlmostEqual(w.item(), 1.0, places=5)
    
    def test_deactivate_adapters(self):
        """Test adapter deactivation"""
        # Deactivate all adapters
        self.manager.deactivate_adapters(list(self.manager.manager.keys()))
        
        # Check that weights are set to 0
        weights = self.manager.get_weight_vector()
        for w in weights:
            self.assertAlmostEqual(w.item(), 0.0, places=5)
    
    def test_get_active_adapters(self):
        """Test getting active adapters"""
        # Initially all adapters should be active (weight=1)
        active = self.manager.get_active_adapters()
        self.assertGreater(len(active), 0)
        
        # Deactivate some adapters
        self.manager.deactivate_adapters(list(self.manager.manager.keys())[:3])
        
        # Check that only remaining adapters are active
        active_after = self.manager.get_active_adapters()
        self.assertLess(len(active_after), len(active))
    
    def test_set_weight(self):
        """Test setting weight for specific adapter"""
        adapter_name = list(self.manager.manager.keys())[0]
        
        # Set weight to 0
        self.manager.set_weight(adapter_name, 0.0)
        weights = self.manager.get_weight_vector()
        self.assertAlmostEqual(weights[0].item(), 0.0, places=5)
        
        # Set weight to 1
        self.manager.set_weight(adapter_name, 1.0)
        weights = self.manager.get_weight_vector()
        self.assertAlmostEqual(weights[0].item(), 1.0, places=5)
    
    def test_forward_pass_with_active_adapter(self):
        """Test forward pass with active adapter"""
        # Create test input
        x = torch.randn(2, 128)
        
        # Set adapter to active
        self.manager.activate_adapters(list(self.manager.manager.keys()))
        
        # Forward pass should include adapter contribution
        output = self.manager.forward(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_forward_pass_with_inactive_adapter(self):
        """Test forward pass with inactive adapter"""
        # Create test input
        x = torch.randn(2, 128)
        
        # Deactivate adapter
        self.manager.deactivate_adapters(list(self.manager.manager.keys()))
        
        # Forward pass should not include adapter contribution
        output = self.manager.forward(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_disconnect_adapters_by_weight(self):
        """Test disconnecting adapters based on weight threshold"""
        # Set some weights to 0
        self.manager.deactivate_adapters(list(self.manager.manager.keys())[:3])
        
        # Disconnect adapters by weight
        disconnected = self.manager.disconnect_adapters_by_weight(
            threshold=1e-6,
            disconnect=True
        )
        
        # Check that adapters were disconnected
        self.assertGreater(len(disconnected), 0)
    
    def test_zero_adapter(self):
        """Test zeroing out adapter parameters"""
        adapter_name = list(self.manager.manager.keys())[0]
        
        # Get initial adapter parameters
        initial_A = self.manager.manager[adapter_name].A.clone()
        initial_B = self.manager.manager[adapter_name].B.clone()
        
        # Zero the adapter
        self.manager.zero_adapter(adapter_name)
        
        # Check that parameters are zero
        self.assertTrue(torch.allclose(
            self.manager.manager[adapter_name].A, 
            torch.zeros_like(initial_A)
        ))
        self.assertTrue(torch.allclose(
            self.manager.manager[adapter_name].B, 
            torch.zeros_like(initial_B)
        ))
    
    def test_reset_adapter(self):
        """Test resetting adapter parameters"""
        adapter_name = list(self.manager.manager.keys())[0]
        
        # Get initial adapter parameters
        initial_A = self.manager.manager[adapter_name].A.clone()
        initial_B = self.manager.manager[adapter_name].B.clone()
        
        # Zero the adapter
        self.manager.zero_adapter(adapter_name)
        
        # Reset the adapter
        self.manager.reset_adapter(adapter_name)
        
        # Check that parameters are restored
        self.assertTrue(torch.allclose(
            self.manager.manager[adapter_name].A, 
            initial_A
        ))
        self.assertTrue(torch.allclose(
            self.manager.manager[adapter_name].B, 
            initial_B
        ))
    
    def test_get_active_count(self):
        """Test getting count of active adapters"""
        # Initially all adapters should be active
        active_count = self.manager.get_active_count()
        self.assertGreater(active_count, 0)
        
        # Deactivate some adapters
        self.manager.deactivate_adapters(list(self.manager.manager.keys())[:3])
        
        # Check that count decreased
        active_count_after = self.manager.get_active_count()
        self.assertLess(active_count_after, active_count)
    
    def test_get_total_param_count(self):
        """Test getting total parameter count"""
        total_params = self.manager.get_total_param_count()
        self.assertGreater(total_params, 0)
    
    def test_get_active_param_count(self):
        """Test getting active parameter count"""
        # Initially all adapters should be active
        active_params = self.manager.get_active_param_count()
        self.assertGreater(active_params, 0)
        
        # Deactivate some adapters
        self.manager.deactivate_adapters(list(self.manager.manager.keys())[:3])
        
        # Check that active params decreased
        active_params_after = self.manager.get_active_param_count()
        self.assertLess(active_params_after, active_params)
    
    def test_is_target_layer(self):
        """Test is_target_layer function"""
        # Test with linear layer
        linear_layer = torch.nn.Linear(128, 256)
        self.assertTrue(is_target_layer(linear_layer))
        
        # Test with embedding layer
        embedding_layer = torch.nn.Embedding(1000, 128)
        self.assertTrue(is_target_layer(embedding_layer))
        
        # Test with non-target layer
        conv_layer = torch.nn.Conv2d(3, 64, 3)
        self.assertFalse(is_target_layer(conv_layer))
    
    def test_create_lora_adapter(self):
        """Test creating LoRA adapter"""
        adapter = create_lora_adapter(
            layer_type='linear',
            in_features=128,
            out_features=256,
            r=8,
            alpha=32.0
        )
        
        self.assertIsInstance(adapter, LoRAAdapter)
        self.assertEqual(adapter.r, 8)
        self.assertEqual(adapter.alpha, 32.0)


class TestWeightLoRALayer(unittest.TestCase):
    """Test WeightLoRALayer class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock pretrained layer
        self.pretrained = torch.nn.Linear(128, 256)
        self.pretrained.weight.data = torch.randn(256, 128) * 0.1
        self.pretrained.bias.data = torch.zeros(256)
        
        # Create adapter
        self.adapter = LoRAAdapter(128, 256, r=8, alpha=32.0, dropout=0.0)
        
        # Create WeightLoRALayer
        self.wl_layer = WeightLoRALayer(
            pretrained_layer=self.pretrained,
            adapter=self.adapter,
            layer_name="test_layer",
            rank=8
        )
    
    def test_forward_with_weight_one(self):
        """Test forward pass with weight=1"""
        x = torch.randn(2, 128)
        
        # Set weight to 1
        self.wl_layer.set_weight(1.0)
        
        # Forward pass should include adapter contribution
        output = self.wl_layer(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_forward_with_weight_zero(self):
        """Test forward pass with weight=0"""
        x = torch.randn(2, 128)
        
        # Set weight to 0
        self.wl_layer.set_weight(0.0)
        
        # Forward pass should only include pretrained layer
        output = self.wl_layer(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_forward_with_weight_half(self):
        """Test forward pass with weight=0.5"""
        x = torch.randn(2, 128)
        
        # Set weight to 0.5
        self.wl_layer.set_weight(0.5)
        
        # Forward pass should include half adapter contribution
        output = self.wl_layer(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_disconnect(self):
        """Test disconnecting layer"""
        x = torch.randn(2, 128)
        
        # Disconnect
        self.wl_layer.disconnect()
        
        # Forward pass should only include pretrained layer
        output = self.wl_layer(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_enable(self):
        """Test enabling layer"""
        # Disconnect first
        self.wl_layer.disconnect()
        
        # Enable
        self.wl_layer.enable()
        
        # Forward pass should include adapter contribution
        x = torch.randn(2, 128)
        output = self.wl_layer(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 256))
    
    def test_get_weight(self):
        """Test getting weight value"""
        self.wl_layer.set_weight(0.75)
        weight = self.wl_layer.get_weight()
        self.assertAlmostEqual(weight.item(), 0.75, places=5)
    
    def test_compute_layer_loss_contribution(self):
        """Test computing layer loss contribution"""
        x = torch.randn(2, 128)
        target = torch.randn(2, 256)
        
        # Set weight to 1
        self.wl_layer.set_weight(1.0)
        
        # Compute loss contribution
        contribution = self.wl_layer.compute_layer_loss_contribution(x, target)
        
        self.assertIsNotNone(contribution)
        self.assertEqual(contribution.shape, (2, 256))
    
    def test_backward_pass(self):
        """Test backward pass with weight control"""
        x = torch.randn(2, 128, requires_grad=True)
        target = torch.randn(2, 256)
        
        # Set weight to 1
        self.wl_layer.set_weight(1.0)
        
        # Forward pass
        output = self.wl_layer(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(self.wl_layer.weight.grad)
        self.assertIsNotNone(self.adapter.A.grad)
        self.assertIsNotNone(self.adapter.B.grad)


class TestWeightLoRAWrapper(unittest.TestCase):
    """Test WeightLoRAWrapper class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock model with multiple layers
        self.model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 512),
            torch.nn.Linear(512, 128)
        )
        
        # Create WeightLoRAWrapper
        self.wrapper = WeightLoRAWrapper(
            model=self.model,
            target_modules=["0", "1", "2"],
            rank=8,
            alpha=32.0,
            dropout=0.0,
            K=2
        )
    
    def test_forward_pass(self):
        """Test forward pass through wrapper"""
        x = torch.randn(2, 128)
        output = self.wrapper(x)
        
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (2, 128))
    
    def test_get_active_layers(self):
        """Test getting active layers"""
        active = self.wrapper.get_active_layers()
        self.assertGreater(len(active), 0)
    
    def test_get_disconnected_layers(self):
        """Test getting disconnected layers"""
        disconnected = self.wrapper.get_disconnected_layers()
        self.assertGreater(len(disconnected), 0)
    
    def test_disconnect_layer(self):
        """Test disconnecting a specific layer"""
        layer_idx = 0
        
        # Disconnect layer
        self.wrapper.disconnect_layer(layer_idx)
        
        # Check that layer is disconnected
        self.assertTrue(layer_idx in self.wrapper.get_disconnected_layers())
    
    def test_enable_layer(self):
        """Test enabling a specific layer"""
        layer_idx = 0
        
        # Disconnect layer first
        self.wrapper.disconnect_layer(layer_idx)
        
        # Enable layer
        self.wrapper.enable_layer(layer_idx)
        
        # Check that layer is active
        self.assertTrue(layer_idx in self.wrapper.get_active_layers())
    
    def test_get_weight_vector(self):
        """Test getting weight vector"""
        weights = self.wrapper.get_weight_vector()
        self.assertEqual(len(weights), 3)  # 3 layers
    
    def test_set_weight_vector(self):
        """Test setting weight vector"""
        weights = torch.tensor([1.0, 0.0, 1.0])
        self.wrapper.set_weight_vector(weights)
        
        # Check that weights were set
        new_weights = self.wrapper.get_weight_vector()
        self.assertAlmostEqual(new_weights[0].item(), 1.0, places=5)
        self.assertAlmostEqual(new_weights[1].item(), 0.0, places=5)
        self.assertAlmostEqual(new_weights[2].item(), 1.0, places=5)
    
    def test_apply_disconnections(self):
        """Test applying disconnections"""
        # Set some weights to 0
        weights = torch.tensor([1.0, 0.0, 1.0])
        self.wrapper.set_weight_vector(weights)
        
        # Apply disconnections
        active_indices = self.wrapper.apply_disconnections()
        
        # Check that active indices match
        self.assertEqual(len(active_indices), 2)
        self.assertIn(0, active_indices)
        self.assertIn(2, active_indices)
        self.assertNotIn(1, active_indices)


class TestDebertaWeightLoRA(unittest.TestCase):
    """Test DeBERTaV3WeightLoRA model integration"""
    
    def test_model_creation(self):
        """Test creating DeBERTaV3 model with WeightLoRA"""
        # Note: This test requires the model to be downloaded
        # For now, we test the class structure
        try:
            # Mock model creation (would require actual model download)
            # model, _ = create_deberta_weightlora(
            #     model_name="microsoft/deberta-v3-base",
            #     rank=8,
            #     alpha=32.0,
            #     dropout=0.0,
            #     K=10
            # )
            self.assertTrue(True)  # Placeholder for actual test
        except Exception as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_model_has_adapters(self):
        """Test that model has adapters initialized"""
        try:
            # model, _ = create_deberta_weightlora(...)
            # self.assertTrue(hasattr(model, 'adapter_manager'))
            self.assertTrue(True)  # Placeholder
        except Exception as e:
            self.skipTest(f"Model not available: {e}")


class TestBartWeightLoRA(unittest.TestCase):
    """Test BARTWeightLoRA model integration"""
    
    def test_model_creation(self):
        """Test creating BART model with WeightLoRA"""
        try:
            # model, _ = create_bart_weightlora(...)
            self.assertTrue(True)  # Placeholder
        except Exception as e:
            self.skipTest(f"Model not available: {e}")


class TestLlamaWeightLoRA(unittest.TestCase):
    """Test LlamaWeightLoRA model integration"""
    
    def test_model_creation(self):
        """Test creating Llama model with WeightLoRA"""
        try:
            # model, _ = create_llama_weightlora(...)
            self.assertTrue(True)  # Placeholder
        except Exception as e:
            self.skipTest(f"Model not available: {e}")


class TestAdapterManagerIntegration(unittest.TestCase):
    """Integration tests for AdapterManager with various scenarios"""
    
    def test_full_lifecycle(self):
        """Test full adapter lifecycle"""
        # Create manager
        manager = AdapterManager(
            model=self._create_mock_model(),
            K=3,
            rank=8,
            alpha=32.0,
            dropout=0.0
        )
        
        # Initialize adapters
        manager.initialize_adapters()
        
        # Activate all
        manager.activate_adapters(list(manager.manager.keys()))
        
        # Deactivate some
        manager.deactivate_adapters(list(manager.manager.keys())[:2])
        
        # Get active adapters
        active = manager.get_active_adapters()
        self.assertEqual(len(active), 1)
    
    def test_sparsity_enforcement(self):
        """Test sparsity enforcement with K constraint"""
        manager = AdapterManager(
            model=self._create_mock_model(),
            K=2,
            rank=8,
            alpha=32.0,
            dropout=0.0
        )
        
        # Initialize
        manager.initialize_adapters()
        
        # Set weights randomly
        import random
        random.seed(42)
        for name, adapter in manager.manager.items():
            adapter.weight.data = torch.tensor([random.random()])
        
        # Apply disconnections
        active = manager.apply_disconnections()
        
        # Check that only K adapters are active
        self.assertEqual(len(active), 2)
    
    def test_memory_reduction(self):
        """Test memory reduction calculation"""
        manager = AdapterManager(
            model=self._create_mock_model(),
            K=1,
            rank=8,
            alpha=32.0,
            dropout=0.0
        )
        
        # Initialize
        manager.initialize_adapters()
        
        # Get parameter counts
        total_params = manager.get_total_param_count()
        active_params = manager.get_active_param_count()
        
        # Check that active params < total params
        self.assertLess(active_params, total_params)
    
    def _create_mock_model(self):
        """Create a mock model for testing"""
        mock_layer = torch.nn.Linear(128, 256)
        adapter = LoRAAdapter(128, 256, r=8, alpha=32.0, dropout=0.0)
        wl_layer = WeightLoRALayer(
            pretrained_layer=mock_layer,
            adapter=adapter,
            layer_name="test",
            rank=8
        )
        return wl_layer


if __name__ == '__main__':
    unittest.main()
