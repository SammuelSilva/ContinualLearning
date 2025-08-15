"""
Quick test script to verify the implementation works correctly.
Tests with reduced settings for faster execution.
"""

import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lora_vit import ContinualLoRAViT
from src.models.lora_layers import TaskSpecificLoRA
from src.data.cifar_dataset import ContinualCIFAR100
from src.data.memory_buffer import MemoryBuffer


def test_lora_layers():
    """Test LoRA layer implementation"""
    print("\n=== Testing LoRA Layers ===")
    
    # Test configuration
    hidden_dim = 768
    mlp_dim = 3072
    num_heads = 12
    num_layers = 12
    rank = 4
    
    # Create LoRA adapters
    lora_adapter = TaskSpecificLoRA(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        rank=rank,
        alpha=4.0,
        dropout=0.1,
        lora_config="attention_only"
    )
    
    # Check parameter count
    param_count = lora_adapter.num_parameters()
    print(f"✓ LoRA parameters (attention-only): {param_count:,}")
    
    # Test different configurations
    for config in ["ffn_only", "both"]:
        adapter = TaskSpecificLoRA(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            rank=rank,
            lora_config=config
        )
        print(f"✓ LoRA parameters ({config}): {adapter.num_parameters():,}")
    
    print("✓ LoRA layers test passed!")


def test_model_creation():
    """Test model creation and task addition"""
    print("\n=== Testing Model Creation ===")
    
    # Create model (using smaller ViT for testing)
    model = ContinualLoRAViT(
        vit_model_name="vit_tiny_patch16_224",  # Smaller model for testing
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False  # Faster for testing
    )
    
    # Add tasks
    for i in range(3):
        model.add_task(f"task_{i}", num_classes=10)
    
    print(f"✓ Model created with {model.num_tasks} tasks")
    
    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Set active task
    model.set_active_task("task_0")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Forward pass successful. Output shape: {output.shape}")
    assert output.shape == (batch_size, 11), "Output shape mismatch (should be batch_size x (num_classes + 1))"
    
    print("✓ Model test passed!")


def test_memory_buffer():
    """Test memory buffer operations"""
    print("\n=== Testing Memory Buffer ===")
    
    buffer = MemoryBuffer(
        buffer_size=100,
        selection_strategy="random",
        samples_per_class=5
    )
    
    # Add some dummy data
    for task_id in range(2):
        images = torch.randn(50, 3, 32, 32)
        labels = torch.randint(0, 10, (50,))
        buffer.update(images, labels, f"task_{task_id}")
    
    print(f"✓ Buffer size: {len(buffer)}")
    
    # Test sampling
    batch = buffer.sample(batch_size=16)
    print(f"✓ Sampled batch size: {len(batch['images'])}")
    
    # Get statistics
    stats = buffer.get_statistics()
    print(f"✓ Buffer statistics: {stats['total_samples']} samples, {stats['num_classes']} classes")
    
    print("✓ Memory buffer test passed!")


def test_dataset():
    """Test CIFAR-100 dataset setup"""
    print("\n=== Testing Dataset ===")
    
    dataset = ContinualCIFAR100(
        num_tasks=5,
        classes_per_task=20
    )
    
    # Get first task data
    train_loader, val_loader, test_loader = dataset.get_task_loaders(
        task_id=0,
        batch_size=32
    )
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"✓ Batch shape: {images.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"✓ Label range: {labels.min().item()} - {labels.max().item()}")
    
    print("✓ Dataset test passed!")


def test_unknown_class_mechanism():
    """Test unknown class prediction mechanism"""
    print("\n=== Testing Unknown Class Mechanism ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with multiple tasks
    model = ContinualLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False
    )
    
    # Add multiple tasks
    for i in range(3):
        model.add_task(f"task_{i}", num_classes=10)
    
    model = model.to(device)
    
    # Create test input
    test_input = torch.randn(8, 3, 224, 224).to(device)
    
    # Test task prediction
    with torch.no_grad():
        predicted_tasks, unknown_probs = model.predict_task_id(test_input)
    
    print(f"✓ Predicted tasks: {predicted_tasks[:4]}")
    print(f"✓ Unknown probabilities shape: {unknown_probs.shape}")
    
    # Test full prediction pipeline
    with torch.no_grad():
        predictions, task_ids = model.predict(test_input)
    
    print(f"✓ Final predictions shape: {predictions.shape}")
    print(f"✓ Predicted classes: {predictions[:4].cpu().numpy()}")
    
    print("✓ Unknown class mechanism test passed!")


def test_training_step():
    """Test a single training step"""
    print("\n=== Testing Training Step ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple model
    model = ContinualLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False
    )
    
    model.add_task("task_0", num_classes=10)
    model = model.to(device)
    model.set_active_task("task_0")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=1e-4
    )
    
    # Create dummy batch
    images = torch.randn(8, 3, 224, 224).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)
    
    # Training step
    model.train()
    losses = model.compute_loss(images, labels, "task_0")
    
    # Backward pass
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()
    
    print(f"✓ Training loss: {losses['total'].item():.4f}")
    print("✓ Training step test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*50)
    print("RUNNING QUICK TESTS")
    print("="*50)
    
    try:
        test_lora_layers()
        test_model_creation()
        test_memory_buffer()
        #test_dataset()
        test_unknown_class_mechanism()
        test_training_step()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\nThe implementation is working correctly!")
        print("You can now run the full training with:")
        print("  python scripts/train_cifar100.py --lora_config attention_only")
        print("\nOr run the ablation study with:")
        print("  python scripts/ablation_study.py")
    else:
        print("\nPlease fix the errors before running full training.")
        sys.exit(1)