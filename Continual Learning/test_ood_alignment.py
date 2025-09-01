#!/usr/bin/env python3
"""
Quick test to verify the OOD alignment implementation
"""

import torch
import sys
import os

# Add project root to path
sys.path.append('/home/sammuel/cloudwalk/others/ContinualLearning/Continual Learning')

from src.trainers.hierarchical_trainer import HierarchicalTrainer
from src.data.memory_buffer import MemoryBuffer

def test_ood_alignment_interface():
    """Test that the OOD alignment methods exist and have correct signatures"""
    
    print("Testing OOD Alignment Interface...")
    
    # Create dummy trainer (no need for real model for interface test)
    trainer = HierarchicalTrainer(
        model=None,
        memory_buffer=None,
        ood_alignment_lr=1e-5,
        ood_alignment_epochs=3
    )
    
    # Test 1: Check initialization parameters
    assert hasattr(trainer, 'ood_alignment_lr'), "Missing ood_alignment_lr parameter"
    assert hasattr(trainer, 'ood_alignment_epochs'), "Missing ood_alignment_epochs parameter"
    assert trainer.ood_alignment_lr == 1e-5, "Incorrect ood_alignment_lr value"
    assert trainer.ood_alignment_epochs == 3, "Incorrect ood_alignment_epochs value"
    
    # Test 2: Check methods exist
    assert hasattr(trainer, 'ood_alignment_phase'), "Missing ood_alignment_phase method"
    assert hasattr(trainer, 'train_task_with_ood_alignment'), "Missing train_task_with_ood_alignment method"
    assert hasattr(trainer, '_evaluate_ood_alignment'), "Missing _evaluate_ood_alignment method"
    
    print("✓ All OOD alignment interface tests passed!")

def test_memory_buffer():
    """Test memory buffer functionality needed for OOD alignment"""
    
    print("Testing Memory Buffer...")
    
    # Create memory buffer
    buffer = MemoryBuffer(buffer_size=100, samples_per_class=10)
    
    # Test empty buffer
    assert len(buffer) == 0, "Buffer should start empty"
    
    # Add some dummy data
    images = torch.randn(20, 3, 224, 224)
    labels = torch.randint(0, 5, (20,))
    
    buffer.update(images[:10], labels[:10], "task_0")
    buffer.update(images[10:], labels[10:], "task_1")
    
    assert len(buffer) == 20, "Buffer should contain 20 samples"
    
    # Test sampling
    sample_data = buffer.sample(batch_size=8)
    assert 'images' in sample_data, "Sample should contain images"
    assert 'labels' in sample_data, "Sample should contain labels"
    assert 'task_ids' in sample_data, "Sample should contain task_ids"
    
    # Test get_all_data
    all_data = buffer.get_all_data()
    assert len(all_data['images']) == 20, "Should get all 20 samples"
    assert len(all_data['task_ids']) == 20, "Should get all 20 task IDs"
    
    print("✓ Memory buffer tests passed!")

def test_ood_alignment_with_empty_buffer():
    """Test OOD alignment behavior with empty buffer"""
    
    print("Testing OOD Alignment with Empty Buffer...")
    
    # Create trainer with empty buffer
    buffer = MemoryBuffer()
    trainer = HierarchicalTrainer(
        model=None,
        memory_buffer=buffer,
        ood_alignment_lr=1e-5,
        ood_alignment_epochs=2
    )
    
    # Test OOD alignment with empty buffer
    result = trainer.ood_alignment_phase("task_0", 0)
    assert result == {}, "Should return empty dict for empty buffer"
    
    print("✓ Empty buffer test passed!")

if __name__ == "__main__":
    print("Running OOD Alignment Tests...")
    print("=" * 50)
    
    try:
        test_ood_alignment_interface()
        test_memory_buffer() 
        test_ood_alignment_with_empty_buffer()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! OOD Alignment implementation is ready.")
        print("\nUsage example:")
        print("```python")
        print("# Standard training")
        print("trainer = HierarchicalTrainer(")
        print("    model=model,")
        print("    memory_buffer=buffer,")
        print("    ood_alignment_lr=1e-5,")
        print("    ood_alignment_epochs=5")
        print(")")
        print("")
        print("# Train with OOD alignment")
        print("val_acc, ood_metrics = trainer.train_task_with_ood_alignment(")
        print("    task_id='task_1',")
        print("    train_loader=train_loader,")
        print("    val_loader=val_loader,")
        print("    task_idx=1,")
        print("    enable_ood_alignment=True")
        print(")")
        print("```")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)