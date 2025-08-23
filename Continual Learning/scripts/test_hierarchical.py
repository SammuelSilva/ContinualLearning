"""
Comprehensive test script for Hierarchical LoRA-ViT implementation.
Tests both standard and hierarchical modes with all features.
"""

import os
import sys
import torch
import numpy as np
import time
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all modules we need to test
from src.models.lora_vit import ContinualLoRAViT
from src.models.hierarchical_lora import HierarchicalLoRAViT, OrthogonalMergedBlock
from src.models.lora_layers import TaskSpecificLoRA, LoRALayer
from src.models.orthogonal_utils import (
    OrthogonalLoRAMerger, 
    OrthogonalProjector,
    compute_orthogonality_score,
    CombinedOrthogonalAdapter,
    BlockDiagonalAdapter
)
from src.models.merge_strategies import (
    AdaptiveMergeStrategy,
    ProgressiveMergeScheduler,
    TaskSimilarityAnalyzer
)
from src.data.cifar_dataset import ContinualCIFAR100
from src.data.memory_buffer import MemoryBuffer
from src.trainers.hierarchical_trainer import HierarchicalTrainer
from src.utils.visualization import HierarchicalVisualizer, MetricsAnimator


# ============== Basic Component Tests ==============

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


def test_standard_model_creation():
    """Test standard model creation and task addition"""
    print("\n=== Testing Standard Model Creation ===")
    
    # Create model (using smaller ViT for testing)
    model = ContinualLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False
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
    assert output.shape == (batch_size, 11), "Output shape mismatch"
    
    print("✓ Standard model test passed!")


# ============== Hierarchical Component Tests ==============

def test_hierarchical_model_creation():
    """Test hierarchical model creation with merging"""
    print("\n=== Testing Hierarchical Model Creation ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create hierarchical model
    model = HierarchicalLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False,
        tasks_per_block=3,
        use_orthogonal_merge=True
    )
    
    model = model.to(device)
    
    # Add tasks to trigger merging
    print("Adding tasks to test hierarchical structure...")
    for i in range(7):  # This should create 2 merged blocks + 1 active task
        model.add_task(f"task_{i}", num_classes=10)
        print(f"  Added task_{i}")
    
    # Check hierarchical structure
    stats = model.get_statistics()
    print(f"\n✓ Hierarchical structure created:")
    print(f"  - Merged blocks: {stats['num_merged_blocks']}")
    print(f"  - Tasks in active block: {stats['tasks_in_active_block']}")
    print(f"  - Total tasks: {stats['total_tasks']}")
    
    assert stats['num_merged_blocks'] == 2, "Should have 2 merged blocks"
    assert stats['tasks_in_active_block'] == 1, "Should have 1 task in active block"
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test task in merged block
    with torch.no_grad():
        output = model(dummy_input, task_id="task_0")
        print(f"✓ Forward pass for merged task successful. Shape: {output.shape}")
    
    # Test task in active block
    with torch.no_grad():
        output = model(dummy_input, task_id="task_6")
        print(f"✓ Forward pass for active task successful. Shape: {output.shape}")
    
    # Visualize hierarchy
    model.visualize_hierarchy()
    
    print("✓ Hierarchical model test passed!")


def test_orthogonal_merging():
    """Test orthogonal merging utilities"""
    print("\n=== Testing Orthogonal Merging ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create some LoRA layers with intentional overlap (not random)
    lora_layers = []
    base_matrix = torch.randn(4, 768)  # Base matrix to create overlap
    
    for i in range(3):
        lora = LoRALayer(
            in_features=768,
            out_features=768,
            rank=4,
            alpha=4.0,
            dropout=0.1
        )
        
        # Create overlapping A matrices by adding noise to a base
        # This ensures they're not already orthogonal
        lora.lora_A.data = base_matrix + 0.3 * torch.randn(4, 768)
        lora.lora_B.data = torch.randn(768, 4)
        
        lora_layers.append(lora)
    
    # Test orthogonality score before merging
    ortho_score_before = compute_orthogonality_score(lora_layers)
    print(f"✓ Orthogonality score before merging: {ortho_score_before:.3f}")
    
    # Test QR orthogonalization
    projector = OrthogonalProjector()
    orthogonal_loras = projector.qr_orthogonalization(lora_layers)
    ortho_score_after = compute_orthogonality_score(orthogonal_loras)
    print(f"✓ Orthogonality score after QR: {ortho_score_after:.3f}")
    
    # Check improvement or maintenance of orthogonality
    if ortho_score_before < 0.95:  # If there was room for improvement
        assert ortho_score_after >= ortho_score_before, "QR should improve or maintain orthogonality"
        if ortho_score_after > ortho_score_before:
            print(f"✓ Orthogonality improved by {ortho_score_after - ortho_score_before:.3f}")
    else:  # Already highly orthogonal
        assert ortho_score_after >= 0.95, "QR should maintain high orthogonality"
        print(f"✓ Maintained high orthogonality")
    
    # Test that orthogonalized adapters still work
    test_input = torch.randn(4, 768).to(device)
    for lora in orthogonal_loras:
        output = lora(test_input)
        assert output.shape == (4, 768), f"Output shape mismatch: {output.shape}"
    print(f"✓ Orthogonalized adapters produce correct output shapes")
    
    # Test SVD merging
    merged_lora = projector.svd_merge(lora_layers, target_rank=8)
    print(f"✓ SVD merge successful. Merged rank: {merged_lora.rank}")
    
    # Test that merged adapter works
    merged_output = merged_lora(test_input)
    assert merged_output.shape == (4, 768), f"Merged output shape mismatch"
    print(f"✓ Merged adapter produces correct output shape")
    
    # Test merger with different strategies
    merger = OrthogonalLoRAMerger(merge_strategy="qr")
    lora_dict = {f"task_{i}": lora for i, lora in enumerate(lora_layers)}
    merged_adapter = merger.merge_adapters(lora_dict)
    print(f"✓ QR merger successful")
    
    # Test block diagonal adapter
    block_diagonal = BlockDiagonalAdapter(lora_layers)
    test_input = torch.randn(4, 768).to(device)
    output = block_diagonal(test_input)
    print(f"✓ Block diagonal adapter working. Output shape: {output.shape}")
    
    # Additional test: verify orthogonality mathematically
    if len(orthogonal_loras) >= 2:
        # Check that dot products between different adapters are close to zero
        A1 = orthogonal_loras[0].lora_A.data
        A2 = orthogonal_loras[1].lora_A.data
        
        # Compute row-wise dot products
        dot_products = []
        for i in range(A1.shape[0]):
            for j in range(A2.shape[0]):
                dot_prod = torch.dot(A1[i], A2[j]).abs().item()
                dot_products.append(dot_prod)
        
        avg_dot_product = sum(dot_products) / len(dot_products)
        print(f"✓ Average absolute dot product between adapters: {avg_dot_product:.6f}")
        
        # Should be close to zero for orthogonal vectors
        if ortho_score_after > 0.95:
            assert avg_dot_product < 0.1, "Orthogonal adapters should have small dot products"
    
    print("✓ Orthogonal merging test passed!")


def test_hierarchical_task_prediction():
    """Test hierarchical task prediction with dual unknown"""
    print("\n=== Testing Hierarchical Task Prediction ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with multiple blocks
    model = HierarchicalLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False,
        tasks_per_block=2,
        use_orthogonal_merge=True
    )
    
    model = model.to(device)
    
    # Add tasks to create multiple blocks
    for i in range(5):
        model.add_task(f"task_{i}", num_classes=10)
    
    # Test input
    test_input = torch.randn(8, 3, 224, 224).to(device)
    
    # Test hierarchical prediction
    with torch.no_grad():
        predicted_tasks, confidences = model.predict_task_id_hierarchical(test_input)
    
    print(f"✓ Predicted tasks: {predicted_tasks[:4]}")
    print(f"✓ Task confidences: {confidences[:4].cpu().numpy()}")
    
    assert len(predicted_tasks) == 8, "Should predict for all samples"
    assert all(task in [f"task_{i}" for i in range(5)] for task in predicted_tasks)
    
    print("✓ Hierarchical task prediction test passed!")


def test_merge_strategies():
    """Test adaptive merge strategies"""
    print("\n=== Testing Merge Strategies ===")
    
    # Create adaptive strategy
    adaptive = AdaptiveMergeStrategy(
        similarity_threshold=0.7,
        min_tasks_for_merge=2,
        max_tasks_for_merge=5
    )
    
    # Create dummy adapters
    adapters = {}
    for i in range(4):
        adapters[f"task_{i}"] = LoRALayer(768, 768, rank=4)
    
    # Test merge decision
    performances = {f"task_{i}": 0.85 + np.random.random() * 0.1 for i in range(4)}
    should_merge = adaptive.should_merge(adapters, performances)
    print(f"✓ Should merge decision: {should_merge}")
    
    # Test strategy selection
    strategy = adaptive.select_merge_strategy(adapters, memory_constraint=None)
    print(f"✓ Selected merge strategy: {strategy}")
    assert strategy in ['qr', 'svd', 'gram_schmidt', 'blockwise']
    
    # Test progressive scheduler
    scheduler = ProgressiveMergeScheduler(
        initial_block_size=3,
        growth_factor=1.5,
        max_block_size=10
    )
    
    next_merge = scheduler.get_next_merge_point(0)
    print(f"✓ Next merge point: task {next_merge}")
    assert next_merge == 3, "First merge should be at initial_block_size"
    
    # Test task similarity analyzer
    analyzer = TaskSimilarityAnalyzer()
    
    # Create dummy features
    task_features = {}
    for i in range(4):
        features = torch.randn(100, 768)  # 100 samples, 768 dims
        task_features[f"task_{i}"] = features
    
    similarity = analyzer.compute_feature_similarity(
        task_features["task_0"],
        task_features["task_1"]
    )
    print(f"✓ Task similarity: {similarity:.3f}")
    
    print("✓ Merge strategies test passed!")


def test_memory_buffer():
    """Test memory buffer operations"""
    print("\n=== Testing Memory Buffer ===")
    
    buffer = MemoryBuffer(
        buffer_size=100,
        selection_strategy="herding",
        samples_per_class=5
    )
    
    # Add some dummy data with features for herding
    for task_id in range(2):
        images = torch.randn(50, 3, 32, 32)
        labels = torch.randint(0, 10, (50,))
        features = torch.randn(50, 768)  # Add features for herding
        buffer.update(images, labels, f"task_{task_id}", features)
    
    print(f"✓ Buffer size: {len(buffer)}")
    
    # Test sampling
    batch = buffer.sample(batch_size=16)
    print(f"✓ Sampled batch size: {len(batch['images'])}")
    
    # Test exclusion
    batch_excluded = buffer.sample(batch_size=16, exclude_task="task_0")
    print(f"✓ Sampled with exclusion: {len(batch_excluded['images'])}")
    
    # Get statistics
    stats = buffer.get_statistics()
    print(f"✓ Buffer statistics: {stats['total_samples']} samples, {stats['num_classes']} classes")
    
    print("✓ Memory buffer test passed!")


def test_hierarchical_trainer():
    """Test hierarchical trainer with dual unknown loss"""
    print("\n=== Testing Hierarchical Trainer ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model and components
    model = HierarchicalLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False,
        tasks_per_block=3,
        use_orthogonal_merge=True
    )
    model = model.to(device)
    
    memory_buffer = MemoryBuffer(
        buffer_size=100,
        selection_strategy="random",
        samples_per_class=5
    )
    
    trainer = HierarchicalTrainer(
        model=model,
        memory_buffer=memory_buffer,
        device=device,
        learning_rate=1e-4,
        weight_decay=0.01,
        lambda_task_unknown=0.5,
        lambda_block_unknown=0.3,
        save_dir="./test_checkpoints",
        num_tasks=5
    )
    
    # Add a task
    model.add_task("task_0", num_classes=10)
    model.set_active_task("task_0")
    
    # Create dummy batch
    images = torch.randn(8, 3, 224, 224).to(device)
    labels = torch.randint(0, 10, (8,)).to(device)
    
    # Test hierarchical loss computation
    losses = trainer.compute_hierarchical_loss(
        images, labels, "task_0", memory_batch=None
    )
    
    print(f"✓ Hierarchical loss computed:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}: {value.item():.4f}")
    
    assert 'classification' in losses
    assert 'total' in losses
    
    print("✓ Hierarchical trainer test passed!")


def test_visualization():
    """Test visualization components"""
    print("\n=== Testing Visualization ===")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer = HierarchicalVisualizer(save_dir=tmpdir)
        
        # Test hierarchy tree plot
        hierarchy_stats = {
            'blocks': [
                {'block_id': 0, 'num_tasks': 3, 'task_ids': ['task_0', 'task_1', 'task_2']},
                {'block_id': 1, 'num_tasks': 3, 'task_ids': ['task_3', 'task_4', 'task_5']}
            ],
            'active_block': {'block_id': 2, 'task_ids': ['task_6']},
            'num_merged_blocks': 2,
            'tasks_in_active_block': 1,
            'total_tasks': 7
        }
        
        tree_path = os.path.join(tmpdir, 'hierarchy.html')
        visualizer.plot_hierarchy_tree(hierarchy_stats, save_path=tree_path)
        assert os.path.exists(tree_path), "Hierarchy tree should be created"
        print(f"✓ Hierarchy tree created at {tree_path}")
        
        # Test memory efficiency plot
        mem_path = os.path.join(tmpdir, 'memory.png')
        visualizer.plot_memory_efficiency(
            num_tasks=20,
            tasks_per_block=5,
            lora_rank=4,
            save_path=mem_path
        )
        assert os.path.exists(mem_path), "Memory plot should be created"
        print(f"✓ Memory efficiency plot created at {mem_path}")
        
        # Test orthogonality matrix
        adapters = {}
        for i in range(3):
            adapters[f"task_{i}"] = LoRALayer(768, 768, rank=4)
        
        ortho_path = os.path.join(tmpdir, 'orthogonality.png')
        visualizer.plot_orthogonality_matrix(adapters, save_path=ortho_path)
        assert os.path.exists(ortho_path), "Orthogonality matrix should be created"
        print(f"✓ Orthogonality matrix created at {ortho_path}")
        
    print("✓ Visualization test passed!")

def test_full_training_step():
    """Test a complete training step with hierarchical model"""
    print("\n=== Testing Full Training Step ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create hierarchical model
    model = HierarchicalLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False,
        tasks_per_block=2,
        use_orthogonal_merge=True
    )
    
    model = model.to(device)
    
    # Add tasks to create a merged block
    for i in range(3):
        model.add_task(f"task_{i}", num_classes=10)
    
    # Test training on task in merged block
    model.set_active_task("task_0")
    
    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    print(f"  Trainable parameters for merged task: {len(trainable_params)}")
    
    # Only create optimizer if we have parameters to train
    if trainable_params:
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        
        images = torch.randn(4, 3, 224, 224).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        # Forward pass
        model.train()
        losses = model.compute_loss(images, labels, "task_0")
        
        # Backward pass
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        print(f"✓ Training loss (merged block): {losses['total'].item():.4f}")
    else:
        print("  Note: Task in merged block has no trainable parameters (frozen)")
    
    # Test training on active task (not in merged block)
    model.set_active_task("task_2")
    trainable_params = model.get_trainable_parameters()
    print(f"  Trainable parameters for active task: {len(trainable_params)}")
    
    if trainable_params:
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        
        images = torch.randn(4, 3, 224, 224).to(device)
        labels = torch.randint(0, 10, (4,)).to(device)
        
        losses = model.compute_loss(images, labels, "task_2")
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        print(f"✓ Training loss (active task): {losses['total'].item():.4f}")
    
    print("✓ Full training step test passed!")

def test_performance_benchmark():
    """Benchmark performance of hierarchical vs standard"""
    print("\n=== Performance Benchmark ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_iterations = 10
    
    # Standard model
    standard_model = ContinualLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False
    ).to(device)
    
    # Hierarchical model
    hierarchical_model = HierarchicalLoRAViT(
        vit_model_name="vit_tiny_patch16_224",
        lora_rank=4,
        lora_alpha=4.0,
        lora_dropout=0.1,
        lora_config="attention_only",
        use_pretrained=False,
        tasks_per_block=5,
        use_orthogonal_merge=True
    ).to(device)
    
    # Add same number of tasks
    for i in range(10):
        standard_model.add_task(f"task_{i}", num_classes=10)
        hierarchical_model.add_task(f"task_{i}", num_classes=10)
    
    # Benchmark standard model
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    standard_model.set_active_task("task_5")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = standard_model(dummy_input)
    standard_time = (time.time() - start_time) / num_iterations
    
    # Benchmark hierarchical model
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = hierarchical_model(dummy_input, task_id="task_5")
    hierarchical_time = (time.time() - start_time) / num_iterations
    
    print(f"✓ Standard model time: {standard_time*1000:.2f}ms")
    print(f"✓ Hierarchical model time: {hierarchical_time*1000:.2f}ms")
    print(f"✓ Speedup: {standard_time/hierarchical_time:.2f}x")
    
    # Memory comparison
    standard_params = sum(p.numel() for p in standard_model.parameters())
    hierarchical_params = sum(p.numel() for p in hierarchical_model.parameters())
    
    print(f"✓ Standard model params: {standard_params:,}")
    print(f"✓ Hierarchical model params: {hierarchical_params:,}")
    print(f"✓ Memory reduction: {(1 - hierarchical_params/standard_params)*100:.1f}%")
    
    print("✓ Performance benchmark passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE HIERARCHICAL LORA TESTS")
    print("="*60)
    
    test_suite = [
        # Basic tests
        ("LoRA Layers", test_lora_layers),
        ("Standard Model", test_standard_model_creation),
        ("Memory Buffer", test_memory_buffer),
        
        # Hierarchical tests
        ("Hierarchical Model", test_hierarchical_model_creation),
        ("Orthogonal Merging", test_orthogonal_merging),
        ("Hierarchical Task Prediction", test_hierarchical_task_prediction),
        ("Merge Strategies", test_merge_strategies),
        ("Hierarchical Trainer", test_hierarchical_trainer),
        
        # Integration tests
        ("Visualization", test_visualization),
        ("Full Training Step", test_full_training_step),
        ("Performance Benchmark", test_performance_benchmark),
    ]
    
    failed_tests = []
    
    for test_name, test_func in test_suite:
        try:
            test_func()
            print(f"✅ {test_name} test passed")
        except Exception as e:
            print(f"❌ {test_name} test failed: {str(e)}")
            failed_tests.append((test_name, e))
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    
    if not failed_tests:
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nThe hierarchical implementation is working correctly!")
        print("\nYou can now run experiments with:")
        print("  python scripts/train_hierarchical.py --use_hierarchical --use_orthogonal_merge")
        print("\nOr run comparative experiments:")
        print("  python scripts/train_hierarchical.py --ablation_mode different_blocks")
        return True
    else:
        print(f"❌ {len(failed_tests)} TESTS FAILED:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {str(error)[:50]}...")
        print("="*60)
        print("\nPlease fix the errors before running full training.")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)